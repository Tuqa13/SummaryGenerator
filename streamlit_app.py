from utils.helpers import generate_summaries
from bs4 import BeautifulSoup
import concurrent.futures
import streamlit as st
import requests
import textwrap
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")


def fetch_and_process_content(my_url):
    """This method fetches content from a given URL, extracts and joins text from
    paragraphs in the HTML content, and returns the processed text."""
    try:
        response = requests.get(my_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        contents = ' '.join([p.text.strip() for p in paragraphs if p.text.strip()])
        return contents if contents else None
    except requests.exceptions.RequestException as exp:
        st.error(f"Error fetching articles: {exp}")
        return None


def chat_with_openai(prompt):
    """This  method interacts with the OpenAI API to generate a chat response based
    on a given prompt. It sends the prompt to the OpenAI API, receives a response,
    and returns the stripped text of the model's reply."""
    try:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as ex:
        st.error(f"Error in chat_with_openai: {ex}")
        return f"Error generating chat response. Details: {ex}"


st.set_page_config(
    page_title="Article Summarizer with Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("Article Summarizer with Chatbot")

article_urls_list = []
num_urls = st.number_input("Number of URLs", value=1, min_value=1)
for i in range(num_urls):
    url = st.text_input(f"Enter URL {i + 1}:")
    article_urls_list.append(url)

loading_text = st.empty()
summarize_button = st.button("Summarize and Chat", key="summarize_button")
if 'state' not in st.session_state:
    st.session_state.state = {}

with st.spinner("Summarizing and Chatting..."):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(fetch_and_process_content, article_url): article_url
                   for article_url in article_urls_list}
        for i, (article_url, future) in enumerate(zip(article_urls_list, concurrent.futures.as_completed(futures)),
                                                  start=1):
            try:
                content = future.result()
                if content:
                    st.write(f"**Article {i} Summary:**")
                    st.text(textwrap.fill(generate_summaries(content), width=150))
                    question_input_key = f"question_input_{i}"
                    user_question = st.text_area(f"Ask a question - Article {i}:", key=question_input_key)
                    chat_button_key = f"chat_button_{i}"
                    chat_button_pressed = st.button("Chat with AI", key=chat_button_key)
                    if user_question and chat_button_pressed:
                        loading_text.text("Chatting with AI...")
                        chat_prompt = f"Chat with AI about the content of Article {i}: {content[:300]}... Question: " \
                                      f"{user_question}"
                        chat_response = chat_with_openai(chat_prompt)
                        with st.expander(f"Chat with AI - Article {i}"):
                            chat_response_key = f"chat_response_{i}"
                            st.text_area("AI's Response:", value=chat_response, key=chat_response_key)
            except Exception as e:
                st.error(f"Error processing article: {e}")

loading_text.empty()
