from utils.helpers import summarizer
import concurrent.futures
import streamlit as st
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")


def main():
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
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(summarizer.fetch_content, article_url): article_url
                           for article_url in article_urls_list}
                for i, (article_url, future) in enumerate(zip(article_urls_list,
                                                              concurrent.futures.as_completed(futures)), start=1):
                    try:
                        content = future.result()
                        if content:
                            summarizer.display_article_summary(i, content)
                            summarizer.process_user_interaction(i, content, loading_text)
                    except Exception as e:
                        st.error(f"Error processing article: {e}")

        except Exception as e:
            st.error(f"Error: {e}")

        loading_text.empty()


if __name__ == '__main__':
    main()
