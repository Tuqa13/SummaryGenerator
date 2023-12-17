from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from flask import jsonify, request
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import concurrent.futures
import streamlit as st
import textwrap
import requests
import logging
import openai
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-3.5-turbo"
MAX_TOKENS = 4097
CONCISE_SUMMARY_TEMPLATE = """Write a concise summary of the following:\n\n{text}\n\nCONCISE SUMMARY IN ENGLISH:"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArticleSummarizer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ArticleSummarizer, cls).__new__(cls)
            cls._instance.llm = cls._instance._initialize_openai_model()
        return cls._instance

    def _initialize_openai_model(self):
        return ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name=MODEL_NAME)

    def _get_article_urls(self):
        """Extracts article URLs from the request form."""
        return request.form.getlist('article_urls[]')

    def _validate_article_urls(self, article_urls):
        """Validates the presence of article URLs."""
        if not article_urls:
            logger.error("No article URLs provided.")
            return jsonify({"error": "No article URLs provided."}), 400

    def _process_article(self, article_url):
        """Processes a single article URL, fetching content and generating a summary."""
        try:
            articles_data = self._fetch_articles(article_url)

            if articles_data:
                summary = self._generate_summaries(articles_data)

                if summary:
                    return summary
                else:
                    logger.error(f"Error generating summary for {article_url}")
                    return f"Error generating summary for {article_url}"
            else:
                logger.error(f"Error fetching articles from {article_url}")
                return f"Error fetching articles from {article_url}"

        except Exception as e:
            logger.error(f"Error processing article: {e}")
            return f"Error processing article: {e}"

    def summarize(self):
        """Summarizes articles based on provided URLs."""
        try:
            article_urls = self._get_article_urls()
            self._validate_article_urls(article_urls)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                summaries = list(executor.map(self._process_article, article_urls))

            return jsonify({"summaries": summaries})

        except Exception as e:
            logger.error(f"Error in Flask app: {e}")
            return jsonify({"error": "Internal server error."}), 500

    def fetch_content(self, url):
        """Fetches content from a given URL and extracts text from paragraphs in the HTML content."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            contents = ' '.join([p.text.strip() for p in paragraphs if p.text.strip()])
            return contents if contents else None
        except requests.exceptions.RequestException as exp:
            logger.error(f"Error fetching articles: {exp}")
            return None

    def _chat_with_openai(self, prompt):
        """Interacts with the OpenAI API to generate a chat response based on a given prompt."""
        try:
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=prompt,
                max_tokens=150
            )
            return response.choices[0].text.strip()
        except Exception as ex:
            logger.error(f"Error in chat_with_openai: {ex}")
            return f"Error generating chat response. Details: {ex}"

    def display_article_summary(self, i, content):
        """Displays the summary of an article."""
        st.write(f"**Article {i} Summary:**")
        st.text(textwrap.fill(self._generate_summaries(content), width=150))

    def process_user_interaction(self, i, content, loading_text):
        """Handles user interaction, including asking questions and chatting with AI."""
        question_input_key = f"question_input_{i}"
        user_question = st.text_area(f"Ask a question - Article {i}:", key=question_input_key)

        chat_button_key = f"chat_button_{i}"
        chat_button_pressed = st.button("Chat with AI", key=chat_button_key)

        if user_question and chat_button_pressed:
            loading_text.text("Chatting with AI...")
            chat_prompt = f"Chat with AI about the content of Article {i}: {content[:300]}... Question: {user_question}"
            chat_response = self._chat_with_openai(chat_prompt)
            with st.expander(f"Chat with AI - Article {i}"):
                chat_response_key = f"chat_response_{i}"
                st.text_area("AI's Response:", value=chat_response, key=chat_response_key)

    @staticmethod
    def _fetch_articles(url):
        """
        This function performs an HTTP GET request to fetch the content of a given URL and returns the response text.
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching articles: {e}")
            return None

    def _create_summary_chain(self, llm, articles_data):
        """
        Creates and returns a summarization chain based on the provided ChatOpenAI model and articles data.
        """
        prompt = PromptTemplate(template=CONCISE_SUMMARY_TEMPLATE, input_variables=["text"])
        articles_data_truncated = articles_data[:MAX_TOKENS]
        documents = [Document(page_content=articles_data_truncated)]

        chain_type = "stuff" if len(articles_data) < MAX_TOKENS else "map_reduce"
        return load_summarize_chain(llm, chain_type=chain_type, map_prompt=prompt, combine_prompt=prompt, verbose=True)

    def _generate_summaries(self, articles_data):
        """
        Generates summaries using the ChatOpenAI model and the provided articles data.
        """
        try:
            llm = self._initialize_openai_model()
            chain = self._create_summary_chain(llm, articles_data)
            documents = [Document(page_content=articles_data[:MAX_TOKENS])]

            summary = chain.run(documents)
            logger.info("Summary generated successfully.")
            print(textwrap.fill(summary, width=100))
            return summary

        except Exception as e:
            logger.error(f"Error generating summaries: {e}")
            return None


summarizer = ArticleSummarizer()
