from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import textwrap
import requests
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-3.5-turbo"
MAX_TOKENS = 4097
CONCISE_SUMMARY_TEMPLATE = """Write a concise summary of the following:\n\n{text}\n\nCONCISE SUMMARY IN ENGLISH:"""


def fetch_articles(url):
    """
    This function performs an HTTP GET request to fetch the content of a given URL and returns the response text.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching articles: {e}")
        return None


def initialize_openai_model():
    """
    Initializes and returns the ChatOpenAI model with the required parameters.
    """
    return ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name=MODEL_NAME)


def create_summary_chain(llm, articles_data):
    """
    Creates and returns a summarization chain based on the provided ChatOpenAI model and articles data.
    """
    prompt = PromptTemplate(template=CONCISE_SUMMARY_TEMPLATE, input_variables=["text"])
    articles_data_truncated = articles_data[:MAX_TOKENS]
    documents = [Document(page_content=articles_data_truncated)]

    chain_type = "stuff" if len(articles_data) < MAX_TOKENS else "map_reduce"
    return load_summarize_chain(llm, chain_type=chain_type, map_prompt=prompt, combine_prompt=prompt, verbose=True)


def generate_summaries(articles_data):
    """
    Generates summaries using the ChatOpenAI model and the provided articles data.
    """
    try:
        llm = initialize_openai_model()
        chain = create_summary_chain(llm, articles_data)
        documents = [Document(page_content=articles_data[:MAX_TOKENS])]

        summary = chain.run(documents)
        print("Summary:")
        print(textwrap.fill(summary, width=100))
        return summary

    except Exception as e:
        print(f"Error generating summaries: {e}")
        return None
