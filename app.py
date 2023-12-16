from utils.helpers import generate_summaries, fetch_articles
from flask import Flask, request, jsonify
import concurrent.futures

app = Flask(__name__)


def get_article_urls():
    """Extracts article URLs from the request form."""
    return request.form.getlist('article_urls[]')


def validate_article_urls(article_urls):
    """Validates the presence of article URLs."""
    if not article_urls:
        return jsonify({"error": "No article URLs provided."}), 400


def process_article(article_url):
    """Processes a single article URL, fetching content and generating a summary."""
    try:
        articles_data = fetch_articles(article_url)

        if articles_data:
            summary = generate_summaries(articles_data)

            if summary:
                return summary
            else:
                return f"Error generating summary for {article_url}"
        else:
            return f"Error fetching articles from {article_url}"

    except Exception as e:
        return f"Error processing article: {e}"


def summarize():
    """Summarizes articles based on provided URLs."""
    try:
        article_urls = get_article_urls()
        validate_article_urls(article_urls)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            summaries = list(executor.map(process_article, article_urls))

        return jsonify({"summaries": summaries})

    except Exception as e:
        print(f"Error in Flask app: {e}")
        return jsonify({"error": "Internal server error."}), 500


@app.route('/summarize', methods=['POST'])
def handle_summarize_request():
    """Handles the summarize request."""
    return summarize()


if __name__ == '__main__':
    app.run(debug=True)
