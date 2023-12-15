from utils.helpers import generate_summaries, fetch_articles
from flask import Flask, request, jsonify
import concurrent.futures

app = Flask(__name__)


@app.route('/summarize', methods=['POST'])
def summarize():
    """
    This method condenses article content into a brief summary using natural language processing or another
    summarization technique.
    """
    if request.method == 'POST':
        try:
            article_urls = request.form.getlist('article_urls[]')

            if not article_urls:
                return jsonify({"error": "No article URLs provided."}), 400

            with concurrent.futures.ThreadPoolExecutor() as executor:
                summaries = list(executor.map(process_article, article_urls))

            return jsonify({"summaries": summaries})

        except Exception as e:
            print(f"Error in Flask app: {e}")
            return jsonify({"error": "Internal server error."}), 500


def process_article(article_url):
    """
    This method fetch articles from given URLs, generate summaries using the generate_summaries function,
    and handle potential errors during the process.
    """
    try:
        articles_data = fetch_articles(article_url)

        if articles_data is not None:
            summary = generate_summaries(articles_data)

            if summary is not None:
                return summary
            else:
                return f"Error generating summary for {article_url}"
        else:
            return f"Error fetching articles from {article_url}"

    except Exception as e:
        return f"Error processing article: {e}"


if __name__ == '__main__':
    app.run(debug=True)
