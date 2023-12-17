from utils.helpers import summarizer
from flask import *


app = Flask(__name__)


@app.route('/summarize', methods=['POST'])
def handle_summarize_request():
    """Handles the summarize request."""
    return summarizer.summarize()


if __name__ == '__main__':
    app.run(debug=True)
