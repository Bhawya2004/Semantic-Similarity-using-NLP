# ============================================================
# app.py — Flask Backend API
# ============================================================
# This file creates a simple Flask web server with one endpoint
# that accepts two sentences and returns their similarity score
# (adjusted by sentiment analysis).
# ============================================================

# Step 1: Import required libraries
from flask import Flask, request, jsonify  # Flask for building the API
from flask_cors import CORS                # CORS to allow requests from Streamlit
from model import get_similarity           # Import our similarity function

# Step 2: Create the Flask application
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing


# Step 3: Define the /similarity endpoint
@app.route('/similarity', methods=['POST'])
def similarity():
    """
    API endpoint that receives two sentences and returns their
    similarity score (adjusted by sentiment analysis).

    Expected JSON input:
    {
        "sentence1": "first sentence here",
        "sentence2": "second sentence here"
    }

    JSON output:
    {
        "similarity_score": 0.16,
        "original_similarity": 0.82,
        "sentiment1": 0.6369,
        "sentiment2": -0.5719,
        "sentiment_adjusted": true
    }
    """

    # Get the JSON data from the request
    data = request.get_json()

    # Extract the two sentences
    sentence1 = data.get('sentence1', '')
    sentence2 = data.get('sentence2', '')

    # Check if both sentences are provided
    if not sentence1 or not sentence2:
        return jsonify({'error': 'Please provide both sentence1 and sentence2'}), 400

    # Compute similarity using our model function
    result = get_similarity(sentence1, sentence2)

    # Return the result as JSON
    return jsonify({
        'similarity_score': result['final_similarity'],
        'original_similarity': result['original_similarity'],
        'sentiment1': result['sentiment1'],
        'sentiment2': result['sentiment2'],
        'sentiment_adjusted': result['sentiment_adjusted']
    })


# Step 4: Run the Flask server
if __name__ == '__main__':
    print("Starting Flask server on http://localhost:5001")
    app.run(debug=True, port=5001)
