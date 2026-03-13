# ============================================================
# model.py — Core Semantic Similarity + Sentiment Analysis
# ============================================================
# This file loads a pretrained Sentence Transformer model and
# a VADER Sentiment Analyzer. It computes semantic similarity
# between two sentences and adjusts the score based on
# sentiment to avoid false matches (e.g., "I love X" vs "I hate X").
# ============================================================

# Step 1: Import required libraries
from sentence_transformers import SentenceTransformer       # For generating sentence embeddings
from sklearn.metrics.pairwise import cosine_similarity      # For computing cosine similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # For sentiment analysis

# Step 2: Load the pretrained Sentence Transformer model
# 'all-MiniLM-L6-v2' converts sentences into 384-dimensional vectors
# that capture semantic meaning.
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Step 3: Load the VADER Sentiment Analyzer
# VADER (Valence Aware Dictionary and sEntiment Reasoner) detects
# whether a sentence is positive, negative, or neutral.
sentiment_analyzer = SentimentIntensityAnalyzer()


def get_sentiment(sentence):
    """
    Returns the compound sentiment score for a sentence.

    The compound score ranges from:
        -1 (most negative) to +1 (most positive)

    Example:
        "I love chocolate"  → +0.6369 (positive)
        "I hate chocolate"  → -0.5719 (negative)
    """
    scores = sentiment_analyzer.polarity_scores(sentence)
    return scores['compound']


def get_similarity(sentence1, sentence2):
    """
    Computes the semantic similarity between two sentences,
    adjusted by sentiment analysis.

    Steps:
        1. Generate embeddings for both sentences
        2. Compute cosine similarity (topic similarity)
        3. Get sentiment scores for both sentences
        4. If sentiments are opposite, reduce the similarity score

    Returns:
        dict with original_similarity, final_similarity,
        sentiment1, sentiment2, and sentiment_adjusted flag
    """

    # Step 4: Generate embeddings for both sentences
    embeddings = model.encode([sentence1, sentence2])

    # Step 5: Compute Cosine Similarity between the two embeddings
    similarity = cosine_similarity(
        [embeddings[0]],  # Embedding of sentence 1
        [embeddings[1]]   # Embedding of sentence 2
    )
    original_score = round(float(similarity[0][0]), 4)

    # Step 6: Get sentiment scores for both sentences
    sentiment1 = get_sentiment(sentence1)
    sentiment2 = get_sentiment(sentence2)

    # Step 7: Adjust similarity based on sentiment
    # If one sentence is positive and the other is negative,
    # their sentiments will multiply to a negative number.
    # In that case, reduce the similarity score significantly.
    adjusted = False
    final_score = original_score

    if sentiment1 * sentiment2 < 0:
        # Opposite sentiments detected — reduce similarity
        final_score = round(original_score * 0.2, 4)
        adjusted = True

    return {
        'original_similarity': original_score,
        'final_similarity': final_score,
        'sentiment1': round(sentiment1, 4),
        'sentiment2': round(sentiment2, 4),
        'sentiment_adjusted': adjusted
    }


# ============================================================
# Quick test — runs only when this file is executed directly
# ============================================================
if __name__ == "__main__":
    # Test 1: Opposite sentiment
    print("=" * 55)
    print("  TEST 1: Opposite Sentiment")
    print("=" * 55)
    result = get_similarity("I love chocolate", "I hate chocolate")
    print(f"Sentence 1: I love chocolate")
    print(f"Sentence 2: I hate chocolate")
    print(f"Sentiment 1: {result['sentiment1']}")
    print(f"Sentiment 2: {result['sentiment2']}")
    print(f"Original Similarity: {result['original_similarity']}")
    print(f"Sentiment Adjusted: {result['sentiment_adjusted']}")
    print(f"Final Similarity Score: {result['final_similarity']}")

    # Test 2: Same sentiment
    print()
    print("=" * 55)
    print("  TEST 2: Same Sentiment")
    print("=" * 55)
    result2 = get_similarity("I love machine learning", "I enjoy studying artificial intelligence")
    print(f"Sentence 1: I love machine learning")
    print(f"Sentence 2: I enjoy studying artificial intelligence")
    print(f"Sentiment 1: {result2['sentiment1']}")
    print(f"Sentiment 2: {result2['sentiment2']}")
    print(f"Original Similarity: {result2['original_similarity']}")
    print(f"Sentiment Adjusted: {result2['sentiment_adjusted']}")
    print(f"Final Similarity Score: {result2['final_similarity']}")
