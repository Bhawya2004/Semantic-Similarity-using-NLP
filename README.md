# Semantic Similarity Detection

**Using Sentence Transformers, Cosine Similarity, and Sentiment Analysis**

An NLP project that computes the true semantic similarity between two sentences by combining embeddings with sentiment analysis — preventing false matches like "I love chocolate" vs "I hate chocolate".

---

## The Problem

Traditional similarity methods consider sentences like:

| Sentence 1       | Sentence 2       | Cosine Similarity |
| ---------------- | ---------------- | ----------------- |
| I love chocolate | I hate chocolate | ~0.82 (High!)     |

This is **incorrect** — the sentences have **opposite meanings**.

## The Solution

We add **Sentiment Analysis** to the pipeline:

```
User enters two sentences
        ↓
Sentence Transformer → embeddings (384-dim vectors)
        ↓
Cosine Similarity → topic similarity score
        ↓
VADER Sentiment Analysis → detect positive/negative feelings
        ↓
If sentiments are opposite → reduce the similarity score
        ↓
Final Adjusted Similarity Score
```

---

## Project Structure

```
semantic_similarrity/
│
├── model_building.ipynb    # Step-by-step notebook with explanations
├── model.py                # Core logic (embeddings + sentiment + similarity)
├── app.py                  # Flask backend API (port 5001)
├── streamlit_app.py        # Streamlit frontend UI
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Setup & Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Flask Backend

```bash
python3 app.py
```

The API starts on `http://localhost:5001`.

### 3. Run Streamlit Frontend (in a new terminal)

```bash
streamlit run streamlit_app.py
```

### 4. Run Jupyter Notebook

```bash
jupyter notebook
```

Open `model_building.ipynb` and run all cells.

---

## API Usage

**Endpoint:** `POST /similarity`

**Request:**

```json
{
  "sentence1": "I love chocolate",
  "sentence2": "I hate chocolate"
}
```

**Response:**

```json
{
  "similarity_score": 0.1607,
  "original_similarity": 0.8036,
  "sentiment1": 0.6369,
  "sentiment2": -0.5719,
  "sentiment_adjusted": true
}
```

**cURL Example:**

```bash
curl -X POST http://localhost:5001/similarity \
  -H "Content-Type: application/json" \
  -d '{"sentence1": "I love chocolate", "sentence2": "I hate chocolate"}'
```

---

## Example Outputs

| Sentence 1            | Sentence 2             | Original | Final    | Adjusted? |
| --------------------- | ---------------------- | -------- | -------- | --------- |
| I love chocolate      | I hate chocolate       | 0.80     | **0.16** | ⚠️ Yes    |
| I love ML             | I enjoy studying AI    | 0.57     | 0.57     | No        |
| The movie was amazing | The movie was terrible | 0.78     | **0.16** | ⚠️ Yes    |
| Python is great       | Python is great        | 1.00     | 1.00     | No        |

---

## Technologies Used

| Technology                  | Purpose                                     |
| --------------------------- | ------------------------------------------- |
| Sentence Transformers       | Generate sentence embeddings                |
| Cosine Similarity (sklearn) | Measure topic similarity between embeddings |
| VADER Sentiment             | Detect positive/negative sentiment          |
| Flask                       | Backend REST API                            |
| Streamlit                   | Frontend web UI                             |

---

## Viva Preparation — Key Concepts

### What are Embeddings?

- A list of numbers (a vector) representing a sentence's meaning
- `"I love ML"` → `[0.12, -0.45, 0.78, ...]` (384 numbers)
- Similar meanings → similar embeddings

### What are Sentence Transformers?

- Pretrained deep learning models that convert sentences into fixed-size vectors
- We use `all-MiniLM-L6-v2` — lightweight, produces 384-dim embeddings
- Already trained on millions of sentence pairs

### What is Cosine Similarity?

- Measures the angle between two vectors
- Same direction → high score (close to 1)
- Perpendicular → no similarity (close to 0)
- **Limitation:** Only captures topic similarity, not emotional meaning

### What is VADER Sentiment Analysis?

- **VADER** = Valence Aware Dictionary and sEntiment Reasoner
- Detects whether a sentence is positive, negative, or neutral
- Returns a compound score from -1 (negative) to +1 (positive)
- Works well for social media text, reviews, and short sentences

### How Does the Sentiment Adjustment Work?

1. Compute cosine similarity between embeddings (topic similarity)
2. Get VADER compound sentiment scores for both sentences
3. If `sentiment1 × sentiment2 < 0` → sentiments are **opposite**
4. In that case, multiply similarity by 0.2 to **reduce** it
5. Result: sentences with opposite feelings get a low final score

### Why Not Just Use Cosine Similarity?

- Cosine similarity captures **what** a sentence talks about (topic)
- It does NOT capture **how** the sentence feels about it (sentiment)
- `"I love chocolate"` and `"I hate chocolate"` have similar embeddings because the topic is the same
- Sentiment analysis fixes this by detecting the emotional difference

### Common Viva Questions

**Q: Why use VADER instead of a deep learning sentiment model?**

> VADER is simple, fast, and doesn't require training. It works well for short sentences and is easy to explain in a project. Deep learning models would add unnecessary complexity.

**Q: What happens if both sentences are neutral?**

> If both sentiments are neutral (compound ≈ 0), their product is ≈ 0, so no adjustment is applied. The original cosine similarity score is used.

**Q: Is the 0.2 multiplier the best value?**

> It's a simple heuristic. In production, you could fine-tune this value or use a more sophisticated formula. For this project, 0.2 clearly demonstrates the concept.

**Q: What are the applications?**

> Duplicate detection (avoiding false matches), opinion mining, review comparison, plagiarism detection with meaning awareness, and chatbot response matching.

---

## License

This project is for educational purposes only.
