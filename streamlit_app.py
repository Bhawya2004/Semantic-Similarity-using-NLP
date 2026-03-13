# ============================================================
# streamlit_app.py — Streamlit Frontend & Core Logic
# ============================================================
# This file creates a simple web interface where users can
# enter two sentences and check their semantic similarity.
# 
# Note for Deployment: 
# We import the logic directly from model.py instead of using 
# a Flask API so that this runs seamlessly on Streamlit Cloud.
# ============================================================

# Step 1: Import required libraries
import streamlit as st  # Streamlit for building the UI
from model import get_similarity  # Import our similarity function directly

# Optional: Add caching to model loading if we want to optimize, 
# but for this simple project, model.py already loads it globally.

# Step 2: Set up the page title and description
st.set_page_config(page_title="Semantic Similarity Checker", page_icon="🔍")

st.title("🔍 Semantic Similarity Checker")
st.write("Enter two sentences below and click **Check Similarity** to see how semantically similar they are.")
st.caption("This system uses **Sentence Transformers** + **Cosine Similarity** + **Sentiment Analysis** to detect true semantic meaning.")

st.divider()

# Step 3: Create text input fields for the two sentences
sentence1 = st.text_input("📝 Sentence 1", placeholder="e.g., I love chocolate")
sentence2 = st.text_input("📝 Sentence 2", placeholder="e.g., I hate chocolate")

st.write("")  # Add some spacing

# Step 4: Create the "Check Similarity" button
if st.button("🚀 Check Similarity", type="primary"):

    # Validate that both sentences are entered
    if not sentence1 or not sentence2:
        st.warning("⚠️ Please enter both sentences.")
    else:
        # Step 5: Compute similarity directly using our imported model function
        with st.spinner("Loading model and calculating similarity... (this may take a moment on first run)"):
            try:
                # Call the function from model.py directly! No API needed.
                result = get_similarity(sentence1, sentence2)

                # Step 6: Display the result
                final_score = result['final_similarity']
                original_score = result['original_similarity']
                sentiment1 = result['sentiment1']
                sentiment2 = result['sentiment2']
                adjusted = result['sentiment_adjusted']

                st.divider()
                st.subheader("📊 Result")

                # Display scores side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Original Similarity", value=f"{original_score}")
                with col2:
                    st.metric(
                        label="Final Similarity Score",
                        value=f"{final_score}",
                        delta=f"{round(final_score - original_score, 4)}" if adjusted else None
                    )

                # Progress bar for final score
                st.progress(min(max(final_score, 0.0), 1.0))

                # Sentiment details
                st.subheader("💭 Sentiment Analysis")
                col3, col4 = st.columns(2)
                with col3:
                    label1 = "Positive 😊" if sentiment1 > 0.05 else ("Negative 😠" if sentiment1 < -0.05 else "Neutral 😐")
                    st.metric(label=f"Sentence 1: {label1}", value=f"{sentiment1}")
                with col4:
                    label2 = "Positive 😊" if sentiment2 > 0.05 else ("Negative 😠" if sentiment2 < -0.05 else "Neutral 😐")
                    st.metric(label=f"Sentence 2: {label2}", value=f"{sentiment2}")

                # Interpretation
                st.divider()
                if adjusted:
                    st.warning("⚠️ **Sentiment Adjustment Applied** — The sentences have opposite sentiment, so the similarity score was reduced.")
                    st.info(f"The sentences talk about the same topic but have **opposite meaning**.")
                elif final_score >= 0.8:
                    st.success("✅ The sentences are **highly similar** in meaning and sentiment.")
                elif final_score >= 0.5:
                    st.info("ℹ️ The sentences are **moderately similar** in meaning.")
                else:
                    st.warning("⚠️ The sentences are **not very similar** in meaning.")
            
            except Exception as e:
                st.error(f"❌ An error occurred during calculation: {e}")
