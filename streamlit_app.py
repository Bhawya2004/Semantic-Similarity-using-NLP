# ============================================================
# streamlit_app.py — Streamlit Frontend UI
# ============================================================
# This file creates a simple web interface where users can
# enter two sentences and check their semantic similarity.
# The system also considers sentiment to avoid false matches.
# ============================================================

# Step 1: Import required libraries
import streamlit as st  # Streamlit for building the UI
import requests         # To send HTTP requests to the Flask API

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
        # Step 5: Send sentences to the Flask API
        try:
            with st.spinner("Calculating similarity..."):
                response = requests.post(
                    "http://localhost:5001/similarity",
                    json={
                        "sentence1": sentence1,
                        "sentence2": sentence2
                    }
                )

            # Step 6: Display the result
            if response.status_code == 200:
                result = response.json()
                final_score = result['similarity_score']
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

            else:
                st.error("❌ Error from API. Please check if the Flask server is running.")

        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to the Flask API. Make sure the Flask server is running on http://localhost:5001")
