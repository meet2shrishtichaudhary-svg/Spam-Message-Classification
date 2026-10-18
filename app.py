"""Streamlit Web Interface for Spam Message Classification.

Provides a modern, interactive, beginner-friendly web application for
classifying SMS messages, viewing model evaluation metrics, and exploring
dataset characteristics.
"""

from pathlib import Path
import json
import streamlit as st

# Import backend prediction and data modules
from src.predict import get_predictor, predict_message
from src.preprocess import clean_text

# Page configuration
st.set_page_config(
    page_title="SMS Spam Message Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern, polished appearance
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }

    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        color: #f8fafc;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        line-height: 1.6;
        margin-bottom: 0;
    }

    /* Card Panels */
    .custom-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }

    /* Result Badges */
    .result-badge-spam {
        background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-size: 1.35rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 10px 20px -3px rgba(239, 68, 68, 0.35);
        margin: 1rem 0;
        letter-spacing: 0.5px;
    }
    .result-badge-ham {
        background: linear-gradient(135deg, #10b981 0%, #047857 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-size: 1.35rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 10px 20px -3px rgba(16, 185, 129, 0.35);
        margin: 1rem 0;
        letter-spacing: 0.5px;
    }

    /* Metric Boxes */
    .metric-box {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.25rem;
    }

    /* Disclaimer Notice */
    .disclaimer-box {
        background: rgba(245, 158, 11, 0.1);
        border-left: 4px solid #f59e0b;
        color: #fbbf24;
        padding: 0.85rem 1rem;
        border-radius: 6px;
        font-size: 0.85rem;
        margin-top: 1.25rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "message_input" not in st.session_state:
    st.session_state["message_input"] = ""


def set_example_message(text: str):
    """Callback to populate the text area with an illustrative example."""
    st.session_state["message_input"] = text


def clear_message_input():
    """Callback to reset the text area."""
    st.session_state["message_input"] = ""


# Header Section
st.markdown("""
<div class="hero-container">
    <div class="hero-title">
        <span>🛡️</span> Spam Message Classification System
    </div>
    <p class="hero-subtitle">
        An educational Natural Language Processing (NLP) application trained on the Kaggle SMS Spam Collection.
        Detects unwanted promotional broadcasts, phishing alerts, and scams using TF-IDF feature extraction
        and Machine Learning classifiers.
    </p>
</div>
""", unsafe_allow_html=True)

# Check model artifacts availability
models_dir = Path("models")
reports_dir = Path("reports")
model_file = models_dir / "best_model.joblib"
vectorizer_file = models_dir / "tfidf_vectorizer.joblib"
metadata_file = models_dir / "model_metadata.json"
metrics_file = reports_dir / "metrics.json"
cm_image_file = reports_dir / "confusion_matrix.png"

artifacts_ready = model_file.is_file() and vectorizer_file.is_file()

# Automatic self-healing training if artifacts are missing on first cloud launch
if not artifacts_ready:
    dataset_file = Path("dataset/spam.csv")
    if dataset_file.is_file():
        with st.spinner("⏳ First launch setup: Initializing and training Spam Classifier model..."):
            try:
                from src.train import train_models
                train_models(dataset_path=dataset_file, models_dir=models_dir, reports_dir=reports_dir, verbose=False)
                artifacts_ready = model_file.is_file() and vectorizer_file.is_file()
            except Exception as e:
                st.error(f"Automatic model training failed: {e}")

# Main Application Layout: Two Columns
col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    st.subheader("✉️ Test an SMS Message")
    st.caption("Enter any message below or pick one of the quick test presets.")

    # Preset Example Buttons
    st.write("**Quick Test Presets (Illustrative Examples):**")
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("🎁 Winner Prize (Spam)", use_container_width=True):
            set_example_message("WINNER! You have won a guaranteed £1000 prize or cash reward. Call 08452810075 now to claim.")
        if st.button("💬 Friendly Chat (Ham)", use_container_width=True):
            set_example_message("Hey, are we still meeting for lunch at the cafeteria around 1:30 PM?")

    with btn_col2:
        if st.button("🚨 Phishing Security Alert (Spam)", use_container_width=True):
            set_example_message("URGENT: Your mobile account has been locked. Verify identity immediately at http://secure-update-alert.com")
        if st.button("📚 Study Group Note (Ham)", use_container_width=True):
            set_example_message("Can you please email me the machine learning notes from today's lecture when you're free?")

    # Text Area Input
    user_message = st.text_area(
        label="SMS Message Content:",
        value=st.session_state["message_input"],
        height=140,
        placeholder="Type or paste an SMS message here (e.g. 'Congratulations! You won a gift card...')"
    )

    action_col1, action_col2 = st.columns([1, 1])
    with action_col1:
        check_clicked = st.button("🔍 Check Message", type="primary", use_container_width=True)
    with action_col2:
        st.button("🗑️ Clear Input", on_click=clear_message_input, use_container_width=True)

    # Classification Execution
    if check_clicked:
        if not artifacts_ready:
            st.error(
                "⚠️ Trained model artifacts not found! Please run the training script first:\n\n"
                "`python src/train.py`"
            )
        elif not user_message.strip():
            st.warning("⚠️ Input message cannot be empty. Please enter text before checking.")
        else:
            try:
                res = predict_message(user_message.strip())

                st.markdown("### 📋 Classification Outcome")
                if res["is_spam"]:
                    st.markdown(
                        '<div class="result-badge-spam">🚨 CLASSIFIED AS: SPAM (UNWANTED)</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="result-badge-ham">✅ CLASSIFIED AS: HAM (LEGITIMATE MESSAGE)</div>',
                        unsafe_allow_html=True
                    )

                # Probabilities & Confidence
                st.write("**Estimated Model Probabilities:**")
                prob_col1, prob_col2 = st.columns(2)
                with prob_col1:
                    st.metric(
                        label="Spam Probability",
                        value=f"{res['spam_probability'] * 100:.1f}%",
                        help="Estimated probability that this message belongs to the Spam class."
                    )
                    st.progress(res["spam_probability"])
                with prob_col2:
                    st.metric(
                        label="Ham (Legit) Probability",
                        value=f"{res['ham_probability'] * 100:.1f}%",
                        help="Estimated probability that this message is legitimate."
                    )
                    st.progress(res["ham_probability"])

                # Preprocessing Traceability Accordion
                with st.expander("🔍 View Preprocessed Tokens (Feature Pipeline Inspection)"):
                    st.write("**Raw Input:**", res["raw_message"])
                    st.write("**Preprocessed String:**", f"`{res['cleaned_text']}`")
                    st.caption("Normalized via lowercase, URL & currency replacement, and whitespace standardization.")

                # Responsible Use Disclaimer
                st.markdown(
                    '<div class="disclaimer-box">'
                    '<strong>⚠️ Educational Project Disclaimer:</strong> The predicted category and percentage above '
                    'are estimated statistical model outputs, not a guarantee of correctness. False positives and '
                    'false negatives can occur. This tool is designed for educational minor project demonstrations '
                    'and should not replace production cybersecurity filters.'
                    '</div>',
                    unsafe_allow_html=True
                )

            except Exception as err:
                st.error(f"Prediction error: {err}")

with col_right:
    st.subheader("📊 Model Evaluation & Benchmarks")
    st.caption("Held-out test set performance (80% training / 20% test stratified split).")

    # Load and display saved evaluation metrics
    if metrics_file.is_file():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                saved_metrics = json.load(f)

            model_name = saved_metrics.get("model_name", "Champion Model")
            st.info(f"🏆 Active Selected Classifier: **{model_name}**")

            # Metric Cards Grid
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{saved_metrics['accuracy'] * 100:.2f}%</div>
                    <div class="metric-label">Overall Accuracy</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{saved_metrics['precision'] * 100:.2f}%</div>
                    <div class="metric-label">Spam Precision (PPV)</div>
                </div>
                """, unsafe_allow_html=True)

            with m_col2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{saved_metrics['f1_score'] * 100:.2f}%</div>
                    <div class="metric-label">Spam F1-Score (Primary)</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{saved_metrics['recall'] * 100:.2f}%</div>
                    <div class="metric-label">Spam Recall (Sensitivity)</div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            # Display Confusion Matrix Plot
            if cm_image_file.is_file():
                st.write("**Confusion Matrix on Held-Out Test Set (1,034 messages):**")
                st.image(str(cm_image_file), caption="Confusion Matrix: Multinomial Naive Bayes (Spam = 1, Ham = 0)", use_container_width=True)

            cm_data = saved_metrics.get("confusion_matrix", {})
            with st.expander("📋 Detailed Confusion Matrix Breakdown"):
                st.write(f"• **True Negatives (TN):** {cm_data.get('true_negatives_tn', 0)} (Ham correctly identified as Ham)")
                st.write(f"• **False Positives (FP):** {cm_data.get('false_positives_fp', 0)} (Ham mistakenly flagged as Spam)")
                st.write(f"• **False Negatives (FN):** {cm_data.get('false_negatives_fn', 0)} (Spam missed and delivered to inbox)")
                st.write(f"• **True Positives (TP):** {cm_data.get('true_positives_tp', 0)} (Spam correctly flagged)")

        except Exception as e:
            st.warning(f"Could not load evaluation metrics: {e}")
    else:
        st.warning(
            "Evaluation metrics not found. Run `python src/train.py` to generate the evaluation report and confusion matrix plot."
        )

# Sidebar: System Architecture & Viva Quick Reference
with st.sidebar:
    st.header("⚙️ Project Architecture")
    st.markdown("""
    **Pipeline Workflow:**
    1. **Data Loading:** Kaggle SMS collection (`v1, v2` columns normalized).
    2. **Deduplication:** Dropped 403 exact duplicates (`5,169` clean messages).
    3. **Preprocessing:** URL tokens (`httpaddr`), currency tokens (`currencysymb`), numbers (`phonenbr`), lowercase.
    4. **Feature Extraction:** TF-IDF n-grams `(1, 2)` fit strictly on training set.
    5. **Classification:** Multinomial Naive Bayes vs Logistic Regression.
    6. **Selection Metric:** Spam-class F1-Score (balances precision & recall under 87:13 imbalance).
    """)

    st.divider()
    st.header("📖 Viva / Viva-Voce Notes")
    st.markdown("""
    - **Why not Accuracy alone?**
      The dataset has 87.4% Ham. A naive classifier predicting Ham 100% of the time gets 87.4% accuracy with 0% Spam recall!
    - **Why Naive Bayes for SMS?**
      Word frequencies conditionally independent given class works remarkably well for sparse text TF-IDF matrices and trains in milliseconds.
    - **False Positive vs False Negative:**
      A False Positive (flagging a real family SMS as spam) is far worse than a False Negative (letting a spam promo through).
    """)

    st.divider()
    st.caption("B.Tech AI & ML Minor Project • Spam Message Classification")
