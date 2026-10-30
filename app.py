
from pathlib import Path
import pandas as pd
import streamlit as st
import joblib

ROOT = Path(__file__).resolve().parent

# Correct model path: model file is in the root folder
MODEL_PATH = ROOT / "email_classifier.joblib"
REPORTS = ROOT / "reports"

st.set_page_config(
    page_title="Email Classification AI",
    page_icon="📧",
    layout="wide"
)

st.title("📧 AI-Based Email Classification System")
st.caption("NLP + TF-IDF + Machine Learning")

# Load trained model
if not MODEL_PATH.exists():
    st.error("Trained model file not found: email_classifier.joblib")
    st.stop()

try:
    bundle = joblib.load(MODEL_PATH)
    model = bundle["pipeline"]
    categories = bundle["categories"]
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()

# Sidebar
st.sidebar.header("📌 Project Information")
st.sidebar.write(
    "This application uses NLP and Machine Learning "
    "to classify customer support tickets."
)
st.sidebar.success("Actual 4,000-row dataset used for training.")
st.sidebar.write("Target column: queue")
st.sidebar.write("Input columns: subject and body")

st.sidebar.subheader("Available Categories")
for category in categories:
    st.sidebar.write(f"• {category}")

st.sidebar.divider()
st.sidebar.subheader("Selected Model")
st.sidebar.write(bundle.get("model_name", "Trained classifier"))

# Main interface
left, right = st.columns([3, 2])

with left:
    st.subheader("📩 Test an Email")

    subject = st.text_input(
        "Email Subject",
        placeholder="Example: Unable to access my account"
    )

    body = st.text_area(
        "Email Body",
        height=200,
        placeholder="Enter your customer support message..."
    )

    predict = st.button(
        "Classify Email",
        type="primary",
        use_container_width=True
    )

with right:
    st.subheader("Available Categories")
    for category in categories:
        st.write(f"• {category}")

    st.subheader("Model Information")
    st.info(bundle.get("model_name", "Trained classifier"))

# Prediction history
if "history" not in st.session_state:
    st.session_state.history = []

if predict:
    text = (subject.strip() + " " + body.strip()).strip()

    if not text:
        st.error("Please enter an email subject or body.")

    else:
        try:
            prediction = model.predict([text])[0]

            confidence = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba([text])[0]
                confidence = float(max(probabilities))

            st.divider()
            st.subheader("🎯 Prediction Result")

            st.success(f"Predicted Category: {prediction}")

            if confidence is not None:
                st.metric(
                    "Model Confidence",
                    f"{confidence * 100:.2f}%"
                )

                st.caption(
                    "Confidence is a model score, "
                    "not a guarantee of correctness."
                )

            st.session_state.history.insert(
                0,
                {
                    "Subject": subject,
                    "Predicted Category": prediction,
                    "Confidence": (
                        f"{confidence * 100:.2f}%"
                        if confidence is not None
                        else "Not available"
                    )
                }
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

# Prediction history
st.divider()
st.subheader("📋 Prediction History")

if st.session_state.history:

    st.dataframe(
        pd.DataFrame(st.session_state.history),
        use_container_width=True,
        hide_index=True
    )

    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

else:
    st.info("Your predictions will appear here.")

# Training reports
st.divider()

with st.expander("📊 View Training Reports"):

    comparison = REPORTS / "model_comparison.csv"
    analysis = REPORTS / "dataset_analysis.txt"
    matrix = REPORTS / "confusion_matrix.png"

    if comparison.exists():
        st.subheader("Model Comparison")
        st.dataframe(
            pd.read_csv(comparison),
            use_container_width=True,
            hide_index=True
        )

    if analysis.exists():
        st.subheader("Dataset Analysis")
        st.text(
            analysis.read_text(encoding="utf-8")
        )

    if matrix.exists():
        st.subheader("Confusion Matrix")
        st.image(
            str(matrix),
            caption="Confusion Matrix for Selected Model"
        )

    if not any([comparison.exists(), analysis.exists(), matrix.exists()]):
        st.info("Training report files are not available yet.")
