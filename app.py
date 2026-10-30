from pathlib import Path
import pandas as pd
import streamlit as st
import joblib

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "email_classifier.joblib"
REPORTS = ROOT / "reports"

st.set_page_config(page_title="Email Classification AI", page_icon="📧", layout="wide")
st.title("📧 AI-Based Email Classification System")
st.caption("NLP + TF-IDF + Machine Learning")

if not MODEL_PATH.exists():
    st.warning("The model has not been trained yet. Run `python train_model.py` in the project folder, then refresh this page.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["pipeline"]
categories = bundle["categories"]
st.sidebar.header("About this project")
st.sidebar.write("Classifies an email using its subject and body, then suggests the predicted department/category.")
st.sidebar.info("Demo dataset is synthetic. Replace it with a real public dataset before reporting final performance.")

left, right = st.columns([3, 2])
with left:
    st.subheader("Test an email")
    subject = st.text_input("Email subject", placeholder="e.g., Password reset request")
    body = st.text_area("Email body", height=200, placeholder="Type or paste the email content here...")
    predict = st.button("Classify Email", type="primary", use_container_width=True)

with right:
    st.subheader("Available categories")
    for category in categories:
        st.write(f"• {category}")
    st.subheader("Model")
    st.write(bundle.get("model_name", "Trained classifier"))

if "history" not in st.session_state:
    st.session_state.history = []

if predict:
    text = (subject.strip() + " " + body.strip()).strip()
    if not text:
        st.error("Please enter an email subject or body.")
    else:
        prediction = model.predict([text])[0]
        confidence = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([text])[0]
            confidence = float(max(probs))
        st.divider()
        st.subheader("Prediction result")
        st.success(f"Predicted category: **{prediction}**")
        if confidence is not None:
            st.metric("Model confidence", f"{confidence * 100:.1f}%")
            st.caption("Confidence is a model score, not a guarantee of correctness.")
        st.session_state.history.insert(0, {
            "Subject": subject,
            "Predicted category": prediction,
            "Confidence": f"{confidence * 100:.1f}%" if confidence is not None else "Not available",
        })

st.divider()
st.subheader("Prediction history")
if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("Your predictions in this session will appear here.")

with st.expander("View training reports"):
    comparison = REPORTS / "model_comparison.csv"
    analysis = REPORTS / "dataset_analysis.txt"
    matrix = REPORTS / "confusion_matrix.png"
    if comparison.exists():
        st.write("Model comparison")
        st.dataframe(pd.read_csv(comparison), use_container_width=True, hide_index=True)
    if analysis.exists():
        st.write("Dataset analysis")
        st.text(analysis.read_text(encoding="utf-8"))
    if matrix.exists():
        st.image(str(matrix), caption="Confusion matrix for selected model")
