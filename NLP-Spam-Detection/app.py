"""Streamlit interface for the AI-powered spam and phishing detector."""
import json
from io import BytesIO
import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from config import DATASET_PATH, METRICS_PATH
from predict import SpamDetector
from utils.helpers import append_history, clear_history, load_history

st.set_page_config(page_title="Message Guard", page_icon="🛡️", layout="wide")

def apply_theme() -> None:
    dark = st.sidebar.toggle("Dark mode", value=False)
    if dark:
        st.markdown("<style>.stApp {background:#101827;color:#e5e7eb}.stMetric {background:#1f2937}</style>", unsafe_allow_html=True)

@st.cache_resource
def detector() -> SpamDetector:
    return SpamDetector()

@st.cache_data
def metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}

def result_pdf(message: str, result: dict) -> bytes:
    """Make a small downloadable report for one result."""
    buffer = BytesIO(); pdf = canvas.Canvas(buffer, pagesize=letter)
    text = pdf.beginText(48, 740); text.setFont("Helvetica", 11)
    lines = ["Message Guard - Analysis Report", "", f"Prediction: {result['prediction'].title()}",
             f"Confidence: {result['confidence']:.1%}", f"Risk: {result['risk_score']}/100 ({result['risk_level']})", "",
             "Explanation:", result['explanation'], "", "Message:"]
    for line in lines + [message[i:i+90] for i in range(0, len(message), 90)]:
        text.textLine(line)
    pdf.drawText(text); pdf.save(); return buffer.getvalue()

def home() -> None:
    st.title("🛡️ Message Guard")
    st.subheader("AI-powered spam and phishing detection")
    st.write("Analyse messages with NLP and machine learning, understand the risk signals, and keep a private local history.")
    cols = st.columns(3)
    for col, label, value in zip(cols, ["Classes", "Best model", "Training rows"], ["Safe · Spam · Phishing", metrics().get("best_model", "Train model"), metrics().get("dataset_rows", "—")]):
        col.metric(label, value)
    st.info("Start at **Analyze Message**. Train or retrain the model with `python train_model.py` whenever you update the dataset.")

def analyze() -> None:
    st.title("Analyze Message")
    sample = "URGENT! Verify your account now at https://secure-check.example or it will be suspended!!"
    message = st.text_area("Paste a text message or email", value=st.session_state.get("message", ""), height=170, placeholder=sample)
    if st.button("Analyze message", type="primary", use_container_width=True):
        try:
            with st.spinner("Checking language patterns and risk indicators…"):
                result = detector().analyze(message)
            append_history(message, result["prediction"], result["confidence"], result["risk_score"])
            st.session_state.result, st.session_state.message = result, message
        except (FileNotFoundError, ValueError) as error:
            st.error(str(error)); return
    result = st.session_state.get("result")
    if not result:
        return
    color = {"ham": "✅", "spam": "⚠️", "phishing": "🚨"}[result["prediction"]]
    st.subheader(f"{color} {result['prediction'].title()}")
    a, b, c = st.columns(3)
    a.metric("Confidence", f"{result['confidence']:.1%}")
    b.metric("Risk score", f"{result['risk_score']}/100")
    c.metric("Risk level", result["risk_level"])
    st.progress(result["risk_score"] / 100)
    st.write(result["explanation"])
    words = [word.upper() for values in result["indicators"]["keywords"].values() for word in values]
    st.caption("Detected keywords: " + (", ".join(words) if words else "None"))
    st.code(
        f"Prediction: {result['prediction'].title()}\nConfidence: {result['confidence']:.1%}\n"
        f"Risk: {result['risk_score']}/100 ({result['risk_level']})\n{result['explanation']}",
        language=None,
    )
    if result["indicators"]["urls"]:
        st.warning("Suspicious URL detected: " + ", ".join(result["indicators"]["urls"]))
    chart = pd.DataFrame({"Class": list(result["probabilities"]), "Probability": list(result["probabilities"].values())})
    st.plotly_chart(px.bar(chart, x="Class", y="Probability", range_y=[0, 1], color="Class"), use_container_width=True)
    st.download_button("Download result as PDF", result_pdf(st.session_state.message, result), "message-analysis.pdf", "application/pdf")

def dashboard() -> None:
    st.title("Statistics Dashboard")
    data, info, history = pd.read_csv(DATASET_PATH), metrics(), load_history()
    left, right = st.columns(2)
    left.plotly_chart(px.pie(data, names="label", title="Dataset class distribution"), use_container_width=True)
    if info:
        scores = pd.DataFrame([{"Model": name, "Accuracy": value["accuracy"], "F1": value["f1"]} for name, value in info["models"].items()])
        right.plotly_chart(px.bar(scores, x="Model", y=["Accuracy", "F1"], barmode="group", title="Model comparison"), use_container_width=True)
        st.caption(f"Selected model: {info['best_model']} · accuracy: {info['models'][info['best_model']]['accuracy']:.1%}")
    if not history.empty:
        history["Date"] = pd.to_datetime(history["Date"])
        daily = history.groupby(history["Date"].dt.date).size().reset_index(name="Analyses")
        st.plotly_chart(px.line(daily, x="Date", y="Analyses", markers=True, title="Daily analysis count"), use_container_width=True)
        st.plotly_chart(px.histogram(history, x="Risk Score", nbins=10, title="Risk distribution"), use_container_width=True)
    else:
        st.info("Analyse messages to populate prediction activity charts.")

def history_page() -> None:
    st.title("Prediction History")
    history = load_history(); search = st.text_input("Search messages or predictions")
    if search:
        history = history[history.astype(str).apply(lambda row: row.str.contains(search, case=False).any(), axis=1)]
    st.dataframe(history, use_container_width=True, hide_index=True)
    st.download_button("Export history as CSV", history.to_csv(index=False).encode(), "prediction-history.csv", "text/csv")
    if st.button("Delete all history"):
        clear_history(); st.rerun()

def about() -> None:
    st.title("About")
    st.write("This educational project compares Multinomial Naive Bayes, Logistic Regression, and SVM on TF-IDF features. The best weighted-F1 model is saved locally.")
    st.warning("Predictions are decision support, not a replacement for security controls. Do not open unexpected links or disclose credentials.")

apply_theme()
page = st.sidebar.radio("Navigate", ["Home", "Analyze Message", "Dashboard", "History", "About"])
{"Home": home, "Analyze Message": analyze, "Dashboard": dashboard, "History": history_page, "About": about}[page]()
