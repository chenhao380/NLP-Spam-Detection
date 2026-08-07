"""Streamlit interface for the AI-powered spam and phishing detector.

This is a merged, single-file version of the project: it combines
config.py, preprocess.py, utils/helpers.py, evaluate.py (metrics helper),
predict.py, and utils/__init__.py into this one app.py.
"""
import csv
import json
import re
import string
from datetime import datetime
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import nltk
import pandas as pd
import plotly.express as px
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

# ---------------------------------------------------------------------------
# config.py — Application paths and shared settings
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "spam.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "spam_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
HISTORY_PATH = BASE_DIR / "prediction_history.csv"
RANDOM_STATE = 42
LABELS = ["ham", "spam", "phishing"]


# ---------------------------------------------------------------------------
# evaluate.py — Model evaluation and metric serialization
# ---------------------------------------------------------------------------
def evaluate_model(model: Any, x_test: Any, y_test: Any, labels: list[str]) -> dict[str, Any]:
    """Return standard classification metrics for a fitted model."""
    predicted = model.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predicted, labels=labels, average="weighted", zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=labels).tolist(),
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# preprocess.py — Reusable NLP preprocessing utilities
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _resources() -> tuple[set[str], WordNetLemmatizer]:
    """Ensure NLTK data exists and return preprocessing resources."""
    for package, location in [("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab"),
                              ("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet")]:
        try:
            nltk.data.find(location)
        except LookupError:
            nltk.download(package, quiet=True)
    return set(stopwords.words("english")), WordNetLemmatizer()


def preprocess_text(text: object) -> str:
    """Normalize text, remove numbers/punctuation/stopwords, then lemmatize."""
    if not isinstance(text, str):
        return ""
    stops, lemmatizer = _resources()
    cleaned = text.lower()
    cleaned = re.sub(r"\d+", " ", cleaned)
    cleaned = cleaned.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    tokens = word_tokenize(cleaned)
    return " ".join(lemmatizer.lemmatize(token) for token in tokens if token not in stops and len(token) > 1)


# ---------------------------------------------------------------------------
# utils/helpers.py — Risk analysis, message explanation, prediction history
# ---------------------------------------------------------------------------
KEYWORDS = {
    "urgency": ["urgent", "immediately", "now", "act fast", "limited", "hurry", "today"],
    "promotional": ["free", "prize", "winner", "won", "offer", "cash", "gift"],
    "security": ["verify", "password", "account suspended", "login", "confirm", "security alert"],
    "action": ["click", "claim", "reply", "call", "subscribe", "open link"],
}
URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s<>]+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d\s()\-]{6,}\d)")


def find_indicators(message: str) -> dict[str, Any]:
    """Locate linguistic and structural warning signals in a message."""
    lowered = message.lower()
    found = {category: [term for term in terms if term in lowered] for category, terms in KEYWORDS.items()}
    found = {key: value for key, value in found.items() if value}
    return {
        "keywords": found,
        "urls": URL_PATTERN.findall(message),
        "emails": EMAIL_PATTERN.findall(message),
        "phones": PHONE_PATTERN.findall(message),
        "caps": len(re.findall(r"\b[A-Z]{3,}\b", message)),
        "repeated_punctuation": bool(re.search(r"[!?]{2,}", message)),
    }


def calculate_risk(probabilities: dict[str, float], indicators: dict[str, Any]) -> tuple[int, str]:
    """Calculate transparent 0-100 risk score from model and observed indicators."""
    base = 100 * (probabilities.get("spam", 0) + probabilities.get("phishing", 0))
    keyword_count = sum(len(items) for items in indicators["keywords"].values())
    score = base + min(keyword_count * 4, 16) + (12 if indicators["urls"] else 0)
    score += 5 if indicators["emails"] else 0
    score += 5 if indicators["phones"] else 0
    score += min(indicators["caps"] * 2, 8) + (5 if indicators["repeated_punctuation"] else 0)
    score = min(100, round(score))
    level = "Low Risk" if score <= 30 else "Medium Risk" if score <= 70 else "High Risk"
    return score, level


def explanation(prediction: str, indicators: dict[str, Any]) -> str:
    """Turn detected signals into a concise human-readable reason."""
    parts = []
    words = [word.upper() for group in indicators["keywords"].values() for word in group]
    if words:
        parts.append("suspicious language: " + ", ".join(words))
    if indicators["urls"]:
        parts.append("a URL")
    if indicators["repeated_punctuation"]:
        parts.append("repeated punctuation")
    if prediction == "ham" and not parts:
        return "No strong spam or phishing signals were detected."
    return "The message was flagged because it contains " + (", ".join(parts) or "patterns associated with unsafe messages") + "."


def append_history(message: str, prediction: str, confidence: float, risk: int) -> None:
    """Persist one analysis result to the local CSV history."""
    new_file = not HISTORY_PATH.exists()
    with HISTORY_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(["Date", "Message", "Prediction", "Confidence", "Risk Score"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), message, prediction, round(confidence, 4), risk])


def load_history() -> pd.DataFrame:
    """Return saved predictions, or an empty table with the expected columns."""
    columns = ["Date", "Message", "Prediction", "Confidence", "Risk Score"]
    return pd.read_csv(HISTORY_PATH) if HISTORY_PATH.exists() else pd.DataFrame(columns=columns)


def clear_history() -> None:
    """Remove all saved prediction history."""
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()


# ---------------------------------------------------------------------------
# predict.py — Model-backed prediction engine
# ---------------------------------------------------------------------------
class SpamDetector:
    """Load saved artifacts and produce explainable message classifications."""
    def __init__(self) -> None:
        if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
            raise FileNotFoundError("No model artifacts found. Run: python train_model.py")
        self.model = joblib.load(MODEL_PATH)
        self.vectorizer = joblib.load(VECTORIZER_PATH)

    def analyze(self, message: str) -> dict[str, Any]:
        """Classify a non-empty message and calculate its risk explanation."""
        if not message or not message.strip():
            raise ValueError("Please enter a message to analyse.")
        features = self.vectorizer.transform([preprocess_text(message)])
        probabilities = dict(zip(self.model.classes_, self.model.predict_proba(features)[0]))
        prediction = max(probabilities, key=probabilities.get)
        indicators = find_indicators(message)
        risk, risk_level = calculate_risk(probabilities, indicators)
        return {"prediction": prediction, "confidence": float(probabilities[prediction]), "probabilities": probabilities,
                "risk_score": risk, "risk_level": risk_level, "indicators": indicators,
                "explanation": explanation(prediction, indicators)}


# ---------------------------------------------------------------------------
# app.py — Streamlit UI
# ---------------------------------------------------------------------------
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