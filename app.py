"""Streamlit interface for the AI-powered spam and phishing detector."""
import json
from io import BytesIO
import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from email import policy
from email.parser import BytesParser

from config import DATASET_PATH, METRICS_PATH
from predict import SpamDetector
from train_model import train
from utils.helpers import append_history, clear_history, load_history

st.set_page_config(page_title="Message Guard", page_icon="🛡️", layout="wide")

PAGES = ["Home", "Analyze Message", "Dashboard", "History", "About"]

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "started" not in st.session_state:
    st.session_state.started = False
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Home"


# ---------------------------------------------------------------------------
# Cloudflare-inspired look & feel
# ---------------------------------------------------------------------------
CLOUDFLARE_CSS = """
<style>
    /* ---- global type & spacing, Cloudflare learning-center feel ---- */
    html, body, [class*="css"]  {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    .stApp {
        background-color: #ffffff;
    }

    /* tone down Streamlit's default header spacing */
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 1000px;
    }

    /* section dividers, subtle like cloudflare's hairlines */
    hr {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 2rem 0;
    }

    /* Cloudflare orange for primary buttons */
    .stButton > button[kind="primary"] {
        background-color: #f6821f;
        border: none;
        border-radius: 4px;
        color: #ffffff;
        font-weight: 600;
        padding: 0.65rem 1.4rem;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #d96f11;
        color: #ffffff;
    }

    /* card look for metrics / info blocks */
    div[data-testid="stMetric"] {
        background: #fafafa;
        border: 1px solid #ececec;
        border-radius: 8px;
        padding: 0.9rem 1rem;
    }

    /* --- hero block for the Home / landing page --- */
    .cf-hero {
        text-align: center;
        padding: 3.5rem 1rem 2.5rem 1rem;
    }
    .cf-hero .cf-icon {
        font-size: 3.2rem;
        line-height: 1;
        margin-bottom: 0.75rem;
    }
    .cf-hero h1 {
        font-size: 2.1rem;
        font-weight: 700;
        color: #14181f;
        margin-bottom: 0.4rem;
    }
    .cf-hero p.cf-sub {
        font-size: 1.05rem;
        color: #4b5563;
        max-width: 560px;
        margin: 0 auto 0.25rem auto;
        line-height: 1.55;
    }

    .cf-card {
        border: 1px solid #ececec;
        border-radius: 8px;
        padding: 1.4rem 1.6rem;
        background: #fcfcfc;
        margin-bottom: 1rem;
    }
</style>
"""


def apply_theme() -> None:
    st.markdown(CLOUDFLARE_CSS, unsafe_allow_html=True)
    dark = st.sidebar.toggle("Dark mode", value=False)
    if dark:
        st.markdown(
            "<style>.stApp {background:#101827;color:#e5e7eb}"
            ".stMetric, div[data-testid='stMetric'], .cf-card {background:#1f2937;border-color:#2d3748}"
            ".cf-hero h1 {color:#f3f4f6}.cf-hero p.cf-sub{color:#cbd5e1}</style>",
            unsafe_allow_html=True,
        )


@st.cache_resource
def detector() -> SpamDetector:
    """Load an existing model, or train one automatically on first deployment."""
    if not METRICS_PATH.exists():
        with st.spinner("Preparing the AI model for first use…"):
            train()
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


def go_to(page_name: str) -> None:
    """Central helper: change page + rerun (avoids duplicated rerun logic)."""
    st.session_state.nav_page = page_name
    st.rerun()


def home() -> None:
    # --- hero section, Cloudflare-style: big icon + big title + short line ---
    st.markdown(
        """
        <div class="cf-hero">
            <div class="cf-icon">🛡️</div>
            <h1>Email Detection</h1>
            <p class="cf-sub">
                Message Guard uses AI to tell you whether a message or email is
                <strong>Safe</strong>, <strong>Spam</strong>, or <strong>Phishing</strong> —
                instantly, with a clear explanation for every result.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if st.button("Get Started", type="primary", use_container_width=True):
            st.session_state.started = True
            go_to("Analyze Message")

    st.divider()

    st.markdown("## System Overview")

    cols = st.columns(3)
    cols[0].metric("Detection Classes", "Safe · Spam · Phishing")
    cols[1].metric("Best AI Model", metrics().get("best_model", "Train model"))
    cols[2].metric("Training Dataset", metrics().get("dataset_rows", "—"))

    st.divider()

    st.markdown("## Navigation Guide")

    st.markdown(
        """
    **Home** — Learn about the purpose of Message Guard and view system information.

    **Analyze Message** — Paste an email or text message to detect whether it is
    Safe, Spam, or Phishing. Shows prediction, confidence, risk score, explanation,
    suspicious keywords/URLs, and a downloadable PDF report.

    **Dashboard** — Dataset class distribution, model comparison, daily analysis
    activity, and risk score distribution.

    **History** — View, search, export, or delete previous prediction records.

    **About** — Technologies used, machine learning models, and project purpose.
    """
    )

    st.divider()

    st.markdown("## How Message Guard Works")
    st.markdown(
        """
    1. Paste a message or email into **Analyze Message**.
    2. The system preprocesses the text using NLP techniques.
    3. The trained machine learning model analyses the message.
    4. A prediction, confidence score, risk score, and explanation are generated.
    5. The analysis is saved to the local prediction history.
    """
    )


def analyze() -> None:
    st.title("Analyze Message")

    sample = "URGENT! Verify your account now at https://secure-check.example or it will be suspended!!"

    # Upload file
    uploaded_file = st.file_uploader(
        "Or drag and drop an email or text file",
        type=["txt", "eml"]
    )

    uploaded_message = ""

    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".eml"):
                email_message = BytesParser(
                    policy=policy.default
                ).parse(uploaded_file)

                if email_message.is_multipart():
                    body = email_message.get_body(preferencelist=("plain",))
                    uploaded_message = body.get_content() if body else ""
                else:
                    uploaded_message = email_message.get_content()

            else:
                uploaded_message = uploaded_file.read().decode(
                    "utf-8",
                    errors="ignore"
                )

        except Exception as e:
            st.error(f"Unable to read file: {e}")
            return

    # Text area
    message = st.text_area(
        "Paste a text message or email",
        value=uploaded_message or st.session_state.get("message", ""),
        height=170,
        placeholder=sample
    )

    if st.button("Analyze Message", type="primary", use_container_width=True):
        try:
            with st.spinner("Checking language patterns and risk indicators..."):
                result = detector().analyze(message)

            append_history(
                message,
                result["prediction"],
                result["confidence"],
                result["risk_score"]
            )

            st.session_state.result = result
            st.session_state.message = message

        except (FileNotFoundError, ValueError) as error:
            st.error(str(error))
            return

    result = st.session_state.get("result")

    if not result:
        return

    color = {
        "ham": "✅",
        "spam": "⚠️",
        "phishing": "🚨"
    }[result["prediction"]]

    st.subheader(f"{color} {result['prediction'].title()}")

    a, b, c = st.columns(3)

    a.metric("Confidence", f"{result['confidence']:.1%}")
    b.metric("Risk Score", f"{result['risk_score']}/100")
    c.metric("Risk Level", result["risk_level"])

    st.progress(result["risk_score"] / 100)

    st.write(result["explanation"])

    words = [
        word.upper()
        for values in result["indicators"]["keywords"].values()
        for word in values
    ]

    st.caption(
        "Detected Keywords: " +
        (", ".join(words) if words else "None")
    )

    st.code(
        f"Prediction: {result['prediction'].title()}\n"
        f"Confidence: {result['confidence']:.1%}\n"
        f"Risk: {result['risk_score']}/100 ({result['risk_level']})\n"
        f"{result['explanation']}",
        language=None,
    )

    if result["indicators"]["urls"]:
        st.warning(
            "Suspicious URL detected: " +
            ", ".join(result["indicators"]["urls"])
        )

    chart = pd.DataFrame({
        "Class": list(result["probabilities"]),
        "Probability": list(result["probabilities"].values())
    })

    st.plotly_chart(
        px.bar(
            chart,
            x="Class",
            y="Probability",
            range_y=[0, 1],
            color="Class"
        ),
        use_container_width=True
    )

    st.download_button(
        "Download Result as PDF",
        result_pdf(st.session_state.message, result),
        "message-analysis.pdf",
        "application/pdf"
    )


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

pages = {
    "Home": home,
    "Analyze Message": analyze,
    "Dashboard": dashboard,
    "History": history_page,
    "About": about,
}

if st.session_state.started:
    # Full navigation unlocked after "Get Started"
    current_index = PAGES.index(st.session_state.nav_page) if st.session_state.nav_page in PAGES else 0
    selected = st.sidebar.radio("Navigate", PAGES, index=current_index)
    if selected != st.session_state.nav_page:
        st.session_state.nav_page = selected
else:
    # Locked state: no interactive/disabled widget (avoids the hover glitch),
    # just a plain caption explaining how to unlock the rest of the app.
    st.sidebar.markdown("**Navigate**")
    st.sidebar.markdown("Home")
    st.sidebar.caption("Click Get Started on the Home page to unlock the other sections.")
    st.session_state.nav_page = "Home"

pages[st.session_state.nav_page]()