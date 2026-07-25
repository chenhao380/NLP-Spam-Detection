"""Streamlit interface for the AI-powered spam and phishing detector."""
import json
from io import BytesIO
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
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
# Global styling (applies outside the hero iframe: buttons, cards, layout)
# ---------------------------------------------------------------------------
GLOBAL_CSS = """
<style>
    :root {
        --cf-color-bg: #faf8f3;
        --cf-color-surface: #ffffff;
        --cf-color-border-100: #e4ddcf;
        --cf-color-border-200: #cdc2ac;
        --cf-color-ink-900: #201d18;
        --cf-color-ink-500: #6f6558;
        --cf-color-accent-100: #fbe3c8;
        --cf-color-accent-500: #f6821f;
        --cf-color-accent-600: #d06c11;
    }

    html, body, [class*="css"]  {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    .stApp {
        background-color: #ffffff;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1000px;
    }

    hr {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 2rem 0;
    }

    /* fade-in-up animation applied to page sections as they render */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .cf-fade {
        animation: fadeInUp 0.6s ease-out both;
    }

    /* ---------------------------------------------------------------
       Hero framing: full-bleed dashed rules + light dot-grid backdrop
       --------------------------------------------------------------- */
    .cf-dashed-line {
        position: relative;
        left: 50%;
        right: 50%;
        margin-left: -50vw;
        margin-right: -50vw;
        width: 100vw;
        border-top: 1px dashed var(--cf-color-border-200);
        height: 0;
    }
    .cf-hero-frame {
        position: relative;
        left: 50%;
        right: 50%;
        margin-left: -50vw;
        margin-right: -50vw;
        width: 100vw;
        background-color: var(--cf-color-bg);
        background-image: radial-gradient(var(--cf-color-border-100) 1px, transparent 1px);
        background-size: 20px 20px;
        padding: 0.5rem 1rem 1.75rem;
        display: flex;
        justify-content: center;
    }
    .cf-hero-frame-inner {
        width: 100%;
        max-width: 620px;
    }
    /* screen-reader-only heading: keeps a real <h1> for accessibility
       without showing literal title text in the visual design */
    .cf-sr-only {
        position: absolute;
        width: 1px; height: 1px;
        padding: 0; margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        white-space: nowrap;
        border: 0;
    }
    .cf-hero-copy {
        text-align: center;
        max-width: 480px;
        margin: 0.6rem auto 1.25rem;
        color: var(--cf-color-ink-500);
        font-size: 1.02rem;
        line-height: 1.55;
    }

    /* Get Started / primary buttons: pill-shaped, orange gradient */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--cf-color-accent-500), #ff9d3d);
        border: none;
        border-radius: 999px;
        color: #ffffff;
        font-weight: 700;
        font-size: 1.0rem;
        padding: 0.7rem 1.7rem;
        box-shadow: 0 4px 14px rgba(246, 130, 31, 0.35);
        transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) scale(1.015);
        box-shadow: 0 0 0 6px var(--cf-color-accent-100), 0 8px 22px rgba(246, 130, 31, 0.4);
        background: linear-gradient(135deg, var(--cf-color-accent-600), var(--cf-color-accent-500));
        color: #ffffff;
    }
    .stButton > button[kind="primary"]:active {
        transform: translateY(0px) scale(0.99);
    }

    /* secondary pill button: outlined, transparent */
    .stButton > button:not([kind="primary"]) {
        background: transparent;
        border: 1.5px solid var(--cf-color-border-200);
        border-radius: 999px;
        color: var(--cf-color-ink-900);
        font-weight: 600;
        font-size: 1.0rem;
        padding: 0.7rem 1.7rem;
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }
    .stButton > button:not([kind="primary"]):hover {
        border-color: var(--cf-color-accent-500);
        box-shadow: 0 0 0 5px var(--cf-color-accent-100);
        transform: translateY(-2px);
        color: var(--cf-color-ink-900);
    }
    .stButton > button:not([kind="primary"]):active {
        transform: translateY(0px);
    }

    div[data-testid="stMetric"] {
        background: #fafafa;
        border: 1px solid #ececec;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        transition: box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        box-shadow: 0 4px 14px rgba(0,0,0,0.06);
    }
    /* let long metric values wrap onto a second line instead of being
       clipped with an ellipsis (e.g. "Safe · Spam · Phishing") */
    div[data-testid="stMetricValue"] {
        white-space: normal !important;
        overflow-wrap: break-word;
        line-height: 1.25;
        font-size: 1.5rem !important;
    }
</style>
"""

# ---------------------------------------------------------------------------
# Hero illustration: a small canvas animation standing in for the wordmark.
# The literal "Message Guard" title is kept as an sr-only <h1> in the outer
# page (see home()) so screen readers still get a real heading, while the
# visual focus is this scanning-radar sketch, faded out at the edges with a
# radial mask so it blends into the dot-grid backdrop behind it.
# ---------------------------------------------------------------------------
def render_hero_illustration() -> None:
    html = """
    <div class="cf-illo-wrap">
      <style>
        * { box-sizing: border-box; }
        html, body { margin: 0; background: transparent; }
        .cf-illo-wrap {
            width: 100%;
            display: flex;
            justify-content: center;
            animation: cfIlloFade 0.8s ease-out both;
        }
        @keyframes cfIlloFade {
            from { opacity: 0; transform: translateY(8px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        .cf-illo-mask {
            width: 320px;
            height: 200px;
            -webkit-mask-image: radial-gradient(circle at 50% 50%, black 55%, transparent 90%);
            mask-image: radial-gradient(circle at 50% 50%, black 55%, transparent 90%);
        }
        canvas { display: block; }
      </style>
      <div class="cf-illo-mask">
        <canvas id="cfIllo" width="320" height="200"></canvas>
      </div>
    </div>
    <script>
      (function () {
        const canvas = document.getElementById('cfIllo');
        const ctx = canvas.getContext('2d');
        const cx = canvas.width / 2, cy = canvas.height / 2;
        const ringColor = 'rgba(205, 194, 172, 0.55)';
        const sweepColor = 'rgba(246, 130, 31, 0.28)';
        const safeColor = '#cdc2ac';
        const flagColor = '#f6821f';

        // orbiting "messages": most are calm dots, one periodically flags
        const dots = Array.from({ length: 9 }, (_, i) => ({
            radius: 34 + (i % 3) * 26,
            angle: (i / 9) * Math.PI * 2,
            speed: 0.004 + (i % 3) * 0.0015,
            flagged: false,
        }));
        let sweepAngle = 0;
        let flagTimer = 0;
        let flaggedIndex = 2;

        function draw() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // concentric radar rings
            for (let r = 26; r <= 86; r += 20) {
                ctx.beginPath();
                ctx.arc(cx, cy, r, 0, Math.PI * 2);
                ctx.strokeStyle = ringColor;
                ctx.lineWidth = 1;
                ctx.stroke();
            }

            // rotating sweep wedge
            sweepAngle += 0.018;
            const grad = ctx.createConicGradient
                ? ctx.createConicGradient(sweepAngle, cx, cy)
                : null;
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.arc(cx, cy, 86, sweepAngle, sweepAngle + 0.9);
            ctx.closePath();
            ctx.fillStyle = sweepColor;
            ctx.fill();
            ctx.restore();

            // center shield glyph
            ctx.beginPath();
            ctx.moveTo(cx, cy - 10);
            ctx.lineTo(cx + 8, cy - 5);
            ctx.lineTo(cx + 8, cy + 6);
            ctx.quadraticCurveTo(cx, cy + 14, cx, cy + 14);
            ctx.quadraticCurveTo(cx, cy + 14, cx - 8, cy + 6);
            ctx.lineTo(cx - 8, cy - 5);
            ctx.closePath();
            ctx.fillStyle = '#3d372c';
            ctx.fill();

            // orbiting message dots; one flags red-orange when swept
            flagTimer += 1;
            if (flagTimer > 150) {
                flagTimer = 0;
                flaggedIndex = Math.floor(Math.random() * dots.length);
            }
            dots.forEach((d, i) => {
                d.angle += d.speed;
                const x = cx + Math.cos(d.angle) * d.radius;
                const y = cy + Math.sin(d.angle) * d.radius * 0.6;
                const isFlagged = i === flaggedIndex && flagTimer < 60;
                ctx.beginPath();
                ctx.arc(x, y, isFlagged ? 4.5 : 3, 0, Math.PI * 2);
                ctx.fillStyle = isFlagged ? flagColor : safeColor;
                ctx.fill();
                if (isFlagged) {
                    ctx.beginPath();
                    ctx.arc(x, y, 8, 0, Math.PI * 2);
                    ctx.strokeStyle = 'rgba(246, 130, 31, 0.4)';
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                }
            });

            requestAnimationFrame(draw);
        }
        draw();
      })();
    </script>
    """
    components.html(html, height=210, scrolling=False)


def apply_theme() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    dark = st.sidebar.toggle("Dark mode", value=False)
    if dark:
        st.markdown(
            "<style>.stApp {background:#101827;color:#e5e7eb}"
            "div[data-testid='stMetric'] {background:#1f2937;border-color:#2d3748}</style>",
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
    st.markdown('<div class="cf-dashed-line"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cf-hero-frame"><div class="cf-hero-frame-inner">',
        unsafe_allow_html=True,
    )

    render_hero_illustration()

    st.markdown(
        '<h1 class="cf-sr-only">Message Guard — AI Email &amp; SMS Threat Detection</h1>'
        '<p class="cf-hero-copy">Paste a message and get an instant read on '
        'whether it&rsquo;s safe, spam, or phishing.</p>',
        unsafe_allow_html=True,
    )

    _, c1, c2, _ = st.columns([1.1, 1, 1, 1.1])
    with c1:
        if st.button("Get Started", type="primary", use_container_width=True):
            st.session_state.started = True
            go_to("Analyze Message")
    with c2:
        if st.button("See How It Works", use_container_width=True):
            st.session_state.started = True
            go_to("About")

    st.markdown('</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="cf-dashed-line"></div>', unsafe_allow_html=True)

    st.markdown('<div class="cf-fade">', unsafe_allow_html=True)
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
    st.markdown('</div>', unsafe_allow_html=True)


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
    # Locked state: plain static text (no disabled widget, avoids hover glitches)
    st.sidebar.markdown("**Navigate**")
    st.sidebar.markdown("Home")
    st.sidebar.caption("Click Get Started on the Home page to unlock the other sections.")
    st.session_state.nav_page = "Home"

pages[st.session_state.nav_page]()