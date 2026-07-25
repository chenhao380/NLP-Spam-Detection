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

    /* Get Started / primary buttons: prominent, animated on hover */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #f6821f, #ff9d3d);
        border: none;
        border-radius: 6px;
        color: #ffffff;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.75rem 1.6rem;
        box-shadow: 0 4px 14px rgba(246, 130, 31, 0.35);
        transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) scale(1.015);
        box-shadow: 0 8px 22px rgba(246, 130, 31, 0.45);
        background: linear-gradient(135deg, #e0740f, #f6821f);
        color: #ffffff;
    }
    .stButton > button[kind="primary"]:active {
        transform: translateY(0px) scale(0.99);
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
</style>
"""

# ---------------------------------------------------------------------------
# Hero section: dark cybersecurity background + mouse-triggered glitch title
# Rendered as a self-contained HTML component (own CSS + JS, no Streamlit
# rerun involved) so the scramble animation is instant and client-side only.
# ---------------------------------------------------------------------------
def render_hero(title: str = "EMAIL DETECTION", dark: bool = False) -> None:
    # Theme tokens: plain white surface in light mode, plain black in dark
    # mode (no gradient wash) so the toggle reads clearly as two states.
    if dark:
        bg = "#000000"
        grid_line = "rgba(255, 255, 255, 0.07)"
        glyph_color = "rgba(255, 255, 255, 0.16)"
        glyph_flash = "rgba(246, 130, 31, 0.55)"
        title_color = "#f5f7fa"
        title_glow = "rgba(56, 189, 248, 0.25)"
        sub_color = "#aab3c2"
        icon_glow = "rgba(56, 189, 248, 0.45)"
    else:
        bg = "#ffffff"
        grid_line = "rgba(15, 23, 42, 0.06)"
        glyph_color = "rgba(15, 23, 42, 0.10)"
        glyph_flash = "rgba(246, 130, 31, 0.65)"
        title_color = "#14181f"
        title_glow = "rgba(246, 130, 31, 0.12)"
        sub_color = "#5b6472"
        icon_glow = "rgba(246, 130, 31, 0.35)"

    html = f"""
    <div class="hero-wrap">
      <style>
        * {{ box-sizing: border-box; }}
        .hero-wrap {{
            position: relative;
            width: 100%;
            min-height: 380px;
            border-radius: 14px;
            overflow: hidden;
            background: {bg};
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            animation: heroFadeIn 0.8s ease-out both;
        }}
        @keyframes heroFadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}

        /* faint circuit / grid overlay for the cybersecurity feel */
        .hero-grid {{
            position: absolute;
            inset: 0;
            background-image:
                linear-gradient({grid_line} 1px, transparent 1px),
                linear-gradient(90deg, {grid_line} 1px, transparent 1px);
            background-size: 32px 32px;
            mask-image: radial-gradient(circle at 50% 40%, black 0%, transparent 75%);
        }}

        /* background "garbled code" layer: a grid of faint monospace
           glyphs, a handful of which flicker to a random character (and
           to the accent colour) every tick, like static behind the title */
        .hero-glyphs {{
            position: absolute;
            inset: 0;
            display: grid;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 12px;
            line-height: 1;
            color: {glyph_color};
            user-select: none;
            pointer-events: none;
            mask-image: radial-gradient(circle at 50% 45%, transparent 0%, transparent 28%, black 60%, black 100%);
        }}
        .hero-glyphs span.flash {{
            color: {glyph_flash};
        }}

        .hero-inner {{
            position: relative;
            z-index: 2;
            text-align: center;
            padding: 2.5rem 1.25rem;
            max-width: 720px;
        }}

        .hero-icon {{
            font-size: 2.4rem;
            margin-bottom: 0.5rem;
            filter: drop-shadow(0 0 10px {icon_glow});
        }}

        .glitch-title {{
            font-size: clamp(2.1rem, 6vw, 3.6rem);
            font-weight: 800;
            letter-spacing: 0.06em;
            color: {title_color};
            margin: 0 0 0.9rem 0;
            cursor: default;
            text-shadow: 0 0 18px {title_glow};
            user-select: none;
        }}
        .glitch-title span.char {{
            display: inline-block;
            min-width: 0.15em;
        }}

        .hero-sub {{
            font-size: clamp(0.92rem, 2vw, 1.05rem);
            color: {sub_color};
            line-height: 1.6;
            margin: 0 auto;
        }}

        @media (max-width: 640px) {{
            .hero-wrap {{ min-height: 320px; border-radius: 10px; }}
            .hero-inner {{ padding: 1.75rem 1rem; }}
        }}
      </style>

      <div class="hero-grid"></div>
      <div class="hero-glyphs" id="heroGlyphs"></div>
      <div class="hero-inner">
        <div class="hero-icon">🛡️</div>
        <h1 class="glitch-title" id="glitchTitle"></h1>
        <p class="hero-sub">
          AI-powered spam and phishing detection system that analyses emails and
          messages using NLP and Machine Learning.
        </p>
      </div>
    </div>

    <script>
      (function() {{
        const target = {json.dumps(title)};
        const el = document.getElementById('glitchTitle');
        const glitchChars = "!<>-_\\\\/[]{{}}—=+*^?#$%&0123456789";
        let frame = null;
        let running = false;

        function buildSpans(text) {{
            el.innerHTML = "";
            for (const ch of text) {{
                const span = document.createElement('span');
                span.className = 'char';
                span.textContent = ch === ' ' ? '\\u00A0' : ch;
                el.appendChild(span);
            }}
        }}

        function randomChar() {{
            return glitchChars[Math.floor(Math.random() * glitchChars.length)];
        }}

        function playScramble() {{
            if (running) return;
            running = true;
            const spans = Array.from(el.querySelectorAll('.char'));
            const total = spans.length;
            const revealDelayPerChar = 55; // ms between each letter locking in
            let startTime = performance.now();

            function tick(now) {{
                const elapsed = now - startTime;
                const revealCount = Math.min(total, Math.floor(elapsed / revealDelayPerChar));

                for (let i = 0; i < total; i++) {{
                    const original = target[i] === ' ' ? '\\u00A0' : target[i];
                    if (i < revealCount) {{
                        spans[i].textContent = original;
                    }} else if (original === '\\u00A0') {{
                        spans[i].textContent = original;
                    }} else {{
                        spans[i].textContent = randomChar();
                    }}
                }}

                if (revealCount < total) {{
                    frame = requestAnimationFrame(tick);
                }} else {{
                    running = false;
                }}
            }}

            frame = requestAnimationFrame(tick);
        }}

        function resetTitle() {{
            if (frame) cancelAnimationFrame(frame);
            running = false;
            buildSpans(target);
        }}

        buildSpans(target);
        el.addEventListener('mouseenter', playScramble);
        el.addEventListener('mouseleave', resetTitle);

        // ---- background glyph noise ("garbled code" backdrop) ----
        (function() {{
            const wrap = document.querySelector('.hero-wrap');
            const layer = document.getElementById('heroGlyphs');
            const cell = 22; // px per glyph cell
            const noiseChars = "01AXF$#%&*<>/\\\\{{}}[]=+;:";

            function buildGrid() {{
                const cols = Math.ceil(wrap.clientWidth / cell);
                const rows = Math.ceil(wrap.clientHeight / cell);
                layer.style.gridTemplateColumns = `repeat(${{cols}}, ${{cell}}px)`;
                layer.style.gridTemplateRows = `repeat(${{rows}}, ${{cell}}px)`;
                layer.innerHTML = "";
                const total = cols * rows;
                for (let i = 0; i < total; i++) {{
                    const span = document.createElement('span');
                    span.textContent = noiseChars[Math.floor(Math.random() * noiseChars.length)];
                    span.style.textAlign = 'center';
                    layer.appendChild(span);
                }}
            }}

            buildGrid();
            window.addEventListener('resize', buildGrid);

            // periodically flicker a handful of cells to a new (and briefly
            // accent-coloured) character, then let them settle back down
            setInterval(function() {{
                const spans = layer.children;
                if (!spans.length) return;
                const hits = 6 + Math.floor(Math.random() * 8);
                for (let n = 0; n < hits; n++) {{
                    const idx = Math.floor(Math.random() * spans.length);
                    const span = spans[idx];
                    span.textContent = noiseChars[Math.floor(Math.random() * noiseChars.length)];
                    span.classList.add('flash');
                    setTimeout(((s) => () => s.classList.remove('flash'))(span), 220);
                }}
            }}, 160);
        }})();
      }})();
    </script>
    """
    components.html(html, height=420, scrolling=False)


def apply_theme() -> bool:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    dark = st.sidebar.toggle("Dark mode", value=False)
    st.session_state.dark_mode = dark
    if dark:
        st.markdown(
            "<style>.stApp {background:#101827;color:#e5e7eb}"
            "div[data-testid='stMetric'] {background:#1f2937;border-color:#2d3748}</style>",
            unsafe_allow_html=True,
        )
    return dark


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
    render_hero("EMAIL DETECTION", dark=st.session_state.get("dark_mode", False))

    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if st.button("Get Started", type="primary", use_container_width=True):
            st.session_state.started = True
            go_to("Analyze Message")

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