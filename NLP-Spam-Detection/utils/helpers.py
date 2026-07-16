"""Risk analysis, message explanation, and prediction history helpers."""
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd

from config import HISTORY_PATH

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
    """Calculate transparent 0–100 risk score from model and observed indicators."""
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
