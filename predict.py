"""Model-backed prediction engine."""
from typing import Any
import joblib
from config import MODEL_PATH, VECTORIZER_PATH
from preprocess import preprocess_text
from utils.helpers import calculate_risk, explanation, find_indicators


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
