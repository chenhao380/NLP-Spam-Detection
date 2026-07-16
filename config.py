"""Application paths and shared settings."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "spam.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "spam_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
HISTORY_PATH = BASE_DIR / "prediction_history.csv"
RANDOM_STATE = 42
LABELS = ["ham", "spam", "phishing"]

