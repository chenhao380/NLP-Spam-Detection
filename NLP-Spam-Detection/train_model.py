"""Train, compare, and save the best spam/phishing classifier."""
import json
import logging
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from config import DATASET_PATH, LABELS, METRICS_PATH, MODEL_DIR, MODEL_PATH, RANDOM_STATE, VECTORIZER_PATH
from evaluation import evaluate_model
from preprocess import preprocess_text

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_dataset() -> pd.DataFrame:
    """Load, validate, de-duplicate, and clean the configured dataset."""
    data = pd.read_csv(DATASET_PATH)
    if not {"message", "label"}.issubset(data.columns):
        raise ValueError("Dataset must contain 'message' and 'label' columns.")
    data = data[["message", "label"]].dropna().drop_duplicates()
    data["label"] = data["label"].str.lower().str.strip()
    data = data[data["label"].isin(LABELS)]
    if data.empty or data["label"].nunique() < 2:
        raise ValueError("Dataset needs at least two valid label classes.")
    return data


def train() -> dict:
    """Train candidates, select highest weighted F1, and save artifacts."""
    data = load_dataset()
    data["processed"] = data["message"].map(preprocess_text)
    x_train, x_test, y_train, y_test = train_test_split(
        data["processed"], data["label"], test_size=0.2, random_state=RANDOM_STATE,
        stratify=data["label"]
    )
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)
    candidates = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Support Vector Machine": SVC(kernel="linear", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
    }
    labels = sorted(data["label"].unique())
    results, best_name, best_model, best_score = {}, "", None, -1.0
    for name, model in candidates.items():
        model.fit(x_train_vec, y_train)
        metrics = evaluate_model(model, x_test_vec, y_test, labels)
        results[name] = metrics
        if metrics["f1"] > best_score:
            best_name, best_model, best_score = name, model, metrics["f1"]
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    payload = {"best_model": best_name, "dataset_rows": len(data), "class_distribution": data["label"].value_counts().to_dict(), "models": results}
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Saved %s (weighted F1: %.4f)", best_name, best_score)
    return payload


if __name__ == "__main__":
    train()
