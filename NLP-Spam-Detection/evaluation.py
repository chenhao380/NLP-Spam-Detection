"""Model evaluation and metric serialization."""
from typing import Any
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


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

