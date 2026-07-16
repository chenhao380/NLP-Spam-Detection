"""Reusable NLP preprocessing utilities."""
import re
import string
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


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

