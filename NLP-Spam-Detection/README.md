# AI-Powered Spam and Phishing Detection System

A Streamlit university project that classifies messages as **Safe (ham)**, **Spam**, or **Phishing**, then explains risk indicators. It uses preprocessing, TF-IDF features, and compares Multinomial Naive Bayes, Logistic Regression, and a linear SVM. The highest weighted-F1 model is saved automatically.

## Installation and run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

`train_model.py` loads `dataset/spam.csv`, removes missing/duplicate records, performs an 80/20 stratified split, evaluates every model, and writes trained artifacts to `model/`. Run it again after changing the dataset.

## Project layout

```
app.py                 Streamlit pages and visual dashboard
train_model.py         Dataset cleaning, training, comparison, persistence
predict.py             Explainable prediction engine
preprocess.py          NLTK text normalization and lemmatization
evaluation.py          Accuracy, precision, recall, F1 and confusion matrix
utils/helpers.py       Risk scoring, indicators and local history helpers
dataset/spam.csv       CSV training data: message,label
model/                 Generated model, vectorizer and metrics
assets/ screenshots/   Reserved for project visuals
```

## Features

- Text lowercasing, punctuation/number removal, stopword removal, tokenization and lemmatization
- URL, email, phone, urgency, promotional and account-security signal detection
- Transparent 0–100 risk score and Low/Medium/High categorization
- Probability chart, PDF result export, dark mode, dataset/model dashboard
- Searchable CSV prediction history with delete and export controls

## Dataset

The included data is intentionally small for demonstration. For credible evaluation, replace it with a larger balanced, consented dataset retaining the two columns `message,label`; valid labels are `ham`, `spam`, and `phishing`.

## Screenshots

Add application screenshots to `screenshots/` after running the app.

## Future improvements

Use a larger curated dataset, calibrate probabilities on held-out data, add URL reputation services, user authentication, and a database-backed audit history.
