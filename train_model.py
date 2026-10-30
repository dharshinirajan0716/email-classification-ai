from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "sample_emails.csv"
SUBJECT_COLUMN = "subject"
BODY_COLUMN = "body"
LABEL_COLUMN = "category"
MODEL_PATH = ROOT / "models" / "email_classifier.joblib"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)

def clean_text(value):
    text = "" if pd.isna(value) else str(value)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = {SUBJECT_COLUMN, BODY_COLUMN, LABEL_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}. Expected: {sorted(required)}")

    before = len(df)
    df = df.dropna(subset=[LABEL_COLUMN]).copy()
    df[SUBJECT_COLUMN] = df[SUBJECT_COLUMN].fillna("")
    df[BODY_COLUMN] = df[BODY_COLUMN].fillna("")
    df["text"] = (df[SUBJECT_COLUMN].astype(str) + " " + df[BODY_COLUMN].astype(str)).map(clean_text)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text", LABEL_COLUMN]).reset_index(drop=True)

    counts = df[LABEL_COLUMN].value_counts()
    analysis = [
        "AI-BASED EMAIL CLASSIFICATION — DATASET ANALYSIS",
        f"Rows before cleaning: {before}",
        f"Rows after cleaning: {len(df)}",
        f"Duplicate/missing-label rows removed or excluded: {before - len(df)}",
        f"Number of categories: {df[LABEL_COLUMN].nunique()}",
        "",
        "Category counts:",
        counts.to_string(),
        "",
        "Missing values in original columns:",
    ]
    analysis.extend(pd.read_csv(DATA_PATH).isna().sum().to_string().splitlines())
    analysis += ["", "Note: included dataset is synthetic demo data; replace with a public dataset for final evaluation."]
    (REPORTS / "dataset_analysis.txt").write_text("\n".join(analysis), encoding="utf-8")
    counts.rename_axis("category").reset_index(name="count").to_csv(REPORTS / "category_distribution.csv", index=False)

    # Use stratification only if each class has at least 2 examples.
    stratify = df[LABEL_COLUMN] if counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df[LABEL_COLUMN], test_size=0.25, random_state=42, stratify=stratify
    )

    candidates = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced_subsample"),
    }
    results, fitted = [], {}
    for name, estimator in candidates.items():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000, stop_words="english")),
            ("classifier", estimator),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="weighted", zero_division=0)
        acc = accuracy_score(y_test, pred)
        results.append({"model": name, "accuracy": acc, "precision_weighted": p, "recall_weighted": r, "f1_weighted": f1})
        fitted[name] = pipe

    results_df = pd.DataFrame(results).sort_values(["f1_weighted", "accuracy"], ascending=False)
    results_df.to_csv(REPORTS / "model_comparison.csv", index=False)
    best_name = results_df.iloc[0]["model"]
    best_model = fitted[best_name]
    y_pred = best_model.predict(X_test)
    (REPORTS / "classification_report.txt").write_text(
        f"Selected model: {best_name}\n\n" + classification_report(y_test, y_pred, zero_division=0),
        encoding="utf-8"
    )
    labels = sorted(df[LABEL_COLUMN].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(9, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(REPORTS / "confusion_matrix.png", dpi=160)
    plt.close()

    joblib.dump({
        "pipeline": best_model,
        "categories": labels,
        "model_name": best_name,
        "text_columns": [SUBJECT_COLUMN, BODY_COLUMN],
    }, MODEL_PATH)

    print("Training completed.")
    print(f"Best model: {best_name}")
    print(results_df.to_string(index=False))
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Reports folder: {REPORTS}")

if __name__ == "__main__":
    main()
