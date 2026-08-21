
"""
IPL Match Winner Prediction - Training Script

Trains a RandomForestClassifier using the same 9 features used in the
original Model building notebook, but fixes the original hard-coded path,
target inconsistency, missing-value handling and model-saving workflow.
"""

from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

FEATURES = [
    "runs_scored",
    "extras",
    "current_score",
    "wickets_down",
    "balls_remaining",
    "wickets_remaining",
    "current_run_rate",
    "required_run_rate",
    "target_score",
]

def find_dataset(user_path=None):
    if user_path:
        path = Path(user_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Dataset not found: {path}")

    candidates = [
        Path("compressed_data.csv.gz"),
        Path("ipl.csv"),
        Path("data/compressed_data.csv.gz"),
        Path("data/ipl.csv"),
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Dataset not found. Put compressed_data.csv.gz or ipl.csv "
        "in the project folder."
    )

def load_and_prepare_data(path):
    print(f"Loading dataset: {path}")
    df = pd.read_csv(path, low_memory=False)

    required = FEATURES + ["batting_team", "winner"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # Median imputation is safer than silently converting every bad value to 0.
    X = X.fillna(X.median())
    X = X.fillna(0)

    batting_team = df["batting_team"].astype(str).str.strip()
    winner = df["winner"].astype(str).str.strip()

    # Correct target from the original Model building notebook:
    # 1 = batting team eventually won, 0 = batting team did not win.
    y = (batting_team == winner).astype(int)

    # Remove rows where winner is genuinely missing.
    invalid = (
        df["winner"].isna()
        | winner.str.lower().isin(["", "nan", "none", "null"])
    )

    X = X.loc[~invalid].copy()
    y = y.loc[~invalid].copy()

    if y.nunique() < 2:
        raise ValueError("Target contains fewer than two classes.")

    return df, X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print("\n===== MODEL RESULTS =====")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return model, metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to IPL CSV/CSV.GZ dataset")
    parser.add_argument("--model", default="ipl_model.pkl", help="Output model file")
    args = parser.parse_args()

    dataset = find_dataset(args.data)
    _, X, y = load_and_prepare_data(dataset)

    print(f"Rows used: {len(X):,}")
    print(f"Features: {FEATURES}")

    model, metrics = train_model(X, y)

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "metrics": metrics
        },
        args.model
    )

    print(f"\nModel saved successfully: {args.model}")

if __name__ == "__main__":
    main()
