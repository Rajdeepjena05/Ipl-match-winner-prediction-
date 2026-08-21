
"""
IPL Match Winner Prediction - Prediction Script
"""

from pathlib import Path
import argparse
import joblib
import pandas as pd

DEFAULT_MODEL = "ipl_model.pkl"

def load_model(model_path):
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"{model_path} not found. Run training.py first."
        )

    saved = joblib.load(path)

    if isinstance(saved, dict):
        return saved["model"], saved["features"]

    # Compatibility with a plain sklearn .pkl file.
    features = [
        "runs_scored", "extras", "current_score", "wickets_down",
        "balls_remaining", "wickets_remaining", "current_run_rate",
        "required_run_rate", "target_score"
    ]
    return saved, features

def predict(model, features, values):
    X = pd.DataFrame([values], columns=features)

    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    return prediction, probability

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    model, features = load_model(args.model)

    # Example match situation.
    values = [
        120,   # runs_scored
        5,     # extras
        125,   # current_score
        3,     # wickets_down
        60,    # balls_remaining
        7,     # wickets_remaining
        8.33,  # current_run_rate
        9.00,  # required_run_rate
        180    # target_score
    ]

    prediction, probability = predict(model, features, values)

    print("\n===== IPL PREDICTION =====")

    if prediction == 1:
        print("Prediction: BATTING TEAM LIKELY TO WIN")
    else:
        print("Prediction: BATTING TEAM LIKELY TO LOSE")

    print(f"Win probability: {probability * 100:.2f}%")
    print(f"Lose probability: {(1 - probability) * 100:.2f}%")

if __name__ == "__main__":
    main()
