
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="IPL Match Winner Prediction",
    page_icon="🏏",
    layout="centered"
)

st.title("🏏 IPL Match Winner Prediction")
st.write(
    "Enter the current match situation to predict whether "
    "the batting team is likely to win."
)

MODEL_PATH = Path("ipl_model.pkl")

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

@st.cache_resource
def load_saved_model():
    saved = joblib.load(MODEL_PATH)

    if isinstance(saved, dict):
        return saved["model"], saved.get("features", FEATURES)

    return saved, FEATURES

if not MODEL_PATH.exists():
    st.error(
        "ipl_model.pkl is missing. First run training.py and upload "
        "the generated model file to the same GitHub folder as app.py."
    )
    st.stop()

try:
    model, model_features = load_saved_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

st.subheader("Match Situation")

col1, col2 = st.columns(2)

with col1:
    runs_scored = st.number_input(
        "Runs Scored", min_value=0.0, value=120.0, step=1.0
    )
    extras = st.number_input(
        "Extras", min_value=0.0, value=5.0, step=1.0
    )
    current_score = st.number_input(
        "Current Score", min_value=0.0, value=125.0, step=1.0
    )
    wickets_down = st.number_input(
        "Wickets Down", min_value=0, max_value=10, value=3, step=1
    )
    balls_remaining = st.number_input(
        "Balls Remaining", min_value=0, max_value=120, value=60, step=1
    )

with col2:
    wickets_remaining = st.number_input(
        "Wickets Remaining", min_value=0, max_value=10, value=7, step=1
    )
    current_run_rate = st.number_input(
        "Current Run Rate", min_value=0.0, value=8.33, step=0.01
    )
    required_run_rate = st.number_input(
        "Required Run Rate", min_value=0.0, value=9.00, step=0.01
    )
    target_score = st.number_input(
        "Target Score", min_value=0.0, value=180.0, step=1.0
    )

st.divider()

if st.button("🔮 Predict Winner", type="primary", use_container_width=True):

    values = [
        runs_scored,
        extras,
        current_score,
        wickets_down,
        balls_remaining,
        wickets_remaining,
        current_run_rate,
        required_run_rate,
        target_score,
    ]

    input_df = pd.DataFrame([values], columns=model_features)

    try:
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])

        if prediction == 1:
            st.success(
                f"🏆 Batting Team Likely to WIN\n\n"
                f"Estimated win probability: {probability * 100:.2f}%"
            )
        else:
            st.warning(
                f"🏏 Batting Team Likely to LOSE\n\n"
                f"Estimated win probability: {probability * 100:.2f}%"
            )

        st.progress(
            probability,
            text=f"Win probability: {probability * 100:.2f}%"
        )

        with st.expander("Input values"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.caption(
    "Educational ML project. The prediction is an estimate, not a guarantee."
)
