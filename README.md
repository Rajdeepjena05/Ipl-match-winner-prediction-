
# 🏏 IPL Match Winner Prediction using Machine Learning

## 1. Project Overview

This project predicts whether the **current batting team is likely to win an IPL match** from the current match situation.

The project is based on the same nine numerical features used in the original GitHub model-building notebook:

1. `runs_scored`
2. `extras`
3. `current_score`
4. `wickets_down`
5. `balls_remaining`
6. `wickets_remaining`
7. `current_run_rate`
8. `required_run_rate`
9. `target_score`

The target is:

```python
batting_team_won = (batting_team == winner).astype(int)
```

Therefore:

- `1` = batting team eventually won
- `0` = batting team did not win

This target definition is important because the original repository's **Model building** notebook used this binary target, while the original **Model saving** notebook incorrectly used the text `winner` column directly. This version makes the training and deployment target consistent.

## 2. Model Used

The project uses:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

### Why Random Forest?

Random Forest is an ensemble classification algorithm made from many Decision Trees.

Each tree learns different patterns from the training data. The trees then vote on the final class.

For this IPL project, Random Forest is suitable because:

- the prediction problem is classification;
- the input variables are numerical;
- relationships between match-state variables and the result can be nonlinear;
- it does not require feature scaling;
- it provides class probabilities through `predict_proba()`;
- it is relatively robust to noisy data.

## 3. Original Model Result

The original repository's Model building notebook reports:

```text
Model Accuracy: 0.7342966517496091
```

which is approximately:

```text
73.43%
```

This result belongs to the original notebook's particular random train/test split. The corrected training script should be run again on the repository dataset to produce the final reproducible model and current metrics.

## 4. Data Preparation

The training script:

- loads `compressed_data.csv.gz` or `ipl.csv`;
- uses `low_memory=False` to avoid the mixed-type warning seen in the original notebook;
- converts the nine model features to numeric values;
- converts infinity values to missing values;
- fills missing numeric values using training-data medians;
- creates the binary target from `batting_team` and `winner`;
- removes rows without a valid winner.

## 5. Train-Test Split

The project uses:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

Therefore:

- 80% of the usable data is used for training;
- 20% is used for testing;
- `random_state=42` makes the split reproducible;
- `stratify=y` keeps the class distribution approximately consistent.

## 6. Model Evaluation

The training script calculates:

### Accuracy

Percentage of predictions that are correct.

### Precision

Of the cases predicted as a batting-team win, how many were actually wins.

### Recall

Of the actual batting-team wins, how many were detected by the model.

### F1 Score

Harmonic mean of precision and recall.

### ROC-AUC

Measures how well the model separates the two classes across probability thresholds.

### Confusion Matrix

Shows:

- True Negatives
- False Positives
- False Negatives
- True Positives

## 7. Model Saving

The trained model is saved using Joblib:

```python
joblib.dump(saved_model, "ipl_model.pkl")
```

The saved object contains:

```text
model
features
metrics
```

This makes the Streamlit application independent of the training notebook.

## 8. Streamlit Application

The Streamlit application is `app.py`.

It provides input boxes for:

- Runs Scored
- Extras
- Current Score
- Wickets Down
- Balls Remaining
- Wickets Remaining
- Current Run Rate
- Required Run Rate
- Target Score

After clicking **Predict Winner**, the application displays:

- predicted result;
- estimated batting-team win probability;
- input values used for prediction.

## 9. Project Files

```text
IPL-Match-Winner-Prediction/
│
├── training.py
├── prediction.py
├── app.py
├── requirements.txt
├── README.md
├── IPL_Match_Winner_Prediction.ipynb
├── compressed_data.csv.gz
└── ipl_model.pkl
```

## 10. Installation

Create a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 11. Train the Model

If your dataset is named `compressed_data.csv.gz`:

```bash
python training.py
```

Or specify a dataset manually:

```bash
python training.py --data compressed_data.csv.gz
```

After successful training:

```text
ipl_model.pkl
```

will be created.

## 12. Test Prediction

Run:

```bash
python prediction.py
```

The script uses an example match situation and prints the predicted result and probability.

## 13. Run Streamlit

Run:

```bash
streamlit run app.py
```

The application will open in your browser.

## 14. Deploy on Streamlit Community Cloud

Upload these files to GitHub:

```text
app.py
training.py
prediction.py
requirements.txt
README.md
ipl_model.pkl
```

Then:

1. Open Streamlit Community Cloud.
2. Select your GitHub repository.
3. Select branch `main`.
4. Set the main file to `app.py`.
5. Click Deploy.

The model file must be present in the deployed repository.

## 15. Important Limitation

This model predicts the outcome from the match-state variables supplied to it. It is not a guaranteed real-time IPL prediction system.

The model's probability should be interpreted as an ML estimate based on the training data, not as certainty.

## 16. Dataset Credit

The GitHub repository credits:

**Kaggle Uploader:** Sanjeev Singh  
**Original Data Provider:** Cricsheet

Please retain the original dataset's licensing and attribution requirements when redistributing the data.

## 17. Author

**Rajdeep Jena**

IPL Match Winner Prediction — Machine Learning Project.
