🏏 IPL Match Prediction Using Machine Learning

📌 Project Overview

This project uses Machine Learning to predict the outcome of IPL cricket matches based on match-related features.

The project covers the complete Machine Learning workflow:

- Data Understanding
- Exploratory Data Analysis (EDA)
- Data Cleaning
- Data Preprocessing
- Feature Selection
- Feature Scaling
- Train-Test Split
- Model Building
- Model Evaluation
- Hyperparameter Tuning
- Model Saving using Joblib

---

📂 Dataset

Dataset: "ipl.csv"

The dataset contains IPL match-related information that can be used to analyze match situations and build a prediction model.

Example Features

Some of the important features used in the project include:

- "runs_scored"
- "extras"
- "current_score"
- "wickets_down"
- "balls_remaining"
- "wickets_remaining"
- "current_run_rate"
- "required_run_rate"
- "target_score"

---

🎯 Objective

The main objective of this project is to build a Machine Learning model that can learn from IPL match data and make predictions based on the current match situation.

---

🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

---

🔄 Machine Learning Workflow

IPL Dataset
     ↓
Data Understanding
     ↓
Data Cleaning
     ↓
EDA
     ↓
Data Preprocessing
     ↓
Feature Selection
     ↓
Feature Scaling
     ↓
Train-Test Split
     ↓
Model Building
     ↓
Model Evaluation
     ↓
Hyperparameter Tuning
     ↓
Save Model

---

📊 Exploratory Data Analysis

EDA is performed to understand the dataset and identify useful patterns.

The project includes:

- Dataset shape
- Data types
- Missing-value analysis
- Duplicate-value analysis
- Summary statistics
- Univariate analysis
- Bivariate analysis
- Correlation analysis
- Data visualization

---

🧹 Data Preprocessing

The following preprocessing techniques are applied:

1. Handling missing values
2. Removing duplicate records
3. Correcting data types
4. Handling outliers where required
5. Encoding categorical variables
6. Feature scaling
7. Feature selection

---

🤖 Model Building

Suitable Machine Learning algorithms are used to train the model.

The dataset is divided into:

- Training data – used to train the model
- Testing data – used to evaluate the model

Example:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

---

📈 Model Evaluation

The trained model is evaluated using appropriate classification metrics such as:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

These metrics help determine how well the model performs on unseen data.

---

⚙️ Hyperparameter Tuning

Hyperparameter tuning is performed to improve model performance.

Techniques such as:

- "GridSearchCV"
- "RandomizedSearchCV"

can be used to find better model parameters.

---

💾 Model Saving

The trained model can be saved using Joblib.

import joblib

joblib.dump(model, "ipl_model.pkl")

The saved model can later be loaded for making predictions.

model = joblib.load("ipl_model.pkl")

---

📁 Project Structure

IPL-Match-Prediction/
│
├── ipl.csv
├── IPL_Match_Prediction.ipynb
├── requirements.txt
├── README.md
└── ipl_model.pkl

---

🚀 How to Run the Project

1. Clone the repository

git clone <your-github-repository-url>

2. Open the project folder

cd IPL-Match-Prediction

3. Install required libraries

pip install -r requirements.txt

4. Open Jupyter Notebook

jupyter notebook

5. Open

IPL_Match_Prediction.ipynb

Run the cells step by step.

---

📌 Dataset Credits

Special thanks to Sanjeev Singh, the Kaggle uploader/source credited for the dataset used in this project.

Original Data Provider

The underlying cricket data is attributed to Cricsheet, which provides structured cricket data for analysis and research.

Kaggle Uploader: Sanjeev Singh
Original Data Provider: Cricsheet

«Dataset credit is given to the original uploader and data provider. This project is intended for educational and Machine Learning practice purposes.»

---

⚠️ Disclaimer

This project is created for educational and academic purposes.

Machine Learning predictions are based on historical data and selected features. They should not be considered guaranteed predictions of actual IPL match results.

---

👨‍💻 Author

Rajdeep Jena

Machine Learning / Data Science Project

---

⭐ Acknowledgement

Thanks to the cricket-data community, Cricsheet, and the Kaggle uploader Sanjeev Singh for making the dataset available for analysis and Machine Learning practice.

If you find this project useful, consider giving the repository a ⭐ on GitHub.
