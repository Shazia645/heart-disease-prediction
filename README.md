# Heart Disease Prediction

A machine learning project that predicts whether a patient has heart disease
based on their medical attributes, using the UCI Heart Disease dataset.

## Dataset

[Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
(originally 1025 rows, 14 columns) — sourced from Kaggle.

## What was done

1. **Data cleaning** — found and removed 723 duplicate rows, leaving 302
   unique patient records.
2. **Exploratory analysis** — checked missing values, outliers, and target
   class balance (164 vs 138 — well balanced).
3. **Label verification** — used `groupby('target')` on key columns
   (age, max heart rate, exercise angina, etc.) to confirm which label
   value actually represents "disease" rather than assuming.
4. **Model comparison** — trained and evaluated three algorithms:

   | Model | Accuracy |
   |---|---|
   | Logistic Regression | 77.05% |
   | **Random Forest** | **83.61%** ✅ |
   | K-Nearest Neighbors | 73.77% |

5. **Final model** — Random Forest was selected as the best performer and
   saved as `heart_rf_model.pkl` for reuse without retraining.

## Files

- `heart_diseases.ipynb` — full workflow: cleaning, EDA, training, evaluation
- `heart_clean.csv` — cleaned dataset (duplicates removed)
- `heart_rf_model.pkl` — trained Random Forest model, ready to load with `joblib`

## How to use the saved model

\`\`\`python
import joblib
import pandas as pd

model = joblib.load("heart_rf_model.pkl")

patient = pd.DataFrame([{
    "age": 58, "sex": 1, "cp": 2, "trestbps": 140, "chol": 230,
    "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
    "oldpeak": 1.2, "slope": 1, "ca": 0, "thal": 2
}])

patient = patient[model.feature_names_in_]
prediction = model.predict(patient)