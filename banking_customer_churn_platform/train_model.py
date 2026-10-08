import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

DATA_PATH = "data/churn.csv"
MODEL_PATH = "churn_model.pkl"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. Put your CSV there and run this script again."
    )

df = pd.read_csv(DATA_PATH)

drop_cols = ["RowNumber", "CustomerId", "Surname"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

required = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

X = df.drop("Exited", axis=1)
y = df["Exited"]

categorical_cols = ["Geography", "Gender"]
numeric_cols = [c for c in X.columns if c not in categorical_cols]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model.fit(X_train, y_train)
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

print("\n=== MODEL RESULTS ===")
print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))

joblib.dump(model, MODEL_PATH)
print(f"\nSaved trained model to: {MODEL_PATH}")
