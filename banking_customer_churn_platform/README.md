# Banking Customer Churn Prediction & Analytics Platform

A beginner-friendly portfolio project using Python, Scikit-learn, Streamlit and Plotly.

## 1. Requirements
Install Python 3.10 or newer.

## 2. Open the project
Open a terminal in this folder.

Windows:
```bash
cd banking_customer_churn_platform
```

## 3. Create a virtual environment (recommended)

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

## 4. Install packages
```bash
pip install -r requirements.txt
```

## 5. Dataset
A sample `data/churn.csv` is already included so the project can run immediately.

For a real portfolio project, replace it with a larger churn dataset using the same columns:
- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary
- Exited

`Exited = 1` means churned and `Exited = 0` means stayed.

## 6. Train the machine-learning model
```bash
python train_model.py
```

This creates:
`churn_model.pkl`

You will also see accuracy, ROC-AUC, classification report and confusion matrix.

## 7. Start the dashboard
```bash
streamlit run app.py
```

If that command is not recognised:
```bash
python -m streamlit run app.py
```

Open the local address shown by Streamlit, normally localhost port 8501.

## How the project works

Dataset
→ Data cleaning
→ Remove identifiers
→ Train/test split
→ Scaling + one-hot encoding
→ Random Forest training
→ Model evaluation
→ Save model
→ Streamlit dashboard
→ Customer churn probability
→ Low/Medium/High risk classification

## Important
The included CSV is synthetic demonstration data. Replace it with a sufficiently large real/public dataset before reporting model performance in a CV or dissertation.
