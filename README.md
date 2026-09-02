# Loan Default Risk Predictor

A small end-to-end machine-learning demo: train a model to estimate the probability
that a borrower defaults on a loan, then score borrowers through a Streamlit app —
either by typing their details in, or by looking them up in a mock bank database.

> Trained on **synthetic data** for demonstration only. Not for real credit decisions.

## What's inside

| File | Role |
|---|---|
| `loan_default_prediction.ipynb` | Generates synthetic data, trains + compares 3 models (Logistic Regression, Random Forest, Gradient Boosting), evaluates on a held-out test set, saves the winning pipeline |
| `app.py` | Streamlit app that loads the saved pipeline and predicts default risk |
| `build_bank_db.py` | Builds `bank.db` (SQLite) from `data/synthetic_loans.csv` — a normalised 3-table schema (`customers`, `financial_profile`, `loan_applications`) |
| `db.py` | Read-only query layer that joins those tables back into the model's feature row |
| `models/loan_default_pipeline.joblib` | The trained `preprocessor + model` pipeline |

The best model on the synthetic data is **Gradient Boosting**, ~0.82 test ROC-AUC.

## Run it locally

```bash
pip install -r requirements.txt

# 1. open the notebook and run all cells  -> creates the data + trained model
# 2. build the mock bank database
python build_bank_db.py
# 3. launch the app
streamlit run app.py
```

Then open http://localhost:8501. If it prompts for an email, just press Enter.

## Deploy (free)

Push this repo to GitHub, then deploy on [Streamlit Community Cloud](https://share.streamlit.io):
point it at `app.py`. `bank.db` is rebuilt automatically on first launch from the
committed CSV, so customer lookup works on the hosted app too.
