"""Streamlit app for interactive loan default risk scoring.

Run with:
    streamlit run app.py

Loads the pipeline saved by loan_default_prediction.ipynb (models/loan_default_pipeline.joblib).
Borrower features can be typed in by hand, or pulled from the mock bank database
(bank.db) by customer id -- build that first with `python build_bank_db.py`.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import db

MODEL_PATH = Path("models/loan_default_pipeline.joblib")

DEFAULTS = {
    "age": 35,
    "annual_income": 55000.0,
    "employment_status": "Employed",
    "credit_score": 680,
    "loan_amount": 15000.0,
    "debt_to_income_ratio": 0.35,
    "late_payments_12m": 0,
    "credit_utilization": 0.40,
    "savings_rate": 0.10,
    "avg_monthly_transactions": 35,
}
EMPLOYMENT_OPTIONS = ["Employed", "Self-Employed", "Unemployed", "Retired"]

st.set_page_config(page_title="Loan Default Risk", page_icon="\U0001F4B0", layout="centered")
st.title("Loan Default Risk Predictor")
st.caption(
    "Estimates the probability a borrower defaults, based on a model trained on "
    "static borrower features and dynamic financial-behavior features."
)

if not MODEL_PATH.exists():
    st.error(
        f"No trained model found at `{MODEL_PATH}`.\n\n"
        "Run all cells in `loan_default_prediction.ipynb` first to generate the data "
        "and train + save the model."
    )
    st.stop()

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
feature_cols = bundle["feature_cols"]
model_name = bundle["model_name"]
test_auc = bundle["test_roc_auc"]


@st.cache_resource
def ensure_bank_db() -> bool:
    """Build bank.db on first run if it's missing (e.g. a fresh cloud deploy)."""
    if db.db_available():
        return True
    try:
        import build_bank_db

        build_bank_db.main()
    except Exception as exc:  # degrade gracefully to manual entry
        st.warning(f"Could not build the customer database ({exc}). Manual entry still works.")
    return db.db_available()


db_ready = ensure_bank_db()

st.sidebar.header("Model info")
st.sidebar.write(f"**Model:** {model_name}")
st.sidebar.write(f"**Test ROC-AUC:** {test_auc:.3f}")
st.sidebar.caption("Trained on synthetic data — for demonstration purposes only, not real credit decisions.")


def borrower_form(defaults: dict) -> dict:
    """Render the borrower-detail inputs pre-filled from `defaults`; return the values."""
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=int(defaults["age"]))
        annual_income = st.number_input(
            "Annual income ($)", min_value=0.0, value=float(defaults["annual_income"]), step=1000.0
        )
        employment_status = st.selectbox(
            "Employment status",
            EMPLOYMENT_OPTIONS,
            index=EMPLOYMENT_OPTIONS.index(defaults["employment_status"]),
        )
        credit_score = st.slider(
            "Credit score", min_value=300, max_value=850, value=int(defaults["credit_score"])
        )
        loan_amount = st.number_input(
            "Loan amount requested ($)", min_value=0.0, value=float(defaults["loan_amount"]), step=500.0
        )

    with col2:
        debt_to_income_ratio = st.slider(
            "Debt-to-income ratio", 0.0, 1.2, round(float(defaults["debt_to_income_ratio"]), 2), step=0.01
        )
        late_payments_12m = st.number_input(
            "Late payments in the last 12 months",
            min_value=0, max_value=12, value=int(defaults["late_payments_12m"]),
        )
        credit_utilization = st.slider(
            "Credit utilization", 0.0, 1.0, round(float(defaults["credit_utilization"]), 2), step=0.01
        )
        savings_rate = st.slider(
            "Monthly savings rate", -0.1, 0.5, round(float(defaults["savings_rate"]), 2), step=0.01
        )
        avg_monthly_transactions = st.number_input(
            "Average monthly transactions",
            min_value=0, max_value=200, value=int(defaults["avg_monthly_transactions"]),
        )

    return {
        "age": age,
        "annual_income": annual_income,
        "employment_status": employment_status,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "debt_to_income_ratio": debt_to_income_ratio,
        "late_payments_12m": late_payments_12m,
        "credit_utilization": credit_utilization,
        "savings_rate": savings_rate,
        "avg_monthly_transactions": avg_monthly_transactions,
    }


st.subheader("Borrower details")

defaults = dict(DEFAULTS)
history_outcome = None

modes = ["Manual entry"]
if db_ready:
    modes.insert(0, "Look up bank customer")

mode = st.radio("Input source", modes, horizontal=True)

if mode == "Look up bank customer":
    customers = db.list_customers()
    labels = {cid: f"{cid} — {name}" for cid, name in customers}
    chosen = st.selectbox(
        "Customer", [cid for cid, _ in customers], format_func=lambda c: labels[c]
    )
    record = db.get_customer(chosen)
    if record is None:
        st.warning("Customer not found in the database.")
    else:
        defaults.update({k: record[k] for k in DEFAULTS})
        history_outcome = record["default_outcome"]
        st.caption(
            f"Loaded **{record['full_name']}** · application dated {record['application_date']}. "
            "Fields below are pre-filled from the bank database — adjust for what-if scenarios."
        )

values = borrower_form(defaults)

if st.button("Predict default risk", type="primary"):
    row = pd.DataFrame([values])[feature_cols]

    probability = pipeline.predict_proba(row)[0, 1]

    if probability < 0.15:
        tier, color = "Low risk", "green"
    elif probability < 0.40:
        tier, color = "Medium risk", "orange"
    else:
        tier, color = "High risk", "red"

    st.metric("Predicted default probability", f"{probability:.1%}")
    st.markdown(f"**Risk tier:** :{color}[{tier}]")
    st.progress(min(probability, 1.0))

    if history_outcome is not None:
        actual = "Defaulted" if history_outcome == 1 else "Repaid"
        st.caption(f"Historical outcome on record for this application: **{actual}**.")
