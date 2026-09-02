"""Build a mock bank database (SQLite) from data/synthetic_loans.csv.

Run once:
    python build_bank_db.py

Produces bank.db with a small normalised schema that mimics how a real bank
would store this information across several tables:

    customers           - static borrower / bureau attributes
    financial_profile   - current financial-behaviour aggregates (1 row per customer)
    loan_applications   - one loan request per customer, with its historical outcome

app.py joins these back together to rebuild the 10 feature columns the model
expects. Swapping SQLite for Postgres later is just a connection-string change.
"""

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

CSV_PATH = Path("data/synthetic_loans.csv")
DB_PATH = Path("bank.db")
NAME_SEED = 42

FIRST_NAMES = [
    "Ada", "Chidi", "Ngozi", "Emeka", "Funke", "Tunde", "Zainab", "Ibrahim",
    "Grace", "Samuel", "Blessing", "David", "Aisha", "Kelechi", "Yetunde", "Musa",
    "Chioma", "Olu", "Fatima", "John", "Peace", "Daniel", "Amara", "Segun",
    "Halima", "Victor", "Nkechi", "Ahmed", "Ruth", "Bola", "Esther", "Uche",
    "Maryam", "Femi", "Joy", "Sadiq", "Ifeoma", "Gbenga", "Rebecca", "Ismail",
]
LAST_NAMES = [
    "Okafor", "Balogun", "Eze", "Adeyemi", "Bello", "Nwosu", "Abubakar", "Okonkwo",
    "Oladipo", "Mohammed", "Ibe", "Adebayo", "Sani", "Chukwu", "Ogunleye", "Yusuf",
    "Obi", "Afolabi", "Danjuma", "Onyeka", "Lawal", "Nnamdi", "Aliyu", "Ojo",
    "Ekwueme", "Suleiman", "Ude", "Akinyemi", "Garba", "Anyanwu", "Ibrahim", "Oke",
]


def make_names(n: int) -> list[str]:
    rng = np.random.default_rng(NAME_SEED)
    firsts = rng.choice(FIRST_NAMES, size=n)
    lasts = rng.choice(LAST_NAMES, size=n)
    return [f"{f} {l}" for f, l in zip(firsts, lasts)]


SCHEMA = """
DROP TABLE IF EXISTS loan_applications;
DROP TABLE IF EXISTS financial_profile;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id       TEXT PRIMARY KEY,
    full_name         TEXT NOT NULL,
    age               INTEGER NOT NULL,
    employment_status TEXT NOT NULL,
    annual_income     REAL NOT NULL,
    credit_score      INTEGER NOT NULL
);

CREATE TABLE financial_profile (
    customer_id              TEXT PRIMARY KEY REFERENCES customers(customer_id),
    debt_to_income_ratio     REAL NOT NULL,
    late_payments_12m        INTEGER NOT NULL,
    credit_utilization       REAL NOT NULL,
    savings_rate             REAL NOT NULL,
    avg_monthly_transactions INTEGER NOT NULL
);

CREATE TABLE loan_applications (
    application_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id      TEXT NOT NULL REFERENCES customers(customer_id),
    loan_amount      REAL NOT NULL,
    application_date TEXT NOT NULL,
    default_outcome  INTEGER NOT NULL
);
"""


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(
            f"{CSV_PATH} not found. Run the notebook first to generate the synthetic data."
        )

    df = pd.read_csv(CSV_PATH).reset_index(drop=True)
    df["customer_id"] = [f"CUST-{i + 1:05d}" for i in range(len(df))]
    df["full_name"] = make_names(len(df))

    # spread applications over the last ~2 years for a plausible-looking date column
    rng = np.random.default_rng(NAME_SEED)
    days_ago = rng.integers(1, 730, size=len(df))
    df["application_date"] = (
        pd.Timestamp("today").normalize() - pd.to_timedelta(days_ago, unit="D")
    ).strftime("%Y-%m-%d")

    DB_PATH.unlink(missing_ok=True)
    con = sqlite3.connect(DB_PATH)
    try:
        con.executescript(SCHEMA)

        df[[
            "customer_id", "full_name", "age", "employment_status",
            "annual_income", "credit_score",
        ]].to_sql("customers", con, if_exists="append", index=False)

        df[[
            "customer_id", "debt_to_income_ratio", "late_payments_12m",
            "credit_utilization", "savings_rate", "avg_monthly_transactions",
        ]].to_sql("financial_profile", con, if_exists="append", index=False)

        df[[
            "customer_id", "loan_amount", "application_date", "default",
        ]].rename(columns={"default": "default_outcome"}).to_sql(
            "loan_applications", con, if_exists="append", index=False
        )

        con.commit()
        n = con.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        rate = con.execute(
            "SELECT AVG(default_outcome) FROM loan_applications"
        ).fetchone()[0]
    finally:
        con.close()

    print(f"Built {DB_PATH} | {n:,} customers | historical default rate {rate:.2%}")
    print("Sample IDs: CUST-00001, CUST-00002, CUST-00003 ...")


if __name__ == "__main__":
    main()
