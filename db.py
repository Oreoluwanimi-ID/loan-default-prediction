"""Read-only query layer over the mock bank database (bank.db).

This is the seam a real integration would sit behind: point DB_URL at a
Postgres read-replica instead of the local SQLite file and nothing else
in the app has to change.
"""

from pathlib import Path
import sqlite3

DB_PATH = Path("bank.db")

# columns the trained pipeline expects, in order
FEATURE_COLS = [
    "age",
    "annual_income",
    "employment_status",
    "credit_score",
    "loan_amount",
    "debt_to_income_ratio",
    "late_payments_12m",
    "credit_utilization",
    "savings_rate",
    "avg_monthly_transactions",
]

_LOOKUP_SQL = """
SELECT
    c.customer_id,
    c.full_name,
    c.age,
    c.annual_income,
    c.employment_status,
    c.credit_score,
    la.loan_amount,
    fp.debt_to_income_ratio,
    fp.late_payments_12m,
    fp.credit_utilization,
    fp.savings_rate,
    fp.avg_monthly_transactions,
    la.application_date,
    la.default_outcome
FROM customers c
JOIN financial_profile fp ON fp.customer_id = c.customer_id
JOIN loan_applications  la ON la.customer_id = c.customer_id
WHERE c.customer_id = ?
"""


def db_available() -> bool:
    return DB_PATH.exists()


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def list_customers(limit: int = 500) -> list[tuple[str, str]]:
    """Return [(customer_id, full_name), ...] for a picker."""
    with _connect() as con:
        rows = con.execute(
            "SELECT customer_id, full_name FROM customers "
            "ORDER BY customer_id LIMIT ?",
            (limit,),
        ).fetchall()
    return [(r["customer_id"], r["full_name"]) for r in rows]


def get_customer(customer_id: str) -> dict | None:
    """Join the tables back into one record.

    Returns a dict with every FEATURE_COLS key plus `customer_id`, `full_name`,
    `application_date` and `default_outcome`, or None if the id is unknown.
    """
    with _connect() as con:
        row = con.execute(_LOOKUP_SQL, (customer_id,)).fetchone()
    if row is None:
        return None

    rec = dict(row)
    rec["age"] = int(rec["age"])
    rec["credit_score"] = int(rec["credit_score"])
    rec["late_payments_12m"] = int(rec["late_payments_12m"])
    rec["avg_monthly_transactions"] = int(rec["avg_monthly_transactions"])
    return rec
