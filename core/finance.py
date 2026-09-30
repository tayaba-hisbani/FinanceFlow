import pandas as pd

NUMERIC_CANDIDATES = ["amount", "value", "total", "debit", "credit", "expense", "sales"]
DATE_CANDIDATES = ["date", "transaction_date", "invoice_date"]
ID_CANDIDATES = ["invoice", "invoice_no", "invoice_number", "reference", "transaction_id"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out


def detect_amount_column(df):
    for c in NUMERIC_CANDIDATES:
        if c in df.columns:
            return c
    return None


def numeric_summary(df: pd.DataFrame):
    amount_col = detect_amount_column(df)
    if not amount_col:
        return {"amount_column": None, "total": 0, "count": len(df)}
    values = pd.to_numeric(df[amount_col], errors="coerce").fillna(0)
    return {"amount_column": amount_col, "total": float(values.sum()), "count": len(df)}


def find_duplicates(df: pd.DataFrame):
    for c in ID_CANDIDATES:
        if c in df.columns:
            dup = df[df[c].astype(str).duplicated(keep=False)]
            if not dup.empty:
                return c, dup.to_dict(orient="records")
    return None, []


def basic_reconciliation(df: pd.DataFrame):
    """Deterministic checks; never ask the LLM to calculate totals."""
    summary = numeric_summary(df)
    id_col, duplicates = find_duplicates(df)
    issues = []
    if duplicates:
        issues.append(f"Duplicate identifiers detected in column '{id_col}'.")
    amount_col = summary["amount_column"]
    if amount_col:
        values = pd.to_numeric(df[amount_col], errors="coerce")
        if values.isna().any():
            issues.append(f"Some values in '{amount_col}' are non-numeric or missing.")
        if (values < 0).any():
            issues.append(f"Negative values found in '{amount_col}'; review refunds/adjustments.")
    return {"summary": summary, "issues": issues, "duplicate_count": len(duplicates)}
