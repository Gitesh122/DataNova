"""Smart data detection and profiling."""
from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC, CATEGORICAL, DATETIME, BOOLEAN, TEXT, IDENTIFIER = (
    "Numerical", "Categorical", "Date/Time", "Boolean", "Text", "Identifier",
)


def semantic_type(s: pd.Series) -> str:
    n = len(s)
    nn = s.dropna()
    nunique = nn.nunique()
    if pd.api.types.is_bool_dtype(s) or (nunique == 2 and set(map(str, nn.unique())) <= {"0", "1", "True", "False", "true", "false"}):
        return BOOLEAN
    if pd.api.types.is_datetime64_any_dtype(s):
        return DATETIME
    if pd.api.types.is_numeric_dtype(s):
        name = str(s.name).lower()
        if (name.endswith("id") or name.endswith("_id") or name == "id") and nunique > 0.95 * max(len(nn), 1):
            return IDENTIFIER
        return NUMERIC
    # text-like
    if n and nunique > 0.95 * len(nn) and len(nn) > 20:
        avg_len = nn.astype(str).str.len().mean()
        return TEXT if avg_len > 30 else IDENTIFIER
    if nn.astype(str).str.len().mean() > 60:
        return TEXT
    return CATEGORICAL


def column_types(df: pd.DataFrame) -> dict[str, str]:
    return {c: semantic_type(df[c]) for c in df.columns}


def numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


MEASURE_HINTS = ["sales", "revenue", "amount", "profit", "income", "price", "cost", "visits", "value",
                 "score", "cgpa", "total", "charges", "temperature", "power", "quantity"]
NON_MEASURES = ("latitude", "longitude", "lat", "lon", "lng", "year", "zip", "zipcode", "postal_code")


def ranked_measures(df: pd.DataFrame) -> list[str]:
    """Numeric columns ordered by how likely they are the 'headline' measure."""
    cols = [c for c in numeric_cols(df)
            if str(c).lower() not in NON_MEASURES and not str(c).lower().endswith(("id", "_id"))]

    def rank(c):
        name = str(c).lower()
        return next((i for i, h in enumerate(MEASURE_HINTS) if h in name), len(MEASURE_HINTS))
    return sorted(cols, key=rank)


ADDITIVE_HINTS = ("sales", "revenue", "amount", "profit", "income", "cost", "quantity", "qty", "visits",
                  "signups", "total", "count", "units", "orders", "spend", "charges")


def default_agg(col) -> str:
    """'sum' for additive measures (sales, quantity …), 'mean' for rates/levels (temperature, rating …)."""
    name = str(col).lower()
    if any(k in name for k in ("rate", "pct", "percent", "avg", "mean", "ratio", "score", "rating")):
        return "mean"
    return "sum" if any(h in name for h in ADDITIVE_HINTS) else "mean"


def datetime_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]


def categorical_cols(df: pd.DataFrame, max_unique: int | None = None) -> list[str]:
    out = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        if max_unique is not None and s.nunique() > max_unique:
            continue
        out.append(c)
    return out


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    types = column_types(df)
    rows = []
    for c in df.columns:
        s = df[c]
        row = {
            "Column": c,
            "Semantic type": types[c],
            "Pandas dtype": str(s.dtype),
            "Non-null": int(s.notna().sum()),
            "Missing": int(s.isna().sum()),
            "Missing %": round(100 * s.isna().mean(), 2) if len(s) else 0.0,
            "Unique": int(s.nunique()),
            "Sample": ", ".join(map(str, s.dropna().unique()[:3])),
        }
        if types[c] == NUMERIC:
            row.update({
                "Mean": round(float(s.mean()), 4) if s.notna().any() else None,
                "Std": round(float(s.std()), 4) if s.notna().sum() > 1 else None,
                "Min": float(s.min()) if s.notna().any() else None,
                "Max": float(s.max()) if s.notna().any() else None,
                "Skew": round(float(s.skew()), 3) if s.notna().sum() > 2 else None,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def dataset_overview(df: pd.DataFrame) -> dict:
    types = column_types(df)
    counts = pd.Series(types).value_counts().to_dict()
    mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_cells": int(df.isna().sum().sum()),
        "missing_pct": round(100 * df.isna().mean().mean(), 2) if df.size else 0.0,
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(mem, 2),
        "type_counts": counts,
    }


def outlier_counts(df: pd.DataFrame, method: str = "iqr") -> pd.Series:
    out = {}
    for c in numeric_cols(df):
        s = df[c].dropna()
        if len(s) < 4:
            out[c] = 0
            continue
        if method == "zscore":
            z = (s - s.mean()) / (s.std() or 1)
            out[c] = int((z.abs() > 3).sum())
        else:
            q1, q3 = s.quantile([.25, .75])
            iqr = q3 - q1
            out[c] = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
    return pd.Series(out, dtype="int64")


def quality_score(df: pd.DataFrame) -> dict:
    """0-100 data-quality score from completeness, uniqueness, validity and consistency."""
    if df.empty:
        return {"score": 0, "completeness": 0, "uniqueness": 0, "validity": 0, "consistency": 0}
    completeness = 100 * (1 - df.isna().mean().mean())
    uniqueness = 100 * (1 - df.duplicated().mean())
    num = numeric_cols(df)
    if num:
        oc = outlier_counts(df)
        validity = 100 * (1 - (oc.sum() / max(1, df[num].notna().sum().sum())))
    else:
        validity = 100.0
    # consistency: share of text columns without leading/trailing spaces or mixed casing variants
    cons_scores = []
    for c in categorical_cols(df, max_unique=500):
        s = df[c].dropna().astype(str)
        if s.empty:
            continue
        stripped = (s == s.str.strip()).mean()
        variants = s.str.strip().str.lower().nunique() / max(1, s.nunique())
        cons_scores.append(100 * stripped * variants)
    consistency = float(np.mean(cons_scores)) if cons_scores else 100.0
    score = 0.4 * completeness + 0.2 * uniqueness + 0.2 * validity + 0.2 * consistency
    return {k: round(float(v), 1) for k, v in {
        "score": score, "completeness": completeness, "uniqueness": uniqueness,
        "validity": validity, "consistency": consistency}.items()}


def strong_correlations(df: pd.DataFrame, threshold: float = 0.7, method: str = "pearson") -> pd.DataFrame:
    num = numeric_cols(df)
    if len(num) < 2:
        return pd.DataFrame(columns=["Column 1", "Column 2", "Correlation"])
    corr = df[num].corr(method=method)
    pairs = []
    for i, a in enumerate(num):
        for b in num[i + 1:]:
            v = corr.loc[a, b]
            if pd.notna(v) and abs(v) >= threshold:
                pairs.append((a, b, round(float(v), 3)))
    return pd.DataFrame(pairs, columns=["Column 1", "Column 2", "Correlation"]).sort_values(
        "Correlation", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)
