"""Feature engineering helpers (all return a new DataFrame + a log message)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def extract_date_parts(df, col, parts=("year", "month", "day", "dayofweek", "quarter")):
    out = df.copy()
    s = pd.to_datetime(out[col], errors="coerce")
    made = []
    for p in parts:
        name = f"{col}_{p}"
        if p == "week":
            out[name] = s.dt.isocalendar().week.astype("Int64")
        elif p == "is_weekend":
            out[name] = s.dt.dayofweek >= 5
        elif p == "month_name":
            out[name] = s.dt.month_name()
        else:
            out[name] = getattr(s.dt, p)
        made.append(name)
    return out, f"Extracted {', '.join(made)} from {col}"


def bin_column(df, col, bins=5, method="equal width", labels=None, new_name=None):
    out = df.copy()
    name = new_name or f"{col}_bin"
    if method == "quantile":
        out[name] = pd.qcut(out[col], q=bins, labels=labels, duplicates="drop").astype(str)
    else:
        out[name] = pd.cut(out[col], bins=bins, labels=labels).astype(str)
    return out, f"Binned {col} into {bins} {method} bins → {name}"


def transform_column(df, col, how="log", new_name=None):
    out = df.copy()
    s = out[col].astype(float)
    name = new_name or f"{col}_{how.replace(' ', '_')}"
    if how == "log":
        out[name] = np.log1p(s.clip(lower=0))
    elif how == "sqrt":
        out[name] = np.sqrt(s.clip(lower=0))
    elif how == "square":
        out[name] = s ** 2
    elif how == "z-score":
        out[name] = (s - s.mean()) / (s.std() or 1)
    elif how == "min-max":
        rng = (s.max() - s.min()) or 1
        out[name] = (s - s.min()) / rng
    elif how == "rank":
        out[name] = s.rank(pct=True)
    else:
        raise ValueError(how)
    return out, f"Applied {how} to {col} → {name}"


def one_hot(df, col, drop_original=False, max_categories=30):
    out = df.copy()
    top = out[col].value_counts().index[:max_categories]
    s = out[col].where(out[col].isin(top), "Other")
    dummies = pd.get_dummies(s, prefix=col, dtype=int)
    out = pd.concat([out, dummies], axis=1)
    if drop_original:
        out = out.drop(columns=[col])
    return out, f"One-hot encoded {col} → {dummies.shape[1]} column(s)"


def formula_column(df, name, expression):
    """Create a column from a pandas expression, e.g. ``profit / sales * 100``."""
    out = df.copy()
    out[name] = out.eval(expression, engine="python")
    return out, f"Created {name} = {expression}"


def rename_columns(df, mapping: dict):
    mapping = {k: v for k, v in mapping.items() if v and k != v}
    return df.rename(columns=mapping), f"Renamed {len(mapping)} column(s)"
