"""Configurable data-cleaning pipeline.

Each step is a pure function ``(df, **options) -> (df, message)``. The pipeline
records a human-readable log of every change so it can be shown in the UI and
included in reports. ``auto_clean`` applies sensible defaults (snake_case
column names, median / 'Unknown' imputation, drop duplicates).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .profiling import categorical_cols, numeric_cols


def snake_case(name: str) -> str:
    name = re.sub(r"[^\w\s]", " ", str(name)).strip()
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return re.sub(r"[\s_]+", "_", name).lower() or "col"


# ---------------------------------------------------------------- steps
def standardize_columns(df):
    new = [snake_case(c) for c in df.columns]
    seen, final = {}, []
    for n in new:
        if n in seen:
            seen[n] += 1
            final.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 0
            final.append(n)
    changed = sum(a != b for a, b in zip(df.columns, final))
    out = df.copy()
    out.columns = final
    return out, f"Standardized {changed} column name(s) to snake_case"


def trim_text(df):
    out = df.copy()
    n = 0
    for c in categorical_cols(out):
        s = out[c]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            mask = s.notna()
            new = s.copy()
            new[mask] = s[mask].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
            n += int((new[mask] != s[mask]).sum())
            new = new.replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan, "NULL": np.nan, "null": np.nan, "N/A": np.nan, "n/a": np.nan, "-": np.nan})
            out[c] = new
    return out, f"Trimmed whitespace / normalised blanks in {n} text cell(s)"


def drop_empty(df, row_thresh: float = 1.0, col_thresh: float = 0.95):
    """Drop columns with >= col_thresh missing and rows that are entirely empty."""
    cols = [c for c in df.columns if df[c].isna().mean() >= col_thresh]
    out = df.drop(columns=cols)
    before = len(out)
    out = out.dropna(how="all")
    return out, f"Dropped {len(cols)} mostly-empty column(s){' (' + ', '.join(map(str, cols)) + ')' if cols else ''} and {before - len(out)} empty row(s)"


def remove_duplicates(df, subset: list | None = None, keep: str = "first"):
    before = len(df)
    out = df.drop_duplicates(subset=subset or None, keep=keep).reset_index(drop=True)
    return out, f"Removed {before - len(out)} duplicate row(s)"


def impute_missing(df, numeric_strategy: str = "median", categorical_strategy: str = "mode",
                   constant_value: str = "Unknown", columns: list | None = None):
    out = df.copy()
    cols = columns or list(out.columns)
    filled = 0
    for c in cols:
        s = out[c]
        miss = int(s.isna().sum())
        if not miss:
            continue
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            strat = numeric_strategy
            if strat == "drop rows":
                continue
            if strat == "mean":
                out[c] = s.fillna(s.mean())
            elif strat == "median":
                out[c] = s.fillna(s.median())
            elif strat == "zero":
                out[c] = s.fillna(0)
            elif strat == "interpolate":
                out[c] = s.interpolate(limit_direction="both")
            elif strat == "forward fill":
                out[c] = s.ffill().bfill()
            else:
                continue
        elif pd.api.types.is_datetime64_any_dtype(s):
            out[c] = s.ffill().bfill()
        else:
            strat = categorical_strategy
            if strat == "drop rows":
                continue
            if strat == "mode" and s.mode().size:
                out[c] = s.fillna(s.mode().iloc[0])
            elif strat == "constant":
                out[c] = s.fillna(constant_value)
            elif strat == "forward fill":
                out[c] = s.ffill().bfill()
            else:
                continue
        filled += miss - int(out[c].isna().sum())
    msg = f"Filled {filled} missing value(s) (numeric: {numeric_strategy}, categorical: {categorical_strategy})"
    if "drop rows" in (numeric_strategy, categorical_strategy):
        before = len(out)
        target = [c for c in cols if (numeric_strategy == "drop rows" and c in numeric_cols(out))
                  or (categorical_strategy == "drop rows" and c not in numeric_cols(out))]
        out = out.dropna(subset=target).reset_index(drop=True)
        msg += f"; dropped {before - len(out)} row(s) with missing values"
    return out, msg


def handle_outliers(df, method: str = "iqr", action: str = "cap", columns: list | None = None,
                    factor: float = 1.5, z_thresh: float = 3.0):
    out = df.copy()
    cols = columns or numeric_cols(out)
    mask_any = pd.Series(False, index=out.index)
    touched = 0
    for c in cols:
        s = out[c]
        if s.dropna().nunique() < 5:
            continue
        if method == "zscore":
            mu, sd = s.mean(), s.std() or 1
            lo, hi = mu - z_thresh * sd, mu + z_thresh * sd
        else:
            q1, q3 = s.quantile([.25, .75])
            iqr = q3 - q1
            lo, hi = q1 - factor * iqr, q3 + factor * iqr
        mask = (s < lo) | (s > hi)
        touched += int(mask.sum())
        if action == "cap":
            out[c] = s.clip(lo, hi)
        elif action == "set missing":
            out.loc[mask, c] = np.nan
        mask_any |= mask.fillna(False)
    if action == "remove":
        out = out[~mask_any].reset_index(drop=True)
    return out, f"Outliers ({method.upper()}): {touched} value(s) → {action}"


def convert_types(df, conversions: dict[str, str]):
    out = df.copy()
    done = []
    for col, target in conversions.items():
        if col not in out.columns:
            continue
        try:
            if target == "numeric":
                out[col] = pd.to_numeric(out[col].astype(str).str.replace(r"[,\s$€£₹%]", "", regex=True), errors="coerce")
            elif target == "datetime":
                out[col] = pd.to_datetime(out[col], errors="coerce", format="mixed")
            elif target == "category":
                out[col] = out[col].astype("category")
            elif target == "text":
                out[col] = out[col].astype(str)
            elif target == "boolean":
                out[col] = out[col].astype(str).str.lower().isin(["1", "true", "yes", "y", "t"])
            done.append(f"{col}→{target}")
        except Exception as exc:  # noqa: BLE001
            done.append(f"{col}: failed ({exc})")
    return out, "Converted types: " + (", ".join(done) if done else "none")


def normalize_case(df, columns: list | None = None, case: str = "title"):
    out = df.copy()
    cols = columns or categorical_cols(out, max_unique=200)
    for c in cols:
        s = out[c]
        m = s.notna()
        f = {"title": str.title, "lower": str.lower, "upper": str.upper}[case]
        out.loc[m, c] = s[m].astype(str).map(f)
    return out, f"Normalised text case ({case}) in {len(cols)} column(s)"


# ---------------------------------------------------------------- pipeline
@dataclass
class CleaningConfig:
    standardize_names: bool = True
    trim_text: bool = True
    drop_empty: bool = True
    remove_duplicates: bool = True
    impute: bool = True
    numeric_strategy: str = "median"
    categorical_strategy: str = "constant"
    constant_value: str = "Unknown"
    outliers: bool = False
    outlier_method: str = "iqr"
    outlier_action: str = "cap"
    drop_columns: list = field(default_factory=list)


def run_pipeline(df: pd.DataFrame, cfg: CleaningConfig) -> tuple[pd.DataFrame, list[str]]:
    log = []
    out = df.copy()
    if cfg.drop_columns:
        out = out.drop(columns=[c for c in cfg.drop_columns if c in out.columns])
        log.append(f"Dropped column(s): {', '.join(map(str, cfg.drop_columns))}")
    steps = [
        (cfg.standardize_names, standardize_columns, {}),
        (cfg.trim_text, trim_text, {}),
        (cfg.drop_empty, drop_empty, {}),
        (cfg.remove_duplicates, remove_duplicates, {}),
        (cfg.outliers, handle_outliers, {"method": cfg.outlier_method, "action": cfg.outlier_action}),
        (cfg.impute, impute_missing, {"numeric_strategy": cfg.numeric_strategy,
                                      "categorical_strategy": cfg.categorical_strategy,
                                      "constant_value": cfg.constant_value}),
    ]
    for enabled, fn, kw in steps:
        if enabled:
            out, msg = fn(out, **kw)
            log.append(msg)
    if out.empty:
        log.append("⚠️ The cleaned dataset is empty — relax the cleaning options.")
    return out, log


def auto_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One-click cleaning with sensible defaults ."""
    return run_pipeline(df, CleaningConfig())
