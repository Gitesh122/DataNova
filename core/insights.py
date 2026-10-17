"""Automatic, rule-based insights written in plain English.

The generator scans the dataset for things an analyst would point out:
data-quality problems, skewed distributions, strong correlations, dominant
categories, time trends, seasonality and anomalies. Each insight carries a
severity so the UI can colour it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .profiling import categorical_cols, datetime_cols, numeric_cols, outlier_counts, quality_score, ranked_measures


def _ins(kind, title, text, level="info", metric=None):
    return {"kind": kind, "title": title, "text": text, "level": level, "metric": metric}


def generate_insights(df: pd.DataFrame, max_items: int = 25) -> list[dict]:
    out: list[dict] = []
    if df.empty:
        return [_ins("quality", "Empty dataset", "There are no rows to analyse.", "error")]

    q = quality_score(df)
    lvl = "success" if q["score"] >= 85 else "warning" if q["score"] >= 65 else "error"
    out.append(_ins("quality", f"Data quality score: {q['score']}/100",
                    f"Completeness {q['completeness']}%, uniqueness {q['uniqueness']}%, "
                    f"validity {q['validity']}%, consistency {q['consistency']}%.", lvl, q["score"]))

    miss = df.isna().mean().sort_values(ascending=False)
    for c, m in miss[miss > 0.05].head(3).items():
        out.append(_ins("quality", f"'{c}' is {m:.0%} missing",
                        "Consider imputing it on the Clean page or excluding it from models.",
                        "warning" if m < .4 else "error"))
    dup = df.duplicated().sum()
    if dup:
        out.append(_ins("quality", f"{dup:,} duplicate rows", "Duplicates inflate totals; remove them on the Clean page.", "warning"))

    num = ranked_measures(df)
    for c in num:
        s = df[c].dropna()
        if len(s) < 10 or s.nunique() < 3:
            continue
        sk = s.skew()
        if abs(sk) > 1.5:
            out.append(_ins("distribution", f"'{c}' is heavily {'right' if sk > 0 else 'left'}-skewed (skew {sk:.2f})",
                            f"Median {s.median():,.2f} vs mean {s.mean():,.2f}. A log transform may help models.", "info"))
    oc = outlier_counts(df)
    for c, n in oc.sort_values(ascending=False).head(3).items():
        if n and n / len(df) > 0.005:
            out.append(_ins("anomaly", f"{n:,} outliers in '{c}' ({n / len(df):.1%} of rows)",
                            "Values beyond 1.5×IQR. Inspect them on AI Insights → Anomalies or cap them on the Clean page.", "warning"))

    if len(num) >= 2:
        corr = df[num].corr()
        pairs = []
        for i, a in enumerate(num):
            for b in num[i + 1:]:
                v = corr.loc[a, b]
                if pd.notna(v) and abs(v) >= 0.6:
                    pairs.append((abs(v), a, b, v))
        for _, a, b, v in sorted(pairs, reverse=True)[:4]:
            out.append(_ins("correlation", f"'{a}' and '{b}' move {'together' if v > 0 else 'in opposite directions'} (r = {v:.2f})",
                            "Strong linear relationship — useful for prediction, but watch for multicollinearity.", "success"))

    for c in categorical_cols(df, max_unique=50)[:5]:
        vc = df[c].value_counts(normalize=True)
        if len(vc) >= 2 and vc.iloc[0] > 0.5:
            out.append(_ins("category", f"'{vc.index[0]}' dominates '{c}' ({vc.iloc[0]:.0%})",
                            f"Classes are imbalanced across {len(vc)} values.", "info"))
        if num and 2 <= df[c].nunique() <= 30:
            target = num[0]
            g = df.groupby(c, observed=True)[target].mean().sort_values(ascending=False)
            if len(g) >= 2 and g.iloc[-1] != 0:
                ratio = g.iloc[0] / g.iloc[-1] if g.iloc[-1] else np.inf
                if np.isfinite(ratio) and ratio > 1.3:
                    out.append(_ins("category", f"'{g.index[0]}' has the highest average {target} by {c}",
                                    f"{g.iloc[0]:,.2f} vs lowest '{g.index[-1]}' {g.iloc[-1]:,.2f} ({ratio:.1f}×).", "success"))

    dts = datetime_cols(df)
    if dts and num:
        dcol = dts[0]
        for c in num[:3]:
            d = df[[dcol, c]].dropna().sort_values(dcol)
            if len(d) < 20:
                continue
            t = (d[dcol] - d[dcol].min()).dt.total_seconds().values
            if t.max() == 0:
                continue
            slope, intercept = np.polyfit(t / t.max(), d[c].values, 1)
            base = abs(d[c].mean()) or 1
            change = slope / base * 100
            if abs(change) > 5:
                out.append(_ins("trend", f"'{c}' is trending {'up' if slope > 0 else 'down'} over time",
                                f"Linear trend implies ≈{change:+.0f}% change across the period "
                                f"{d[dcol].min():%Y-%m-%d} → {d[dcol].max():%Y-%m-%d}.",
                                "success" if slope > 0 else "warning"))
            # day-of-week seasonality
            dow = d.groupby(d[dcol].dt.day_name())[c].mean()
            if len(dow) == 7 and dow.mean():
                spread = (dow.max() - dow.min()) / abs(dow.mean())
                if spread > 0.15:
                    out.append(_ins("trend", f"Weekly pattern in '{c}'",
                                    f"Highest on {dow.idxmax()}, lowest on {dow.idxmin()} ({spread:.0%} spread).", "info"))
                    break
    return out[:max_items]


def summary_paragraph(df: pd.DataFrame) -> str:
    num, cats, dts = ranked_measures(df), categorical_cols(df), datetime_cols(df)
    parts = [f"The dataset has **{len(df):,} rows** and **{df.shape[1]} columns** "
             f"({len(numeric_cols(df))} numeric, {len(cats)} categorical, {len(dts)} date)."]
    if dts:
        d = df[dts[0]].dropna()
        if not d.empty:
            parts.append(f"It covers **{d.min():%d %b %Y} – {d.max():%d %b %Y}**.")
    if num:
        c = num[0]
        parts.append(f"'{c}' ranges from {df[c].min():,.2f} to {df[c].max():,.2f} (median {df[c].median():,.2f}).")
    miss = df.isna().mean().mean()
    parts.append("There are no missing values." if miss == 0 else f"About {miss:.1%} of cells are missing.")
    return " ".join(parts)
