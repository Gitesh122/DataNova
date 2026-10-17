"""Statistical tests with plain-English interpretations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _verdict(p, alpha=0.05):
    return "significant" if p < alpha else "not significant"


def normality(series: pd.Series, alpha=0.05):
    s = series.dropna().astype(float)
    if len(s) < 8:
        raise ValueError("Need at least 8 values.")
    if len(s) <= 5000:
        stat, p = stats.shapiro(s)
        test = "Shapiro-Wilk"
    else:
        stat, p = stats.normaltest(s)
        test = "D'Agostino K²"
    msg = (f"{test}: W/K={stat:.4f}, p={p:.4g}. "
           + ("The data looks **normally distributed**." if p >= alpha else
              "The data is **not normally distributed** — prefer non-parametric tests (Mann-Whitney, Kruskal)."))
    return {"test": test, "statistic": stat, "p_value": p, "interpretation": msg}


def two_group_test(df, value_col, group_col, group_a, group_b, parametric=True, alpha=0.05):
    a = df.loc[df[group_col] == group_a, value_col].dropna().astype(float)
    b = df.loc[df[group_col] == group_b, value_col].dropna().astype(float)
    if len(a) < 3 or len(b) < 3:
        raise ValueError("Each group needs at least 3 values.")
    if parametric:
        stat, p = stats.ttest_ind(a, b, equal_var=False)
        test = "Welch's t-test"
    else:
        stat, p = stats.mannwhitneyu(a, b)
        test = "Mann-Whitney U"
    pooled = np.sqrt((a.var() + b.var()) / 2) or 1
    d = (a.mean() - b.mean()) / pooled
    msg = (f"{test}: statistic={stat:.3f}, p={p:.4g}. Mean {value_col}: {group_a}={a.mean():.3f}, "
           f"{group_b}={b.mean():.3f} (Cohen's d={d:.2f}). The difference is **{_verdict(p, alpha)}** at α={alpha}.")
    return {"test": test, "statistic": stat, "p_value": p, "effect_size": d, "interpretation": msg}


def anova(df, value_col, group_col, parametric=True, alpha=0.05):
    groups = [g[value_col].dropna().astype(float) for _, g in df.groupby(group_col, observed=True) if g[value_col].notna().sum() >= 2]
    if len(groups) < 2:
        raise ValueError("Need at least two groups with data.")
    if parametric:
        stat, p = stats.f_oneway(*groups)
        test = "One-way ANOVA"
    else:
        stat, p = stats.kruskal(*groups)
        test = "Kruskal-Wallis"
    msg = (f"{test} across {len(groups)} groups of {group_col}: statistic={stat:.3f}, p={p:.4g}. "
           f"Group means differ **{_verdict(p, alpha)}**ly." if p < alpha else
           f"{test} across {len(groups)} groups of {group_col}: statistic={stat:.3f}, p={p:.4g}. "
           f"No significant difference between group means.")
    return {"test": test, "statistic": stat, "p_value": p, "interpretation": msg}


def chi_square(df, col_a, col_b, alpha=0.05):
    table = pd.crosstab(df[col_a], df[col_b])
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("Both columns need at least two categories.")
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.values.sum()
    v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    msg = (f"Chi-square test of independence: χ²={chi2:.3f}, dof={dof}, p={p:.4g}, Cramér's V={v:.3f}. "
           + (f"**{col_a} and {col_b} are associated.**" if p < alpha else f"No evidence that {col_a} and {col_b} are related."))
    return {"test": "Chi-square", "statistic": chi2, "p_value": p, "cramers_v": v, "table": table, "interpretation": msg}


def correlation_test(df, col_a, col_b, method="pearson", alpha=0.05):
    d = df[[col_a, col_b]].dropna().astype(float)
    fn = {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}[method]
    r, p = fn(d[col_a], d[col_b])
    strength = "very strong" if abs(r) > .8 else "strong" if abs(r) > .6 else "moderate" if abs(r) > .4 else "weak" if abs(r) > .2 else "negligible"
    direction = "positive" if r > 0 else "negative"
    msg = (f"{method.title()} r={r:.3f}, p={p:.4g} → a {strength} {direction} relationship, "
           f"which is **{_verdict(p, alpha)}**.")
    return {"test": f"{method.title()} correlation", "statistic": r, "p_value": p, "interpretation": msg}
