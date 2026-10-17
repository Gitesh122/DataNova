"""Natural-language query engine (offline, rule-based).

Understands questions such as::

    how many rows are there
    average sales by region            total profit per category
    top 5 state_code by sales          bottom 3 products by profit
    histogram of age / distribution of cgpa
    trend of visits over time          monthly sales trend
    correlation  /  correlation between sales and profit
    sales vs profit                    outliers in sales
    describe sales                     missing values
    show rows where sales > 500        count by segment

Column names are matched fuzzily, so "customer rating" finds
``customer_rating``. The optional LLM assistant (core/llm.py) can answer
open-ended questions when an API key is configured.
"""
from __future__ import annotations

import difflib
import re

import pandas as pd
import plotly.express as px

from .profiling import datetime_cols, numeric_cols
from .viz import build_chart, correlation_heatmap

AGG_WORDS = {
    "average": "mean", "avg": "mean", "mean": "mean", "total": "sum", "sum": "sum", "count": "count",
    "number": "count", "max": "max", "maximum": "max", "highest": "max", "min": "min", "minimum": "min",
    "lowest": "min", "median": "median", "std": "std", "unique": "nunique", "distinct": "nunique",
}
EXAMPLES = [
    "average sales by region", "top 5 category by profit", "histogram of sales", "monthly sales trend",
    "correlation between sales and profit", "outliers in sales", "show rows where discount > 0.2",
    "count by segment", "describe profit", "missing values", "how many rows",
]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def match_column(text: str, columns, cutoff: float = 0.75):
    """Return the best-matching column for a phrase (or None)."""
    t = _norm(text)
    if not t:
        return None
    normed = {_norm(c): c for c in columns}
    if t in normed:
        return normed[t]
    for n, c in sorted(normed.items(), key=lambda kv: -len(kv[0])):
        if n and (re.search(rf"\b{re.escape(n)}\b", t) or re.search(rf"\b{re.escape(t)}\b", n)):
            return c
    m = difflib.get_close_matches(t, list(normed), n=1, cutoff=cutoff)
    return normed[m[0]] if m else None


def columns_in(text: str, columns) -> list:
    """All columns mentioned in the text, in order of appearance."""
    t = " " + _norm(text) + " "
    found = []
    for c in sorted(columns, key=lambda c: -len(_norm(c))):
        n = _norm(c)
        if n and re.search(rf"\b{re.escape(n)}\b", t):
            pos = t.find(" " + n + " ")
            found.append((pos, c))
            t = t.replace(" " + n + " ", " " + "#" * len(n) + " ")
    if not found:  # fuzzy single words
        for w in _norm(text).split():
            if len(w) > 3:
                c = match_column(w, columns, cutoff=0.8)
                if c and c not in [f[1] for f in found]:
                    found.append((_norm(text).find(w), c))
    return [c for _, c in sorted(found)]


def answer(query: str, df: pd.DataFrame, template: str = "plotly_white") -> dict:
    q = _norm(query)
    cols = list(df.columns)
    num, dts = numeric_cols(df), datetime_cols(df)
    mentioned = columns_in(query, cols)
    res = {"text": None, "table": None, "fig": None, "intent": None}

    if not q:
        res["text"] = "Type a question about your data."
        return res

    if re.search(r"\bhow many (rows|records)\b|\brow count\b|\bsize of\b|\bshape\b", q):
        res.update(intent="shape", text=f"The data has **{len(df):,} rows** and **{df.shape[1]} columns**.")
        return res

    if "missing" in q or "null" in q:
        m = df.isna().sum()
        m = m[m > 0].sort_values(ascending=False)
        res.update(intent="missing", table=m.rename("missing").to_frame(),
                   text="No missing values 🎉" if m.empty else f"{len(m)} column(s) have missing values.")
        return res

    # filter: rows where X > 5
    fm = re.search(r"(?:where|with|rows)\s+(.+?)\s*(>=|<=|!=|==|=|>|<|is|equals|contains)\s*['\"]?([\w .\-]+)['\"]?$",
                   query.strip(), flags=re.I)
    if fm and ("where" in q or "rows" in q or "show" in q):
        col = match_column(fm.group(1), cols)
        op, val = fm.group(2).lower(), fm.group(3).strip()
        if col is not None:
            s = df[col]
            try:
                if op == "contains":
                    mask = s.astype(str).str.contains(val, case=False, na=False)
                else:
                    v = float(val) if pd.api.types.is_numeric_dtype(s) else val
                    if pd.api.types.is_datetime64_any_dtype(s):
                        v = pd.to_datetime(val)
                    op2 = {"=": "==", "is": "==", "equals": "=="}.get(op, op)
                    if isinstance(v, str):
                        mask = s.astype(str).str.lower() == v.lower() if op2 == "==" else s.astype(str).str.lower() != v.lower()
                    else:
                        mask = eval(f"s {op2} v", {"s": s, "v": v})  # noqa: S307 (operator is whitelisted above)
                sub = df[mask]
                res.update(intent="filter", table=sub.head(500),
                           text=f"**{len(sub):,}** row(s) where `{col}` {op} {val} (showing up to 500).")
                return res
            except Exception as exc:  # noqa: BLE001
                res["text"] = f"Couldn't apply that filter: {exc}"
                return res

    if re.search(r"\bcorrelat|\brelationship\b|\brelated\b", q):
        if len(mentioned) >= 2 and all(c in num for c in mentioned[:2]):
            a, b = mentioned[:2]
            r = df[[a, b]].corr().iloc[0, 1]
            res.update(intent="corr_pair", fig=build_chart(df, "scatter", x=a, y=b, template=template, trendline=True),
                       text=f"Pearson correlation between **{a}** and **{b}** is **{r:.3f}**.")
        elif len(num) >= 2:
            res.update(intent="corr", fig=correlation_heatmap(df, template=template), text="Correlation matrix of numeric columns.")
        else:
            res["text"] = "Need at least two numeric columns for correlation."
        return res

    if re.search(r"\b(histogram|distribution|spread)\b", q):
        col = next((c for c in mentioned), num[0] if num else None)
        if col is not None:
            res.update(intent="hist", fig=build_chart(df, "histogram", x=col, template=template),
                       text=f"Distribution of **{col}**.")
            return res

    if re.search(r"\b(outlier|anomal|unusual)", q):
        col = next((c for c in mentioned if c in num), num[0] if num else None)
        if col is not None:
            s = df[col]
            q1, q3 = s.quantile([.25, .75])
            iqr = q3 - q1
            mask = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
            res.update(intent="outliers", table=df[mask].head(500), fig=build_chart(df, "box", y=col, template=template),
                       text=f"Found **{int(mask.sum()):,}** outliers in **{col}** (outside "
                            f"{q1 - 1.5 * iqr:,.2f} – {q3 + 1.5 * iqr:,.2f}).")
            return res

    if re.search(r"\b(trend|over time|timeline|monthly|weekly|daily|yearly|per month|by month)\b", q) and dts:
        col = next((c for c in mentioned if c in num), num[0] if num else None)
        if col is not None:
            freq = "MS" if "month" in q else "W" if "week" in q else "YS" if "year" in q else "D" if "dai" in q else None
            d = df[[dts[0], col]].dropna()
            if freq is None:
                span = (d[dts[0]].max() - d[dts[0]].min()).days
                freq = "D" if span < 120 else "W" if span < 730 else "MS"
            agg = next((AGG_WORDS[w] for w in q.split() if w in AGG_WORDS), "sum")
            g = d.set_index(dts[0])[col].resample(freq).agg(agg if agg != "count" else "count").reset_index()
            fig = px.line(g, x=dts[0], y=col, markers=len(g) < 60, template=template,
                          title=f"{agg}({col}) per {dict(D='day', W='week', MS='month', YS='year')[freq]}")
            fig.update_traces(line=dict(color="#4F46E5", width=3))
            change = (g[col].iloc[-1] - g[col].iloc[0]) / abs(g[col].iloc[0]) * 100 if len(g) > 1 and g[col].iloc[0] else 0
            res.update(intent="trend", fig=fig, table=g.tail(24),
                       text=f"**{col}** went from {g[col].iloc[0]:,.2f} to {g[col].iloc[-1]:,.2f} ({change:+.1f}%).")
            return res

    top = re.search(r"\b(top|bottom|best|worst|largest|smallest)\s+(\d+)?", q)
    if top:
        n = int(top.group(2) or 5)
        asc = top.group(1) in ("bottom", "worst", "smallest")
        cat = next((c for c in mentioned if c not in num), None)
        val = next((c for c in mentioned if c in num), num[0] if num else None)
        if cat is not None and val is not None:
            agg = next((AGG_WORDS[w] for w in q.split() if w in AGG_WORDS), "sum")
            g = df.groupby(cat, observed=True)[val].agg(agg).sort_values(ascending=asc).head(n).reset_index()
            fig = px.bar(g, x=cat, y=val, template=template, title=f"{'Bottom' if asc else 'Top'} {n} {cat} by {agg}({val})",
                         color_discrete_sequence=["#06B6D4"])
            res.update(intent="topn", table=g, fig=fig, text=f"{'Bottom' if asc else 'Top'} {n} **{cat}** by {agg} of **{val}**.")
            return res
        if val is not None:
            sub = df.sort_values(val, ascending=asc).head(n)
            res.update(intent="topn_rows", table=sub, text=f"{'Lowest' if asc else 'Highest'} {n} rows by **{val}**.")
            return res

    by = re.search(r"\b(?:by|per|for each|across|grouped by)\s+(.+)$", q)
    agg = next((AGG_WORDS[w] for w in q.split() if w in AGG_WORDS), None)
    if by:
        group = match_column(by.group(1), cols) or next((c for c in reversed(mentioned) if c not in num), None)
        val = next((c for c in mentioned if c in num and c != group), None)
        if group is not None:
            agg = agg or ("sum" if val is not None else "count")
            if val is None or agg == "count":
                g = df.groupby(group, observed=True).size().sort_values(ascending=False).reset_index(name="count")
                ycol = "count"
            else:
                g = df.groupby(group, observed=True)[val].agg(agg).sort_values(ascending=False).reset_index()
                ycol = val
            fig = px.bar(g.head(30), x=group, y=ycol, template=template, color_discrete_sequence=["#4F46E5"],
                         title=f"{agg}({val or 'rows'}) by {group}")
            top_row = g.iloc[0]
            res.update(intent="groupby", table=g, fig=fig,
                       text=f"{agg.title()} of **{val or 'rows'}** by **{group}** — highest is "
                            f"**{top_row[group]}** ({top_row[ycol]:,.2f}).")
            return res

    vs = re.search(r"(.+?)\s+(?:vs|versus|against)\s+(.+)", q)
    if vs:
        a, b = match_column(vs.group(1), cols), match_column(vs.group(2), cols)
        if a is not None and b is not None:
            kind = "scatter" if a in num and b in num else "box"
            fig = build_chart(df, kind, x=b, y=a, template=template) if kind == "scatter" else \
                build_chart(df, "box", x=b if b not in num else a, y=a if a in num else b, template=template)
            res.update(intent="vs", fig=fig, text=f"**{a}** vs **{b}**.")
            return res

    if agg and mentioned:
        val = next((c for c in mentioned if c in num), None)
        if val is not None:
            v = df[val].agg(agg)
            res.update(intent="scalar", text=f"The {agg} of **{val}** is **{v:,.4g}**.")
            return res
        if agg == "nunique" or "unique" in q:
            c = mentioned[0]
            res.update(intent="unique", text=f"**{c}** has **{df[c].nunique():,}** unique values.",
                       table=df[c].value_counts().head(50).rename("count").to_frame())
            return res

    if re.search(r"\b(describe|summary|summarise|summarize|stats|statistics|about)\b", q):
        if mentioned:
            c = mentioned[0]
            res.update(intent="describe", table=df[c].describe().to_frame(), text=f"Summary statistics for **{c}**.")
        else:
            res.update(intent="describe_all", table=df.describe(include="all").T, text="Summary statistics for all columns.")
        return res

    if mentioned:
        c = mentioned[0]
        if c in num:
            res.update(intent="col", fig=build_chart(df, "histogram", x=c, template=template), table=df[c].describe().to_frame(),
                       text=f"Here is an overview of **{c}**.")
        else:
            vc = df[c].value_counts().head(20).reset_index()
            vc.columns = [c, "count"]
            res.update(intent="col", table=vc, fig=px.bar(vc, x=c, y="count", template=template,
                                                          color_discrete_sequence=["#4F46E5"]),
                       text=f"Most frequent values of **{c}**.")
        return res

    res["text"] = ("I couldn't understand that. Try one of: " + ", ".join(f"*{e}*" for e in EXAMPLES[:6]) + ".")
    return res
