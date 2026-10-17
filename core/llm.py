"""Optional AI analyst powered by the Anthropic API.

If ``ANTHROPIC_API_KEY`` is set (environment variable or Streamlit secret) and
the ``anthropic`` package is installed, users can ask open-ended questions.
Only a compact *summary* of the dataset (schema, statistics, a few sample
rows) is sent — never the full file.
"""
from __future__ import annotations

import os

import pandas as pd

DEFAULT_MODEL = os.environ.get("DATANOVA_LLM_MODEL", "claude-sonnet-5-5")


def available(api_key: str | None = None) -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(api_key or os.environ.get("ANTHROPIC_API_KEY"))


def dataset_context(df: pd.DataFrame, max_cols: int = 40) -> str:
    cols = df.columns[:max_cols]
    lines = [f"Rows: {len(df)}, Columns: {df.shape[1]}", "Schema:"]
    for c in cols:
        s = df[c]
        lines.append(f"- {c} ({s.dtype}), missing={s.isna().mean():.1%}, unique={s.nunique()}")
    try:
        lines.append("\nNumeric summary:\n" + df[cols].describe().T.round(3).to_string())
    except ValueError:
        pass
    lines.append("\nSample rows:\n" + df[cols].head(8).to_string())
    return "\n".join(lines)[:15000]


def ask(question: str, df: pd.DataFrame, history: list[dict] | None = None, api_key: str | None = None,
        model: str | None = None) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    system = (
        "You are DataNova's data analyst assistant. Answer questions about the user's dataset using the summary "
        "provided. Be concise, quantitative and practical. If the summary is not enough to answer exactly, say what "
        "you'd compute and suggest which DataNova page (Visualize, Machine Learning, Forecasting, Statistics) to use.\n\n"
        "DATASET SUMMARY\n" + dataset_context(df)
    )
    messages = (history or [])[-10:] + [{"role": "user", "content": question}]
    resp = client.messages.create(model=model or DEFAULT_MODEL, max_tokens=1200, system=system, messages=messages)
    return "".join(getattr(b, "text", "") for b in resp.content)
