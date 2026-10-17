import os

import plotly.express as px
import streamlit as st

from core import insights as ins
from core import llm, nlq, profiling
from views import ui


def _api_key():
    try:
        return st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    except Exception:  # noqa: BLE001 - no secrets file
        return os.environ.get("ANTHROPIC_API_KEY")


def render():
    ui.page_header("AI Insights & Ask", "Automatic findings, plain-English questions and segment deep-dives.", "💡")
    df = ui.require_data()
    t1, t2, t3, t4 = st.tabs(["💡 Automatic insights", "🗣️ Ask your data", "🧩 Segment breakdown", "🤖 AI analyst (optional)"])

    with t1:
        st.markdown(ins.summary_paragraph(df))
        items = ins.generate_insights(df)
        kinds = sorted({i["kind"] for i in items})
        pick = st.multiselect("Show", kinds, default=kinds, key="ins_kinds")
        for i in items:
            if i["kind"] in pick:
                ui.insight_box(i)
        if st.button("📌 Add insights to report"):
            ui.add_section("Automatic insights", "text",
                           ins.summary_paragraph(df) + "\n" + "\n".join(f"- **{x['title']}** — {x['text']}" for x in items),
                           source="AI Insights")
            st.toast("Added")

    with t2:
        st.caption("Examples: " + " · ".join(f"`{e}`" for e in nlq.EXAMPLES))
        with st.form("nlq_form", clear_on_submit=False):
            q = st.text_input("Ask a question about your data", placeholder="e.g. average sales by region")
            go = st.form_submit_button("Ask", type="primary")
        if go and q:
            res = nlq.answer(q, df, ui.template())
            st.session_state["nlq_history"].insert(0, (q, res))
            st.session_state["nlq_history"] = st.session_state["nlq_history"][:10]
            ui.log("nl_query", q)
        for i, (q, res) in enumerate(st.session_state["nlq_history"][:5]):
            with st.container(border=True):
                st.markdown(f"**🗣️ {q}**")
                st.markdown(res["text"] or "")
                if res.get("fig") is not None:
                    ui.show_fig(res["fig"], f"nlq_fig_{i}_{abs(hash(q))}", title=q, source="Ask your data")
                if res.get("table") is not None:
                    st.dataframe(res["table"], width="stretch", height=min(400, 40 + 35 * len(res["table"])))

    with t3:
        cats = [c for c in profiling.categorical_cols(df, max_unique=60) if df[c].nunique() >= 2]
        if not cats:
            st.info("No categorical columns to segment by.")
        else:
            default = "funding_type" if "funding_type" in cats else cats[0]
            seg = st.selectbox("Segment by", cats, index=cats.index(default))
            measures = profiling.ranked_measures(df)
            show = st.multiselect("Measures", measures, default=measures[:3])
            summary = df.groupby(seg, observed=True).agg(rows=(seg, "size"), **{f"avg_{m}": (m, "mean") for m in show},
                                                         **{f"total_{m}": (m, "sum") for m in show[:1]})
            summary["share_%"] = (100 * summary["rows"] / summary["rows"].sum()).round(1)
            ui.show_table(summary.round(3).sort_values("rows", ascending=False), f"seg_{seg}", f"Breakdown by {seg}", "AI Insights")
            if show:
                fig = px.bar(summary.reset_index(), x=seg, y=f"avg_{show[0]}", color=seg, template=ui.template(),
                             color_discrete_sequence=ui.palette_colors(), title=f"Average {show[0]} by {seg}")
                ui.show_fig(fig, f"segfig_{seg}", source="AI Insights")
            for val in df[seg].astype(str).value_counts().index[:12]:
                with st.expander(f"{val}"):
                    sub = df[df[seg].astype(str) == val]
                    cols = st.columns(min(4, len(show) + 1))
                    cols[0].metric("Rows", f"{len(sub):,}")
                    for c, m in zip(cols[1:], show):
                        overall = df[m].mean()
                        c.metric(f"avg {m}", f"{sub[m].mean():,.2f}",
                                 f"{(sub[m].mean() - overall) / abs(overall) * 100:+.1f}% vs all" if overall else None)
                    st.dataframe(sub.head(100), width="stretch")

    with t4:
        key = _api_key()
        manual = None
        if not key:
            st.info("Connect an AI model to ask open-ended questions (e.g. *“What should I investigate first?”*). "
                    "Set `ANTHROPIC_API_KEY` as an environment variable or in `.streamlit/secrets.toml`, "
                    "or paste a key below for this session. Only a summary of the data (schema, statistics, 8 sample rows) is sent.")
            manual = st.text_input("API key (session only)", type="password")
        key = key or manual
        if not llm.available(key):
            if key:
                st.warning("Install the `anthropic` package to enable the AI analyst: `pip install anthropic`")
            return
        for m in st.session_state["chat"]:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        prompt = st.chat_input("Ask the AI analyst about this dataset…")
        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        reply = llm.ask(prompt, df, st.session_state["chat"], api_key=key)
                    except Exception as exc:  # noqa: BLE001
                        reply = f"⚠️ AI request failed: {exc}"
                st.markdown(reply)
            st.session_state["chat"] += [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]
        if st.session_state["chat"] and st.button("Clear chat"):
            st.session_state["chat"] = []
            st.rerun()
