import streamlit as st

from core import insights as ins
from core import viz
from views import ui


def _render_rec(df, r, key):
    if r["kind"] == "heatmap_corr":
        return viz.correlation_heatmap(df, template=ui.template())
    return viz.build_chart(df, r["kind"], x=r.get("x"), y=r.get("y"), color=r.get("color"), agg=r.get("agg", "sum"),
                           template=ui.template(), palette=ui.palette())


def render():
    ui.page_header("Auto Dashboard", "Zero-click dashboard: KPIs, the most useful charts and key insights, generated from your data.", "⚡")
    df = ui.require_data()

    kpis = viz.kpi_values(df)
    cols = st.columns(len(kpis))
    for c, (label, val, delta) in zip(cols, kpis):
        c.metric(label, val, delta)

    st.markdown(f"<div class='ad-insight'><b>At a glance</b>{ins.summary_paragraph(df).replace('**', '')}</div>",
                unsafe_allow_html=True)

    recs = viz.recommend_charts(df, max_charts=8)
    figs = []
    left, right = st.columns(2)
    for i, r in enumerate(recs):
        try:
            fig = _render_rec(df, r, i)
            fig.update_layout(height=380)
            figs.append((r, fig))
            with (left if i % 2 == 0 else right):
                st.plotly_chart(fig, width="stretch", key=f"auto_{i}")
                st.caption(f"💡 {r['why']}")
        except Exception as exc:  # noqa: BLE001
            with (left if i % 2 == 0 else right):
                st.caption(f"(Skipped a chart: {exc})")

    st.subheader("💡 Key insights")
    found = ins.generate_insights(df, max_items=10)
    c1, c2 = st.columns(2)
    for i, item in enumerate(found):
        with (c1 if i % 2 == 0 else c2):
            ui.insight_box(item)

    st.divider()
    if st.button("📌 Add the whole dashboard to the report", type="primary"):
        ui.add_section("Key metrics", "metrics", [(a, b) for a, b, _ in kpis], source="Auto Dashboard")
        ui.add_section("Executive summary", "text", ins.summary_paragraph(df), source="Auto Dashboard")
        for r, fig in figs:
            ui.add_section(fig.layout.title.text or r["why"], "figure", fig, source="Auto Dashboard", note=r["why"])
        ui.add_section("Key insights", "text", "\n".join(f"- **{x['title']}** — {x['text']}" for x in found),
                       source="Auto Dashboard")
        st.success(f"Added {len(figs) + 3} sections. Open **Reports & Sharing** to export.")
