import pandas as pd
import plotly.express as px
import streamlit as st

from core import data_io, profiling
from core.viz import AGGREGATIONS
from views import ui


def render():
    ui.page_header("Explore & Pivot", "Browse, search, group and pivot the (filtered) data.", "🧭")
    df = ui.require_data()
    t1, t2, t3 = st.tabs(["🔎 Data grid", "🧮 Group-by summary", "🔄 Pivot table"])

    with t1:
        c1, c2, c3 = st.columns([2, 2, 1])
        q = c1.text_input("Search all columns", placeholder="type to search…")
        cols = c2.multiselect("Columns to show", list(df.columns), default=list(df.columns)[:15])
        n = c3.number_input("Rows", 10, 100000, 500, step=100)
        view = df
        if q:
            mask = view.astype(str).apply(lambda s: s.str.contains(q, case=False, na=False, regex=False)).any(axis=1)
            view = view[mask]
        view = view[cols or list(df.columns)]
        st.caption(f"{len(view):,} matching rows")
        st.dataframe(view.head(int(n)), width="stretch", height=480)
        d1, d2, _ = st.columns([1, 1, 4])
        d1.download_button("⬇️ CSV (filtered)", data_io.to_csv_bytes(view), "filtered_data.csv", "text/csv")
        if d2.button("📊 Prepare Excel", help="Build an .xlsx of the filtered rows"):
            st.session_state["explore_xlsx"] = data_io.to_excel_bytes({"data": view})
        if st.session_state.get("explore_xlsx"):
            d2.download_button("⬇️ Excel (filtered)", st.session_state["explore_xlsx"], "filtered_data.xlsx")

    with t2:
        cats = profiling.categorical_cols(df, max_unique=500) + profiling.datetime_cols(df)
        num = profiling.numeric_cols(df)
        c1, c2, c3 = st.columns(3)
        by = c1.multiselect("Group by", cats, default=cats[:1])
        vals = c2.multiselect("Values", num, default=profiling.ranked_measures(df)[:2])
        aggs = c3.multiselect("Aggregations", AGGREGATIONS, default=["count", "sum", "mean"])
        if by and aggs:
            if vals:
                g = df.groupby(by, observed=True)[vals].agg(aggs)
                g.columns = [f"{a}({v})" for v, a in g.columns]
            else:
                g = df.groupby(by, observed=True).size().to_frame("count")
            g = g.reset_index()
            sort_col = st.selectbox("Sort by", list(g.columns), index=len(by))
            g = g.sort_values(sort_col, ascending=False)
            ui.show_table(g.round(4), "groupby", f"Summary by {', '.join(by)}", "Explore")
            if len(by) == 1 and len(g) <= 60:
                fig = px.bar(g, x=by[0], y=sort_col, template=ui.template(), color_discrete_sequence=ui.palette_colors(),
                             title=f"{sort_col} by {by[0]}")
                ui.show_fig(fig, "groupby_fig", source="Explore")

    with t3:
        cats = profiling.categorical_cols(df, max_unique=200) + profiling.datetime_cols(df)
        num = profiling.numeric_cols(df)
        if not cats:
            st.info("Need at least one categorical column.")
            return
        c1, c2, c3, c4 = st.columns(4)
        rows = c1.selectbox("Rows", cats, key="pv_r")
        with c2:
            colv = ui.col_select("Columns", [c for c in cats if c != rows], "pv_c", none_option=True)
        with c3:
            val = ui.col_select("Values", num, "pv_v", default=(profiling.ranked_measures(df) or [None])[0], none_option=True)
        agg = c4.selectbox("Aggregation", AGGREGATIONS, index=1 if val else 0, key="pv_a")
        d = df.copy()
        for k in (rows, colv):
            if k and pd.api.types.is_datetime64_any_dtype(d[k]):
                d[k] = d[k].dt.to_period("M").astype(str)
        if val and agg != "count":
            pv = pd.pivot_table(d, index=rows, columns=colv, values=val, aggfunc=agg, margins=True, observed=True)
        else:
            pv = pd.crosstab(d[rows], d[colv], margins=True) if colv else d[rows].value_counts().to_frame("count")
        ui.show_table(pv.round(3), "pivot", f"Pivot: {agg}({val or 'rows'}) by {rows}{' × ' + colv if colv else ''}", "Explore")
        if colv and pv.shape[0] <= 60 and pv.shape[1] <= 60:
            heat = pv.drop(index="All", columns="All", errors="ignore")
            fig = px.imshow(heat, text_auto=".3s", aspect="auto", color_continuous_scale="Tealgrn", template=ui.template(),
                            title="Pivot heatmap")
            ui.show_fig(fig, "pivot_heat", source="Explore")
