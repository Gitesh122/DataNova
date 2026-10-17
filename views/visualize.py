import streamlit as st

from core import profiling, viz
from views import ui

QUICK = [("histogram", "histogram_icon.png", "Histogram"), ("box", "box_icon.png", "Box plot"),
         ("pie", "pie_icon.png", "Pie chart"), ("line", "line_icon.png", "Line chart"), ("map", "map_icon.png", "Map")]


def _spec_ui(df, prefix, default_kind="bar"):
    num, cats, dts = profiling.numeric_cols(df), profiling.categorical_cols(df, max_unique=100), profiling.datetime_cols(df)
    measures = profiling.ranked_measures(df) or num
    kinds = list(viz.CHART_TYPES)
    kind = st.selectbox("Chart type", kinds, index=kinds.index(default_kind),
                        format_func=lambda k: viz.CHART_TYPES[k][0], key=f"{prefix}_kind")
    allc = list(df.columns)
    c1, c2, c3, c4 = st.columns(4)
    spec = {"kind": kind}
    default_x = {"histogram": (measures or allc)[0], "ecdf": (measures or allc)[0],
                 "line": (dts or allc)[0], "area": (dts or allc)[0],
                 "scatter": (num or allc)[0], "bubble": (num or allc)[0], "heatmap": (num or allc)[0],
                 "density_contour": (num or allc)[0], "scatter_3d": (num or allc)[0],
                 "choropleth": viz.find_state_col(df)}.get(kind, (cats or allc)[0])
    with c1:
        spec["x"] = ui.col_select("X / category", allc, f"{prefix}_x", default=default_x, none_option=kind in ("box", "violin", "map", "scatter_matrix", "parallel"))
    with c2:
        y_default = measures[1] if kind in ("scatter", "bubble", "heatmap", "density_contour", "scatter_3d") and len(measures) > 1 else (measures[0] if measures else None)
        spec["y"] = ui.col_select("Y / value", num, f"{prefix}_y", default=y_default if kind not in ("histogram", "pie", "donut", "ecdf") else None,
                                  none_option=True)
    with c3:
        spec["color"] = ui.col_select("Colour / group", [c for c in allc if df[c].nunique() <= 30 or c in num], f"{prefix}_color", none_option=True)
    with c4:
        spec["agg"] = st.selectbox("Aggregation", viz.AGGREGATIONS, index=1, key=f"{prefix}_agg",
                                   help="Used by bar, line, pie, treemap, funnel, choropleth")
    with st.expander("More options"):
        o1, o2, o3, o4 = st.columns(4)
        with o1:
            spec["size"] = ui.col_select("Size (bubble/map)", num, f"{prefix}_size", none_option=True)
            spec["z"] = ui.col_select("Z (3D / heatmap)", num, f"{prefix}_z", none_option=True)
        with o2:
            spec["facet"] = ui.col_select("Facet (small multiples)", [c for c in cats if df[c].nunique() <= 12], f"{prefix}_facet", none_option=True)
            spec["orientation"] = "h" if st.checkbox("Horizontal bars", key=f"{prefix}_h") else "v"
        with o3:
            spec["top_n"] = st.number_input("Top N categories", 3, 200, 25, key=f"{prefix}_top")
            spec["sort"] = st.selectbox("Sort", ["desc", "asc", "none"], key=f"{prefix}_sort")
        with o4:
            spec["log_y"] = st.checkbox("Log Y axis", key=f"{prefix}_log")
            spec["trendline"] = st.checkbox("Trendline (scatter)", key=f"{prefix}_trend")
            spec["bins"] = st.slider("Histogram bins", 5, 150, 30, key=f"{prefix}_bins")
        spec["title"] = st.text_input("Custom title", key=f"{prefix}_title") or None
    return spec


def render():
    ui.page_header("Chart Studio", "Build any chart in seconds, then pin it to your dashboard or add it to the report.", "📊")
    df = ui.require_data()
    t_build, t_quick, t_board = st.tabs(["🎨 Chart builder", "⚡ Quick charts", f"📌 My dashboard ({len(st.session_state['pinned'])})"])

    with t_build:
        spec = _spec_ui(df, "cb")
        try:
            fig = viz.build_chart(df, template=ui.template(), palette=ui.palette(), **spec)
            ui.show_fig(fig, "chart_builder", source="Chart Studio", height=520)
            if st.button("📌 Pin to my dashboard"):
                st.session_state["pinned"].append(spec)
                ui.log("pin_chart", spec["kind"])
                st.toast("Pinned!", icon="📌")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Can't draw this combination yet: {exc}")

    with t_quick:
        st.caption("Icon gallery — pick a column, click a chart.")
        col = st.selectbox("Column to visualise", list(df.columns), key="qc_col")
        cols = st.columns(len(QUICK))
        for c, (kind, icon, label) in zip(cols, QUICK):
            with c:
                img = ui.b64(ui.ASSETS / "icons" / icon)
                if img:
                    st.markdown(f"<div class='ad-icon' style='text-align:center'><img src='data:image/png;base64,{img}' width='64'></div>",
                                unsafe_allow_html=True)
                if st.button(label, key=f"qc_{kind}", width="stretch"):
                    st.session_state["qc_kind"] = kind
        kind = st.session_state.get("qc_kind")
        if kind:
            try:
                is_num = col in profiling.numeric_cols(df)
                dts = profiling.datetime_cols(df)
                if kind == "box":
                    fig = viz.build_chart(df, "box", y=col if is_num else (profiling.numeric_cols(df) or [None])[0],
                                          x=None if is_num else col, template=ui.template(), palette=ui.palette())
                elif kind == "pie":
                    fig = viz.build_chart(df, "donut", x=col, agg="count", template=ui.template(), palette=ui.palette())
                elif kind == "line":
                    if dts and is_num:
                        fig = viz.build_chart(df, "line", x=dts[0], y=col, agg=profiling.default_agg(col), template=ui.template())
                    else:
                        fig = viz.build_chart(df, "line", x=col, template=ui.template())
                elif kind == "map":
                    fig = viz.build_chart(df, "map", x=col, color=col if df[col].nunique() <= 20 else None, template=ui.template(),
                                          palette=ui.palette())
                else:
                    fig = viz.build_chart(df, "histogram" if is_num else "bar", x=col, agg="count", template=ui.template(),
                                          palette=ui.palette())
                ui.show_fig(fig, f"quick_{kind}", source="Chart Studio")
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

    with t_board:
        pins = st.session_state["pinned"]
        if not pins:
            st.info("Pin charts from the builder to assemble your own dashboard here.")
        else:
            ncol = st.radio("Columns", [1, 2, 3], index=1, horizontal=True)
            grid = st.columns(ncol)
            for i, spec in enumerate(list(pins)):
                with grid[i % ncol]:
                    try:
                        fig = viz.build_chart(df, template=ui.template(), palette=ui.palette(), **spec)
                        fig.update_layout(height=360)
                        st.plotly_chart(fig, width="stretch", key=f"pin_{i}")
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"Pinned chart no longer valid: {exc}")
                        fig = None
                    b1, b2 = st.columns(2)
                    if b1.button("🗑 Unpin", key=f"unpin_{i}"):
                        pins.pop(i)
                        st.rerun()
                    if fig is not None and b2.button("📌 To report", key=f"pinrep_{i}"):
                        ui.add_section(fig.layout.title.text or "Chart", "figure", fig, source="My dashboard")
                        st.toast("Added")
            if st.button("📌 Add all pinned charts to report"):
                for spec in pins:
                    try:
                        fig = viz.build_chart(df, template=ui.template(), palette=ui.palette(), **spec)
                        ui.add_section(fig.layout.title.text or "Chart", "figure", fig, source="My dashboard")
                    except Exception:  # noqa: BLE001
                        pass
                st.success("Added.")
