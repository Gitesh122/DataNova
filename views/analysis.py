"""Data Profile page and Statistics page."""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import profiling, stats_tests, viz
from views import ui


# ================================================================ PROFILE
def render_profile():
    ui.page_header("Data Profile", "Smart data detection — what is in your data and how healthy is it?", "🔍")
    df = ui.require_data()
    raw = ui.raw_df()
    ov = profiling.dataset_overview(df)
    q = profiling.quality_score(df)

    c = st.columns(6)
    c[0].metric("Rows", f"{ov['rows']:,}", delta=f"{ov['rows'] - len(raw):,} vs raw" if ov['rows'] != len(raw) else None)
    c[1].metric("Columns", ov["columns"])
    c[2].metric("Missing cells", f"{ov['missing_cells']:,}", delta=f"{ov['missing_pct']}%", delta_color="inverse")
    c[3].metric("Duplicate rows", f"{ov['duplicate_rows']:,}")
    c[4].metric("Memory", f"{ov['memory_mb']} MB")
    c[5].metric("Quality score", f"{q['score']}/100")

    t1, t2, t3, t4, t5 = st.tabs(["🧬 Columns", "🩺 Quality", "📈 Distributions", "🔗 Relationships", "💡 Recommendations"])
    with t1:
        prof = profiling.column_profile(df)
        types = prof["Semantic type"].value_counts()
        st.markdown(" ".join(f"<span class='ad-pill'>{t}: {n}</span>" for t, n in types.items()), unsafe_allow_html=True)
        st.write("")
        ui.show_table(prof, "profile_cols", "Column profile (smart data detection)", "Data Profile")
        st.markdown("**Summary statistics**")
        st.dataframe(df.describe(include="all").T, width="stretch")
    with t2:
        g1, g2 = st.columns([1, 2])
        with g1:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=q["score"], title={"text": "Data quality"},
                                         gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#4F46E5"},
                                                "steps": [{"range": [0, 65], "color": "rgba(255,86,48,.25)"},
                                                          {"range": [65, 85], "color": "rgba(255,171,0,.25)"},
                                                          {"range": [85, 100], "color": "rgba(54,179,126,.25)"}]}))
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10), template=ui.template())
            st.plotly_chart(fig, width="stretch")
        with g2:
            dims = {k: v for k, v in q.items() if k != "score"}
            fig = px.bar(x=list(dims.values()), y=[k.title() for k in dims], orientation="h", range_x=[0, 100],
                         template=ui.template(), title="Quality dimensions (%)", color_discrete_sequence=["#06B6D4"],
                         labels={"x": "%", "y": ""}, text=list(dims.values()))
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
            st.plotly_chart(fig, width="stretch")
        if df.isna().any().any():
            ui.show_fig(viz.missing_heatmap(df, ui.template()), "missing_map", "Missing-value map", "Data Profile")
        else:
            st.success("No missing values in the current data. 🎉")
        oc = profiling.outlier_counts(df)
        if oc.sum():
            st.markdown("**Outliers per numeric column (1.5 × IQR rule)**")
            st.bar_chart(oc[oc > 0])
    with t3:
        num = profiling.numeric_cols(df)
        cats = profiling.categorical_cols(df, max_unique=40)
        pick = st.multiselect("Columns", num + cats, default=(num[:4] + cats[:2]), key="prof_dist")
        cols = st.columns(2)
        for i, c in enumerate(pick):
            with cols[i % 2]:
                if c in num:
                    fig = px.histogram(df, x=c, nbins=40, marginal="box", template=ui.template(),
                                       color_discrete_sequence=[ui.palette_colors()[i % 8]], title=c)
                else:
                    vc = df[c].astype(str).value_counts().head(15)
                    fig = px.bar(x=vc.values, y=vc.index, orientation="h", template=ui.template(), title=c,
                                 color_discrete_sequence=[ui.palette_colors()[i % 8]], labels={"x": "count", "y": ""})
                    fig.update_yaxes(autorange="reversed")
                fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, width="stretch", key=f"pd_{c}")
    with t4:
        num = profiling.numeric_cols(df)
        if len(num) < 2:
            st.info("Need at least two numeric columns.")
        else:
            thr = st.slider("Correlation threshold", 0.3, 0.95, 0.7, 0.05)
            sc = profiling.strong_correlations(df, thr)
            if sc.empty:
                st.info(f"No pairs with |r| ≥ {thr}.")
            else:
                ui.show_table(sc, "strong_corr", f"Strong correlations (|r| ≥ {thr})", "Data Profile")
            ui.show_fig(viz.correlation_heatmap(df, template=ui.template()), "prof_corr", source="Data Profile")
    with t5:
        st.write("Charts DataNova recommends for this data (open **Auto Dashboard** to see them all rendered):")
        for r in viz.recommend_charts(df):
            label = "Correlation heatmap" if r["kind"] == "heatmap_corr" else viz.CHART_TYPES[r["kind"]][0]
            st.markdown(f"- **{label}** — {r['why']}")


# ================================================================ STATISTICS
def render_stats():
    ui.page_header("Statistics", "Correlations and hypothesis tests, with plain-English interpretations.", "📐")
    df = ui.require_data()
    num = profiling.numeric_cols(df)
    cats = [c for c in profiling.categorical_cols(df, max_unique=50) if df[c].nunique() >= 2]
    t1, t2, t3, t4 = st.tabs(["🔗 Correlation", "🧪 Hypothesis tests", "📏 Normality", "📋 Crosstab"])

    with t1:
        if len(num) < 2:
            st.info("Need at least two numeric columns.")
        else:
            c1, c2 = st.columns([1, 3])
            method = c1.radio("Method", ["pearson", "spearman", "kendall"], help="Pearson = linear; Spearman/Kendall = rank (robust to outliers)")
            sel = c2.multiselect("Columns", num, default=num[:12])
            if len(sel) >= 2:
                ui.show_fig(viz.correlation_heatmap(df[sel], method, ui.template()), f"st_corr_{method}", source="Statistics")
                a, b = st.columns(2)
                x = a.selectbox("Test a pair: X", sel, key="ct_x")
                y = b.selectbox("Y", [c for c in sel if c != x], key="ct_y")
                res = stats_tests.correlation_test(df, x, y, method)
                st.info(res["interpretation"])
                ui.show_fig(viz.build_chart(df, "scatter", x=x, y=y, template=ui.template(), trendline=True,
                                            palette=ui.palette()), f"st_pair_{x}_{y}", source="Statistics")

    with t2:
        test = st.selectbox("Test", ["Compare two groups (t-test / Mann-Whitney)", "Compare many groups (ANOVA / Kruskal)",
                                     "Association of two categories (chi-square)"])
        alpha = st.select_slider("Significance level α", [0.01, 0.05, 0.1], value=0.05)
        try:
            if test.startswith("Compare two"):
                if not (num and cats):
                    st.info("Needs a numeric and a categorical column.")
                else:
                    c1, c2 = st.columns(2)
                    v = c1.selectbox("Numeric value", num, key="tt_v")
                    g = c2.selectbox("Group column", cats, key="tt_g")
                    levels = df[g].dropna().astype(str).value_counts().index.tolist()
                    c3, c4, c5 = st.columns(3)
                    a = c3.selectbox("Group A", levels, key="tt_a")
                    b = c4.selectbox("Group B", [l for l in levels if l != a], key="tt_b")
                    param = c5.toggle("Parametric (t-test)", value=True)
                    d = df.assign(**{g: df[g].astype(str)})
                    res = stats_tests.two_group_test(d, v, g, a, b, param, alpha)
                    st.success(res["interpretation"]) if res["p_value"] < alpha else st.info(res["interpretation"])
                    ui.show_fig(px.box(d[d[g].isin([a, b])], x=g, y=v, color=g, template=ui.template(), points="outliers",
                                       title=f"{v}: {a} vs {b}", color_discrete_sequence=ui.palette_colors()),
                                f"tt_{v}_{g}", source="Statistics")
            elif test.startswith("Compare many"):
                if not (num and cats):
                    st.info("Needs a numeric and a categorical column.")
                else:
                    c1, c2, c3 = st.columns(3)
                    v = c1.selectbox("Numeric value", num, key="an_v")
                    g = c2.selectbox("Group column", cats, key="an_g")
                    param = c3.toggle("Parametric (ANOVA)", value=True)
                    res = stats_tests.anova(df, v, g, param, alpha)
                    st.success(res["interpretation"]) if res["p_value"] < alpha else st.info(res["interpretation"])
                    means = df.groupby(g, observed=True)[v].agg(["count", "mean", "median", "std"]).round(3)
                    st.dataframe(means, width="stretch")
                    ui.show_fig(viz.build_chart(df, "violin", x=g, y=v, template=ui.template(), palette=ui.palette()),
                                f"an_{v}_{g}", source="Statistics")
            else:
                if len(cats) < 2:
                    st.info("Needs two categorical columns.")
                else:
                    c1, c2 = st.columns(2)
                    a = c1.selectbox("Category A", cats, key="chi_a")
                    b = c2.selectbox("Category B", [c for c in cats if c != a], key="chi_b")
                    res = stats_tests.chi_square(df, a, b, alpha)
                    st.success(res["interpretation"]) if res["p_value"] < alpha else st.info(res["interpretation"])
                    ui.show_fig(px.imshow(res["table"], text_auto=True, color_continuous_scale="Teal", aspect="auto",
                                          template=ui.template(), title=f"{a} × {b} counts"), f"chi_{a}_{b}", source="Statistics")
        except ValueError as exc:
            st.warning(str(exc))

    with t3:
        if not num:
            st.info("No numeric columns.")
        else:
            v = st.selectbox("Column", num, key="norm_v")
            try:
                res = stats_tests.normality(df[v])
                st.info(res["interpretation"])
            except ValueError as exc:
                st.warning(str(exc))
            import numpy as np
            from scipy import stats as sps
            s = df[v].dropna().astype(float)
            s = s.sample(min(len(s), 3000), random_state=0).sort_values()
            theo = sps.norm.ppf((np.arange(1, len(s) + 1) - 0.5) / len(s))
            fig = px.scatter(x=theo, y=s.values, labels={"x": "Theoretical quantiles", "y": "Sample quantiles"},
                             template=ui.template(), title=f"Q-Q plot of {v}", color_discrete_sequence=["#06B6D4"])
            fig.add_trace(go.Scatter(x=theo, y=s.mean() + s.std() * theo, mode="lines", name="Normal", line=dict(color="#FF5630")))
            ui.show_fig(fig, f"qq_{v}", source="Statistics")

    with t4:
        if len(cats) < 2:
            st.info("Needs two categorical columns.")
        else:
            c1, c2, c3 = st.columns(3)
            a = c1.selectbox("Rows", cats, key="xt_a")
            b = c2.selectbox("Columns", [c for c in cats if c != a], key="xt_b")
            norm = c3.selectbox("Show", ["Counts", "Row %", "Column %"])
            import pandas as pd
            nmap = {"Counts": False, "Row %": "index", "Column %": "columns"}
            tab = pd.crosstab(df[a], df[b], normalize=nmap[norm], margins=norm == "Counts")
            if norm != "Counts":
                tab = (tab * 100).round(1)
            ui.show_table(tab, f"xt_{a}_{b}", f"Crosstab {a} × {b} ({norm})", "Statistics")
