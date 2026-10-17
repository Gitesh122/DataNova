import pandas as pd
import plotly.express as px
import streamlit as st

from core import ml, profiling
from views import ui


def render():
    ui.page_header("Machine Learning Studio", "AutoML for prediction & classification, clustering, anomaly detection and PCA — no code.", "🤖")
    df = ui.require_data()
    t_sup, t_clu, t_ano, t_pca = st.tabs(["🎯 Predict (AutoML)", "🫧 Clustering", "🚨 Anomaly detection", "🧭 PCA"])
    with t_sup:
        supervised(df)
    with t_clu:
        clustering(df)
    with t_ano:
        anomalies(df)
    with t_pca:
        pca(df)


def _feature_candidates(df):
    return [c for c in df.columns if profiling.semantic_type(df[c]) not in (profiling.IDENTIFIER, profiling.TEXT)]


def supervised(df):
    cand = _feature_candidates(df)
    if len(cand) < 2:
        st.info("Need at least two usable columns.")
        return
    c1, c2 = st.columns([1, 2])
    target = c1.selectbox("🎯 Target (what to predict)", cand, index=cand.index(profiling.ranked_measures(df)[0])
                          if profiling.ranked_measures(df) and profiling.ranked_measures(df)[0] in cand else 0, key="ml_target")
    inferred = ml.infer_task(df[target].dropna())
    task = c1.radio("Task", ["regression", "classification"], index=0 if inferred == "regression" else 1, horizontal=True,
                    help=f"Auto-detected: {inferred}", key=f"ml_task_{target}")
    feats = c2.multiselect("Features (inputs)", [c for c in cand if c != target],
                           default=[c for c in cand if c != target][:10], key=f"ml_feats_{target}")
    registry = ml.REGRESSORS if task == "regression" else ml.CLASSIFIERS
    defaults = ml.DEFAULT_REG if task == "regression" else ml.DEFAULT_CLF
    m1, m2, m3 = st.columns([3, 1, 1])
    models = m1.multiselect("Models to compare", list(registry), default=defaults, key=f"ml_models_{task}")
    test_size = m2.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    folds = m3.selectbox("CV folds", [0, 3, 5, 10], index=1, help="0 = skip cross-validation")
    if st.button("🚀 Train & compare", type="primary", disabled=not (feats and models)):
        with st.spinner("Training models…"):
            try:
                res = ml.train_and_compare(df, target, feats, task, models, test_size, folds)
                st.session_state["ml_result"] = res
                ui.log("train_models", {"target": target, "task": task, "best": res["best_name"]})
            except Exception as exc:  # noqa: BLE001
                st.error(f"Training failed: {exc}")
    res = st.session_state.get("ml_result")
    if not res or res["target"] != target:
        return
    st.success(f"🏆 Best model: **{res['best_name']}** · trained on {res['n_train']:,} rows, tested on {res['n_test']:,}")
    board = res["leaderboard"].copy()
    num_cols = board.select_dtypes("number").columns
    st.dataframe(board.style.format({c: "{:.4f}" for c in num_cols}).highlight_max(
        subset=[c for c in num_cols if c in ("R²", "Accuracy", "F1", "ROC AUC", "Precision", "Recall")], color="rgba(54,179,126,.3)"),
        width="stretch")
    ui.report_button(f"Model leaderboard — {target}", "table", board.round(4), "ml_board", "Machine Learning")
    key_metric = "R²" if res["task"] == "regression" else "F1"
    fig = px.bar(board, x="Model", y=key_metric, template=ui.template(), color="Model",
                 color_discrete_sequence=ui.palette_colors(), title=f"{key_metric} by model")
    ui.show_fig(fig, "ml_cmp", source="Machine Learning")

    g1, g2 = st.columns(2)
    if res["task"] == "regression":
        f1, f2 = ml.regression_figures(res, ui.template())
    else:
        f1, f2 = ml.classification_figures(res, ui.template())
    with g1:
        ui.show_fig(f1, "ml_f1", source="Machine Learning")
    with g2:
        if f2 is not None:
            ui.show_fig(f2, "ml_f2", source="Machine Learning")

    with st.expander("🔑 Feature importance (permutation)", expanded=True):
        with st.spinner("Computing importance…"):
            imp = ml.feature_importance(res)
        fig = px.bar(imp.head(15).iloc[::-1], x="Importance", y="Feature", orientation="h", error_x="Std",
                     template=ui.template(), color_discrete_sequence=["#06B6D4"], title="What drives the prediction?")
        ui.show_fig(fig, "ml_imp", source="Machine Learning")

    with st.expander("🔮 Make a prediction (what-if)"):
        X = res["X_test"]
        vals = {}
        cols = st.columns(3)
        for i, f in enumerate(res["features"]):
            with cols[i % 3]:
                s = X[f]
                if pd.api.types.is_numeric_dtype(s):
                    vals[f] = st.number_input(f, value=float(s.median()), key=f"pred_{f}")
                else:
                    opts = s.astype(str).value_counts().index.tolist()
                    vals[f] = st.selectbox(f, opts, key=f"pred_{f}")
        if st.button("Predict", type="primary"):
            out = ml.predict_one(res, vals)
            p = out["prediction"]
            st.metric(f"Predicted {res['target']}", f"{p:,.4f}" if isinstance(p, (float, int)) else str(p))
            if "probabilities" in out:
                st.bar_chart(pd.Series(out["probabilities"], name="probability"))

    with st.expander("📦 Batch predictions & export"):
        st.caption("Score every row of the current data with the best model and download the result, or download the model itself.")
        if st.button("Score all rows"):
            pred = ml.predict_frame(res, df)
            scored = df.copy()
            scored[f"predicted_{res['target']}"] = pred
            st.dataframe(scored.head(200), width="stretch")
            st.download_button("⬇️ Download scored data (CSV)", scored.to_csv(index=False).encode(), "predictions.csv", "text/csv")
        st.download_button("⬇️ Download trained model (.pkl)", ml.model_to_bytes(res), f"datanova_{res['best_name'].replace(' ', '_')}.pkl",
                           help="Load with pickle; payload['best_model'].predict(DataFrame) — needs scikit-learn")


def clustering(df):
    cand = [c for c in _feature_candidates(df) if not pd.api.types.is_datetime64_any_dtype(df[c])]
    num = profiling.numeric_cols(df)
    feats = st.multiselect("Features to cluster on", cand, default=[c for c in profiling.ranked_measures(df) if c in cand][:4] or num[:3], key="cl_feats")
    if len(feats) < 1:
        st.info("Choose at least one feature.")
        return
    c1, c2, c3 = st.columns(3)
    algo = c1.selectbox("Algorithm", ["K-Means", "Agglomerative", "DBSCAN"])
    k = c2.slider("Clusters (k)", 2, 12, 3, disabled=algo == "DBSCAN")
    eps = c3.slider("DBSCAN eps", 0.1, 5.0, 0.8, 0.1, disabled=algo != "DBSCAN")
    if algo == "K-Means" and st.checkbox("Help me choose k (elbow & silhouette)"):
        with st.spinner("Testing k = 2…10"):
            el = ml.elbow_analysis(df, feats)
        best_k = int(el.loc[el["Silhouette"].idxmax(), "k"])
        fig = px.line(el, x="k", y=["Inertia", "Silhouette"], markers=True, facet_row="variable", template=ui.template(),
                      title=f"Elbow & silhouette — best silhouette at k = {best_k}")
        fig.update_yaxes(matches=None)
        fig.update_layout(height=450)
        ui.show_fig(fig, "elbow", source="Machine Learning")
    if st.button("Run clustering", type="primary"):
        try:
            with st.spinner("Clustering…"):
                out, sil = ml.cluster(df, feats, algo, k, eps)
            st.session_state["cluster_out"] = (out, sil, feats, algo)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
    if "cluster_out" in st.session_state:
        out, sil, feats_used, algo_used = st.session_state["cluster_out"]
        st.metric("Silhouette score", f"{sil:.3f}" if sil is not None else "n/a", help="−1…1, higher = better separated clusters")
        fig = px.scatter(out, x="_pc1", y="_pc2", color="cluster", template=ui.template(), opacity=.75,
                         color_discrete_sequence=ui.palette_colors(), labels={"_pc1": "PC1", "_pc2": "PC2"},
                         title=f"{algo_used} clusters (PCA projection)", hover_data=feats_used[:4])
        ui.show_fig(fig, "clusters", source="Machine Learning")
        prof = ml.cluster_profile(out, feats_used)
        ui.show_table(prof, "cluster_prof", "Cluster profiles", "Machine Learning")
        if st.button("➕ Add 'cluster' column to the dataset"):
            d = ui.clean_df().copy()
            d.loc[out.index, "cluster"] = out["cluster"]
            ui.set_clean(d, f"Added cluster labels ({algo_used})")
            st.success("Added. Use it as a colour or filter anywhere.")


def anomalies(df):
    num = profiling.numeric_cols(df)
    if not num:
        st.info("Need numeric columns.")
        return
    c1, c2 = st.columns([2, 1])
    feats = c1.multiselect("Columns to check", num, default=profiling.ranked_measures(df)[:3] or num[:3], key="an_feats")
    method = c2.selectbox("Method", ["Isolation Forest", "Local Outlier Factor", "Z-score", "IQR"])
    cont = st.slider("Expected anomaly share", 0.005, 0.2, 0.03, 0.005, disabled=method in ("Z-score", "IQR"))
    if not feats:
        return
    try:
        out = ml.detect_anomalies(df, feats, method, cont)
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        return
    n = int(out["anomaly"].sum())
    a, b = st.columns(2)
    a.metric("Anomalies found", f"{n:,}")
    b.metric("Share of rows", f"{n / max(1, len(out)):.2%}")
    x = feats[0]
    y = feats[1] if len(feats) > 1 else None
    dts = profiling.datetime_cols(df)
    plot = out.assign(status=out["anomaly"].map({True: "Anomaly", False: "Normal"}))
    if dts and y is None:
        fig = px.scatter(plot.sort_values(dts[0]), x=dts[0], y=x, color="status", template=ui.template(),
                         color_discrete_map={"Anomaly": "#FF5630", "Normal": "#4F46E5"}, title=f"Anomalies in {x} over time")
    elif y:
        fig = px.scatter(plot, x=x, y=y, color="status", template=ui.template(), opacity=.75,
                         color_discrete_map={"Anomaly": "#FF5630", "Normal": "#4F46E5"}, title=f"Anomalies: {x} vs {y}")
    else:
        fig = px.strip(plot, x=x, color="status", template=ui.template(),
                       color_discrete_map={"Anomaly": "#FF5630", "Normal": "#4F46E5"}, title=f"Anomalies in {x}")
    ui.show_fig(fig, "anom_fig", source="Machine Learning")
    top = out[out["anomaly"]].sort_values("anomaly_score", ascending=False)
    ui.show_table(top.head(200), "anom_tbl", f"Top anomalies ({method})", "Machine Learning")
    per_col = pd.DataFrame({"Column": feats, "Mean (normal)": [out.loc[~out.anomaly, c].mean() for c in feats],
                            "Mean (anomaly)": [out.loc[out.anomaly, c].mean() for c in feats]}).round(3)
    st.dataframe(per_col, width="stretch")
    if n and st.button("🧹 Remove these anomalies from the dataset"):
        d = ui.clean_df()
        ui.set_clean(d.drop(index=[i for i in top.index if i in d.index]), f"Removed {n} anomalies ({method})")
        st.rerun()


def pca(df):
    cand = [c for c in _feature_candidates(df) if not pd.api.types.is_datetime64_any_dtype(df[c])]
    feats = st.multiselect("Features", cand, default=profiling.numeric_cols(df)[:8], key="pca_feats")
    if len(feats) < 2:
        st.info("Choose at least two features.")
        return
    color = ui.col_select("Colour points by", profiling.categorical_cols(df, 30), "pca_color", none_option=True)
    ev, loadings, coords = ml.pca_analysis(df, feats)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(ev, x="Component", y="Explained variance %", template=ui.template(), color_discrete_sequence=["#4F46E5"],
                     title="Explained variance (scree plot)")
        fig.add_scatter(x=ev["Component"], y=ev["Cumulative %"], mode="lines+markers", name="Cumulative %")
        ui.show_fig(fig, "pca_scree", source="Machine Learning")
    with c2:
        plot = coords.join(df[[color]]) if color else coords
        fig = px.scatter(plot, x="PC1", y="PC2", color=color, template=ui.template(), opacity=.7,
                         color_discrete_sequence=ui.palette_colors(), title="Data in 2 principal components")
        ui.show_fig(fig, "pca_scatter", source="Machine Learning")
    ui.show_table(loadings, "pca_load", "PCA loadings", "Machine Learning")
