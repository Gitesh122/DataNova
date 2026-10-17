import streamlit as st

from core import db
from core.data_io import SAMPLE_DATASETS
from views import ui

FEATURES = [
    ("📂", "Data Hub", "CSV, Excel (all sheets), JSON, Parquet, PDF tables, URLs and samples. Merge & append datasets.", "data"),
    ("🔍", "Data Profile", "Smart type detection, quality score, missing-value map and correlations.", "profile"),
    ("🧹", "Clean & Transform", "One-click or custom cleaning, outliers, type fixes, feature engineering — with undo.", "clean"),
    ("🧭", "Explore & Pivot", "Searchable grid, group-by summaries and pivot tables.", "explore"),
    ("⚡", "Auto Dashboard", "KPIs, recommended charts and insights generated automatically.", "dashboard"),
    ("📊", "Chart Studio", "21 chart types incl. maps, 3D and treemaps, plus your own pinned dashboard.", "charts"),
    ("🗿", "3D Sculpture", "Turn data into a 3D landscape and export an STL for 3D printing.", "sculpture"),
    ("📐", "Statistics", "Correlations and hypothesis tests explained in plain English.", "statistics"),
    ("🤖", "Machine Learning", "AutoML, clustering, anomaly detection, PCA and live predictions.", "ml"),
    ("🔮", "Forecasting", "Six models with back-testing and prediction intervals.", "forecast"),
    ("💡", "AI Insights & Ask", "Automatic findings and questions in plain English.", "insights"),
    ("📄", "Reports & Sharing", "Export HTML / PDF / Excel, e-mail reports or create share links.", "reports"),
]


def _link(key, label, icon=None):
    page = ui.PAGES.get(key)
    if page is not None:
        st.page_link(page, label=label, icon=icon)


def render():
    ui.page_header("Welcome to DataNova",
                   "Your smart analytics workspace — from raw file to shareable insight in minutes. No code required.", "✨")
    entry = ui.active_entry()

    if entry is None:
        st.markdown("#### 🚀 Start with a sample dataset")
        cols = st.columns(len(SAMPLE_DATASETS))
        for c, (name, fn) in zip(cols, SAMPLE_DATASETS.items()):
            if c.button(name, key=f"home_s_{name}", width="stretch"):
                ui.add_dataset(name.split(" ", 1)[1], fn(), "sample", auto_clean=True)
                st.rerun()
        c1, c2, _ = st.columns([1.2, 1.2, 3])
        with c1:
            _link("data", "Upload your own data", ":material/upload_file:")
        with c2:
            _link("deploy", "Deploy this app", ":material/rocket_launch:")
    else:
        df = ui.working_df()
        st.success(f"**{st.session_state['active']}** is loaded — {len(df):,} rows × {df.shape[1]} columns.", icon="✅")
        c = st.columns(4)
        c[0].metric("Datasets", len(st.session_state["datasets"]))
        c[1].metric("Rows (filtered)", f"{len(df):,}")
        c[2].metric("Report sections", len(st.session_state["report"]))
        c[3].metric("Pinned charts", len(st.session_state["pinned"]))
        l1, l2, l3, l4 = st.columns(4)
        with l1:
            _link("dashboard", "Open Auto Dashboard", ":material/dashboard:")
        with l2:
            _link("ml", "Train a model", ":material/model_training:")
        with l3:
            _link("insights", "Ask a question", ":material/auto_awesome:")
        with l4:
            _link("reports", "Build a report", ":material/description:")

    st.markdown("#### 🧭 How it works")
    steps = [("Load", "files, URLs, samples"), ("Profile & clean", "fix quality issues"),
             ("Visualise", "auto & custom charts"), ("Model & forecast", "ML in one click"),
             ("Report & share", "PDF, link, e-mail")]
    cols = st.columns(len(steps))
    for i, (c, (t, sub)) in enumerate(zip(cols, steps), 1):
        c.markdown(f"<div class='dn-card dn-step'><div class='n'>0{i}</div><h4>{t}</h4><p>{sub}</p></div>",
                   unsafe_allow_html=True)

    st.markdown("#### ✨ Everything inside")
    for row in range(0, len(FEATURES), 4):
        cols = st.columns(4)
        for c, (icon, title, text, key) in zip(cols, FEATURES[row:row + 4]):
            with c:
                ui.feature_card(icon, title, text)
                _link(key, f"Open {title}", ":material/arrow_forward:")
        st.write("")

    if ui.username() != "guest":
        with st.expander("🕑 Your recent activity"):
            acts = db.recent_activity(ui.username(), 15)
            if acts:
                st.dataframe([{k: a[k] for k in ("created_at", "action", "detail")} for a in acts], width="stretch")
            else:
                st.caption("Nothing yet.")
