"""DataNova — Automated Data Intelligence Platform (Streamlit app entry point).

Run:  streamlit run app.py
"""
from __future__ import annotations

import os
import re

import pandas as pd
import streamlit as st

from core import APP_NAME, APP_TAGLINE, CREDIT, __version__, auth, sharing
from core.profiling import categorical_cols, datetime_cols, numeric_cols
from views import ui

st.set_page_config(page_title=f"{APP_NAME} · {APP_TAGLINE}", page_icon=str(ui.ASSETS / "icons" / "logo.png"), layout="wide",
                   initial_sidebar_state="expanded",
                   menu_items={"About": f"{APP_NAME} v{__version__} — {APP_TAGLINE}. {CREDIT}."})

# Clear message instead of cryptic errors when an old Streamlit is installed
_ver = tuple(int(x) for x in re.findall(r"\d+", st.__version__)[:2])
if _ver < (1, 50):
    st.error(f"DataNova needs Streamlit 1.50 or newer — you have {st.__version__}.\n\n"
             "Run this in your terminal, then restart the app:\n\n`python -m pip install -U -r requirements.txt`")
    st.stop()

ui.init_state()
ui.inject_css()
st.logo(str(ui.ASSETS / "icons" / "wordmark.png"), size="large", icon_image=str(ui.ASSETS / "icons" / "logo.png"))


# ---------------------------------------------------------------- shared report viewer (?share=<id>)
def share_view(share_id: str):
    row = sharing.get_share(share_id)
    if not row:
        st.error("This shared report does not exist or was deleted.")
        st.stop()
    if row.get("expired"):
        st.warning("This shared link has expired.")
        st.stop()
    if row.get("visibility") == "Private" and row.get("password_hash"):
        if not st.session_state.get(f"share_ok_{share_id}"):
            st.subheader(f"🔒 {row['title']}")
            pw = st.text_input("This report is password protected", type="password")
            if st.button("Open report", type="primary"):
                if sharing.check_password(row, pw):
                    st.session_state[f"share_ok_{share_id}"] = True
                    st.rerun()
                st.error("Wrong password.")
            st.stop()
    if not st.session_state.get(f"share_counted_{share_id}"):
        sharing.register_view(share_id)
        st.session_state[f"share_counted_{share_id}"] = True
    st.caption(f"Shared by **{row['owner']}** · {row['created_at'][:10]} · {row['views'] + 1} views")
    ui.embed_html(row["html"] or "<p>Report file missing.</p>", height=1400)
    st.download_button("⬇️ Download this report (HTML)", row["html"] or "", file_name=f"{row['title']}.html",
                       mime="text/html")
    st.stop()


share_id = st.query_params.get("share")
if share_id:
    share_view(share_id)


# ---------------------------------------------------------------- authentication
def login_page():
    logo = ui.b64(ui.ASSETS / "icons" / "logo.png")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown(f"""
        <div class="dn-auth-brand">
          <div class="snow"></div>
          <img src="data:image/png;base64,{logo}" width="58" alt="">
          <h1>{APP_NAME}</h1>
          <p class="lead">{APP_TAGLINE}</p>
          <ul>
            <li>⚡ Upload any file — get a dashboard in seconds</li>
            <li>🧹 One-click cleaning with undo</li>
            <li>🤖 AutoML, forecasting &amp; anomaly detection</li>
            <li>💬 Ask questions in plain English</li>
            <li>📄 Export &amp; share polished reports</li>
          </ul>
          <p class="credit">{CREDIT}</p>
        </div>""", unsafe_allow_html=True)
    with right:
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        st.markdown("### Welcome 👋")
        st.caption("Sign in to save preferences and share reports — or explore as a guest.")
        t_login, t_reg = st.tabs(["Sign in", "Create account"])
        with t_login:
            with st.form("login_form", border=False):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                ok = st.form_submit_button("Sign in", type="primary", width="stretch")
            if ok:
                user = auth.authenticate_user(u, p)
                if user:
                    st.session_state["user"] = user
                    prefs = auth.get_preferences(user["username"])
                    st.session_state["settings"].update(prefs.get("settings", {}))
                    st.rerun()
                st.error("Invalid username or password.")
        with t_reg:
            with st.form("register_form", border=False):
                full = st.text_input("Full name (optional)")
                email = st.text_input("E-mail (optional)")
                u = st.text_input("Choose a username")
                p1 = st.text_input("Password", type="password", help="At least 6 characters")
                p2 = st.text_input("Confirm password", type="password")
                ok = st.form_submit_button("Create account", type="primary", width="stretch")
            if p1:
                score, label = auth.password_strength(p1)
                st.progress(score / 4, text=f"Password strength: {label}")
            if ok:
                if p1 != p2:
                    st.error("Passwords do not match.")
                else:
                    success, msg = auth.register_user(u, p1, email, full)
                    (st.success if success else st.error)(msg + (" You can sign in now." if success else ""))
        st.markdown("<div class='dn-or'><span>or</span></div>", unsafe_allow_html=True)
        if st.button("Continue as guest", icon=":material/visibility:", width="stretch",
                     help="Explore everything; nothing is saved to an account"):
            st.session_state["user"] = {"username": "guest", "role": "guest", "full_name": "Guest"}
            st.rerun()
        admin_rule = ("Admins are set by the app owner." if os.environ.get("DATANOVA_ADMINS")
                      else "The first account created becomes the administrator.")
        st.caption(f"🔒 Passwords are salted & hashed ({auth.BACKEND_NAME}). {admin_rule}")
    ui.footer()


if st.session_state["user"] is None:
    if os.environ.get("DATANOVA_REQUIRE_LOGIN", "1") == "0":
        st.session_state["user"] = {"username": "guest", "role": "guest", "full_name": "Guest"}
    else:
        login_page()
        st.stop()


# ---------------------------------------------------------------- sidebar: user, dataset, filters
def sidebar():
    user = st.session_state["user"]
    with st.sidebar:
        name = user.get("full_name") or user["username"]
        initials = "".join(w[0] for w in str(name).split()[:2]).upper() or "U"
        st.markdown(f"""<div class="dn-user">
            <div class="dn-avatar">{initials}</div>
            <div><b>{name}</b><span>{user.get('role', 'user').title()} · {len(st.session_state['report'])} in report</span></div>
            </div>""", unsafe_allow_html=True)
        names = list(st.session_state["datasets"])
        if names:
            idx = names.index(st.session_state["active"]) if st.session_state["active"] in names else 0
            st.session_state["active"] = st.selectbox("📂 Active dataset", names, index=idx, key="sb_active")
            df = ui.clean_df()
            wdf = ui.working_df()
            c1, c2 = st.columns(2)
            c1.metric("Rows", f"{len(wdf):,}", delta=f"{len(wdf) - len(df):,} filtered" if len(wdf) != len(df) else None,
                      delta_color="off")
            c2.metric("Columns", df.shape[1])
            filters_panel(df)
        else:
            st.info("No data yet — open **Data Hub**.")
        st.divider()
        if st.button("Log out", icon=":material/logout:", width="stretch"):
            ui.log("logout")
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
        st.caption(f"✨ {APP_NAME} v{__version__} · {CREDIT}")


def filters_panel(df: pd.DataFrame):
    """Intelligent global filters: numeric ranges (IQR-aware), date ranges, top categories, text search."""
    name = st.session_state["active"]
    current = st.session_state["filters"].setdefault(name, {})
    with st.expander(f"🧠 Smart filters ({len(current)} active)", expanded=bool(current)):
        cols = st.multiselect("Filter by columns", list(df.columns), default=[c for c in current if c in df.columns],
                              key=f"flt_cols_{name}")
        new = {}
        nums, dts, cats = set(numeric_cols(df)), set(datetime_cols(df)), set(categorical_cols(df))
        for c in cols:
            s = df[c].dropna()
            if s.empty:
                continue
            if c in dts:
                lo, hi = s.min().date(), s.max().date()
                prev = current.get(c, ("dates", (lo, hi)))[1]
                v = st.date_input(c, value=prev, min_value=lo, max_value=hi, key=f"flt_{name}_{c}")
                if isinstance(v, (tuple, list)) and len(v) == 2:
                    new[c] = ("dates", tuple(v))
            elif c in nums:
                lo, hi = float(s.min()), float(s.max())
                if lo == hi:
                    continue
                q1, q3 = s.quantile([.25, .75])
                st.caption(f"{c}: typical range {q1 - 1.5 * (q3 - q1):,.2f} – {q3 + 1.5 * (q3 - q1):,.2f}")
                prev = current.get(c, ("range", (lo, hi)))[1]
                v = st.slider(c, lo, hi, (max(lo, prev[0]), min(hi, prev[1])), key=f"flt_{name}_{c}")
                if v != (lo, hi):
                    new[c] = ("range", v)
            elif c in cats:
                if s.nunique() <= 200:
                    opts = s.astype(str).value_counts().index.tolist()
                    prev = current.get(c, ("values", []))[1]
                    v = st.multiselect(c, opts, default=[p for p in prev if p in opts], key=f"flt_{name}_{c}",
                                       placeholder="All values")
                    if v:
                        new[c] = ("values", v)
                else:
                    prev = current.get(c, ("text", ""))[1]
                    v = st.text_input(f"{c} contains", value=prev, key=f"flt_{name}_{c}")
                    if v:
                        new[c] = ("text", v)
        st.session_state["filters"][name] = new
        if new and st.button("✖ Clear all filters", width="stretch"):
            st.session_state["filters"][name] = {}
            for k in list(st.session_state.keys()):
                if str(k).startswith(f"flt_{name}_") or k == f"flt_cols_{name}":
                    del st.session_state[k]
            st.rerun()


# ---------------------------------------------------------------- navigation
from views import (analysis, auto_dashboard, clean, data_hub, deploy_page, explore, forecast, home,  # noqa: E402
                   insights, ml_studio, reports_page, sculpture_page, settings_page, visualize)

P = {
    "home": st.Page(home.render, title="Home", icon=":material/home:", url_path="home", default=True),
    "data": st.Page(data_hub.render, title="Data Hub", icon=":material/upload_file:", url_path="data"),
    "profile": st.Page(analysis.render_profile, title="Data Profile", icon=":material/troubleshoot:", url_path="profile"),
    "clean": st.Page(clean.render, title="Clean & Transform", icon=":material/cleaning_services:", url_path="clean"),
    "explore": st.Page(explore.render, title="Explore & Pivot", icon=":material/table_view:", url_path="explore"),
    "dashboard": st.Page(auto_dashboard.render, title="Auto Dashboard", icon=":material/dashboard:", url_path="dashboard"),
    "charts": st.Page(visualize.render, title="Chart Studio", icon=":material/bar_chart:", url_path="charts"),
    "sculpture": st.Page(sculpture_page.render, title="3D Sculpture", icon=":material/view_in_ar:", url_path="sculpture"),
    "statistics": st.Page(analysis.render_stats, title="Statistics", icon=":material/functions:", url_path="statistics"),
    "ml": st.Page(ml_studio.render, title="Machine Learning", icon=":material/model_training:", url_path="ml"),
    "forecast": st.Page(forecast.render, title="Forecasting", icon=":material/trending_up:", url_path="forecast"),
    "insights": st.Page(insights.render, title="AI Insights & Ask", icon=":material/auto_awesome:", url_path="insights"),
    "reports": st.Page(reports_page.render, title="Reports & Sharing", icon=":material/description:", url_path="reports"),
    "deploy": st.Page(deploy_page.render, title="Deploy", icon=":material/rocket_launch:", url_path="deploy"),
    "settings": st.Page(settings_page.render, title="Settings & Help", icon=":material/settings:", url_path="settings"),
}
ui.PAGES = P  # lets pages link to each other with st.page_link
pages = {
    "Start": [P["home"], P["data"]],
    "Prepare": [P["profile"], P["clean"], P["explore"]],
    "Visualise": [P["dashboard"], P["charts"], P["sculpture"]],
    "Analyse": [P["statistics"], P["ml"], P["forecast"], P["insights"]],
    "Share": [P["reports"], P["deploy"], P["settings"]],
}
nav = st.navigation(pages, expanded=True)
sidebar()
nav.run()
ui.footer()
