import pandas as pd
import plotly.express as px
import streamlit as st

from core import APP_NAME, CREDIT, __version__, auth, db, emailer
from core.viz import PALETTES, TEMPLATES
from views import ui

GUIDE = """
**1. Load data** — *Data Hub*: upload CSV / Excel / JSON / Parquet / PDF, fetch a URL, paste text or load a sample.
Turn on *Auto-clean on load* for instant tidy data.

**2. Profile** — *Data Profile*: column types, quality score, missing-value map, distributions and correlations.

**3. Clean** — *Clean & Transform*: one-click cleaning or a custom pipeline, drop/rename/convert columns,
fix values, handle outliers and engineer features. Use **Undo** any time.

**4. Filter** — sidebar *Smart filters* apply to every page (ranges, dates, categories, text).

**5. Visualise** — *Auto Dashboard* builds KPIs + charts for you; *Chart Studio* builds any of 21 chart types
and lets you pin charts into your own dashboard; *3D Sculpture* exports STL files.

**6. Analyse** — *Statistics* (tests in plain English), *Machine Learning* (AutoML, clustering, anomalies, PCA),
*Forecasting* (6 models with back-testing), *AI Insights & Ask* (automatic findings and natural-language questions).

**7. Report & share** — click **📌 Add to report** anywhere, then *Reports & Sharing* to export HTML / PDF / Excel,
e-mail it or create a password-protected share link.
"""

TROUBLE = """
- **Large files** → `streamlit run app.py --server.maxUploadSize=2048`
- **PDF charts look basic** → install Google Chrome (kaleido uses it) for pixel-perfect chart images.
- **Gmail login fails** → enable 2-step verification and create an *App Password*.
- **Holt-Winters / ARIMA missing** → `pip install statsmodels`
- **AI analyst** → `pip install anthropic` and set `ANTHROPIC_API_KEY`.
- **MongoDB accounts** → set `DATANOVA_MONGO_URI=mongodb://localhost:27017/` before launching.
- **Deploying online** → open the **Deploy** page in the sidebar for a readiness check and step-by-step guide.
"""


def render():
    ui.page_header("Settings & Help", "Appearance, e-mail, account, feedback, admin and the user guide.", "⚙️")
    tabs = ["🎨 Appearance", "📧 E-mail", "👤 Account", "💬 Feedback", "📘 Guide"] + (["🛡️ Admin"] if ui.is_admin() else [])
    t = st.tabs(tabs)

    with t[0]:
        s = st.session_state["settings"]
        c1, c2, c3 = st.columns(3)
        s["template"] = c1.selectbox("Chart theme", TEMPLATES, index=TEMPLATES.index(s["template"]))
        s["palette"] = c2.selectbox("Colour palette", list(PALETTES), index=list(PALETTES).index(s["palette"]))
        s["accent"] = c3.color_picker("Accent colour", s["accent"])
        st.caption("Light / dark mode for the whole app: use the ⋮ menu (top-right) → Settings → Theme. "
                   "Pick *plotly_dark* above for dark charts. *Safe (colour-blind)* is an accessible palette.")
        demo = pd.DataFrame({"category": list("ABCDEF"), "value": [5, 9, 6, 11, 7, 4]})
        st.plotly_chart(px.bar(demo, x="category", y="value", color="category", template=s["template"],
                               color_discrete_sequence=PALETTES[s["palette"]], title="Preview"), width="stretch")
        if ui.username() != "guest" and st.button("💾 Save as my default"):
            prefs = auth.get_preferences(ui.username())
            prefs["settings"] = s
            auth.save_preferences(ui.username(), prefs)
            st.success("Saved — applied automatically next time you log in.")

    with t[1]:
        cfg = st.session_state["email_config"]
        prov = st.selectbox("Provider", list(emailer.PRESETS), index=list(emailer.PRESETS).index(cfg.get("provider", "Gmail")))
        if prov != cfg.get("provider"):
            server, port, sec = emailer.PRESETS[prov]
            cfg.update(provider=prov, smtp_server=server or cfg["smtp_server"], smtp_port=port, security=sec)
        c1, c2, c3 = st.columns([2, 1, 1])
        cfg["smtp_server"] = c1.text_input("SMTP server", cfg["smtp_server"])
        cfg["smtp_port"] = c2.number_input("Port", 1, 65535, int(cfg["smtp_port"]))
        cfg["security"] = c3.selectbox("Security", ["starttls", "ssl", "none"], index=["starttls", "ssl", "none"].index(cfg["security"]))
        c1, c2 = st.columns(2)
        cfg["sender_email"] = c1.text_input("Sender e-mail", cfg["sender_email"])
        cfg["sender_password"] = c2.text_input("Password / App password", cfg["sender_password"], type="password")
        c1, c2 = st.columns(2)
        cfg["sender_name"] = c1.text_input("Sender display name", cfg["sender_name"])
        cfg["default_subject"] = c2.text_input("Default subject", cfg["default_subject"])
        st.info("For Gmail: Google Account → Security → 2-Step Verification → App passwords. "
                "Credentials are kept only in this browser session, never written to disk.")
        if st.button("🔌 Test connection", disabled=not cfg["sender_email"]):
            with st.spinner("Connecting…"):
                ok, msg = emailer.test_connection(cfg)
            (st.success if ok else st.error)(msg)

    with t[2]:
        user = st.session_state["user"]
        st.markdown(f"**Username:** {user['username']}  \n**Role:** {user.get('role')}  \n"
                    f"**E-mail:** {user.get('email') or '—'}  \n**Storage backend:** {auth.BACKEND_NAME}")
        if user["username"] == "guest":
            st.info("You're using a guest session. Log out and create an account to save preferences and share links.")
        else:
            with st.form("pw"):
                old = st.text_input("Current password", type="password")
                new = st.text_input("New password", type="password")
                new2 = st.text_input("Confirm new password", type="password")
                if st.form_submit_button("Change password"):
                    if new != new2:
                        st.error("New passwords do not match.")
                    else:
                        ok, msg = auth.change_password(user["username"], old, new)
                        (st.success if ok else st.error)(msg)
            st.markdown("**Activity history**")
            acts = db.recent_activity(user["username"], 100)
            if acts:
                st.dataframe(pd.DataFrame(acts)[["created_at", "action", "detail"]], width="stretch", height=300)

    with t[3]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**💬 Feedback**")
            with st.form("fb", clear_on_submit=True):
                rating = st.slider("How useful is DataNova?", 1, 5, 4)
                cat = st.selectbox("About", ["General", "Feature request", "Charts", "Machine learning", "Reports", "Usability"])
                text = st.text_area("Tell us how we can improve", placeholder="What would you like to see next?")
                if st.form_submit_button("Submit feedback", type="primary"):
                    db.add_feedback(ui.username(), rating, cat, text)
                    st.success("Thanks for your feedback! 😊")
        with c2:
            st.markdown("**🐞 Report an issue**")
            with st.form("iss", clear_on_submit=True):
                sev = st.select_slider("Severity", ["Low", "Medium", "High", "Critical"])
                page = st.text_input("Where did it happen?", placeholder="e.g. Forecasting page")
                desc = st.text_area("What went wrong?")
                if st.form_submit_button("Submit issue"):
                    if desc.strip():
                        db.add_issue(ui.username(), sev, page, desc)
                        st.success("Issue reported — thank you!")
                    else:
                        st.error("Please describe the issue.")

    with t[4]:
        st.markdown(GUIDE)
        st.markdown("**Troubleshooting**")
        st.markdown(TROUBLE)
        st.caption(f"{APP_NAME} v{__version__} · {CREDIT}")

    if ui.is_admin():
        with t[5]:
            users = auth.list_users()
            fb = db.query("SELECT * FROM feedback ORDER BY id DESC")
            iss = db.query("SELECT * FROM issues ORDER BY id DESC")
            acts = db.recent_activity(None, 500)
            c = st.columns(4)
            c[0].metric("Users", len(users))
            c[1].metric("Feedback", len(fb), f"avg {pd.DataFrame(fb)['rating'].mean():.1f}★" if fb else None)
            c[2].metric("Open issues", sum(1 for i in iss if i["status"] == "open"))
            c[3].metric("Events logged", len(acts))
            st.markdown("**Users**")
            st.dataframe(pd.DataFrame(users), width="stretch")
            st.markdown("**Issues**")
            if iss:
                idf = pd.DataFrame(iss)
                st.dataframe(idf, width="stretch")
                close = st.selectbox("Mark issue as resolved", [i["id"] for i in iss if i["status"] == "open"] or [None])
                if close and st.button("Resolve"):
                    db.execute("UPDATE issues SET status='resolved' WHERE id=?", (close,))
                    st.rerun()
            st.markdown("**Feedback**")
            if fb:
                st.dataframe(pd.DataFrame(fb), width="stretch")
            if acts:
                a = pd.DataFrame(acts)
                st.plotly_chart(px.histogram(a, x="action", color="username", template=ui.template(),
                                             title="Activity by action"), width="stretch")
