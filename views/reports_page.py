import streamlit as st

from core import data_io, emailer, sharing
from core import insights as ins
from core import reports as rep
from views import ui


def _summary_default():
    df = ui.working_df()
    if df is None:
        return ""
    top = ins.generate_insights(df, max_items=6)
    return ins.summary_paragraph(df) + "\n" + "\n".join(f"- {i['title']}" for i in top)


def render():
    ui.page_header("Reports & Sharing", "Assemble a report from anything you pinned, export it, e-mail it or share a link.", "📄")
    secs = st.session_state["report"]
    t_build, t_export, t_mail, t_share = st.tabs(["🧱 Build report", "📥 Export", "📧 E-mail", "🔗 Share link"])

    with t_build:
        c1, c2 = st.columns([2, 1])
        st.session_state.setdefault("rep_title", "DataNova Analysis Report")
        c1.text_input("Report title", key="rep_title")
        user = st.session_state["user"] or {}
        st.session_state.setdefault("rep_author", user.get("full_name") or ui.username())
        c2.text_input("Author", key="rep_author")
        if "rep_summary" not in st.session_state:
            st.session_state["rep_summary"] = _summary_default()
        st.text_area("Executive summary (markdown: **bold**, - bullets)", key="rep_summary", height=140)
        if st.button("✨ Regenerate summary from data"):
            st.session_state["rep_summary"] = _summary_default()
            st.rerun()

        b1, b2, b3 = st.columns(3)
        if b1.button("⚡ Add standard sections", help="Data overview, quality, column profile and insights"):
            from core import profiling
            df = ui.working_df()
            if df is not None:
                ov = profiling.dataset_overview(df)
                q = profiling.quality_score(df)
                ui.add_section("Data overview", "metrics", [("Rows", f"{ov['rows']:,}"), ("Columns", str(ov['columns'])),
                                                            ("Missing %", f"{ov['missing_pct']}%"), ("Duplicates", str(ov['duplicate_rows'])),
                                                            ("Quality score", f"{q['score']}/100")], source="Data Profile")
                ui.add_section("Column profile", "table", profiling.column_profile(df), source="Data Profile")
                e = ui.active_entry()
                if e and e["log"]:
                    ui.add_section("Data cleaning steps", "text", "\n".join(f"- {m}" for m in e["log"]), source="Clean")
                ui.add_section("Automatic insights", "text", "\n".join(f"- **{i['title']}** — {i['text']}"
                                                                      for i in ins.generate_insights(df)), source="AI Insights")
                st.rerun()
        if b2.button("📝 Add a text note"):
            ui.add_section("Notes", "text", "Write your commentary here.", source="Manual")
            st.rerun()
        if b3.button("🗑 Clear report", disabled=not secs):
            st.session_state["report"] = []
            st.rerun()

        if not secs:
            st.info("The report is empty. Click **📌 Add to report** under any chart or table in the app, "
                    "use *Add the whole dashboard* on Auto Dashboard, or add the standard sections above.")
        for i, s in enumerate(secs):
            with st.container(border=True):
                h1, h2, h3, h4, h5 = st.columns([6, 1, 1, 1, 1])
                s["title"] = h1.text_input("Section title", s["title"], key=f"st_{s['id']}", label_visibility="collapsed")
                if h2.button("⬆️", key=f"up_{s['id']}", disabled=i == 0):
                    secs[i - 1], secs[i] = secs[i], secs[i - 1]
                    st.rerun()
                if h3.button("⬇️", key=f"dn_{s['id']}", disabled=i == len(secs) - 1):
                    secs[i + 1], secs[i] = secs[i], secs[i + 1]
                    st.rerun()
                show = h4.toggle("👁", key=f"pv_{s['id']}", help="Preview")
                if h5.button("🗑", key=f"rm_{s['id']}"):
                    secs.pop(i)
                    st.rerun()
                st.caption(f"{s['kind']} · from {s.get('source') or '—'} · {s['created']}")
                if s["kind"] == "text":
                    s["text"] = st.text_area("Text", s["text"], key=f"tx_{s['id']}", label_visibility="collapsed")
                s["note"] = st.text_input("Caption / commentary (optional)", s.get("note", ""), key=f"nt_{s['id']}")
                if show:
                    if s["kind"] == "figure":
                        st.plotly_chart(rep.section_figure(s), width="stretch", key=f"pvf_{s['id']}")
                    elif s["kind"] == "table":
                        st.dataframe(s["table"], width="stretch")
                    elif s["kind"] == "metrics":
                        cols = st.columns(min(5, len(s["metrics"])))
                        for c, (a, b) in zip(cols * 5, s["metrics"]):
                            c.metric(a, b)

    meta = dict(title=st.session_state.get("rep_title", "DataNova Report"), author=st.session_state.get("rep_author", ""),
                dataset=st.session_state.get("active") or "", summary=st.session_state.get("rep_summary", ""))

    with t_export:
        if not secs:
            st.info("Add sections first.")
        else:
            st.write(f"**{len(secs)} section(s)** ready.")
            c1, c2, c3, c4 = st.columns(4)
            html = rep.build_html(secs, **meta)
            c1.download_button("🌐 Interactive HTML", html, "DataNova_Report.html", "text/html", type="primary",
                               width="stretch")
            if c2.button("📄 Build PDF", width="stretch"):
                with st.spinner("Rendering PDF…"):
                    st.session_state["pdf_bytes"] = rep.build_pdf(secs, **meta)
            if st.session_state.get("pdf_bytes"):
                c2.download_button("⬇️ Download PDF", st.session_state["pdf_bytes"], "DataNova_Report.pdf",
                                   "application/pdf", width="stretch")
                if rep._KALEIDO_OK is False:
                    c2.caption("Charts drawn with the built-in renderer. Install Chrome + `kaleido` for pixel-perfect images.")
            df = ui.working_df()
            c3.download_button("📊 Excel workbook", rep.build_excel(secs, df), "DataNova_Report.xlsx", width="stretch",
                               help="Report contents, every table, each chart's data and the dataset")
            if df is not None:
                c4.download_button("🧾 Cleaned data (CSV)", data_io.to_csv_bytes(df), "datanova_data.csv", "text/csv",
                                   width="stretch")
            with st.expander("👀 Preview HTML report", expanded=False):
                ui.embed_html(html, height=800)

    with t_mail:
        cfg = st.session_state["email_config"]
        if not cfg.get("sender_email"):
            st.warning("Configure your SMTP account first in **Settings & Help → E-mail**.")
        to = st.text_input("To (comma-separated)", placeholder="manager@company.com, team@company.com")
        cc = st.text_input("Cc (optional)")
        subject = st.text_input("Subject", cfg.get("default_subject", "DataNova Report"))
        msg = st.text_area("Message", "Hi,\n\nPlease find the latest DataNova analysis attached.\n\nRegards")
        c1, c2, c3 = st.columns(3)
        body_report = c1.checkbox("Include report in e-mail body", True, help="Charts embedded as images")
        att_pdf = c2.checkbox("Attach PDF", True)
        att_html = c3.checkbox("Attach interactive HTML", False)
        if st.button("📨 Send", type="primary", disabled=not (secs and cfg.get("sender_email"))):
            good, bad = emailer.parse_recipients(to)
            cc_good, _ = emailer.parse_recipients(cc)
            if bad:
                st.error(f"Invalid address(es): {', '.join(bad)}")
            elif not good:
                st.error("Add at least one recipient.")
            else:
                with st.spinner("Preparing and sending…"):
                    personal = "".join(f"<p>{line}</p>" for line in msg.split("\n") if line.strip())
                    body = personal + (rep.build_html(secs, **meta, static_images=True) if body_report else "")
                    atts = []
                    if att_pdf:
                        atts.append(("DataNova_Report.pdf", rep.build_pdf(secs, **meta), "application/pdf"))
                    if att_html:
                        atts.append(("DataNova_Report.html", rep.build_html(secs, **meta).encode(), "text/html"))
                    ok, info = emailer.send_email(cfg, good, subject, body, atts, cc_good)
                (st.success if ok else st.error)(info)
                if ok:
                    ui.log("email_report", {"to": good})
                    st.balloons()

    with t_share:
        if ui.username() == "guest":
            st.info("Log in with an account to create share links (links are tied to their owner).")
        elif not secs:
            st.info("Add sections first.")
        else:
            c1, c2, c3 = st.columns(3)
            vis = c1.radio("Access", ["Public", "Private"], horizontal=True, help="Private links require a password")
            exp = c2.selectbox("Expires", list(sharing.EXPIRY), index=1)
            pw = c3.text_input("Password", type="password", disabled=vis == "Public")
            from views.deploy_page import _app_url
            base = st.text_input("App address (for the link)", value=st.session_state.get("base_url", _app_url() or "http://localhost:8501"),
                                 help="Change this to your deployed URL, e.g. https://datanova.streamlit.app")
            st.session_state["base_url"] = base
            if st.button("🔗 Create share link", type="primary", disabled=vis == "Private" and not pw):
                sid = sharing.create_share(rep.build_html(secs, **meta), ui.username(), meta["title"], vis,
                                           pw if vis == "Private" else "", exp)
                st.success("Link created — anyone with it can open a read-only snapshot of this report.")
                st.code(f"{base.rstrip('/')}/?share={sid}")
        if ui.username() != "guest":
            mine = sharing.list_shares(ui.username())
            if mine:
                st.markdown("**Your shared links**")
                for r in mine:
                    c1, c2 = st.columns([6, 1])
                    c1.markdown(f"`?share={r['share_id']}` — **{r['title']}** · {r['visibility']} · "
                                f"expires {r['expires_at'][:10] if r['expires_at'] else 'never'} · 👁 {r['views']}")
                    if c2.button("Revoke", key=f"rv_{r['share_id']}"):
                        sharing.delete_share(r["share_id"], ui.username())
                        st.rerun()
