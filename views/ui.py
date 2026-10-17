"""Shared Streamlit helpers: session state, dataset access, styling, and the
"add to report" plumbing used by every page."""
from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
import streamlit as st

from core import APP_NAME, APP_TAGLINE, CREDIT, __version__, db
from core import reports as rep
from core.viz import PALETTES

ASSETS = Path(__file__).resolve().parent.parent / "assets"
PAGES: dict = {}  # filled by app.py with the st.Page objects (for st.page_link)


# ---------------------------------------------------------------- state
def init_state():
    ss = st.session_state
    ss.setdefault("user", None)                # dict from auth or {"username": "guest", "role": "guest"}
    ss.setdefault("datasets", {})              # name -> {"raw", "clean", "history", "log", "source"}
    ss.setdefault("active", None)              # active dataset name
    ss.setdefault("filters", {})               # dataset name -> {col: filter}
    ss.setdefault("report", [])                # list of report sections
    ss.setdefault("pinned", [])                # chart specs pinned to the custom dashboard
    ss.setdefault("chat", [])                  # AI assistant history
    ss.setdefault("nlq_history", [])
    ss.setdefault("settings", {"template": "plotly_white", "palette": "DataNova", "accent": "#4F46E5",
                               "compact": False})
    ss.setdefault("email_config", {"provider": "Gmail", "smtp_server": "smtp.gmail.com", "smtp_port": 587,
                                   "security": "starttls", "sender_email": "", "sender_password": "",
                                   "sender_name": "DataNova", "default_subject": "DataNova Report"})


def username() -> str:
    u = st.session_state.get("user")
    return (u or {}).get("username", "guest")


def is_admin() -> bool:
    return (st.session_state.get("user") or {}).get("role") == "admin"


def template() -> str:
    return st.session_state["settings"]["template"]


def palette() -> str:
    return st.session_state["settings"]["palette"]


def log(action, detail=""):
    db.log_activity(username(), action, detail)


# ---------------------------------------------------------------- datasets
def add_dataset(name: str, df: pd.DataFrame, source: str = "upload", auto_clean: bool = False):
    from core.cleaning import auto_clean as _auto

    base, i = name, 2
    while name in st.session_state["datasets"]:
        name = f"{base} ({i})"
        i += 1
    entry = {"raw": df, "clean": df.copy(), "history": [], "log": [], "source": source}
    if auto_clean:
        cleaned, msgs = _auto(df)
        entry["clean"], entry["log"] = cleaned, [f"[auto] {m}" for m in msgs]
    st.session_state["datasets"][name] = entry
    st.session_state["active"] = name
    st.session_state["filters"][name] = {}
    log("load_dataset", {"name": name, "rows": len(df), "cols": df.shape[1], "source": source})
    return name


def active_entry() -> dict | None:
    name = st.session_state.get("active")
    return st.session_state["datasets"].get(name) if name else None


def raw_df() -> pd.DataFrame | None:
    e = active_entry()
    return e["raw"] if e else None


def clean_df() -> pd.DataFrame | None:
    e = active_entry()
    return e["clean"] if e else None


def set_clean(df: pd.DataFrame, message: str):
    """Replace the working (clean) data, keeping an undo history (max 15 steps)."""
    e = active_entry()
    e["history"].append((e["clean"], list(e["log"])))
    e["history"] = e["history"][-15:]
    e["clean"] = df
    e["log"].append(message)
    log("transform", message)


def undo():
    e = active_entry()
    if e and e["history"]:
        e["clean"], e["log"] = e["history"].pop()
        return True
    return False


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out = df
    for col, f in (filters or {}).items():
        if col not in out.columns or f is None:
            continue
        kind, val = f
        if kind == "range":
            lo, hi = val
            out = out[(out[col] >= lo) & (out[col] <= hi)]
        elif kind == "dates":
            lo, hi = pd.Timestamp(val[0]), pd.Timestamp(val[1]) + pd.Timedelta(days=1)
            out = out[(out[col] >= lo) & (out[col] < hi)]
        elif kind == "values" and val:
            out = out[out[col].astype(str).isin([str(v) for v in val])]
        elif kind == "text" and val:
            out = out[out[col].astype(str).str.contains(val, case=False, na=False)]
    return out


def working_df() -> pd.DataFrame | None:
    """Cleaned data with the active global filters applied — what every analysis page uses."""
    df = clean_df()
    if df is None:
        return None
    return apply_filters(df, st.session_state["filters"].get(st.session_state["active"], {}))


def require_data() -> pd.DataFrame:
    df = working_df()
    if df is None:
        st.info("No dataset loaded yet. Upload a file on **Data Hub**, or load a sample to explore.")
        c1, c2 = st.columns([1, 3])
        if c1.button("⚡ Load sample sales data", type="primary", key=f"quick_sample_{st.session_state.get('_page', '')}"):
            from core.data_io import sample_sales

            add_dataset("Retail Sales (sample)", sample_sales(), "sample", auto_clean=True)
            st.rerun()
        st.stop()
    if df.empty:
        st.warning("The current filters remove every row. Clear filters in the sidebar.")
        st.stop()
    return df


# ---------------------------------------------------------------- report
def add_section(title, kind, payload, source="", note=""):
    st.session_state["report"].append(rep.make_section(title, kind, payload, note=note, source=source))
    log("add_to_report", title)


def report_button(title, kind, payload, key, source="", label="📌 Add to report"):
    if st.button(label, key=f"rep_{key}", help="Save this to the Report Builder"):
        add_section(title, kind, payload, source=source)
        st.toast(f"Added “{title}” to the report", icon="📌")


def show_fig(fig, key, title=None, source="", report=True, height=None):
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", key=f"fig_{key}", config={"displaylogo": False})
    if report:
        t = title or (fig.layout.title.text if fig.layout.title and fig.layout.title.text else "Chart")
        c1, c2, _ = st.columns([1.2, 1.2, 5])
        with c1:
            report_button(t, "figure", fig, key, source)
        with c2:
            st.download_button("⬇️ HTML", fig.to_html(include_plotlyjs="cdn"), file_name=f"{key}.html",
                               mime="text/html", key=f"dl_{key}", help="Download this interactive chart")


def show_table(df, key, title, source="", report=True, height="auto"):
    st.dataframe(df, width="stretch", height=height)
    if report:
        c1, c2, _ = st.columns([1.2, 1.2, 5])
        with c1:
            report_button(title, "table", df, key, source)
        with c2:
            st.download_button("⬇️ CSV", df.to_csv(index=not isinstance(df.index, pd.RangeIndex)).encode(),
                               file_name=f"{key}.csv", mime="text/csv", key=f"dlt_{key}")


# ---------------------------------------------------------------- styling
def b64(path: Path) -> str:
    try:
        return base64.b64encode(path.read_bytes()).decode()
    except OSError:
        return ""


def inject_css():
    accent = st.session_state["settings"].get("accent", "#4F46E5")
    st.markdown(f"""
    <style>
      :root {{ --dn-accent: {accent}; --dn-accent2: #06B6D4; --dn-line: rgba(128,128,128,.22);
               --dn-soft: rgba(128,128,128,.06); --dn-shadow: 0 1px 2px rgba(16,24,40,.04), 0 4px 16px rgba(16,24,40,.06); }}
      .block-container {{ padding-top: 2.4rem; padding-bottom: 1rem; max-width: 1320px; }}
      h1, h2, h3 {{ letter-spacing: -.01em; }}

      /* ---------- page header ---------- */
      .dn-head {{ position: relative; overflow: hidden; border-radius: 20px; padding: 26px 30px; margin: 0 0 22px;
                 color: #fff; background:
                   radial-gradient(1200px 300px at 100% -20%, rgba(255,255,255,.22), transparent 60%),
                   linear-gradient(115deg, var(--dn-accent) 0%, #6D5BF5 45%, var(--dn-accent2) 100%);
                 box-shadow: 0 10px 30px -12px rgba(79,70,229,.55); }}
      .dn-head .crumb {{ display:inline-block; font-size:.72rem; letter-spacing:.08em; text-transform:uppercase;
                        background: rgba(255,255,255,.18); padding: 3px 10px; border-radius: 999px; margin-bottom: 10px; }}
      .dn-head h1 {{ color:#fff !important; margin:0; font-size: 1.85rem; line-height:1.15; padding:0; }}
      .dn-head p {{ color: rgba(255,255,255,.88) !important; margin: 6px 0 0; font-size: 1rem; max-width: 820px; }}
      .dn-head .snow, .dn-auth-brand .snow {{ position:absolute; inset:0; pointer-events:none; opacity:.14;
                 background-image: radial-gradient(white 1px, transparent 1px); background-size: 24px 24px;
                 animation: dnDrift 18s linear infinite; }}
      @keyframes dnDrift {{ from {{ background-position: 0 0; }} to {{ background-position: 0 480px; }} }}

      /* ---------- cards ---------- */
      .dn-card {{ border: 1px solid var(--dn-line); border-radius: 16px; padding: 18px 18px 16px; min-height: 196px;
                 background: var(--dn-soft); box-shadow: var(--dn-shadow); transition: transform .18s ease, border-color .18s ease; }}
      .dn-card:hover {{ transform: translateY(-3px); border-color: var(--dn-accent); }}
      .dn-card .ic {{ width: 40px; height: 40px; border-radius: 12px; display:flex; align-items:center; justify-content:center;
                     font-size: 1.25rem; background: linear-gradient(135deg, rgba(79,70,229,.16), rgba(6,182,212,.16)); margin-bottom: 10px; }}
      .dn-card h4 {{ margin: 0 0 4px; font-size: 1.02rem; padding: 0; }}
      .dn-card p {{ margin: 0; opacity: .75; font-size: .9rem; line-height: 1.45; }}
      .dn-card.dn-step {{ min-height: 0; text-align:center; }} .dn-step .n {{ font-size: 1.6rem; font-weight: 800;
                 background: linear-gradient(135deg, var(--dn-accent), var(--dn-accent2)); -webkit-background-clip: text;
                 -webkit-text-fill-color: transparent; }}
      .ad-card {{ border: 1px solid var(--dn-line); border-radius: 14px; padding: 14px 16px; background: var(--dn-soft); }}

      /* ---------- insights & pills ---------- */
      .ad-insight {{ border: 1px solid var(--dn-line); border-left: 4px solid var(--dn-accent); padding: 10px 14px; margin: 8px 0;
                    border-radius: 10px; background: var(--dn-soft); }}
      .ad-insight.warning {{ border-left-color: #F59E0B; }} .ad-insight.error {{ border-left-color: #EF4444; }}
      .ad-insight.success {{ border-left-color: #10B981; }}
      .ad-insight b {{ display:block; margin-bottom: 2px; }}
      .ad-pill {{ display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 500;
                 background: rgba(79,70,229,.12); color: var(--dn-accent); margin: 0 6px 6px 0; }}

      /* ---------- widgets ---------- */
      [data-testid="stMetric"] {{ border: 1px solid var(--dn-line); border-radius: 14px; padding: 14px 16px; min-height: 112px;
                                  background: var(--dn-soft); box-shadow: var(--dn-shadow); }}
      [data-testid="stMetricValue"] {{ font-weight: 700; }}
      [data-testid="stFileUploaderDropzone"] {{ border: 2px dashed var(--dn-accent); background: rgba(79,70,229,.04); }}
      .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid var(--dn-line); }}
      .stTabs [data-baseweb="tab"] {{ padding: 8px 14px; border-radius: 10px 10px 0 0; }}
      [data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}
      [data-testid="stExpander"] details {{ border-radius: 12px; }}

      /* ---------- sidebar ---------- */
      .dn-user {{ display:flex; gap: 10px; align-items:center; border: 1px solid var(--dn-line); border-radius: 14px;
                 padding: 10px 12px; margin: 4px 0 12px; background: var(--dn-soft); }}
      .dn-user b {{ display:block; font-size: .95rem; }} .dn-user span {{ font-size: .78rem; opacity: .7; }}
      .dn-avatar {{ width: 38px; height: 38px; border-radius: 50%; display:flex; align-items:center; justify-content:center;
                   color:#fff; font-weight: 700; background: linear-gradient(135deg, var(--dn-accent), var(--dn-accent2)); flex: none; }}

      /* ---------- auth screen ---------- */
      .dn-auth-brand {{ position: relative; overflow:hidden; border-radius: 24px; padding: 40px 36px; color: #fff; min-height: 520px;
                       background: radial-gradient(900px 400px at 0% 0%, rgba(255,255,255,.18), transparent 60%),
                                   linear-gradient(140deg, #4F46E5 0%, #6D5BF5 50%, #06B6D4 100%);
                       box-shadow: 0 20px 40px -18px rgba(79,70,229,.6); }}
      .dn-auth-brand img {{ border-radius: 14px; box-shadow: 0 6px 18px rgba(0,0,0,.18); }}
      .dn-auth-brand h1 {{ color:#fff !important; font-size: 2.6rem; margin: 18px 0 4px; padding: 0; }}
      .dn-auth-brand .lead {{ font-size: 1.1rem; opacity: .92; margin-bottom: 22px; }}
      .dn-auth-brand ul {{ list-style: none; padding: 0; margin: 0; }}
      .dn-auth-brand li {{ padding: 9px 0; border-bottom: 1px solid rgba(255,255,255,.16); font-size: .98rem; }}
      .dn-auth-brand .credit {{ margin-top: 26px; font-size: .85rem; opacity: .85; }}
      .dn-or {{ display:flex; align-items:center; gap: 10px; opacity:.6; margin: 10px 0; font-size:.85rem; }}
      .dn-or::before, .dn-or::after {{ content:""; flex:1; border-top: 1px solid var(--dn-line); }}

      .ad-footer {{ text-align:center; opacity:.7; font-size:.85rem; margin-top: 2.5rem; padding: 1.1rem 0 .4rem;
                   border-top: 1px solid var(--dn-line); }}
      .ad-icon img {{ transition: transform .2s ease; }} .ad-icon img:hover {{ transform: scale(1.1); }}
      .dn-check {{ display:flex; gap:10px; align-items:flex-start; padding: 8px 0; border-bottom: 1px dashed var(--dn-line); }}
      .dn-check .s {{ flex:none; width: 22px; text-align:center; }}
    </style>
    """, unsafe_allow_html=True)


SECTIONS = {
    "Data Hub": "Start", "Welcome to DataNova": "Start", "Data Profile": "Prepare", "Clean & Transform": "Prepare",
    "Explore & Pivot": "Prepare", "Auto Dashboard": "Visualise", "Chart Studio": "Visualise",
    "3D Tactile Data Sculpture": "Visualise", "Statistics": "Analyse", "Machine Learning Studio": "Analyse",
    "Forecasting": "Analyse", "AI Insights & Ask": "Analyse", "Reports & Sharing": "Share",
    "Deploy DataNova": "Share", "Settings & Help": "Share",
}


def page_header(title: str, subtitle: str = "", icon: str = ""):
    st.session_state["_page"] = title
    crumb = SECTIONS.get(title, "DataNova")
    st.markdown(f"""<div class="dn-head"><div class="snow"></div>
        <span class="crumb">{crumb}</span>
        <h1>{icon} {title}</h1><p>{subtitle}</p></div>""", unsafe_allow_html=True)


def feature_card(icon: str, title: str, text: str):
    st.markdown(f"<div class='dn-card'><div class='ic'>{icon}</div><h4>{title}</h4><p>{text}</p></div>",
                unsafe_allow_html=True)


def footer():
    st.markdown(f"<div class='ad-footer'>© 2026 {APP_NAME} v{__version__} — {APP_TAGLINE}<br>"
                f"<b>{CREDIT}</b></div>", unsafe_allow_html=True)


def insight_box(ins: dict):
    st.markdown(f"<div class='ad-insight {ins['level']}'><b>{ins['title']}</b>{ins['text']}</div>",
                unsafe_allow_html=True)


def palette_colors():
    return PALETTES.get(palette(), PALETTES["DataNova"])


def col_select(label, options, key, default=None, none_option=False, help=None):
    opts = (["—"] if none_option else []) + list(options)
    idx = 0
    if default is not None and default in opts:
        idx = opts.index(default)
    v = st.selectbox(label, opts, index=idx, key=key, help=help)
    return None if v == "—" else v


def embed_html(html: str, height: int = 800):
    """Embed a full HTML document (reports). Uses st.iframe on new Streamlit, components.html on older."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:  # Streamlit < 1.50
        import streamlit.components.v1 as components

        components.html(html, height=height, scrolling=True)
