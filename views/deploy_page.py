"""Deploy page: readiness check + step-by-step deployment to Streamlit Community Cloud,
GitHub and Docker hosts, with a downloadable deploy-ready project ZIP."""
from __future__ import annotations

import importlib
import io
import os
import re
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

from core import APP_NAME, CREDIT, __version__, db
from views import ui

ROOT = Path(__file__).resolve().parent.parent
EXCLUDE_DIRS = {".venv", "venv", "env", "__pycache__", ".git", ".pytest_cache", "shares", ".idea", ".vscode"}
EXCLUDE_FILES = {"secrets.toml"}
REQUIRED = ["streamlit", "pandas", "numpy", "plotly", "sklearn", "scipy", "statsmodels", "openpyxl",
            "xlsxwriter", "pdfplumber", "reportlab", "matplotlib", "requests"]


def _app_url() -> str:
    try:
        u = urlparse(st.context.url or "")
        return f"{u.scheme}://{u.netloc}" if u.netloc else ""
    except Exception:  # noqa: BLE001
        return ""


def on_cloud() -> bool:
    return ".streamlit.app" in _app_url() or os.environ.get("HOSTNAME", "") == "streamlit"


def project_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in ROOT.rglob("*"):
            rel = p.relative_to(ROOT)
            if any(part in EXCLUDE_DIRS for part in rel.parts) or p.name in EXCLUDE_FILES:
                continue
            if p.suffix in (".db", ".pyc") or p.is_dir():
                continue
            z.write(p, Path("DataNova") / rel)
    return buf.getvalue()


def checks() -> list[tuple[str, bool | None, str]]:
    out = []
    ver = tuple(int(x) for x in re.findall(r"\d+", st.__version__)[:2])
    out.append((f"Streamlit {st.__version__}", ver >= (1, 50), "Needs 1.50 or newer"))
    py = sys.version_info
    out.append((f"Python {py.major}.{py.minor}", (py.major, py.minor) >= (3, 10), "Needs Python 3.10+"))
    missing = []
    for m in REQUIRED:
        try:
            importlib.import_module(m)
        except Exception:  # noqa: BLE001
            missing.append(m)
    out.append(("Required packages installed", not missing,
                "Missing: " + ", ".join(missing) + " → pip install -r requirements.txt" if missing else "All present"))
    out.append(("requirements.txt present", (ROOT / "requirements.txt").exists(), "Streamlit Cloud installs from it"))
    out.append(("app.py entry point", (ROOT / "app.py").exists(), "Main file path to enter on Streamlit Cloud"))
    out.append(("Theme config (.streamlit/config.toml)", (ROOT / ".streamlit" / "config.toml").exists(), "Fonts, colours, Deploy button"))
    try:
        test = db.DATA_DIR / ".write_test"
        test.write_text("ok")
        test.unlink()
        writable = True
    except OSError:
        writable = False
    out.append(("Data folder writable", writable, str(db.DATA_DIR)))
    secret_admin = bool(os.environ.get("DATANOVA_ADMINS"))
    out.append(("Admin users configured (DATANOVA_ADMINS)", secret_admin if on_cloud() else None,
                "Recommended for public deployments" if not secret_admin else os.environ["DATANOVA_ADMINS"]))
    for mod, label in (("anthropic", "AI analyst (optional)"), ("pyarrow", "Parquet support (optional)")):
        try:
            importlib.import_module(mod)
            out.append((label, True, "installed"))
        except Exception:  # noqa: BLE001
            out.append((label, None, "not installed — optional"))
    return out


def render():
    ui.page_header("Deploy DataNova", "Put your dashboard online for free with Streamlit Community Cloud — or any Docker host.", "🚀")
    url = _app_url()
    if on_cloud():
        st.success(f"🎉 This app is already live at **{url}**. Share that link — or redeploy updates by pushing to GitHub.")
    else:
        st.info("You're running DataNova locally. Follow the 4 steps below to get a public link like "
                "`https://datanova-yourname.streamlit.app`.", icon="💡")

    t_check, t_cloud, t_git, t_secrets, t_docker = st.tabs(
        ["✅ Readiness check", "☁️ Streamlit Cloud (free)", "🐙 Push to GitHub", "🔐 Secrets & settings", "🐳 Docker / other hosts"])

    with t_check:
        res = checks()
        ok = sum(1 for _, s, _ in res if s is True)
        bad = [r for r in res if r[1] is False]
        c1, c2, c3 = st.columns(3)
        c1.metric("Checks passed", f"{ok}/{len(res)}")
        c2.metric("Blocking issues", len(bad))
        c3.metric("Version", f"{APP_NAME} {__version__}")
        for label, status, detail in res:
            icon = "✅" if status is True else "❌" if status is False else "➖"
            st.markdown(f"<div class='dn-check'><span class='s'>{icon}</span><div><b>{label}</b><br>"
                        f"<small style='opacity:.7'>{detail}</small></div></div>", unsafe_allow_html=True)
        if not bad:
            st.success("Everything needed for deployment is in place.")
        else:
            st.error("Fix the ❌ items above before deploying (usually: `python -m pip install -r requirements.txt`).")

        st.markdown("##### 📦 Deploy-ready project")
        if on_cloud() and not ui.is_admin():
            st.caption("Source download is available to admins only on a public deployment.")
        else:
            st.download_button("⬇️ Download deploy-ready ZIP", project_zip(), "DataNova_deploy.zip", "application/zip",
                               type="primary", help="The project without .venv, caches, local database or secrets")

    with t_cloud:
        st.markdown("""
**Streamlit Community Cloud** hosts Streamlit apps for free from a GitHub repository.

1. **Push the project to GitHub** — see the *Push to GitHub* tab (or upload the ZIP's files on github.com).
2. **Sign in** at share.streamlit.io with your GitHub account.
3. Click **Create app → Deploy a public app from GitHub** and fill in:
   - **Repository:** `your-username/DataNova`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** choose a name, e.g. `datanova-gitesh`
4. Open **Advanced settings** → choose **Python 3.12** → paste your secrets (see the *Secrets* tab) → **Deploy**.

The first build takes 3–5 minutes while packages install. After that, every `git push` redeploys automatically.
""")
        b1, b2, b3 = st.columns(3)
        b1.link_button("Open Streamlit Community Cloud", "https://share.streamlit.io/", type="primary", width="stretch",
                       icon=":material/cloud_upload:")
        b2.link_button("Create a GitHub repository", "https://github.com/new", width="stretch", icon=":material/add_box:")
        b3.link_button("Streamlit deploy docs", "https://docs.streamlit.io/deploy/streamlit-community-cloud", width="stretch",
                       icon=":material/menu_book:")
        with st.expander("⚠️ Good to know about the free tier"):
            st.markdown("""
- Apps **sleep after a few days without visitors**; the first visitor wakes them (≈30 s).
- About **1 GB RAM** — keep uploads to tens of MB for smooth ML training.
- The file system is **temporary**: accounts, feedback and share links in the local SQLite database reset when the
  app restarts or redeploys. For permanent accounts set `DATANOVA_MONGO_URI` to a free MongoDB Atlas cluster.
- Running locally, you'll also see Streamlit's own **Deploy** button in the top-right toolbar.
""")

    with t_git:
        st.markdown("Install **Git** (git-scm.com) once, then run these in the VS Code terminal inside the project folder:")
        st.code("""git init
git add .
git commit -m "DataNova first deploy"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/DataNova.git
git push -u origin main""", language="bash")
        st.markdown("To publish later changes:")
        st.code('git add .\ngit commit -m "Update"\ngit push', language="bash")
        st.caption("`.gitignore` already keeps `.venv`, the local database and `secrets.toml` out of GitHub. "
                   "No Git? On github.com create the repo → *Add file → Upload files* → drag in the unzipped project files.")

    with t_secrets:
        st.markdown("Paste this into **Streamlit Cloud → App → Settings → Secrets** (or `.streamlit/secrets.toml` locally). "
                    "All keys are optional.")
        admin = ui.username() if ui.username() != "guest" else "your-username"
        template = f'''# Who becomes admin (comma-separated usernames)
DATANOVA_ADMINS = "{admin}"

# Set to "0" to skip the login screen (everyone enters as guest)
DATANOVA_REQUIRE_LOGIN = "1"

# Permanent accounts in MongoDB Atlas (free tier) instead of temporary SQLite
# DATANOVA_MONGO_URI = "mongodb+srv://user:password@cluster0.xxxxx.mongodb.net/"

# Optional AI analyst (also add 'anthropic' to requirements.txt)
# ANTHROPIC_API_KEY = "sk-ant-..."
'''
        st.code(template, language="toml")
        st.download_button("⬇️ Download secrets.toml", template, "secrets.toml", "text/plain")
        st.caption("Streamlit Cloud exposes top-level secrets as environment variables, which DataNova reads automatically. "
                   "Never commit real secrets to GitHub.")

    with t_docker:
        st.markdown("The project includes a **Dockerfile** — use it on Render, Railway, Hugging Face Spaces, Azure, AWS or a VPS.")
        st.code("""# build & run locally
docker build -t datanova .
docker run -p 8501:8501 -e DATANOVA_ADMINS=gitesh datanova
# then open http://localhost:8501""", language="bash")
        st.markdown("""
- **Render / Railway:** *New Web Service → connect the GitHub repo* — the Dockerfile is detected and `$PORT` is honoured.
- **Hugging Face Spaces:** create a Space with the **Docker** SDK and push the project.
- **Your own server:** `pip install -r requirements.txt` then
  `streamlit run app.py --server.address 0.0.0.0 --server.port 80`
""")

    st.caption(f"{APP_NAME} v{__version__} · {CREDIT}")
