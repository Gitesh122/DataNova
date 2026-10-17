# ✨ DataNova — Smart Analytics & Auto-Dashboard Studio

**Designed & developed by Gitesh Kund**

DataNova turns a raw data file into a cleaned dataset, an automatic dashboard, machine-learning
models, forecasts and a shareable report — **without writing code**. It is built entirely with
**Streamlit** + pandas + Plotly + scikit-learn.

> See [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md) for the full technical explanation of every module.

---

## ✨ Features at a glance

| Area | What you get |
|---|---|
| 🔐 **Accounts** | Register / login (salted PBKDF2 hashes), guest mode, first user = admin, saved preferences, activity history. SQLite by default, MongoDB optional. |
| 📂 **Data Hub** | CSV/TSV (auto delimiter & encoding), Excel (all sheets), JSON, Parquet, **PDF tables → Excel**, public URLs, pasted text, 5 realistic sample datasets. Rename / duplicate / delete datasets, **join & append** datasets, export CSV/Excel/JSON/Parquet. |
| 🔍 **Data Profile** | Smart type detection (numeric, categorical, date, boolean, text, ID), **quality score /100**, missing-value map, distributions, outlier counts, strong correlations, chart recommendations. |
| 🧹 **Clean & Transform** | One-click auto clean, configurable pipeline (imputation strategies, outlier cap/remove), drop/rename/convert columns, find & replace, text case, **feature engineering** (formulas, date parts, bins, log/z-score, one-hot), **undo history**, raw-vs-clean comparison. |
| 🧠 **Smart filters** | Global sidebar filters (IQR-aware ranges, date ranges, categories, text search) that apply to every page. |
| 🧭 **Explore & Pivot** | Searchable grid, group-by summaries with any aggregation, pivot tables + heatmaps, filtered downloads. |
| ⚡ **Auto Dashboard** | KPIs with period-over-period change, 8 auto-recommended charts, written summary and key insights — zero clicks. |
| 📊 **Chart Studio** | **21 chart types** (histogram, box, violin, bar, line, area, scatter, bubble, pie, donut, treemap, sunburst, funnel, density heatmap/contour, 3D scatter, scatter matrix, parallel coordinates, map, US choropleth, ECDF), colour/facet/aggregation/log/trendline options, original icon quick-chart gallery, **pinnable custom dashboard**. |
| 🗿 **3D Sculpture** | Three sculpture styles; rotate in the browser; **download a watertight STL** for 3D printing. |
| 📐 **Statistics** | Pearson/Spearman/Kendall matrices, t-test / Mann-Whitney, ANOVA / Kruskal, chi-square, normality + Q-Q plot, crosstabs — all with plain-English interpretations. |
| 🤖 **Machine Learning** | **AutoML** leaderboard (8 regressors / 7 classifiers, CV), actual-vs-predicted, residuals, confusion matrix, ROC, permutation feature importance, what-if prediction form, batch scoring, model download; **clustering** (K-Means/Agglomerative/DBSCAN + elbow/silhouette), **anomaly detection** (Isolation Forest, LOF, Z-score, IQR), **PCA**. |
| 🔮 **Forecasting** | Naive, seasonal naive, moving average, linear trend + seasonality, Holt-Winters, ARIMA — **back-tested** leaderboard, prediction intervals, seasonal decomposition. |
| 💡 **AI Insights & Ask** | Automatic narrative insights, **natural-language questions** ("average sales by region", "monthly sales trend", "show rows where discount > 0.2"), segment breakdown, optional LLM analyst chat. |
| 📄 **Reports & Sharing** | Add any chart/table/text to a report, reorder & caption it, export **interactive HTML, PDF (no wkhtmltopdf needed), Excel**; **e-mail** via SMTP with attachments; **real share links** (public or password-protected, with expiry and view counts). |
| ⚙️ **Settings & Help** | Chart themes, colour-blind-safe palettes, accent colour, SMTP presets + connection test, change password, feedback & issue reports stored in the DB, **admin panel**, built-in user guide. |

---

## 🛠️ Installation

Requires **Python 3.10+**.

```bash
# 1. (recommended) create a virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run
streamlit run app.py
```

Or just double-click **`run_windows.bat`** (Windows) / run **`./run.sh`** (macOS/Linux).

Open <http://localhost:8501>, create an account (the first one becomes **admin**) or click
**Continue as guest**.

### Optional extras

| Want | Do |
|---|---|
| Pixel-perfect chart images in PDF / e-mail | `pip install kaleido` and have Google Chrome installed (otherwise a built-in renderer is used automatically). |
| Accounts in MongoDB | `pip install pymongo` and set `DATANOVA_MONGO_URI=mongodb://localhost:27017/` |
| AI analyst chat | `pip install anthropic` and set `ANTHROPIC_API_KEY` (env var or `.streamlit/secrets.toml`) |
| Skip the login screen | set `DATANOVA_REQUIRE_LOGIN=0` |
| Files > 1 GB | `streamlit run app.py --server.maxUploadSize=4096` |
| Store data elsewhere | set `DATANOVA_DATA_DIR=/path/to/folder` |

---

## ☁️ Deploy online (free)

1. Push this folder to a GitHub repository.
2. Go to **share.streamlit.io**, then **Create app**. Pick the repo, branch `main`, main file **`app.py`**, then **Deploy**.
3. Optional: in *Advanced settings → Secrets* add `DATANOVA_ADMINS = "your-username"`.

Full guide: [`DEPLOY.md`](DEPLOY.md). Inside the app, open **Deploy** in the sidebar for a readiness check,
copy-paste commands and a deploy-ready ZIP. Docker is supported too (`Dockerfile` included).

---

## 🚀 Beginner's guide (end-to-end)

1. **Log in** (or continue as guest).
2. **Data Hub** → upload a CSV/Excel/PDF *or* click a sample dataset. Keep *Auto-clean on load* on.
3. **Data Profile** → read the quality score, missing-value map and column types.
4. **Clean & Transform** → fix what the profile flagged; create new features; use *Undo* freely.
5. **Sidebar → Smart filters** → focus on a subset (e.g. one region, last year).
6. **Auto Dashboard** → instant KPIs, charts and insights. Click *Add the whole dashboard to the report*.
7. **Chart Studio** → build custom charts, pin favourites to *My dashboard*.
8. **Statistics / Machine Learning / Forecasting** → test hypotheses, train models, forecast the future.
9. **AI Insights & Ask** → type questions in plain English.
10. **Reports & Sharing** → arrange sections, export HTML/PDF/Excel, e-mail it or create a share link.

---

## 🧪 Tests

```bash
pip install pytest
pytest -q            # 46 unit tests covering every core module
```

---

## 📁 Project structure

```
DataNova/
├── app.py                 # entry point: page config, login, share viewer, sidebar, navigation
├── core/                  # pure-Python engine (no Streamlit) — reusable & unit-tested
│   ├── auth.py            # accounts, password hashing, preferences (SQLite / MongoDB)
│   ├── db.py              # SQLite storage: users, activity, feedback, issues, shares
│   ├── data_io.py         # file/PDF/URL loaders, sample datasets, exporters
│   ├── profiling.py       # smart type detection, quality score, correlations
│   ├── cleaning.py        # cleaning steps + configurable pipeline
│   ├── features.py        # feature engineering
│   ├── viz.py             # chart builder (21 types) + chart recommender + KPIs
│   ├── stats_tests.py     # hypothesis tests with interpretations
│   ├── ml.py              # AutoML, clustering, anomalies, PCA, model export
│   ├── forecasting.py     # time-series models, back-testing, intervals
│   ├── insights.py        # automatic narrative insights
│   ├── nlq.py             # natural-language query engine
│   ├── llm.py             # optional LLM analyst
│   ├── reports.py         # HTML / PDF / Excel report builders
│   ├── emailer.py         # SMTP sending
│   ├── sharing.py         # share links
│   └── sculpture.py       # 3D surfaces + STL writer
├── views/                 # one module per Streamlit page + shared ui helpers
├── assets/icons/          # original DataNova icons & logo
├── tests/test_core.py     # pytest suite
├── docs/PROJECT_GUIDE.md  # full explanation of architecture, modules and algorithms
├── .streamlit/config.toml # theme & upload size
├── Dockerfile             # container deployment (Render, Railway, HF Spaces…)
├── DEPLOY.md              # deployment guide
└── requirements.txt
```

---

## 🐞 Troubleshooting

- **`streamlit` not found** → activate your virtual environment, then `pip install -r requirements.txt`.
- **Older Streamlit errors** (e.g. unexpected keyword `width`) → `pip install -U streamlit` (needs ≥ 1.50).
- **Gmail “authentication failed”** → enable 2-Step Verification and use an **App Password**.
- **PDF charts look simple** → install Chrome + `kaleido` for exact Plotly rendering.
- **Holt-Winters / ARIMA missing** → `pip install statsmodels`.
- **Share link opens login page** → links must point at the running app's address; set it in *Reports → Share link*.

---

© 2026 **DataNova** — Designed & developed by **Gitesh Kund**.
