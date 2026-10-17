# DataNova — Project Guide (from start to end)

*Designed & developed by Gitesh Kund*

This guide explains **what** every part of DataNova does, **how** it works internally and **why** it
was designed that way. Use it for your report, viva / presentation and for extending the code.

---

## 1. Problem & goal

Analysts spend most of their time *preparing* data (loading, cleaning, reshaping) instead of
*understanding* it. Existing BI tools are either expensive (Power BI, Tableau) or need coding
(Python notebooks, Dash). **DataNova** automates the whole pipeline:

```
 Load ──► Profile ──► Clean ──► Filter ──► Visualise ──► Analyse / Model / Forecast ──► Report & Share
```

so that a non-programmer gets reliable insight in minutes and a professional saves hours.

---

## 2. Feature highlights

| Area | Highlights |
|---|---|
| Accounts | Login / register / guest; SQLite by default, MongoDB optional; PBKDF2 hashing; admin role |
| Data loading | CSV, TSV, Excel (all sheets), JSON, Parquet, PDF tables, URL, paste; auto delimiter/encoding/date detection; join & append |
| Cleaning | Configurable pipeline, 6 imputation strategies, outlier cap/remove, type conversion, find & replace, feature engineering, **undo** |
| Charts | 21 chart types with colour, facets, aggregation, trendlines; icon quick-chart gallery; pinned custom dashboard |
| Statistics | Pearson/Spearman/Kendall, t-test, ANOVA, chi-square, normality — with English interpretations |
| Machine learning | AutoML leaderboard, CV, ROC, confusion matrix, feature importance, what-if & batch predictions, model download, clustering, anomaly detection, PCA |
| Forecasting | 6 models, back-testing, WAPE/RMSE, prediction intervals, seasonal decomposition |
| Insights | Automatic narrative insights, natural-language questions, segment breakdowns, optional LLM analyst |
| Reports & sharing | Report builder; HTML / PDF / Excel export; SMTP e-mail; password-protected, expiring share links |
| Admin | Users, feedback, issues and activity analytics |
| Quality | 46 unit tests + headless tests of every page |

---

## 3. Architecture

```
            ┌────────────────────────── Streamlit UI (views/) ───────────────────────────┐
 Browser ◄──┤ app.py: login · share viewer · sidebar (dataset + smart filters) · navigation │
            │ home · data_hub · analysis(profile,stats) · clean · explore · auto_dashboard  │
            │ visualize · sculpture_page · ml_studio · forecast · insights · reports · settings│
            └──────────────┬─────────────────────────────────────────────────────────────┘
                           │ calls pure functions (DataFrame in → result out)
            ┌──────────────▼──────────────── core/ (no Streamlit) ─────────────────────────┐
            │ data_io  profiling  cleaning  features  viz  stats_tests  ml  forecasting     │
            │ insights  nlq  llm  reports  emailer  sharing  sculpture  auth  db            │
            └──────────────┬───────────────────────────────────────────────────────────────┘
                           │
            SQLite data/datanova.db  (users, activity, feedback, issues, shares)  · data/shares/*.html
            (optional MongoDB for users · optional SMTP server · optional LLM API)
```

**Why split `core/` from `views/`?** The engine can be unit-tested without a browser, reused in a
notebook or API, and each page stays short and readable.

### Session state (per browser tab)

| Key | Meaning |
|---|---|
| `user` | logged-in user (`username`, `role`, …) |
| `datasets` | `{name: {"raw", "clean", "history", "log", "source"}}` — raw is never modified |
| `active` | name of the dataset every page works on |
| `filters` | per-dataset global filters from the sidebar |
| `report` | list of report sections (figures stored as Plotly JSON) |
| `pinned` | chart specs pinned in Chart Studio |
| `settings` / `email_config` | appearance & SMTP settings |

`ui.working_df()` = *clean data + active filters* → this is what every analysis page receives.

---

## 4. Module-by-module explanation

### 4.1 Authentication — `core/auth.py`, `core/db.py`
* **Hashing:** PBKDF2-HMAC-SHA256, 200,000 iterations, 16-byte random salt, constant-time compare
  (`hmac.compare_digest`). Stored as `pbkdf2_sha256$iterations$salt$hash`.
* **Validation:** username pattern, password ≥ 6 chars, optional e-mail format; live strength meter.
* **Roles:** the first registered account becomes `admin` (sees the Admin tab).
* **Backends:** `_SQLiteUsers` (default) or `_MongoUsers` when `DATANOVA_MONGO_URI` is set.
* **Activity log:** every important action (login, load, transform, train, share…) is written to the
  `activity` table — shown in *Settings → Account* and the admin panel.

### 4.2 Data ingestion — `core/data_io.py`, `views/data_hub.py`
* `read_delimited` sniffs the delimiter (`, ; \t |`) and tries UTF-8 / CP1252 / Latin-1.
* `read_excel_all` loads **every sheet** as its own dataset.
* `read_pdf` uses **pdfplumber** to extract every table on every page, merges tables with identical
  headers (multi-page tables) and falls back to splitting text on 2+ spaces. Results can be downloaded
  as an Excel workbook (“PDF → Excel”).
* `coerce_numeric` converts text like `"$1,200"` or `"45%"` to numbers when ≥ 90 % of values parse.
* `auto_parse_dates` converts text columns that look like dates (≥ 85 % parse).
* **Samples:** retail sales (with lat/long & state codes, deliberate duplicates/missing/outliers),
  student funding (demonstrates the *funding_type* segment breakdown), website traffic (daily time series with
  weekly & yearly seasonality and spikes), customer churn (binary classification), IoT sensors (faults).
* **Merge/append:** pandas `merge` (inner/left/right/outer) and `concat` with a `source_dataset` tag.

### 4.3 Profiling & smart detection — `core/profiling.py`
* `semantic_type` → Numerical, Categorical, Date/Time, Boolean, Text (long free text) or Identifier
  (unique codes / `*_id`).
* **Quality score** = 0.4·completeness + 0.2·uniqueness + 0.2·validity + 0.2·consistency, where
  validity = share of numeric values inside 1.5×IQR fences and consistency penalises stray spaces and
  case variants (“Delhi” vs “delhi ”).
* `ranked_measures` guesses the headline measure (sales, revenue, amount…) used by KPIs, charts and
  insights; `default_agg` chooses **sum** for additive measures and **mean** for levels/rates.

### 4.4 Cleaning — `core/cleaning.py`, `views/clean.py`
Each step is a pure function returning `(new_df, log_message)`:
`standardize_columns` → `trim_text` → `drop_empty` → `remove_duplicates` → `handle_outliers` →
`impute_missing`. `run_pipeline(df, CleaningConfig)` chains the enabled steps.
* Imputation: median, mean, zero, interpolate, forward-fill, drop rows (numeric); mode, constant,
  forward-fill, drop rows (text).
* Outliers: IQR (Q1−1.5·IQR, Q3+1.5·IQR) or z-score (|z|>3) → cap, remove or set missing.
* **Undo:** `ui.set_clean` pushes the previous DataFrame + log onto a 15-step history stack.

### 4.5 Feature engineering — `core/features.py`
Formula columns (`df.eval`, e.g. `profit / sales * 100`), date parts (year, quarter, month, week,
weekday, weekend…), equal-width / quantile bins, log / sqrt / square / z-score / min-max / rank,
one-hot encoding (top 30 categories + “Other”), bulk rename.

### 4.6 Visualisation — `core/viz.py`, `views/visualize.py`, `views/auto_dashboard.py`
* `build_chart(df, kind, x, y, color, size, facet, agg, …)` – one function, 21 chart types. Charts
  that summarise (bar, pie, treemap, funnel, choropleth, line on categories) aggregate with
  `groupby(...).agg(agg)` first. Large data is sampled for scatter-type charts to keep the browser fast.
* `recommend_charts` – rule-based recommender: date+measure → line; category+measure → bar;
  category → donut; measure → histogram; two measures → scatter; category+measure → box;
  two categories → treemap; lat/long → map; US state codes → choropleth; ≥3 measures → heatmap.
* `kpi_values` – headline numbers with “second half vs first half” change when a date exists.
* **Pinned dashboard** stores chart *specs* (not images), so pinned charts re-render with the current
  filters.

### 4.7 Statistics — `core/stats_tests.py`
Welch t-test / Mann-Whitney U (two groups, with Cohen’s d), one-way ANOVA / Kruskal-Wallis,
chi-square independence (with Cramér’s V), Shapiro-Wilk / D’Agostino normality, Pearson / Spearman /
Kendall correlation tests. Each returns statistic, p-value and an English sentence.

### 4.8 Machine learning — `core/ml.py`, `views/ml_studio.py`
* **Pipeline:** `ColumnTransformer` → numeric: median imputation + standard scaling; categorical:
  most-frequent imputation + one-hot (≤ 25 levels). Dates become day numbers. This makes every model
  work on mixed data without manual preparation.
* **Task detection:** numeric target with > 15 unique values → regression, otherwise classification.
* **AutoML:** trains the selected models on a stratified train/test split, adds k-fold CV, ranks by
  R² (regression) or F1 (classification).
  * Regression metrics: R², MAE, RMSE, MSE. Classification: accuracy, precision, recall, F1, ROC-AUC.
* **Explainability:** permutation importance on the test set (model-agnostic).
* **What-if & batch scoring:** `predict_one`, `predict_frame`; the trained pipeline downloads as a
  pickle.
* **Clustering:** K-Means (with elbow + silhouette helper), Agglomerative, DBSCAN; 2-D PCA projection
  and per-cluster profiles; cluster labels can be added back to the dataset.
* **Anomaly detection:** Isolation Forest, Local Outlier Factor, Z-score, IQR, with scores, charts and
  one-click removal.
* **PCA:** scree plot, cumulative variance, loadings, 2-D map.

### 4.9 Forecasting — `core/forecasting.py`, `views/forecast.py`
1. `prepare_series` resamples to the chosen frequency (D/W/M/Q/Y/H), drops incomplete first/last
   periods, interpolates gaps.
2. `backtest` holds out the last *h* periods, fits each model on the rest, and scores MAE, RMSE and
   **WAPE** (= Σ|error| / Σ|actual|, robust when some actuals are tiny).
3. `forecast` refits the chosen model on the full history; the prediction interval width uses the
   back-test residual standard deviation, widening with √horizon.

Models: Naive, Seasonal naive, Moving average, Linear trend + Fourier seasonality (least squares),
Holt-Winters additive (statsmodels), ARIMA(2,1,2) (statsmodels). `decompose` shows trend / seasonal /
residual components.

### 4.10 Insights & natural language — `core/insights.py`, `core/nlq.py`, `core/llm.py`
* **Insights engine** scans for: quality score, columns with > 5 % missing, duplicates, heavy skew
  (|skew| > 1.5), outlier share, strong correlations (|r| ≥ 0.6), dominant categories, best/worst
  segment by the headline measure, linear time trend (± % over the period), and weekday seasonality.
* **NL query engine** (offline): normalises the question, fuzzy-matches column names (`difflib`),
  then detects intents with regular expressions — shape, missing, filter (`where x > 5`),
  correlation, histogram, outliers, trend (daily/weekly/monthly), top/bottom N, group-by aggregates,
  “A vs B”, single statistics, describe. Each answer returns text + table + chart.
* **LLM analyst** (optional): sends only a compact *summary* (schema, describe(), 8 sample rows) to
  the model — never the full dataset.

### 4.11 Reports, e-mail & sharing — `core/reports.py`, `core/emailer.py`, `core/sharing.py`
* A **section** is `{title, kind: figure|table|text|metrics, payload, note, source}`; figures are
  stored as Plotly JSON so they stay interactive.
* **HTML:** responsive, dark-mode aware, table of contents, interactive Plotly charts.
* **PDF:** ReportLab (pure Python). Charts are rendered by kaleido when Chrome is available, otherwise
  by a matplotlib fallback that redraws bar/line/scatter/histogram/pie/heatmap traces.
* **Excel:** contents sheet, every table, each chart’s underlying data, and the dataset.
* **E-mail:** STARTTLS/SSL presets (Gmail, Outlook, Yahoo, Zoho), multiple To/Cc, HTML body with
  static chart images, PDF/HTML attachments, connection test. Credentials stay in the session only.
* **Share links:** a frozen HTML snapshot saved in `data/shares/`, metadata in the `shares` table
  (owner, visibility, password hash, expiry, views). The app serves it at `?share=<id>`.

### 4.12 3D tactile sculpture — `core/sculpture.py`
Styles: *Wave* (sin x + cos y · value), *Data terrain* (values laid out as a height
map), *Pivot terrain* (mean value for each pair of categories). The STL writer builds the top
surface, a flat base and four walls so the mesh is **watertight** and printable.

---

## 5. End-to-end data flow (example)

1. User uploads `sales.xlsx` → `data_io.load_bytes` → one dataset per sheet, dates parsed.
2. Auto-clean on load → `cleaning.auto_clean` → `entry["clean"]`, log recorded.
3. Sidebar filter *Region = West* → stored in `filters` → `working_df()` now returns only West.
4. Auto Dashboard → `viz.kpi_values`, `viz.recommend_charts`, `insights.generate_insights`.
5. *Add whole dashboard to report* → sections appended to `st.session_state.report`.
6. ML Studio → `ml.train_and_compare(working_df, "profit", features)` → leaderboard, plots.
7. Reports → `reports.build_pdf(sections)` → download / `emailer.send_email` / `sharing.create_share`.

---

## 6. Non-functional design

* **Performance:** cached file parsing (`st.cache_data`), sampling for heavy charts, bounded grids,
  SVM skipped above 6,000 rows, Agglomerative limited to 8,000 rows.
* **Security:** salted hashes, no plaintext passwords, SMTP credentials never written to disk, share
  passwords hashed, LLM receives only summaries, parameterised SQL everywhere.
* **Accessibility:** colour-blind-safe palette option, dark-mode-aware styling, text alternatives
  (insights in words, not only charts).
* **Robustness:** every page degrades gracefully (e.g. “Forecasting needs a date column”), errors are
  shown inline instead of crashing.

---

## 7. Testing

* `tests/test_core.py` — 46 pytest tests: loaders, cleaning, profiling, every chart type, regression,
  classification, clustering, anomalies, PCA, forecasting accuracy, statistics, NL intents, insights,
  HTML/PDF/Excel generation, STL validity, auth and sharing.
* Every page was additionally run headless (Streamlit `AppTest`) against all five sample datasets,
  including button flows (cleaning, training, clustering, NL questions, report export, share links).

---

## 8. Ideas for future work

Scheduled/automatic report e-mails · database connectors (PostgreSQL, MySQL, Google Sheets) ·
streaming sources (Kafka) · SHAP explanations · hyper-parameter tuning · role-based dataset sharing
between users · saving whole projects (datasets + report) to the account · PWA/mobile layout.

---

© 2026 DataNova — Designed & developed by **Gitesh Kund**.
