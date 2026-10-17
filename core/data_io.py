"""Data ingestion and export.

Supported inputs: CSV / TSV / TXT (delimiter + encoding auto-detected),
Excel (every sheet), JSON / JSON-lines, Parquet, PDF (tables via pdfplumber,
falling back to text parsing), a public URL to any of those, and five built-in
sample datasets.
"""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

SUPPORTED_EXTENSIONS = ["csv", "tsv", "txt", "xlsx", "xls", "json", "jsonl", "parquet", "pdf"]


# ---------------------------------------------------------------- readers
def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def read_delimited(raw: bytes, sep: str | None = None) -> pd.DataFrame:
    text = _decode(raw)
    if sep is None:
        sample = text[:20000]
        try:
            sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            sep = ","
    return pd.read_csv(io.StringIO(text), sep=sep, low_memory=False)


def read_excel_all(raw: bytes) -> dict[str, pd.DataFrame]:
    sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None)
    return {str(k): v for k, v in sheets.items() if not v.empty}


def read_json(raw: bytes) -> pd.DataFrame:
    text = _decode(raw).strip()
    try:
        return pd.read_json(io.StringIO(text))
    except ValueError:
        try:
            return pd.read_json(io.StringIO(text), lines=True)
        except ValueError:
            import json

            return pd.json_normalize(json.loads(text))


def read_pdf(raw: bytes) -> dict[str, pd.DataFrame]:
    """Extract every table found in a PDF. Falls back to whitespace-split text."""
    import pdfplumber

    tables: dict[str, pd.DataFrame] = {}
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            for tno, table in enumerate(page.extract_tables() or [], start=1):
                rows = [r for r in table if r and any(c not in (None, "") for c in r)]
                if len(rows) < 2:
                    continue
                header = [str(h).strip() if h else f"col_{i}" for i, h in enumerate(rows[0])]
                header = _dedupe(header)
                df = pd.DataFrame(rows[1:], columns=header)
                tables[f"page{pno}_table{tno}"] = df
        if not tables:
            text = "\n".join((p.extract_text() or "") for p in pdf.pages)
            df = _parse_text_table(text)
            if df is not None:
                tables["pdf_text"] = df
    # Merge tables that share the same header (multi-page tables)
    merged: dict[str, pd.DataFrame] = {}
    by_header: dict[tuple, list[str]] = {}
    for name, df in tables.items():
        by_header.setdefault(tuple(df.columns), []).append(name)
    for header, names in by_header.items():
        frames = [tables[n] for n in names]
        merged[names[0] if len(names) == 1 else f"{names[0]}+{len(names) - 1}"] = (
            pd.concat(frames, ignore_index=True)
        )
    return {k: coerce_numeric(v) for k, v in merged.items()}


def _parse_text_table(text: str) -> pd.DataFrame | None:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header, data = None, []
    for line in lines:
        parts = re.split(r"\s{2,}|\t", line)
        if header is None:
            if len(parts) > 2:
                header = _dedupe(parts)
            continue
        if len(parts) == len(header):
            data.append(parts)
    if header and data:
        return pd.DataFrame(data, columns=header)
    return None


def _dedupe(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        n = n or "col"
        if n in seen:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


def coerce_numeric(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Convert text columns that are mostly numbers (e.g. '1,200', '$5', '12%')."""
    df = df.copy()
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        non_null = s.dropna().astype(str).str.strip()
        non_null = non_null[non_null != ""]
        if non_null.empty:
            continue
        cleaned = non_null.str.replace(r"[,\s$€£₹%]", "", regex=True)
        nums = pd.to_numeric(cleaned, errors="coerce")
        if nums.notna().mean() >= threshold:
            full = s.astype(str).str.strip().str.replace(r"[,\s$€£₹%]", "", regex=True)
            df[col] = pd.to_numeric(full.replace({"": np.nan, "nan": np.nan, "None": np.nan}), errors="coerce")
    return df


def auto_parse_dates(df: pd.DataFrame, threshold: float = 0.85) -> tuple[pd.DataFrame, list[str]]:
    """Detect text columns that hold dates and convert them. Returns (df, converted_cols)."""
    df = df.copy()
    converted = []
    for col in df.columns:
        s = df[col]
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue
        sample = s.dropna().astype(str).head(500)
        if sample.empty or sample.str.len().median() < 6:
            continue
        # must contain a date-ish separator to avoid converting plain words / codes
        if not sample.str.contains(r"\d{1,4}[-/.]\d{1,2}|\d{1,2}\s+[A-Za-z]{3}|[A-Za-z]{3,9}\s+\d{1,2}", regex=True).mean() > 0.8:
            continue
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed.notna().mean() >= threshold:
            df[col] = pd.to_datetime(s, errors="coerce", format="mixed")
            converted.append(col)
    return df, converted


def load_bytes(name: str, raw: bytes) -> dict[str, pd.DataFrame]:
    """Load a file into one or more DataFrames keyed by a display name."""
    ext = Path(name).suffix.lower().lstrip(".")
    stem = Path(name).stem
    if ext in ("csv", "txt"):
        frames = {stem: read_delimited(raw)}
    elif ext == "tsv":
        frames = {stem: read_delimited(raw, sep="\t")}
    elif ext in ("xlsx", "xls"):
        sheets = read_excel_all(raw)
        frames = {stem if len(sheets) == 1 else f"{stem} [{k}]": v for k, v in sheets.items()}
    elif ext in ("json", "jsonl"):
        frames = {stem: read_json(raw)}
    elif ext == "parquet":
        frames = {stem: pd.read_parquet(io.BytesIO(raw))}
    elif ext == "pdf":
        tables = read_pdf(raw)
        frames = {stem if len(tables) == 1 else f"{stem} [{k}]": v for k, v in tables.items()}
    else:
        raise ValueError(f"Unsupported file type: .{ext}")
    out = {}
    for k, df in frames.items():
        df.columns = [str(c).strip() for c in df.columns]
        df, _ = auto_parse_dates(df)
        out[k] = df
    if not out:
        raise ValueError("No tabular data could be found in this file.")
    return out


def load_url(url: str) -> dict[str, pd.DataFrame]:
    import requests

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    name = url.split("?")[0].rstrip("/").split("/")[-1] or "url_data.csv"
    if "." not in name:
        ctype = resp.headers.get("content-type", "")
        name += ".json" if "json" in ctype else ".csv"
    return load_bytes(name, resp.content)


# ---------------------------------------------------------------- samples
def _rng(seed=42):
    return np.random.default_rng(seed)


def sample_sales(n: int = 1200) -> pd.DataFrame:
    rng = _rng(1)
    states = {
        "CA": (36.78, -119.42), "TX": (31.97, -99.90), "NY": (40.71, -74.01), "FL": (27.66, -81.52),
        "IL": (40.63, -89.40), "WA": (47.75, -120.74), "GA": (32.17, -82.90), "MA": (42.41, -71.38),
    }
    regions = {"CA": "West", "WA": "West", "TX": "South", "FL": "South", "GA": "South",
               "NY": "East", "MA": "East", "IL": "Central"}
    cats = {"Electronics": 420, "Furniture": 260, "Office Supplies": 45, "Clothing": 70, "Sports": 110}
    dates = pd.date_range("2023-01-01", "2025-06-30", freq="D")
    d = rng.choice(dates, n)
    st_codes = rng.choice(list(states), n, p=[.22, .18, .16, .12, .1, .08, .08, .06])
    cat = rng.choice(list(cats), n)
    qty = rng.integers(1, 12, n)
    season = 1 + 0.25 * np.sin((pd.DatetimeIndex(d).month.values - 3) / 12 * 2 * np.pi)
    growth = 1 + (pd.DatetimeIndex(d) - dates[0]).days.values / 900
    unit = np.array([cats[c] for c in cat]) * rng.uniform(0.7, 1.3, n)
    sales = np.round(unit * qty * season * growth, 2)
    discount = rng.choice([0, 0, 0, .05, .1, .15, .2, .3], n)
    profit = np.round(sales * (0.28 - discount) + rng.normal(0, 25, n), 2)
    lat = np.array([states[s][0] for s in st_codes]) + rng.normal(0, 1.2, n)
    lon = np.array([states[s][1] for s in st_codes]) + rng.normal(0, 1.5, n)
    df = pd.DataFrame({
        "Order ID": [f"ORD-{100000 + i}" for i in range(n)],
        "Order Date": pd.to_datetime(d),
        "Region": [regions[s] for s in st_codes],
        "State Code": st_codes,
        "Category": cat,
        "Segment": rng.choice(["Consumer", "Corporate", "Home Office"], n, p=[.5, .3, .2]),
        "Quantity": qty,
        "Discount": discount,
        "Sales": sales,
        "Profit": profit,
        "Customer Rating": np.clip(np.round(rng.normal(4.0, 0.8, n), 1), 1, 5),
        "Latitude": lat.round(4),
        "Longitude": lon.round(4),
    }).sort_values("Order Date").reset_index(drop=True)
    # sprinkle real-world messiness so the cleaning module has work to do
    idx = rng.choice(n, 40, replace=False)
    df.loc[idx[:20], "Customer Rating"] = np.nan
    df.loc[idx[20:30], "Segment"] = np.nan
    df.loc[idx[30:], "Sales"] = (df.loc[idx[30:], "Sales"] * 4).round(2)  # outliers
    df = pd.concat([df, df.sample(15, random_state=3)], ignore_index=True)  # duplicates
    return df


def sample_students(n: int = 600) -> pd.DataFrame:
    """University funding dataset (demonstrates the segment breakdown on 'funding_type')."""
    rng = _rng(2)
    depts = ["CSE", "IT", "Mechanical", "Civil", "Electrical", "Biotech"]
    ft = rng.choice(["Scholarship", "Research Grant", "Self-Funded", "Government Aid", "Industry Sponsored"],
                    n, p=[.3, .15, .3, .15, .1])
    base = {"Scholarship": 60000, "Research Grant": 120000, "Self-Funded": 0,
            "Government Aid": 40000, "Industry Sponsored": 90000}
    att = np.clip(rng.normal(82, 9, n), 40, 100)
    study = np.clip(rng.normal(14, 5, n), 1, 40)
    cgpa = np.clip(3.5 + 0.03 * att + 0.08 * study + rng.normal(0, .7, n), 4, 10)
    return pd.DataFrame({
        "student_id": [f"S{2300 + i}" for i in range(n)],
        "name": [f"Student {i}" for i in range(n)],
        "department": rng.choice(depts, n),
        "year": rng.integers(1, 5, n),
        "gender": rng.choice(["Male", "Female"], n),
        "funding_type": ft,
        "amount": [max(0, base[f] + rng.normal(0, 8000)) if base[f] else 0 for f in ft],
        "attendance_pct": att.round(1),
        "study_hours_week": study.round(1),
        "cgpa": cgpa.round(2),
        "placed": np.where(cgpa + rng.normal(0, 1, n) > 8.2, "Yes", "No"),
    })


def sample_web_traffic() -> pd.DataFrame:
    rng = _rng(3)
    dates = pd.date_range("2024-01-01", periods=540, freq="D")
    t = np.arange(len(dates))
    weekly = 1 + 0.18 * np.isin(dates.dayofweek, [0, 1, 2, 3]).astype(float)
    visits = (2000 + 3.2 * t + 350 * np.sin(2 * np.pi * t / 365)) * weekly + rng.normal(0, 120, len(t))
    visits[[100, 250, 400]] *= 2.4  # traffic spikes = anomalies
    signups = visits * rng.uniform(0.03, 0.05, len(t))
    return pd.DataFrame({
        "date": dates,
        "visits": visits.round().astype(int),
        "signups": signups.round().astype(int),
        "bounce_rate": np.clip(rng.normal(0.42, 0.05, len(t)), 0.2, 0.8).round(3),
        "avg_session_min": np.clip(rng.normal(3.4, 0.6, len(t)), 0.5, 8).round(2),
        "channel_top": rng.choice(["Organic", "Paid", "Social", "Referral", "Email"], len(t)),
    })


def sample_churn(n: int = 1500) -> pd.DataFrame:
    rng = _rng(4)
    tenure = rng.integers(1, 72, n)
    monthly = rng.uniform(20, 120, n).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .2])
    support = rng.integers(0, 8, n)
    logit = -1.2 - 0.04 * tenure + 0.02 * monthly + 0.35 * support + np.where(contract == "Month-to-month", 1.2, -0.6)
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame({
        "customer_id": [f"C{10000 + i}" for i in range(n)],
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "total_charges": (tenure * monthly * rng.uniform(.95, 1.05, n)).round(2),
        "contract": contract,
        "internet_service": rng.choice(["Fiber", "DSL", "None"], n, p=[.45, .4, .15]),
        "payment_method": rng.choice(["Card", "Bank transfer", "E-check", "Mailed check"], n),
        "support_calls": support,
        "senior_citizen": rng.choice(["Yes", "No"], n, p=[.16, .84]),
        "churn": np.where(churn, "Yes", "No"),
    })


def sample_iot(n: int = 2000) -> pd.DataFrame:
    rng = _rng(5)
    ts = pd.date_range("2025-03-01", periods=n, freq="h")
    temp = 22 + 4 * np.sin(np.arange(n) / 24 * 2 * np.pi) + rng.normal(0, .6, n)
    hum = 55 - 1.5 * (temp - 22) + rng.normal(0, 2, n)
    vib = np.abs(rng.normal(0.3, 0.05, n))
    fault_idx = rng.choice(n, 25, replace=False)
    vib[fault_idx] += rng.uniform(0.5, 1.5, 25)
    temp[fault_idx] += rng.uniform(5, 12, 25)
    return pd.DataFrame({
        "timestamp": ts,
        "machine": rng.choice(["M-01", "M-02", "M-03", "M-04"], n),
        "temperature_c": temp.round(2),
        "humidity_pct": hum.round(1),
        "vibration_g": vib.round(3),
        "power_kw": (5 + 0.4 * temp + rng.normal(0, 0.5, n)).round(2),
        "status": np.where(np.isin(np.arange(n), fault_idx), "Fault", "Normal"),
    })


SAMPLE_DATASETS = {
    "🛒 Retail Sales (orders, regions, map)": sample_sales,
    "🎓 Student Funding & Performance": sample_students,
    "🌐 Website Traffic (time series)": sample_web_traffic,
    "📉 Customer Churn (classification)": sample_churn,
    "🏭 IoT Sensor Readings (anomalies)": sample_iot,
}


# ---------------------------------------------------------------- exports
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _col_width(name, series: pd.Series) -> int:
    """Readable Excel column width; safe for empty / all-missing / mixed columns on any pandas version."""
    def cell_len(v):
        try:
            if v is None or (not isinstance(v, (list, dict, tuple)) and pd.isna(v)):
                return 0
        except (TypeError, ValueError):
            pass
        return len(str(v))
    sample = series.head(2000)
    lens = [cell_len(v) for v in sample]
    p90 = float(np.percentile(lens, 90)) if lens else 0.0
    if not np.isfinite(p90):
        p90 = 0.0
    return int(min(50, max(10, len(str(name)) + 2, p90 + 2)))


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            safe = re.sub(r"[\[\]:*?/\\]", "_", str(name))[:31] or "Sheet"
            out = df.copy()
            for c in out.columns:  # Excel cannot store tz-aware datetimes
                if isinstance(out[c].dtype, pd.DatetimeTZDtype):
                    out[c] = out[c].dt.tz_localize(None)
            out.to_excel(writer, sheet_name=safe, index=False)
            ws = writer.sheets[safe]
            for i, col in enumerate(out.columns):
                ws.set_column(i, i, _col_width(col, out[col]))
    return buf.getvalue()


def to_json_bytes(df: pd.DataFrame) -> bytes:
    return df.to_json(orient="records", date_format="iso", indent=2).encode("utf-8")


def to_parquet_bytes(df: pd.DataFrame) -> bytes | None:
    try:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        return buf.getvalue()
    except Exception:
        return None
