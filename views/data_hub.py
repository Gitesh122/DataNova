
import pandas as pd
import streamlit as st

from core import data_io
from views import ui


@st.cache_data(show_spinner=False, max_entries=20)
def _load(name: str, raw: bytes):
    return data_io.load_bytes(name, raw)


def render():
    ui.page_header("Data Hub", "Bring data in from files, PDFs, URLs or samples — then manage, merge and export it.", "📂")
    t_up, t_sample, t_url, t_paste, t_manage, t_combine = st.tabs(
        ["⬆️ Upload", "🧪 Samples", "🌐 From URL", "📋 Paste", "🗂️ Manage datasets", "🔗 Merge / Append"])

    with t_up:
        auto = st.toggle("Auto-clean on load (snake_case names, fill missing, drop duplicates)", value=True, key="hub_auto")
        files = st.file_uploader("Drop one or more files", type=data_io.SUPPORTED_EXTENSIONS, accept_multiple_files=True,
                                 help="Large files: run `streamlit run app.py --server.maxUploadSize=2048`")
        if files and st.button("Load selected files", type="primary"):
            for f in files:
                try:
                    with st.spinner(f"Reading {f.name}…"):
                        frames = _load(f.name, f.getvalue())
                    for name, df in frames.items():
                        ui.add_dataset(name, df, f"file: {f.name}", auto_clean=auto)
                        st.success(f"✅ Loaded **{name}** — {len(df):,} rows × {df.shape[1]} columns")
                    if f.name.lower().endswith(".pdf"):
                        st.download_button(f"📥 Download tables from {f.name} as Excel",
                                           data_io.to_excel_bytes(frames), file_name=f"{f.name[:-4]}_tables.xlsx",
                                           key=f"pdfx_{f.name}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"❌ {f.name}: {exc}")
        st.caption("Supported: " + ", ".join(f".{e}" for e in data_io.SUPPORTED_EXTENSIONS) +
                   ". Excel workbooks load every sheet; PDFs load every table they contain.")

    with t_sample:
        st.write("Realistic synthetic datasets that exercise every feature (they include a few missing values, duplicates and outliers on purpose).")
        for name, fn in data_io.SAMPLE_DATASETS.items():
            c1, c2 = st.columns([4, 1])
            df = fn()
            c1.markdown(f"**{name}** — {len(df):,} rows · {', '.join(map(str, df.columns[:7]))}{'…' if df.shape[1] > 7 else ''}")
            if c2.button("Load", key=f"hub_s_{name}", width="stretch"):
                ui.add_dataset(name.split(" ", 1)[1], df, "sample", auto_clean=st.session_state.get("hub_auto", True))
                st.rerun()

    with t_url:
        url = st.text_input("Public URL of a CSV / Excel / JSON / Parquet file",
                            placeholder="https://raw.githubusercontent.com/…/data.csv")
        if st.button("Fetch", disabled=not url):
            try:
                with st.spinner("Downloading…"):
                    frames = data_io.load_url(url)
                for name, df in frames.items():
                    ui.add_dataset(name, df, f"url: {url}", auto_clean=st.session_state.get("hub_auto", True))
                st.success(f"Loaded {len(frames)} table(s).")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not load URL: {exc}")

    with t_paste:
        txt = st.text_area("Paste CSV / TSV text (first line = headers)", height=180,
                           placeholder="name,score\nAsha,91\nRavi,84")
        nm = st.text_input("Dataset name", value="Pasted data")
        if st.button("Create dataset", disabled=not txt.strip()):
            try:
                df = data_io.read_delimited(txt.encode())
                df, _ = data_io.auto_parse_dates(df)
                ui.add_dataset(nm, df, "pasted", auto_clean=st.session_state.get("hub_auto", True))
                st.success(f"Created **{nm}** with {len(df)} rows.")
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

    with t_manage:
        ds = st.session_state["datasets"]
        if not ds:
            st.info("No datasets loaded.")
        for name, e in list(ds.items()):
            with st.expander(f"{'⭐ ' if name == st.session_state['active'] else ''}{name} — "
                             f"{len(e['clean']):,} × {e['clean'].shape[1]} · {e['source']}"):
                st.dataframe(e["clean"].head(50), width="stretch")
                c1, c2, c3, c4 = st.columns(4)
                if c1.button("Set active", key=f"act_{name}", disabled=name == st.session_state["active"]):
                    st.session_state["active"] = name
                    st.rerun()
                new = c2.text_input("Rename", value=name, key=f"ren_{name}", label_visibility="collapsed")
                if new != name and c2.button("Rename", key=f"renb_{name}"):
                    ds[new] = ds.pop(name)
                    st.session_state["filters"][new] = st.session_state["filters"].pop(name, {})
                    if st.session_state["active"] == name:
                        st.session_state["active"] = new
                    st.rerun()
                if c3.button("Duplicate", key=f"dup_{name}"):
                    ui.add_dataset(f"{name} copy", e["clean"].copy(), f"copy of {name}")
                    st.rerun()
                if c4.button("🗑 Delete", key=f"del_{name}"):
                    ds.pop(name)
                    st.session_state["filters"].pop(name, None)
                    st.session_state["active"] = next(iter(ds), None)
                    st.rerun()
                st.markdown("**Export cleaned data**")
                d1, d2, d3, d4 = st.columns(4)
                df = e["clean"]
                d1.download_button("CSV", data_io.to_csv_bytes(df), f"{name}.csv", "text/csv", key=f"x1_{name}")
                d2.download_button("Excel", data_io.to_excel_bytes({name: df}), f"{name}.xlsx", key=f"x2_{name}")
                d3.download_button("JSON", data_io.to_json_bytes(df), f"{name}.json", "application/json", key=f"x3_{name}")
                pq = data_io.to_parquet_bytes(df)
                if pq:
                    d4.download_button("Parquet", pq, f"{name}.parquet", key=f"x4_{name}")

    with t_combine:
        ds = st.session_state["datasets"]
        if len(ds) < 2:
            st.info("Load at least two datasets to merge or append them.")
            return
        mode = st.radio("Operation", ["Join (merge on key columns)", "Append (stack rows)"], horizontal=True)
        names = list(ds)
        c1, c2 = st.columns(2)
        left = c1.selectbox("Left / first dataset", names, key="mg_l")
        right = c2.selectbox("Right / second dataset", [n for n in names if n != left], key="mg_r")
        L, R = ds[left]["clean"], ds[right]["clean"]
        if mode.startswith("Join"):
            common = [c for c in L.columns if c in R.columns]
            k1, k2, k3 = st.columns(3)
            lk = k1.multiselect("Left key(s)", list(L.columns), default=common[:1])
            rk = k2.multiselect("Right key(s)", list(R.columns), default=common[:1])
            how = k3.selectbox("Join type", ["inner", "left", "right", "outer"])
            if st.button("Merge", type="primary", disabled=not lk or len(lk) != len(rk)):
                try:
                    out = L.merge(R, left_on=lk, right_on=rk, how=how, suffixes=("", f"_{right[:10]}"))
                    ui.add_dataset(f"{left} ⨝ {right}", out, f"merge ({how})")
                    st.success(f"Created merged dataset with {len(out):,} rows.")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
        else:
            st.caption("Columns are aligned by name; missing columns are filled with blanks.")
            tag = st.checkbox("Add a 'source_dataset' column", value=True)
            if st.button("Append", type="primary"):
                a, b = L.copy(), R.copy()
                if tag:
                    a["source_dataset"], b["source_dataset"] = left, right
                out = pd.concat([a, b], ignore_index=True)
                ui.add_dataset(f"{left} + {right}", out, "append")
                st.success(f"Created appended dataset with {len(out):,} rows.")
