import pandas as pd
import streamlit as st

from core import cleaning, features, profiling
from views import ui


def render():
    ui.page_header("Clean & Transform", "Fix data problems and engineer new features. Every step is logged and can be undone.", "🧹")
    if ui.active_entry() is None:
        ui.require_data()
    e = ui.active_entry()
    df = e["clean"]

    top = st.columns([1, 1, 1, 3])
    if top[0].button("↩️ Undo last step", disabled=not e["history"]):
        ui.undo()
        st.rerun()
    if top[1].button("♻️ Reset to raw data"):
        e["history"].append((e["clean"], list(e["log"])))
        e["clean"], e["log"] = e["raw"].copy(), ["Reset to raw data"]
        st.rerun()
    top[2].metric("Steps applied", len(e["log"]))

    t_auto, t_pipe, t_cols, t_fix, t_feat, t_log = st.tabs(
        ["⚡ One-click", "🛠️ Custom pipeline", "🏷️ Columns", "🔧 Fix values", "🧪 Feature engineering", "📜 Log & compare"])

    with t_auto:
        st.write("Applies: snake_case column names → trim text & normalise blanks (‘N/A’, ‘null’…) → drop empty "
                 "rows/columns → remove duplicates → fill numeric gaps with the median and text gaps with ‘Unknown’.")
        if st.button("Run automatic cleaning", type="primary"):
            out, msgs = cleaning.auto_clean(df)
            ui.set_clean(out, "Auto clean: " + "; ".join(msgs))
            for m in msgs:
                st.write("✅", m)

    with t_pipe:
        with st.form("pipeline"):
            c1, c2, c3 = st.columns(3)
            cfg = cleaning.CleaningConfig(
                standardize_names=c1.checkbox("Standardise column names", True),
                trim_text=c1.checkbox("Trim text / normalise blanks", True),
                drop_empty=c1.checkbox("Drop empty rows & ≥95%-empty columns", True),
                remove_duplicates=c1.checkbox("Remove duplicate rows", True),
                impute=c2.checkbox("Fill missing values", True),
                numeric_strategy=c2.selectbox("Numeric gaps", ["median", "mean", "zero", "interpolate", "forward fill", "drop rows"]),
                categorical_strategy=c2.selectbox("Text gaps", ["constant", "mode", "forward fill", "drop rows"]),
                constant_value=c2.text_input("Constant for text gaps", "Unknown"),
                outliers=c3.checkbox("Handle outliers", False),
                outlier_method=c3.selectbox("Outlier rule", ["iqr", "zscore"], format_func=lambda x: {"iqr": "1.5 × IQR", "zscore": "|z| > 3"}[x]),
                outlier_action=c3.selectbox("Action", ["cap", "remove", "set missing"]),
                drop_columns=st.multiselect("Drop columns first", list(df.columns)),
            )
            go = st.form_submit_button("Run pipeline", type="primary")
        if go:
            out, msgs = cleaning.run_pipeline(df, cfg)
            ui.set_clean(out, "Pipeline: " + "; ".join(msgs))
            for m in msgs:
                st.write("✅", m)
            st.rerun()

    with t_cols:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Drop columns**")
            drop = st.multiselect("Columns to drop", list(df.columns), key="cl_drop")
            if st.button("Drop", disabled=not drop):
                ui.set_clean(df.drop(columns=drop), f"Dropped columns: {', '.join(map(str, drop))}")
                st.rerun()
            st.markdown("**Change column types**")
            col = st.selectbox("Column", list(df.columns), key="cl_typecol")
            st.caption(f"Current dtype: `{df[col].dtype}` · detected as **{profiling.semantic_type(df[col])}**")
            target = st.selectbox("Convert to", ["numeric", "datetime", "category", "text", "boolean"], key="cl_type")
            if st.button("Convert"):
                out, msg = cleaning.convert_types(df, {col: target})
                ui.set_clean(out, msg)
                st.rerun()
        with c2:
            st.markdown("**Rename columns**")
            ed = st.data_editor(pd.DataFrame({"Column": df.columns.astype(str), "New name": df.columns.astype(str)}),
                                disabled=["Column"], hide_index=True, width="stretch", key="cl_rename")
            mapping = dict(zip(df.columns, ed["New name"]))
            if st.button("Apply renames"):
                out, msg = features.rename_columns(df, mapping)
                ui.set_clean(out, msg)
                st.rerun()

    with t_fix:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Find & replace**")
            col = st.selectbox("Column", list(df.columns), key="fr_col")
            vals = df[col].dropna().astype(str).value_counts().head(30).index.tolist()
            find = st.selectbox("Find value", vals, key="fr_find") if vals else None
            repl = st.text_input("Replace with (leave empty for missing)", key="fr_repl")
            if st.button("Replace", disabled=find is None):
                out = df.copy()
                mask = out[col].astype(str) == find
                newval = repl if repl else pd.NA
                if repl and pd.api.types.is_numeric_dtype(out[col]):
                    try:
                        newval = float(repl)
                    except ValueError:
                        pass
                out[col] = out[col].astype(object).where(~mask, newval)
                if pd.api.types.is_numeric_dtype(df[col]):
                    out[col] = pd.to_numeric(out[col], errors="coerce")
                ui.set_clean(out, f"Replaced {int(mask.sum())} × '{find}' in {col}")
                st.rerun()
            st.markdown("**Text case**")
            cc = st.multiselect("Text columns", profiling.categorical_cols(df), key="case_cols")
            case = st.radio("Case", ["title", "lower", "upper"], horizontal=True)
            if st.button("Normalise case", disabled=not cc):
                out, msg = cleaning.normalize_case(df, cc, case)
                ui.set_clean(out, msg)
                st.rerun()
        with c2:
            st.markdown("**Missing values per column**")
            miss = df.isna().sum()
            miss = miss[miss > 0]
            if miss.empty:
                st.success("No missing values.")
            else:
                st.dataframe(miss.rename("missing"), width="stretch")
                mc = st.multiselect("Columns to fill", list(miss.index), default=list(miss.index), key="mv_cols")
                ns = st.selectbox("Numeric strategy", ["median", "mean", "zero", "interpolate", "forward fill"], key="mv_n")
                cs = st.selectbox("Text strategy", ["mode", "constant", "forward fill"], key="mv_c")
                if st.button("Fill selected"):
                    out, msg = cleaning.impute_missing(df, ns, cs, columns=mc)
                    ui.set_clean(out, msg)
                    st.rerun()
            st.markdown("**Outliers in one column**")
            num = profiling.numeric_cols(df)
            if num:
                oc = st.selectbox("Column", num, key="oc_col")
                m = st.selectbox("Rule", ["iqr", "zscore"], key="oc_m")
                a = st.selectbox("Action", ["cap", "remove", "set missing"], key="oc_a")
                if st.button("Handle outliers"):
                    out, msg = cleaning.handle_outliers(df, m, a, [oc])
                    ui.set_clean(out, msg)
                    st.rerun()

    with t_feat:
        num, dts, cats = profiling.numeric_cols(df), profiling.datetime_cols(df), profiling.categorical_cols(df, 50)
        kind = st.radio("Create", ["Formula column", "Date parts", "Bins", "Math transform", "One-hot encoding"], horizontal=True)
        try:
            if kind == "Formula column":
                st.caption("Use column names (snake_case works best) with + − × ÷, e.g. `profit / sales * 100` or `quantity * 2`.")
                c1, c2 = st.columns([1, 3])
                name = c1.text_input("New column name", "new_feature")
                expr = c2.text_input("Expression", f"{num[0]} * 2" if num else "")
                if expr:
                    try:
                        preview, _ = features.formula_column(df.head(5), name, expr)
                        st.dataframe(preview[[name]].T, width="stretch")
                    except Exception as exc:  # noqa: BLE001
                        st.caption(f"Preview error: {exc}")
                if st.button("Create column", type="primary"):
                    out, msg = features.formula_column(df, name, expr)
                    ui.set_clean(out, msg)
                    st.rerun()
            elif kind == "Date parts":
                if not dts:
                    st.info("No date columns (convert one on the Columns tab).")
                else:
                    col = st.selectbox("Date column", dts)
                    parts = st.multiselect("Parts", ["year", "quarter", "month", "month_name", "week", "day", "dayofweek", "hour", "is_weekend"],
                                           default=["year", "month", "dayofweek"])
                    if st.button("Extract", type="primary"):
                        out, msg = features.extract_date_parts(df, col, parts)
                        ui.set_clean(out, msg)
                        st.rerun()
            elif kind == "Bins":
                col = st.selectbox("Numeric column", num)
                c1, c2 = st.columns(2)
                b = c1.slider("Number of bins", 2, 20, 5)
                m = c2.radio("Method", ["equal width", "quantile"], horizontal=True)
                if st.button("Create bins", type="primary"):
                    out, msg = features.bin_column(df, col, b, m)
                    ui.set_clean(out, msg)
                    st.rerun()
            elif kind == "Math transform":
                col = st.selectbox("Numeric column", num)
                how = st.selectbox("Transform", ["log", "sqrt", "square", "z-score", "min-max", "rank"])
                if st.button("Transform", type="primary"):
                    out, msg = features.transform_column(df, col, how)
                    ui.set_clean(out, msg)
                    st.rerun()
            else:
                col = st.selectbox("Categorical column", cats)
                drop_o = st.checkbox("Drop original column")
                if st.button("Encode", type="primary"):
                    out, msg = features.one_hot(df, col, drop_o)
                    ui.set_clean(out, msg)
                    st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not create feature: {exc}")

    with t_log:
        st.markdown("**Transformation log**")
        if e["log"]:
            for i, m in enumerate(e["log"], 1):
                st.write(f"{i}. {m}")
            if st.button("📌 Add cleaning log to report"):
                ui.add_section("Data cleaning steps", "text", "\n".join(f"- {m}" for m in e["log"]), source="Clean")
                st.toast("Added")
        else:
            st.caption("No transformations yet.")
        raw = e["raw"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}", delta=f"{len(df) - len(raw):,}")
        c2.metric("Columns", df.shape[1], delta=df.shape[1] - raw.shape[1])
        c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}", delta=f"{int(df.isna().sum().sum() - raw.isna().sum().sum()):,}", delta_color="inverse")
        c4.metric("Duplicates", int(df.duplicated().sum()), delta=int(df.duplicated().sum() - raw.duplicated().sum()), delta_color="inverse")
        a, b = st.columns(2)
        a.markdown("**Raw (first 100 rows)**")
        a.dataframe(raw.head(100), width="stretch")
        b.markdown("**Cleaned (first 100 rows)**")
        b.dataframe(df.head(100), width="stretch")
