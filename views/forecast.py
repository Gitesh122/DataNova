import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core import forecasting as fc
from core import profiling
from views import ui


def render():
    ui.page_header("Forecasting", "Predict future values with six models, honest back-testing and prediction intervals.", "🔮")
    df = ui.require_data()
    dts, num = profiling.datetime_cols(df), profiling.numeric_cols(df)
    if not dts:
        st.warning("Forecasting needs a date/time column. Convert one on **Clean & Transform → Columns**, or load the "
                   "*Website Traffic* sample.")
        return
    if not num:
        st.warning("No numeric columns to forecast.")
        return
    c1, c2, c3, c4 = st.columns(4)
    dcol = c1.selectbox("Date column", dts)
    measures = profiling.ranked_measures(df) or num
    vcol = c2.selectbox("Value to forecast", measures)
    guess = fc.guess_freq(df[dcol])
    inv = {v: k for k, v in fc.FREQS.items()}
    freq_label = c3.selectbox("Frequency", list(fc.FREQS), index=list(fc.FREQS).index(inv.get(guess, "Daily")))
    freq = fc.FREQS[freq_label]
    agg = c4.selectbox("Aggregate with", ["sum", "mean", "median", "max", "min", "count"],
                       index=0 if profiling.default_agg(vcol) == "sum" else 1)
    series = fc.prepare_series(df, dcol, vcol, freq, agg)
    if len(series) < 12:
        st.warning(f"Only {len(series)} periods at {freq_label.lower()} frequency — choose a finer frequency.")
        return
    o1, o2, o3, o4 = st.columns(4)
    horizon = o1.number_input("Forecast horizon (periods)", 1, 365, min(12, max(3, len(series) // 5)))
    holdout = o2.number_input("Back-test window (periods)", 2, max(2, len(series) // 3), min(max(2, len(series) // 6), 60))
    window = o3.number_input("Moving-average window", 2, 60, 7)
    level = o4.selectbox("Interval", [0.8, 0.9, 0.95, 0.99], index=2, format_func=lambda x: f"{int(x * 100)}%")
    models = st.multiselect("Models", fc.models_available(), default=fc.models_available())
    if not fc.HAS_STATSMODELS:
        st.caption("Install `statsmodels` to unlock Holt-Winters and ARIMA.")

    if not models:
        return
    with st.spinner("Back-testing models…"):
        try:
            board, preds, test = fc.backtest(series, models, freq, int(holdout), int(window))
        except ValueError as exc:
            st.warning(str(exc))
            return
    best = board.dropna(subset=["RMSE"]).iloc[0]["Model"]
    st.markdown(f"**Back-test on the last {holdout} periods** — lower error is better. Best: **{best}**")
    st.dataframe(board.style.format({"MAE": "{:,.2f}", "RMSE": "{:,.2f}", "WAPE %": "{:.2f}"}).highlight_min(
        subset=["MAE", "RMSE", "WAPE %"], color="rgba(54,179,126,.3)"), width="stretch")
    chosen = st.selectbox("Model for the final forecast", board["Model"].tolist(), index=board["Model"].tolist().index(best))
    resid = (test - preds[chosen]).values if chosen in preds else None
    out = fc.forecast(series, chosen, freq, int(horizon), resid, int(window), level)
    fig = fc.forecast_figure(series, out, preds.get(chosen), f"{vcol}: {int(horizon)}-period forecast ({chosen})", ui.template())
    ui.show_fig(fig, "forecast", source="Forecasting", height=500)

    last, nxt, end = series.iloc[-1], out["forecast"].iloc[0], out["forecast"].iloc[-1]
    m1, m2, m3 = st.columns(3)
    m1.metric("Last actual", f"{last:,.2f}")
    m2.metric("Next period", f"{nxt:,.2f}", f"{(nxt - last) / abs(last) * 100:+.1f}%" if last else None)
    m3.metric("End of horizon", f"{end:,.2f}", f"{(end - last) / abs(last) * 100:+.1f}%" if last else None)
    tbl = out.round(2).reset_index().rename(columns={"index": "period"})
    ui.show_table(tbl, "fc_table", f"Forecast table — {vcol}", "Forecasting")

    dec = fc.decompose(series, freq)
    if dec is not None:
        with st.expander("🧩 Seasonal decomposition (trend + seasonality + residual)"):
            f = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=list(dec.columns), vertical_spacing=0.05)
            for i, c in enumerate(dec.columns, 1):
                f.add_trace(go.Scatter(x=dec.index, y=dec[c], name=c, line=dict(color=ui.palette_colors()[i - 1])), row=i, col=1)
            f.update_layout(height=700, template=ui.template(), showlegend=False, title="Decomposition")
            ui.show_fig(f, "decomp", source="Forecasting")
