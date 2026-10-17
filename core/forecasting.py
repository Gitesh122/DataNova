"""Time-series forecasting with back-testing.

Models: Naive, Seasonal naive, Moving average, Linear trend (+ seasonality),
Holt-Winters exponential smoothing and ARIMA (the last two need statsmodels).
Every model is evaluated on a hold-out window (MAE / RMSE / WAPE) before
forecasting the future, and prediction intervals are derived from the
hold-out residuals.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    HAS_STATSMODELS = False

FREQS = {"Daily": "D", "Weekly": "W", "Monthly": "MS", "Quarterly": "QS", "Yearly": "YS", "Hourly": "h"}
SEASON = {"D": 7, "W": 52, "MS": 12, "QS": 4, "YS": 1, "h": 24}


def models_available():
    base = ["Naive", "Seasonal naive", "Moving average", "Linear trend"]
    return base + (["Holt-Winters", "ARIMA"] if HAS_STATSMODELS else [])


def guess_freq(dates: pd.Series) -> str:
    raw = pd.to_datetime(dates).dropna()
    d = pd.Series(raw.sort_values().unique())
    if len(d) < 3:
        return "D"
    step = d.diff().median()
    span = d.iloc[-1] - d.iloc[0]
    # transactional data (several rows per day, spanning > 6 months) reads better weekly
    if step <= pd.Timedelta(days=2) and len(raw) > 1.2 * len(d) and span > pd.Timedelta(days=180):
        return "W"
    if step <= pd.Timedelta(hours=2):
        return "h"
    if step <= pd.Timedelta(days=2):
        return "D"
    if step <= pd.Timedelta(days=10):
        return "W"
    if step <= pd.Timedelta(days=45):
        return "MS"
    if step <= pd.Timedelta(days=120):
        return "QS"
    return "YS"


def prepare_series(df, date_col, value_col, freq="D", agg="sum") -> pd.Series:
    d = df[[date_col, value_col]].dropna().copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna().set_index(date_col).sort_index()
    r = d[value_col].resample(freq)
    s = r.agg(agg)
    counts = r.count()
    # drop incomplete first / last periods (e.g. a half week at the end) – they distort sums
    if len(s) > 4 and agg in ("sum", "count"):
        typical = counts.iloc[1:-1].median()
        if counts.iloc[-1] < 0.6 * typical:
            s = s.iloc[:-1]
        if counts.iloc[0] < 0.6 * typical:
            s = s.iloc[1:]
    if agg in ("sum", "count"):
        s = s.where(counts.reindex(s.index) > 0)  # periods with no data are gaps, not zeros
    return s.interpolate(limit_direction="both").astype(float)


def _fit_predict(name, train: pd.Series, horizon: int, freq: str, window: int = 7):
    m = SEASON.get(freq, 1)
    idx = pd.date_range(train.index[-1], periods=horizon + 1, freq=freq)[1:]
    y = train.values
    if name == "Naive":
        fc = np.repeat(y[-1], horizon)
    elif name == "Seasonal naive":
        if m > 1 and len(y) >= m:
            fc = np.array([y[-m + (i % m)] for i in range(horizon)])
        else:
            fc = np.repeat(y[-1], horizon)
    elif name == "Moving average":
        fc = np.repeat(np.mean(y[-window:]), horizon)
    elif name == "Linear trend":
        t = np.arange(len(y))
        X = [np.ones_like(t), t]
        use_season = m > 1 and len(y) >= 2 * m
        if use_season:
            X += [np.sin(2 * np.pi * t / m), np.cos(2 * np.pi * t / m)]
        X = np.vstack(X).T
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        tf = np.arange(len(y), len(y) + horizon)
        Xf = [np.ones_like(tf), tf]
        if use_season:
            Xf += [np.sin(2 * np.pi * tf / m), np.cos(2 * np.pi * tf / m)]
        fc = np.vstack(Xf).T @ beta
    elif name == "Holt-Winters":
        seasonal = "add" if (m > 1 and len(y) >= 2 * m) else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(y, trend="add", seasonal=seasonal,
                                         seasonal_periods=m if seasonal else None,
                                         initialization_method="estimated").fit()
        fc = model.forecast(horizon)
    elif name == "ARIMA":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = ARIMA(y, order=(2, 1, 2)).fit().forecast(horizon)
    else:
        raise ValueError(name)
    return pd.Series(np.asarray(fc, dtype=float), index=idx)


def _metrics(actual, pred):
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    err = a - p
    # WAPE (weighted absolute % error) is robust when some actual values are tiny, unlike plain MAPE
    denom = np.sum(np.abs(a))
    wape = np.sum(np.abs(err)) / denom * 100 if denom else np.nan
    return {"MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err ** 2))), "WAPE %": float(wape)}


def backtest(series: pd.Series, models: list[str], freq: str, holdout: int, window: int = 7):
    if len(series) < holdout + 8:
        raise ValueError(f"Need at least {holdout + 8} periods; the series has {len(series)}.")
    train, test = series.iloc[:-holdout], series.iloc[-holdout:]
    rows, preds = [], {}
    for name in models:
        try:
            p = _fit_predict(name, train, holdout, freq, window)
            p.index = test.index
            preds[name] = p
            rows.append({"Model": name, **_metrics(test.values, p.values)})
        except Exception as exc:  # noqa: BLE001
            rows.append({"Model": name, "MAE": np.nan, "RMSE": np.nan, "WAPE %": np.nan, "Error": str(exc)[:80]})
    board = pd.DataFrame(rows).sort_values("RMSE", na_position="last").reset_index(drop=True)
    return board, preds, test


def forecast(series: pd.Series, model: str, freq: str, horizon: int, holdout_resid: np.ndarray | None = None,
             window: int = 7, level: float = 0.95):
    fc = _fit_predict(model, series, horizon, freq, window)
    if holdout_resid is not None and len(holdout_resid) > 2:
        sd = float(np.std(holdout_resid))
    else:
        sd = float(series.diff().std() or 0)
    z = {0.8: 1.2816, 0.9: 1.6449, 0.95: 1.96, 0.99: 2.5758}.get(level, 1.96)
    widen = np.sqrt(np.arange(1, horizon + 1))
    lower, upper = fc - z * sd * widen, fc + z * sd * widen
    return pd.DataFrame({"forecast": fc, "lower": lower, "upper": upper})


def forecast_figure(series, fc_df, test_preds=None, title="Forecast", template="plotly_white"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, name="History", line=dict(color="#06B6D4")))
    if test_preds is not None:
        fig.add_trace(go.Scatter(x=test_preds.index, y=test_preds.values, name="Back-test",
                                 line=dict(color="#FFAB00", dash="dot")))
    fig.add_trace(go.Scatter(x=fc_df.index, y=fc_df["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc_df.index, y=fc_df["lower"], fill="tonexty", fillcolor="rgba(79,70,229,0.18)",
                             line=dict(width=0), name="Prediction interval"))
    fig.add_trace(go.Scatter(x=fc_df.index, y=fc_df["forecast"], name="Forecast",
                             line=dict(color="#4F46E5", width=3, dash="dash")))
    fig.update_layout(template=template, title=title, hovermode="x unified",
                      margin=dict(l=20, r=20, t=60, b=20), legend=dict(orientation="h", y=-0.2))
    return fig


def decompose(series: pd.Series, freq: str):
    if not HAS_STATSMODELS:
        return None
    m = SEASON.get(freq, 1)
    if m < 2 or len(series) < 2 * m:
        return None
    res = seasonal_decompose(series, period=m, model="additive", extrapolate_trend="freq")
    return pd.DataFrame({"observed": res.observed, "trend": res.trend, "seasonal": res.seasonal, "residual": res.resid})
