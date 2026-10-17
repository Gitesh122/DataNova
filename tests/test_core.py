"""Unit tests for the DataNova core (run: pytest -q)."""
import os
import tempfile

os.environ.setdefault("DATANOVA_DATA_DIR", tempfile.mkdtemp())

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from core import (auth, cleaning, data_io, features, forecasting, insights, ml, nlq, profiling, reports,  # noqa: E402
                  sculpture, sharing, stats_tests, viz)


@pytest.fixture(scope="module")
def sales():
    df, _ = cleaning.auto_clean(data_io.sample_sales())
    return df


def test_samples_load():
    for fn in data_io.SAMPLE_DATASETS.values():
        df = fn()
        assert len(df) > 100 and df.shape[1] >= 5


def test_csv_roundtrip_and_sniffing():
    raw = b"name;score;when\nAsha;91;2024-01-02\nRavi;84;2024-02-03\n"
    out = data_io.load_bytes("x.csv", raw)
    df = next(iter(out.values()))
    assert list(df.columns) == ["name", "score", "when"]
    assert pd.api.types.is_datetime64_any_dtype(df["when"])


def test_coerce_numeric():
    df = pd.DataFrame({"price": ["$1,200", "300", "45%", None]})
    assert pd.api.types.is_numeric_dtype(data_io.coerce_numeric(df)["price"])


def test_cleaning_pipeline():
    raw = data_io.sample_sales()
    df, log = cleaning.auto_clean(raw)
    assert df.duplicated().sum() == 0
    assert df.isna().sum().sum() == 0
    assert all(c == c.lower() for c in df.columns)
    assert any("duplicate" in m for m in log)


def test_outliers_cap(sales):
    out, _ = cleaning.handle_outliers(sales, "iqr", "cap", ["sales"])
    q1, q3 = sales["sales"].quantile([.25, .75])
    assert out["sales"].max() <= q3 + 1.5 * (q3 - q1) + 1e-9


def test_profiling(sales):
    q = profiling.quality_score(sales)
    assert 0 <= q["score"] <= 100
    types = profiling.column_types(sales)
    assert types["sales"] == profiling.NUMERIC and types["order_date"] == profiling.DATETIME
    assert profiling.ranked_measures(sales)[0] == "sales"


def test_features(sales):
    df, _ = features.formula_column(sales, "margin", "profit / sales * 100")
    assert "margin" in df
    df, _ = features.extract_date_parts(sales, "order_date", ["year", "month"])
    assert {"order_date_year", "order_date_month"} <= set(df.columns)


@pytest.mark.parametrize("kind", list(viz.CHART_TYPES))
def test_every_chart_type(sales, kind):
    x = "region" if kind in ("bar", "pie", "donut", "treemap", "sunburst", "funnel") else \
        "state_code" if kind == "choropleth" else "sales"
    fig = viz.build_chart(sales, kind, x=x, y="profit", color="region", size="quantity")
    assert fig.data


def test_recommendations(sales):
    recs = viz.recommend_charts(sales)
    assert recs and recs[0]["kind"] == "line"


def test_regression_and_prediction(sales):
    res = ml.train_and_compare(sales, "profit", ["sales", "discount", "category", "order_date"], cv_folds=0,
                               models=["Linear Regression", "Random Forest"])
    assert res["task"] == "regression" and res["leaderboard"]["R²"].max() > 0.5
    assert len(ml.predict_frame(res, sales)) == len(sales)


def test_classification():
    df = data_io.sample_churn()
    res = ml.train_and_compare(df, "churn", ["tenure_months", "contract", "support_calls"], cv_folds=0,
                               models=["Logistic Regression"])
    assert res["task"] == "classification" and res["leaderboard"]["Accuracy"].iloc[0] > 0.6
    out = ml.predict_one(res, {"tenure_months": 2, "contract": "Month-to-month", "support_calls": 6})
    assert out["prediction"] in ("Yes", "No")


def test_clustering_anomalies_pca(sales):
    out, sil = ml.cluster(sales, ["sales", "profit"], k=3)
    assert out["cluster"].nunique() == 3 and sil is not None
    an = ml.detect_anomalies(sales, ["sales", "profit"], contamination=0.05)
    assert 0 < an["anomaly"].sum() < len(sales) * 0.1
    ev, _, coords = ml.pca_analysis(sales, ["sales", "profit", "quantity"])
    assert ev["Cumulative %"].iloc[-1] == pytest.approx(100, abs=0.5)


def test_forecasting():
    w = data_io.sample_web_traffic()
    s = forecasting.prepare_series(w, "date", "visits", "D")
    board, preds, test = forecasting.backtest(s, forecasting.models_available(), "D", 30)
    best = board.iloc[0]["Model"]
    assert board.iloc[0]["WAPE %"] < 15
    fc = forecasting.forecast(s, best, "D", 14, (test - preds[best]).values)
    assert len(fc) == 14 and (fc["upper"] >= fc["lower"]).all()


def test_stats(sales):
    assert stats_tests.correlation_test(sales, "sales", "profit")["p_value"] < 0.05
    assert "p=" in stats_tests.anova(sales, "sales", "region")["interpretation"]


@pytest.mark.parametrize("q,intent", [
    ("average sales by region", "groupby"), ("top 3 category by profit", "topn"), ("histogram of sales", "hist"),
    ("monthly sales trend", "trend"), ("correlation", "corr"), ("show rows where discount > 0.2", "filter"),
    ("how many rows", "shape"), ("outliers in profit", "outliers"),
])
def test_nlq(sales, q, intent):
    assert nlq.answer(q, sales)["intent"] == intent


def test_insights(sales):
    items = insights.generate_insights(sales)
    assert items and items[0]["kind"] == "quality"


def test_reports(sales):
    secs = [reports.make_section("KPIs", "metrics", [("Rows", "1")]),
            reports.make_section("Chart", "figure", viz.build_chart(sales, "bar", x="region", y="sales")),
            reports.make_section("Table", "table", sales.head()),
            reports.make_section("Text", "text", "**hi**\n- a")]
    html = reports.build_html(secs, summary="x")
    assert "<strong>hi</strong>" in html and "plotly" in html
    assert reports.build_pdf(secs).startswith(b"%PDF")
    assert reports.build_excel(secs, sales)[:2] == b"PK"


def test_sculpture_stl(sales):
    X, Y, Z, *_ = sculpture.build_grid(sales, "sales", resolution=10)
    stl = sculpture.grid_to_stl(X, Y, Z)
    n = int.from_bytes(stl[80:84], "little")
    assert len(stl) == 84 + 50 * n


def test_auth_and_sharing():
    ok, _ = auth.register_user("alice_t", "secret12")
    assert ok and not auth.register_user("alice_t", "secret12")[0]
    assert auth.authenticate_user("alice_t", "secret12")
    assert auth.authenticate_user("alice_t", "nope") is None
    sid = sharing.create_share("<p>x</p>", "alice_t", "T", "Private", "pw", "1 day")
    row = sharing.get_share(sid)
    assert row["html"] == "<p>x</p>" and sharing.check_password(row, "pw") and not sharing.check_password(row, "x")
