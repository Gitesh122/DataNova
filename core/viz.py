"""Chart builder + automatic chart recommendation.

``build_chart`` turns a simple spec (kind, x, y, color, aggregation …) into a
Plotly figure. ``recommend_charts`` inspects the dataset and proposes the most
useful charts, which the Auto Dashboard page renders without any user input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .profiling import categorical_cols, datetime_cols, numeric_cols, ranked_measures, default_agg

PALETTES = {
    "DataNova": ["#4F46E5", "#06B6D4", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#14B8A6"],
    "Plotly": px.colors.qualitative.Plotly,
    "Vivid": px.colors.qualitative.Vivid,
    "Pastel": px.colors.qualitative.Pastel,
    "Safe (colour-blind)": px.colors.qualitative.Safe,
    "Dark24": px.colors.qualitative.Dark24,
}
TEMPLATES = ["plotly_white", "plotly", "plotly_dark", "ggplot2", "seaborn", "simple_white", "presentation"]

CHART_TYPES = {
    # kind: (label, needs)  needs: x / y / both / any
    "histogram": ("Histogram", "x"),
    "box": ("Box plot", "y"),
    "violin": ("Violin plot", "y"),
    "bar": ("Bar chart", "x"),
    "line": ("Line chart", "both"),
    "area": ("Area chart", "both"),
    "scatter": ("Scatter plot", "both"),
    "bubble": ("Bubble chart", "both"),
    "pie": ("Pie chart", "x"),
    "donut": ("Donut chart", "x"),
    "treemap": ("Treemap", "x"),
    "sunburst": ("Sunburst", "x"),
    "funnel": ("Funnel", "x"),
    "heatmap": ("Density heatmap", "both"),
    "density_contour": ("Density contour", "both"),
    "scatter_3d": ("3D scatter", "both"),
    "scatter_matrix": ("Scatter matrix", "any"),
    "parallel": ("Parallel coordinates", "any"),
    "map": ("Map (lat/lon)", "any"),
    "choropleth": ("US state choropleth", "x"),
    "ecdf": ("ECDF", "x"),
}
AGGREGATIONS = ["count", "sum", "mean", "median", "min", "max", "std", "nunique"]

US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY",
    "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND",
    "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
}


def find_lat_lon(df):
    lat = next((c for c in df.columns if str(c).lower() in ("lat", "latitude")), None)
    lon = next((c for c in df.columns if str(c).lower() in ("lon", "lng", "long", "longitude")), None)
    return lat, lon


def find_state_col(df):
    for c in categorical_cols(df, max_unique=60):
        vals = set(df[c].dropna().astype(str).str.upper().unique())
        if vals and len(vals & US_STATES) / len(vals) > 0.8:
            return c
    return None


def _aggregate(df, x, y, agg, color=None):
    keys = [k for k in (x, color) if k]
    if y is None or agg == "count":
        out = df.groupby(keys, dropna=False, observed=True).size().reset_index(name="count")
        return out, "count"
    out = df.groupby(keys, dropna=False, observed=True)[y].agg(agg).reset_index()
    name = f"{agg}({y})"
    return out.rename(columns={y: name}), name


def _sample(df, n=20000):
    return df.sample(n, random_state=0) if len(df) > n else df


def build_chart(df: pd.DataFrame, kind: str, x=None, y=None, color=None, size=None, facet=None,
                agg: str = "sum", top_n: int | None = 25, template: str = "plotly_white",
                palette: str = "DataNova", title: str | None = None, z=None, orientation="v",
                log_y: bool = False, sort: str = "desc", bins: int = 30, trendline: bool = False) -> go.Figure:
    colors = PALETTES.get(palette, PALETTES["DataNova"])
    if color is not None and color in (x, y) and kind not in ("scatter", "bubble", "scatter_3d", "map", "box", "violin"):
        color = None
    if facet is not None and facet in (x, y, color):
        facet = None
    common = dict(template=template, color_discrete_sequence=colors)
    if facet:
        common["facet_col"] = facet
        common["facet_col_wrap"] = 3
    t = title
    data = df

    if kind == "histogram":
        fig = px.histogram(_sample(data), x=x, color=color, nbins=bins, marginal="box" if y is None else None,
                           y=y, histfunc="sum" if y else "count", log_y=log_y, **common)
        t = t or f"Distribution of {x}"
    elif kind in ("box", "violin"):
        fn = px.box if kind == "box" else px.violin
        kw = {"points": "outliers"} if kind == "box" else {"box": True}
        fig = fn(_sample(data), x=x, y=y, color=color, log_y=log_y, **kw, **common)
        t = t or f"{CHART_TYPES[kind][0]} of {y}" + (f" by {x}" if x else "")
    elif kind == "bar":
        g, val = _aggregate(data, x, y, agg, color)
        if sort != "none":
            g = g.sort_values(val, ascending=(sort == "asc"))
        if top_n and not color:
            g = g.head(top_n)
        if orientation == "h":
            fig = px.bar(g, y=x, x=val, color=color, orientation="h", **common)
        else:
            fig = px.bar(g, x=x, y=val, color=color, log_y=log_y, **common)
        fig.update_traces(texttemplate="%{value:.3s}", textposition="outside") if not color else None
        t = t or f"{val} by {x}"
    elif kind in ("line", "area"):
        if y is None:
            g, val = _aggregate(data, x, None, "count", color)
        elif pd.api.types.is_datetime64_any_dtype(data[x]) or data[x].nunique() < len(data) * 0.8:
            g, val = _aggregate(data, x, y, agg if agg != "count" else "sum", color)
        else:
            g, val = data, y
        g = g.sort_values(x)
        fn = px.line if kind == "line" else px.area
        kw = {"markers": len(g) < 60} if kind == "line" else {}
        fig = fn(g, x=x, y=val, color=color, log_y=log_y, **kw, **common)
        t = t or f"{val} over {x}"
    elif kind in ("scatter", "bubble"):
        d = _sample(data, 8000)
        kw = {}
        if kind == "bubble" and size:
            kw["size"] = size
            kw["size_max"] = 40
        if trendline:
            try:
                import statsmodels  # noqa: F401
                kw["trendline"] = "ols"
            except ImportError:
                pass
        fig = px.scatter(d, x=x, y=y, color=color, opacity=0.75, log_y=log_y, **kw, **common)
        t = t or f"{y} vs {x}"
    elif kind in ("pie", "donut"):
        g, val = _aggregate(data, x, y, agg)
        g = g.sort_values(val, ascending=False)
        if len(g) > 10:
            other = g.iloc[10:][val].sum()
            g = pd.concat([g.head(10), pd.DataFrame({x: ["Other"], val: [other]})], ignore_index=True)
        fig = px.pie(g, names=x, values=val, hole=0.45 if kind == "donut" else 0, template=template,
                     color_discrete_sequence=colors)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        t = t or f"Share of {val} by {x}"
    elif kind in ("treemap", "sunburst"):
        path = [c for c in (x, color) if c]
        d = data.copy()
        for p in path:
            d[p] = d[p].astype(str)
        fn = px.treemap if kind == "treemap" else px.sunburst
        if y:
            fig = fn(d, path=path, values=y, template=template, color_discrete_sequence=colors)
        else:
            d = d.groupby(path).size().reset_index(name="count")
            fig = fn(d, path=path, values="count", template=template, color_discrete_sequence=colors)
        t = t or f"{kind.title()} of {' / '.join(path)}"
    elif kind == "funnel":
        g, val = _aggregate(data, x, y, agg)
        g = g.sort_values(val, ascending=False).head(top_n or 10)
        fig = px.funnel(g, x=val, y=x, template=template, color_discrete_sequence=colors)
        t = t or f"Funnel of {val} by {x}"
    elif kind == "heatmap":
        fig = px.density_heatmap(_sample(data), x=x, y=y, z=z, histfunc=agg if z and agg != "count" else "count",
                                 template=template, color_continuous_scale="Tealgrn", text_auto=True)
        t = t or f"Density of {x} × {y}"
    elif kind == "density_contour":
        fig = px.density_contour(_sample(data, 5000), x=x, y=y, color=color, template=template,
                                 color_discrete_sequence=colors)
        t = t or f"Density contour of {x} × {y}"
    elif kind == "scatter_3d":
        zc = z or next((c for c in numeric_cols(data) if c not in (x, y)), y)
        fig = px.scatter_3d(_sample(data, 5000), x=x, y=y, z=zc, color=color, template=template,
                            color_discrete_sequence=colors, opacity=0.8)
        fig.update_traces(marker=dict(size=3))
        t = t or f"3D: {x}, {y}, {zc}"
    elif kind == "scatter_matrix":
        dims = [c for c in numeric_cols(data)][:5]
        fig = px.scatter_matrix(_sample(data, 3000), dimensions=dims, color=color, template=template,
                                color_discrete_sequence=colors)
        fig.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=.6))
        t = t or "Scatter matrix"
    elif kind == "parallel":
        dims = numeric_cols(data)[:7]
        cc = color if color in numeric_cols(data) else (dims[0] if dims else None)
        fig = px.parallel_coordinates(_sample(data, 3000), dimensions=dims, color=cc, template=template,
                                      color_continuous_scale="Tealgrn")
        t = t or "Parallel coordinates"
    elif kind == "map":
        lat, lon = find_lat_lon(data)
        if not (lat and lon):
            raise ValueError("Map needs latitude and longitude columns (e.g. 'latitude', 'longitude').")
        d = _sample(data, 5000)
        fig = px.scatter_map(d, lat=lat, lon=lon, color=color, size=size, hover_name=x, zoom=2.5,
                             map_style="carto-positron", template=template, color_discrete_sequence=colors) \
            if hasattr(px, "scatter_map") else \
            px.scatter_mapbox(d, lat=lat, lon=lon, color=color, size=size, hover_name=x, zoom=2.5,
                              mapbox_style="carto-positron", template=template)
        t = t or "Geographic distribution"
    elif kind == "choropleth":
        state = x or find_state_col(data)
        if not state:
            raise ValueError("Choropleth needs a column of 2-letter US state codes.")
        d = data.copy()
        d[state] = d[state].astype(str).str.upper()
        g, val = _aggregate(d, state, y, agg)
        fig = px.choropleth(g, locations=state, locationmode="USA-states", color=val, scope="usa",
                            color_continuous_scale="Tealgrn", template=template)
        t = t or f"{val} by state"
    elif kind == "ecdf":
        fig = px.ecdf(_sample(data), x=x, color=color, template=template, color_discrete_sequence=colors)
        t = t or f"Cumulative distribution of {x}"
    else:
        raise ValueError(f"Unknown chart kind: {kind}")

    fig.update_layout(title=dict(text=t, x=0.01), margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=-0.3))
    return fig


def correlation_heatmap(df, method="pearson", template="plotly_white"):
    num = numeric_cols(df)
    corr = df[num].corr(method=method).round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    aspect="auto", template=template)
    fig.update_layout(title=f"{method.title()} correlation matrix", margin=dict(l=20, r=20, t=60, b=20))
    return fig


def missing_heatmap(df, template="plotly_white"):
    d = df.isna().astype(int)
    if len(d) > 400:
        d = d.iloc[np.linspace(0, len(d) - 1, 400).astype(int)]
    fig = px.imshow(d.T, color_continuous_scale=[[0, "#E3F2FD"], [1, "#FF5630"]], aspect="auto",
                    template=template, labels=dict(x="Row", y="Column", color="Missing"))
    fig.update_layout(title="Missing-value map (red = missing)", coloraxis_showscale=False,
                      margin=dict(l=20, r=20, t=60, b=20))
    return fig


def kpi_values(df):
    """Pick headline numbers for the KPI strip."""
    ranked = ranked_measures(df)
    kpis = [("Rows", f"{len(df):,}", None)]
    for c in ranked[:3]:
        s = df[c]
        is_sum = default_agg(c) == "sum"
        val = s.sum() if is_sum else s.mean()
        delta = None
        dts = datetime_cols(df)
        if dts and len(df) > 20:
            d = df[[dts[0], c]].dropna().sort_values(dts[0])
            half = len(d) // 2
            a, b = d[c].iloc[:half], d[c].iloc[half:]
            first, second = (a.sum(), b.sum()) if is_sum else (a.mean(), b.mean())
            if first:
                delta = f"{(second - first) / abs(first) * 100:+.1f}% vs 1st half"
        kpis.append((("Total " if is_sum else "Avg ") + str(c), _fmt(val), delta))
    return kpis


def _fmt(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(v) >= div:
            return f"{v / div:,.2f}{unit}"
    return f"{v:,.2f}"


def recommend_charts(df, max_charts: int = 8) -> list[dict]:
    """Heuristic chart recommendations based on column types."""
    num = numeric_cols(df)
    cats = [c for c in categorical_cols(df, max_unique=30) if df[c].nunique() >= 2]
    dts = datetime_cols(df)
    num = ranked_measures(df)
    recs = []
    if dts and num:
        recs.append(dict(kind="line", x=dts[0], y=num[0], agg=default_agg(num[0]), why=f"{num[0]} trend over time"))
    if cats and num:
        recs.append(dict(kind="bar", x=cats[0], y=num[0], agg=default_agg(num[0]), why=f"Compare {num[0]} across {cats[0]}"))
    if cats:
        recs.append(dict(kind="donut", x=cats[min(1, len(cats) - 1)], y=None, agg="count",
                         why=f"Composition of {cats[min(1, len(cats) - 1)]}"))
    if num:
        recs.append(dict(kind="histogram", x=num[0], why=f"Distribution of {num[0]}"))
    if len(num) >= 2:
        recs.append(dict(kind="scatter", x=num[1], y=num[0], color=cats[0] if cats else None,
                         why=f"Relationship between {num[1]} and {num[0]}"))
    if cats and num:
        recs.append(dict(kind="box", x=cats[0], y=num[min(1, len(num) - 1)],
                         why=f"Spread of {num[min(1, len(num) - 1)]} per {cats[0]}"))
    if len(cats) >= 2:
        recs.append(dict(kind="treemap", x=cats[0], color=cats[1], y=num[0] if num else None,
                         why=f"Hierarchy {cats[0]} → {cats[1]}"))
    lat, lon = find_lat_lon(df)
    if lat and lon:
        recs.append(dict(kind="map", x=cats[0] if cats else None, color=cats[0] if cats else None, why="Where the records are"))
    elif find_state_col(df):
        recs.append(dict(kind="choropleth", x=find_state_col(df), y=num[0] if num else None, agg=default_agg(num[0]) if num else "count", why="Value by US state"))
    if len(num) >= 3:
        recs.append(dict(kind="heatmap_corr", why="How numeric columns relate"))
    return recs[:max_charts]
