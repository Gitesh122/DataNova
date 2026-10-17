"""3D tactile data sculptures.

Turns data into a 3D surface you can rotate in the browser and export as an
STL file for 3D printing (accessible, touchable data). The grid size is bounded
and the STL is written directly with NumPy — no extra 3D library needed.
"""
from __future__ import annotations

import struct

import numpy as np
import pandas as pd
import plotly.graph_objects as go

MODES = ["Wave (classic)", "Data terrain", "Pivot terrain (2 categories)"]


def _smooth(Z, passes=1):
    for _ in range(passes):
        P = np.pad(Z, 1, mode="edge")
        Z = (P[:-2, 1:-1] + P[2:, 1:-1] + P[1:-1, :-2] + P[1:-1, 2:] + 4 * P[1:-1, 1:-1]) / 8
    return Z


def build_grid(df: pd.DataFrame, value_col: str, mode: str = "Data terrain", row_col=None, col_col=None,
               resolution: int = 40, smooth: int = 1, height_scale: float = 1.0):
    vals = pd.to_numeric(df[value_col], errors="coerce").dropna().to_numpy(float)
    if len(vals) < 2:
        raise ValueError("Need at least two numeric values.")
    rng = (vals.max() - vals.min()) or 1.0
    norm = (vals - vals.min()) / rng
    xl = yl = None
    if mode == "Wave (classic)":
        n = min(resolution, len(vals))
        idx = np.linspace(0, len(vals) - 1, n).astype(int)
        v = norm[idx]
        x = np.linspace(0, 10, n)
        X, Y = np.meshgrid(x, x)
        Z = np.sin(X) + np.cos(Y) * v[np.newaxis, :] * 3
    elif mode == "Data terrain":
        n = int(np.clip(np.ceil(np.sqrt(len(vals))), 2, resolution))
        idx = np.linspace(0, len(vals) - 1, n * n).astype(int)
        Z = norm[idx].reshape(n, n) * 10
        X, Y = np.meshgrid(np.arange(n), np.arange(n))
    else:
        if not (row_col and col_col):
            raise ValueError("Choose two categorical columns for a pivot terrain.")
        pv = df.pivot_table(index=row_col, columns=col_col, values=value_col, aggfunc="mean").fillna(0)
        pv = pv.iloc[:resolution, :resolution]
        Z = pv.to_numpy(float)
        Z = (Z - Z.min()) / ((Z.max() - Z.min()) or 1) * 10
        X, Y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
        xl, yl = [str(c) for c in pv.columns], [str(i) for i in pv.index]
    if smooth and mode != "Wave (classic)":
        Z = _smooth(Z, smooth)
    return X.astype(float), Y.astype(float), Z * height_scale, xl, yl


def surface_figure(X, Y, Z, title="3D Tactile Data Sculpture", colorscale="Viridis", template="plotly_white",
                   xl=None, yl=None):
    fig = go.Figure(go.Surface(x=X, y=Y, z=Z, colorscale=colorscale, opacity=0.95,
                               contours={"z": {"show": True, "usecolormap": True, "project_z": True}},
                               colorbar=dict(title="Height")))
    scene = dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Value", aspectmode="manual",
                 aspectratio=dict(x=1, y=1, z=0.5))
    if xl:
        scene["xaxis"] = dict(tickvals=list(range(len(xl))), ticktext=xl, title="")
    if yl:
        scene["yaxis"] = dict(tickvals=list(range(len(yl))), ticktext=yl, title="")
    fig.update_layout(title=title, template=template, scene=scene, margin=dict(l=0, r=0, t=50, b=0), height=600)
    return fig


def grid_to_stl(X, Y, Z, base: float = 1.0, size_mm: float = 100.0) -> bytes:
    """Watertight binary STL: top surface + flat base + four side walls."""
    ny, nx = Z.shape
    sx = size_mm / max(1e-9, (X.max() - X.min()) or 1)
    sy = size_mm / max(1e-9, (Y.max() - Y.min()) or 1)
    zr = (Z.max() - Z.min()) or 1
    Xs, Ys = (X - X.min()) * sx, (Y - Y.min()) * sy
    Zs = (Z - Z.min()) / zr * (size_mm * 0.3) + base * 5
    tris = []

    def quad(a, b, c, d):
        tris.append((a, b, c))
        tris.append((a, c, d))

    P = lambda i, j: (Xs[i, j], Ys[i, j], Zs[i, j])  # noqa: E731
    B = lambda i, j: (Xs[i, j], Ys[i, j], 0.0)  # noqa: E731
    for i in range(ny - 1):
        for j in range(nx - 1):
            quad(P(i, j), P(i, j + 1), P(i + 1, j + 1), P(i + 1, j))
            quad(B(i, j), B(i + 1, j), B(i + 1, j + 1), B(i, j + 1))
    for j in range(nx - 1):
        quad(B(0, j), B(0, j + 1), P(0, j + 1), P(0, j))
        quad(B(ny - 1, j + 1), B(ny - 1, j), P(ny - 1, j), P(ny - 1, j + 1))
    for i in range(ny - 1):
        quad(B(i + 1, 0), B(i, 0), P(i, 0), P(i + 1, 0))
        quad(B(i, nx - 1), B(i + 1, nx - 1), P(i + 1, nx - 1), P(i, nx - 1))
    out = bytearray(b"DataNova data sculpture".ljust(80, b" "))
    out += struct.pack("<I", len(tris))
    for a, b, c in tris:
        a, b, c = np.array(a), np.array(b), np.array(c)
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n) or 1
        out += struct.pack("<12fH", *(n / ln), *a, *b, *c, 0)
    return bytes(out)
