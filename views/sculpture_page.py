import streamlit as st

from core import profiling, sculpture
from views import ui


def render():
    ui.page_header("3D Tactile Data Sculpture", "Turn numbers into a landscape you can rotate — and 3D-print as an STL for touch-based data exploration.", "🗿")
    df = ui.require_data()
    num = profiling.numeric_cols(df)
    if not num:
        st.info("Need at least one numeric column.")
        return
    cats = [c for c in profiling.categorical_cols(df, max_unique=40) if df[c].nunique() >= 2]
    c1, c2, c3 = st.columns(3)
    value = c1.selectbox("Value column", num, index=num.index(profiling.ranked_measures(df)[0]) if profiling.ranked_measures(df) else 0)
    mode = c2.selectbox("Sculpture style", sculpture.MODES if len(cats) >= 2 else sculpture.MODES[:2],
                        help="Wave = sin(x)+cos(y)·value · Terrain = values laid out as a height map · "
                             "Pivot terrain = average value for each pair of categories")
    cs = c3.selectbox("Colour scale", ["Viridis", "Tealgrn", "Plasma", "Cividis", "Turbo", "Earth"])
    rc = cc = None
    if mode.startswith("Pivot"):
        a, b = st.columns(2)
        rc = a.selectbox("Rows (Y axis)", cats, key="sc_r")
        cc = b.selectbox("Columns (X axis)", [c for c in cats if c != rc], key="sc_c")
    o1, o2, o3, o4 = st.columns(4)
    res = o1.slider("Resolution", 10, 80, 40)
    smooth = o2.slider("Smoothing passes", 0, 5, 1)
    hscale = o3.slider("Height scale", 0.2, 3.0, 1.0)
    size = o4.number_input("Print size (mm)", 30, 300, 100)
    try:
        X, Y, Z, xl, yl = sculpture.build_grid(df, value, mode, rc, cc, res, smooth, hscale)
        fig = sculpture.surface_figure(X, Y, Z, f"3D sculpture of {value} — {mode}", cs, ui.template(), xl, yl)
        ui.show_fig(fig, "sculpture", source="3D Sculpture")
        stl = sculpture.grid_to_stl(X, Y, Z, size_mm=float(size))
        st.download_button("🖨️ Download STL for 3D printing", stl, file_name=f"datanova_{value}_sculpture.stl",
                           mime="model/stl", type="primary")
        st.caption(f"Watertight mesh with a solid base · {Z.shape[0]}×{Z.shape[1]} grid · {len(stl) // 50:,} triangles. "
                   "Open it in any slicer (Cura, PrusaSlicer) or viewer.")
    except ValueError as exc:
        st.warning(str(exc))
