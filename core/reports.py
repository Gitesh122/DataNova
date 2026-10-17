"""Report builder: collect sections from any page, then export them as an
interactive HTML report, a PDF (pure-Python via ReportLab — no wkhtmltopdf
needed) or a multi-sheet Excel workbook.
"""
from __future__ import annotations

import html
import io
import re
import uuid
from datetime import datetime

import pandas as pd
import plotly.io as pio

from . import APP_NAME, CREDIT, __version__


# ---------------------------------------------------------------- sections
def make_section(title: str, kind: str, payload, note: str = "", source: str = "") -> dict:
    """kind: 'figure' (plotly fig), 'table' (DataFrame), 'text' (markdown), 'metrics' (list of (label, value))."""
    sec = {"id": uuid.uuid4().hex[:10], "title": title, "kind": kind, "note": note, "source": source,
           "created": datetime.now().strftime("%Y-%m-%d %H:%M")}
    if kind == "figure":
        sec["fig_json"] = payload.to_json()
    elif kind == "table":
        df = payload if isinstance(payload, pd.DataFrame) else pd.DataFrame(payload)
        sec["table"] = df.head(200).reset_index() if not isinstance(df.index, pd.RangeIndex) else df.head(200)
        sec["total_rows"] = len(df)
    elif kind == "text":
        sec["text"] = str(payload)
    elif kind == "metrics":
        sec["metrics"] = [(str(a), str(b)) for a, b in payload]
    else:
        raise ValueError(kind)
    return sec


def section_figure(sec):
    return pio.from_json(sec["fig_json"]) if sec.get("fig_json") else None


def md_to_html(text: str) -> str:
    t = html.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    lines, out, in_list = t.split("\n"), [], False
    for ln in lines:
        if re.match(r"^\s*[-•]\s+", ln):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append("<li>" + re.sub(r"^\s*[-•]\s+", "", ln) + "</li>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            if ln.startswith("### "):
                out.append(f"<h4>{ln[4:]}</h4>")
            elif ln.strip():
                out.append(f"<p>{ln}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


# ---------------------------------------------------------------- HTML
CSS = """
:root{--bg:#f4f7fb;--card:#fff;--ink:#172b4d;--muted:#6b778c;--accent:#4F46E5;--accent2:#06B6D4;--line:#dfe1e6}
@media (prefers-color-scheme: dark){:root{--bg:#0f1420;--card:#171d2b;--ink:#e6edf7;--muted:#97a0af;--line:#2a3244}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,'Segoe UI',Roboto,Arial,sans-serif;line-height:1.55}
.wrap{max-width:1100px;margin:0 auto;padding:32px 20px}
.hero{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff;border-radius:18px;padding:32px 28px;margin-bottom:24px}
.hero h1{margin:0 0 6px;font-size:30px}.hero p{margin:2px 0;opacity:.92}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px;margin:0 0 18px;box-shadow:0 2px 10px rgba(9,30,66,.05)}
.card h2{margin:0 0 4px;font-size:20px}.meta{color:var(--muted);font-size:13px;margin-bottom:12px}
table{border-collapse:collapse;width:100%;font-size:13px;overflow-x:auto;display:block}
th,td{border-bottom:1px solid var(--line);padding:7px 10px;text-align:left;white-space:nowrap}th{background:rgba(79,70,229,.08)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.kpi{border:1px solid var(--line);border-radius:12px;padding:14px}.kpi .l{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}.kpi .v{font-size:22px;font-weight:700}
.toc a{color:var(--accent2);text-decoration:none}.toc li{margin:3px 0}
.note{color:var(--muted);font-style:italic}
footer{text-align:center;color:var(--muted);font-size:13px;margin-top:30px}
img.chart{width:100%;border-radius:10px}
"""


def build_html(sections: list[dict], title: str = "DataNova Analysis Report", author: str = "", dataset: str = "",
               summary: str = "", interactive: bool = True, static_images: bool = False) -> str:
    gen = datetime.now().strftime("%d %b %Y, %H:%M")
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title>"
             f"<style>{CSS}</style>"]
    if interactive and not static_images:
        parts.append("<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>")
    parts.append("</head><body><div class='wrap'>")
    parts.append(f"<div class='hero'><h1>{html.escape(title)}</h1><p>Generated {gen}"
                 + (f" · by {html.escape(author)}" if author else "") + "</p>"
                 + (f"<p>Dataset: <strong>{html.escape(dataset)}</strong></p>" if dataset else "") + "</div>")
    if summary:
        parts.append(f"<div class='card'><h2>Executive summary</h2>{md_to_html(summary)}</div>")
    if len(sections) > 2:
        parts.append("<div class='card toc'><h2>Contents</h2><ol>" + "".join(
            f"<li><a href='#s{s['id']}'>{html.escape(s['title'])}</a></li>" for s in sections) + "</ol></div>")
    for s in sections:
        parts.append(f"<div class='card' id='s{s['id']}'><h2>{html.escape(s['title'])}</h2>"
                     f"<div class='meta'>{html.escape(s.get('source') or '')} · {s.get('created', '')}</div>")
        if s.get("note"):
            parts.append(f"<p class='note'>{html.escape(s['note'])}</p>")
        k = s["kind"]
        if k == "figure":
            fig = section_figure(s)
            if static_images:
                png = figure_png(fig)
                if png:
                    import base64

                    parts.append(f"<img class='chart' src='data:image/png;base64,{base64.b64encode(png).decode()}'>")
                else:
                    parts.append("<p class='note'>(Chart image unavailable — open the HTML report for the interactive chart.)</p>")
            else:
                parts.append(pio.to_html(fig, include_plotlyjs=False, full_html=False,
                                         config={"displaylogo": False, "responsive": True}))
        elif k == "table":
            parts.append(s["table"].to_html(index=False, border=0, float_format=lambda v: f"{v:,.4g}"))
            if s.get("total_rows", 0) > len(s["table"]):
                parts.append(f"<p class='note'>Showing {len(s['table'])} of {s['total_rows']:,} rows.</p>")
        elif k == "text":
            parts.append(md_to_html(s["text"]))
        elif k == "metrics":
            parts.append("<div class='kpis'>" + "".join(
                f"<div class='kpi'><div class='l'>{html.escape(a)}</div><div class='v'>{html.escape(b)}</div></div>"
                for a, b in s["metrics"]) + "</div>")
        parts.append("</div>")
    parts.append(f"<footer>Made with {APP_NAME} v{__version__} · {CREDIT}</footer>")
    parts.append("</div></body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------- images
_KALEIDO_OK: bool | None = None


def figure_png(fig, width=1000, height=560) -> bytes | None:
    """Render a Plotly figure to PNG.

    Uses kaleido (needs Chrome/Chromium) when available, otherwise falls back to
    a matplotlib re-drawing of the figure's traces so PDFs always get charts.
    """
    global _KALEIDO_OK
    if _KALEIDO_OK is not False:
        try:
            png = pio.to_image(fig, format="png", width=width, height=height, scale=2)
            _KALEIDO_OK = True
            return png
        except Exception:  # noqa: BLE001
            _KALEIDO_OK = False
    try:
        return _matplotlib_png(fig, width, height)
    except Exception:  # noqa: BLE001
        return None


def _matplotlib_png(fig, width=1000, height=560) -> bytes | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    palette = ["#4F46E5", "#06B6D4", "#36B37E", "#FFAB00", "#FF5630", "#00875A", "#8777D9", "#4C9AFF"]
    def arr(v):
        """Decode plotly's typed-array JSON ({'dtype','bdata','shape'}) into a list / ndarray."""
        if v is None:
            return None
        if isinstance(v, dict) and "bdata" in v:
            import base64 as b64

            a = np.frombuffer(b64.b64decode(v["bdata"]), dtype=np.dtype(v.get("dtype", "f8")))
            if v.get("shape"):
                shp = v["shape"]
                a = a.reshape([int(x) for x in str(shp).split(",")] if isinstance(shp, str) else shp)
            return a
        return np.asarray(v, dtype=object) if not hasattr(v, "shape") else v

    f, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=120)
    drawn = 0
    has_hist = any(tr.type == "histogram" for tr in fig.data)
    for i, tr in enumerate(fig.data):
      try:
        col = palette[i % len(palette)]
        t = tr.type
        x = arr(getattr(tr, "x", None))
        y = arr(getattr(tr, "y", None))
        x = list(x) if x is not None else None
        y = list(y) if y is not None else None
        name = tr.name or None
        if t == "bar" and x is not None and y is not None:
            if getattr(tr, "orientation", None) == "h":
                ax.barh([str(v) for v in y], x, color=col, label=name)
            else:
                ax.bar([str(v) for v in x], y, color=col, label=name)
                if len(x) > 6:
                    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        elif t in ("scatter", "scattergl") and x is not None and y is not None:
            mode = tr.mode or "markers"
            if "lines" in mode:
                ax.plot(x, y, color=col, label=name, lw=2, ls="--" if (tr.line and tr.line.dash) else "-")
            else:
                ax.scatter(x, y, s=10, alpha=.6, color=col, label=name)
            if tr.fill == "tonexty":
                ax.fill_between(x, y, color=col, alpha=.15)
        elif t == "histogram" and (x is not None or y is not None):
            ax.hist(np.asarray(x if x is not None else y, dtype=float), bins=30, color=col, alpha=.85, label=name)
        elif t == "box" and not has_hist:  # skip marginal box plots drawn above histograms
            vals = np.asarray(y if y is not None else x, dtype=float)
            ax.boxplot(vals[~np.isnan(vals)], vert=True)
        elif t == "pie":
            ax.pie(list(arr(tr.values)), labels=[str(v) for v in arr(tr.labels)], colors=palette, autopct="%1.0f%%",
                   wedgeprops=dict(width=0.55 if tr.hole else 1))
            ax.axis("equal")
        elif t == "heatmap" and getattr(tr, "z", None) is not None:
            z = np.asarray(arr(tr.z), dtype=float)
            im = ax.imshow(z, cmap="RdBu_r" if np.nanmin(z) < 0 else "viridis", aspect="auto")
            if tr.x is not None:
                ax.set_xticks(range(len(tr.x)), [str(v) for v in tr.x], rotation=45, ha="right")
            if tr.y is not None:
                ax.set_yticks(range(len(tr.y)), [str(v) for v in tr.y])
            f.colorbar(im, ax=ax)
        else:
            continue
        drawn += 1
      except Exception:  # noqa: BLE001 - skip traces we cannot redraw
        continue
    if not drawn:
        plt.close(f)
        return None
    title = fig.layout.title.text if fig.layout.title and fig.layout.title.text else ""
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.grid(alpha=.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if any(tr.name for tr in fig.data) and len(fig.data) > 1:
        ax.legend(fontsize=8, frameon=False)
    f.tight_layout()
    buf = io.BytesIO()
    f.savefig(buf, format="png")
    plt.close(f)
    return buf.getvalue()


# ---------------------------------------------------------------- PDF
def build_pdf(sections: list[dict], title: str = "DataNova Analysis Report", author: str = "", dataset: str = "",
              summary: str = "") -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.6 * cm,
                            bottomMargin=1.6 * cm, title=title, author=author or APP_NAME)
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=colors.HexColor("#172B4D"), alignment=TA_LEFT, fontSize=24)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=colors.HexColor("#0065FF"), spaceBefore=6)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=10, leading=14)
    meta = ParagraphStyle("m", parent=body, textColor=colors.HexColor("#6B778C"), fontSize=8.5)
    accent = colors.HexColor("#4F46E5")

    def inline(t):
        t = html.escape(t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        t = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", t)
        return t

    story = [Paragraph(html.escape(title), h1),
             Paragraph(f"Generated {datetime.now():%d %b %Y, %H:%M}" + (f" · {html.escape(author)}" if author else ""), meta)]
    if dataset:
        story.append(Paragraph(f"Dataset: <b>{html.escape(dataset)}</b>", meta))
    story.append(Spacer(1, 10))
    if summary:
        story.append(Paragraph("Executive summary", h2))
        for ln in summary.split("\n"):
            if ln.strip():
                story.append(Paragraph(inline(ln.lstrip("-• ")), body))
        story.append(Spacer(1, 6))

    width = A4[0] - 3.2 * cm
    missing_imgs = 0
    for s in sections:
        block = [Paragraph(html.escape(s["title"]), h2)]
        if s.get("note"):
            block.append(Paragraph(f"<i>{html.escape(s['note'])}</i>", meta))
        k = s["kind"]
        if k == "figure":
            png = figure_png(section_figure(s))
            if png:
                block.append(Image(io.BytesIO(png), width=width, height=width * 0.56))
            else:
                missing_imgs += 1
                block.append(Paragraph("<i>Chart available in the interactive HTML report.</i>", meta))
        elif k == "table":
            df = s["table"].copy()
            cols = list(df.columns)[:8]
            df = df[cols].head(25)

            def fmt(v):
                if isinstance(v, float):
                    return f"{v:,.4g}"
                return str(v)[:28]
            data = [[Paragraph(f"<b>{html.escape(str(c)[:22])}</b>", meta) for c in cols]] + \
                   [[Paragraph(html.escape(fmt(v)), meta) for v in row] for row in df.itertuples(index=False)]
            t = Table(data, repeatRows=1, colWidths=[width / len(cols)] * len(cols))
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6FCFF")),
                ("LINEBELOW", (0, 0), (-1, 0), 1, accent),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F5F7")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]))
            block.append(t)
            if s.get("total_rows", 0) > len(df) or len(s["table"].columns) > len(cols):
                block.append(Paragraph(f"Showing {len(df)} of {s.get('total_rows', len(df)):,} rows"
                                       f" and {len(cols)} of {len(s['table'].columns)} columns.", meta))
        elif k == "text":
            for ln in s["text"].split("\n"):
                if ln.strip():
                    block.append(Paragraph(("• " if ln.lstrip().startswith(("-", "•")) else "") + inline(ln.lstrip("-• ")), body))
        elif k == "metrics":
            cells = [[Paragraph(f"<font size=8 color='#6B778C'>{html.escape(a.upper())}</font><br/><b><font size=14>{html.escape(b)}</font></b>", body)
                      for a, b in s["metrics"][i:i + 4]] for i in range(0, len(s["metrics"]), 4)]
            for row in cells:
                row += [""] * (4 - len(row))
            t = Table(cells, colWidths=[width / 4] * 4)
            t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#DFE1E6")),
                                   ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DFE1E6")),
                                   ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
            block.append(t)
        block.append(Spacer(1, 12))
        story.append(KeepTogether(block) if k != "table" else block[0])
        if k == "table":
            story.extend(block[1:])

    def on_page(canvas, doc_):
        canvas.saveState()
        canvas.setFillColor(accent)
        canvas.rect(0, A4[1] - 6, A4[0], 6, fill=1, stroke=0)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#6B778C"))
        canvas.drawString(1.6 * cm, 1 * cm, f"{APP_NAME} report · {title[:60]} · {CREDIT}")
        canvas.drawRightString(A4[0] - 1.6 * cm, 1 * cm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return buf.getvalue()


# ---------------------------------------------------------------- Excel
def build_excel(sections: list[dict], data: pd.DataFrame | None = None) -> bytes:
    from .data_io import to_excel_bytes

    sheets = {}
    if data is not None:
        sheets["Data"] = data
    summary_rows = []
    for i, s in enumerate(sections, 1):
        summary_rows.append({"#": i, "Section": s["title"], "Type": s["kind"], "Source": s.get("source", ""),
                             "Created": s.get("created", "")})
        if s["kind"] == "table":
            sheets[f"{i}_{s['title']}"[:31]] = s["table"]
        elif s["kind"] == "metrics":
            sheets[f"{i}_{s['title']}"[:31]] = pd.DataFrame(s["metrics"], columns=["Metric", "Value"])
        elif s["kind"] == "figure":
            fig = section_figure(s)
            try:  # export the chart's underlying data
                frames = []
                for tr in fig.data:
                    d = {k: list(v) for k in ("x", "y", "z") if (v := getattr(tr, k, None)) is not None and hasattr(v, "__len__")}
                    if d:
                        n = max(len(v) for v in d.values())
                        d = {k: v + [None] * (n - len(v)) for k, v in d.items() if len(v) <= n}
                        f = pd.DataFrame(d)
                        f.insert(0, "trace", tr.name or "")
                        frames.append(f)
                if frames:
                    sheets[f"{i}_{s['title']}"[:31]] = pd.concat(frames, ignore_index=True).head(100000)
            except Exception:  # noqa: BLE001
                pass
    sheets = {"Report contents": pd.DataFrame(summary_rows), **sheets}
    return to_excel_bytes(sheets)
