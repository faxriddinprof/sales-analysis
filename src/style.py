"""Barcha grafiklar uchun yagona, sodda (ilmiy maqola uslubidagi) ko'rinish."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from .data import ROOT

FIG_DIR = ROOT / "reports" / "figures"

# Ranglar: bir xil ma'no har doim bir xil rangda
BLUE = "#2f5d8a"      # asosiy seriya / guruch
ORANGE = "#d9822b"    # ikkinchi seriya / tushum
GREY = "#8c8c8c"      # yordamchi chiziqlar, baseline
RED = "#b23a3a"       # salbiy ta'sir, xato
GREEN = "#3a7d44"     # ijobiy ta'sir
YEAR_COLORS = {2023: "#7a9cc6", 2024: "#2f5d8a", 2025: "#d9822b"}
SEQ_SCALE = "Blues"
DIV_SCALE = "RdBu"

_template = go.layout.Template(
    layout=dict(
        font=dict(family="Helvetica, Arial, sans-serif", size=13, color="#222"),
        title=dict(font=dict(size=15), x=0, xanchor="left"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=[BLUE, ORANGE, GREEN, RED, GREY, "#6b4c9a"],
        xaxis=dict(showline=True, linecolor="#444", ticks="outside", showgrid=False, zeroline=False),
        yaxis=dict(showline=True, linecolor="#444", ticks="outside", showgrid=True,
                   gridcolor="#ececec", zeroline=False),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", borderwidth=0),
        margin=dict(l=60, r=20, t=60, b=50),
        hoverlabel=dict(bgcolor="white"),
    )
)
pio.templates["paper"] = _template
pio.templates.default = "paper"


def setup_notebook() -> None:
    """Notebookda grafiklar VS Code va Jupyter da ko'rinishi uchun."""
    pio.renderers.default = "plotly_mimetype+notebook_connected"


def save(fig: go.Figure, name: str) -> go.Figure:
    """Grafikni dashboard uchun JSON ko'rinishida saqlaydi va qaytaradi."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_json(FIG_DIR / f"{name}.json")
    return fig


def fmt_mln(x: float) -> str:
    """12_345_678 → '12.3 mln'."""
    return f"{x / 1e6:,.1f} mln"
