from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


REPRO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = REPRO_DIR / "data"
OUTPUT_DIR = REPRO_DIR / "output"

PANEL_WIDTH_MM = 60
COMPOSITE_WIDTH_MM = 120

FIGURE_SIZES_MM = {
    "spatial_alignment": (100, 125),
    "travel_time_comparison": (PANEL_WIDTH_MM, PANEL_WIDTH_MM),
    "absolute_travel_time_difference": (PANEL_WIDTH_MM, 30),
    "relative_travel_time_difference": (PANEL_WIDTH_MM, 30),
    "spatial_temporal_map": (PANEL_WIDTH_MM, 45),
    "spatial_temporal_legend": (30, 30),
    "density_null_alignment": (PANEL_WIDTH_MM, 47.5),
    "density_null_metric": (30, 47.5),
    "non_alignment_summary": (COMPOSITE_WIDTH_MM, 45),
    "non_alignment_importance": (45, 38),
    "non_alignment_dependence": (50, 42.5),
    "non_alignment_colourbar": (12, 42.5),
    "amenity_distance_differentials": (COMPOSITE_WIDTH_MM, 96),
}

EXPORT_PROFILES = {
    "draft": {"dpi": 300, "formats": ("pdf",)},
    "final": {"dpi": 300, "formats": ("png", "pdf")},
}

SAVE_OUTPUTS = True

SYMBOL_FONT = FontProperties(family="Symbol")


def mm_to_inches(value_mm: float) -> float:
    return value_mm / 25.4


def figure_size(name: str) -> tuple[float, float]:
    width_mm, height_mm = FIGURE_SIZES_MM[name]
    return mm_to_inches(width_mm), mm_to_inches(height_mm)


def configure_style(profile: str = "draft") -> None:
    if profile not in EXPORT_PROFILES:
        raise ValueError(f"Unknown export profile: {profile}")

    dpi = EXPORT_PROFILES[profile]["dpi"]
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.titlesize": 7,
            "axes.labelsize": 6,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "legend.fontsize": 5.5,
            "legend.title_fontsize": 5.5,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 0.9,
            "patch.linewidth": 0.55,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Symbol",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.sf": "Arial",
            "mathtext.fallback": "stix",
            "axes.unicode_minus": False,
        }
    )

    # Crop notebook-rendered PNGs to their visible artists while retaining a
    # minimal safety margin for antialiasing and glyph overhang.
    try:
        from matplotlib_inline.backend_inline import InlineBackend

        InlineBackend.instance().print_figure_kwargs = {
            "bbox_inches": "tight",
            "pad_inches": 0.01,
        }
    except ImportError:
        pass


def add_panel_label(ax, label: str, x: float = -0.10, y: float = 1.03) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def save_figure(fig, stem: str, profile: str = "draft") -> list[Path]:
    if profile not in EXPORT_PROFILES:
        raise ValueError(f"Unknown export profile: {profile}")

    if not SAVE_OUTPUTS:
        print(f"disk export disabled: {stem}")
        return []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    settings = EXPORT_PROFILES[profile]
    for fmt in settings["formats"]:
        path = OUTPUT_DIR / f"{stem}.{fmt}"
        kwargs = {
            "facecolor": "white",
            "bbox_inches": "tight",
            "pad_inches": 0.01,
        }
        if fmt == "png":
            kwargs["dpi"] = settings["dpi"]
        fig.savefig(path, format=fmt, **kwargs)
        saved.append(path)
    return saved


def finish_axes(ax, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#d9d9d9", linewidth=0.45, alpha=0.6)
        ax.set_axisbelow(True)


def close_and_report(fig, paths: list[Path]) -> None:
    width_px = round(fig.get_figwidth() * fig.dpi)
    height_px = round(fig.get_figheight() * fig.dpi)
    print(f"rendered pixels: {width_px} x {height_px}")
    for path in paths:
        print(f"saved: {path}")
    plt.show()
