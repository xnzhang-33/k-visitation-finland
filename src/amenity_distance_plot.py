"""Scatterplots of class-level absolute and proportional distance differences."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from adjustText import adjust_text

from distance_differentials import POI_MARKER_AREAS


# Retain the original colours for the ten plotted categories.
PALETTE = {
    name: mpl.colormaps["tab20"](index)
    for name, index in {
        "Transport": 0,
        "Culture": 1,
        "Retail": 2,
        "Service": 3,
        "Groceries": 4,
        "Dining": 6,
        "Civic & Religion": 7,
        "Healthcare": 8,
        "Education": 9,
        "Fitness": 10,
    }.items()
}
LABEL_CLASSES = {
    "Sports Equipment", "Cafe", "Grocery", "Fitness Center", "Hospital", "Clinic",
    "Furniture", "Cinema", "Amusement", "Auto Service", "Library", "Cultural Center",
    "Shopping Center", "School Elem./Secd.", "Auto/Home Supply", "Appliances Store",
    "Shoes", "Bus Stop", "Home Decor", "Daycare", "Restaurant", "Surgical Hospital",
    "Specific Sports", "Meat/Fish Market", "Casino", "Concert Hall", "Nightlife",
    "Home Services", "Beauty Salon", "Airport", "Jewelry Repair",
}


def plot_distances(data, metric, ylabel):
    plot_data = data.dropna(subset=[metric]).rename(columns={"size_bin": "Number of POIs"})
    present_categories = set(plot_data.category)
    categories = [name for name in PALETTE if name in present_categories]
    fig, ax = plt.subplots(figsize=(160 / 25.4, 128 / 25.4))
    fig.subplots_adjust(left=.12, right=.76, bottom=.13, top=.96)
    sns.scatterplot(
        ax=ax, data=plot_data, x="d_prox", y=metric,
        size="Number of POIs", size_order=list(POI_MARKER_AREAS), sizes=POI_MARKER_AREAS,
        hue="category", hue_order=categories, palette=PALETTE,
        alpha=.9, edgecolor="white", linewidth=.55, legend="brief",
    )
    ax.set_xscale("log")
    ax.set_xlim(70, 15_000)
    ax.set_xticks([100, 1_000, 5_000, 10_000], ["100", "1,000", "5,000", "10,000"])
    ax.axhline(plot_data[metric].mean(), color="#8A8F98", linestyle="--", linewidth=.65, alpha=.8)
    ax.axvline(plot_data.d_prox.mean(), color="#8A8F98", linestyle="--", linewidth=.65, alpha=.8)
    ax.set(xlabel=r"$d_{prox}$ (m)", ylabel=ylabel)

    texts = [
        ax.text(row.d_prox, getattr(row, metric), row.original_class, fontsize=6.5, alpha=.82)
        for row in plot_data.itertuples()
        if row.original_class in LABEL_CLASSES
    ]
    np.random.seed(42)
    adjust_text(
        texts, ax=ax, iter_lim=1000,
        arrowprops={"arrowstyle": "-", "color": "#999999", "alpha": .3, "linewidth": .35},
    )
    sns.move_legend(
        ax, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False,
        title=None, borderaxespad=0,
    )
    return fig
