"""
src/visualization/segment_plots.py
────────────────────────────────────
Reusable chart functions for segment profiling and comparison.

Functions:
  - plot_segment_radar()       — radar chart for one or all segments
  - plot_umap_scatter()        — 2D UMAP / t-SNE scatter coloured by segment
  - plot_segment_heatmap()     — feature z-score heatmap across segments
  - plot_temporal_patterns()   — time-of-day usage bar charts
  - plot_poi_spider()          — POI affinity spider chart
  - plot_segment_size_bars()   — horizontal bar chart of segment sizes
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.rcParams["figure.dpi"] = 120
sns.set_theme(style="whitegrid")

SEGMENT_COLORS = [
    "#e74c3c", "#3498db", "#9b59b6", "#2ecc71",
    "#f39c12", "#95a5a6", "#1abc9c", "#e67e22",
]


def _get_color(seg_id: int) -> str:
    return SEGMENT_COLORS[seg_id % len(SEGMENT_COLORS)]


def plot_segment_radar(
    df: pd.DataFrame,
    segment_id: int,
    segment_name: str,
    feature_map: Optional[dict] = None,
    figsize: tuple = (6, 6),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Radar chart comparing one segment to population average."""
    if feature_map is None:
        feature_map = {
            "Mobility":    "avg_daily_distance_km",
            "Data":        "data_usage_gb",
            "Evening":     "evening_data_ratio",
            "Weekend":     "weekend_data_ratio",
            "Work POI":    "poi_office_score",
            "Leisure POI": "poi_mall_score",
            "Roaming":     "roaming_days_monthly",
            "Voice":       "call_minutes_monthly",
        }

    available = {k: v for k, v in feature_map.items() if v in df.columns}
    categories = list(available.keys())
    N = len(categories)
    if N < 3:
        return None

    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

    pop_mean = df[[v for v in available.values()]].mean()
    pop_std  = df[[v for v in available.values()]].std().replace(0, 1)
    seg_data = df[df["segment_id"] == segment_id]
    seg_mean = seg_data[[v for v in available.values()]].mean()

    def norm(val, mean, std):
        return float(np.clip((val - (mean - 2*std)) / (4*std + 1e-6), 0, 1))

    values     = [norm(seg_mean[v], pop_mean[v], pop_std[v]) for v in available.values()] + [0]
    values[-1] = values[0]
    pop_vals   = [0.5] * N + [0.5]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    color = _get_color(segment_id)

    ax.plot(angles, values, color=color, linewidth=2.5)
    ax.fill(angles, values, color=color, alpha=0.25)
    ax.plot(angles, pop_vals, color="#aaaaaa", linewidth=1.5, linestyle="--", label="Pop avg")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.set_title(f"{segment_name}\n(n={len(seg_data):,})", size=11, pad=16, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def plot_umap_scatter(
    X_2d: np.ndarray,
    labels: np.ndarray,
    segment_names: Optional[dict] = None,
    title: str = "Segment UMAP Projection",
    figsize: tuple = (10, 7),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Scatter plot of 2D UMAP projection coloured by segment."""
    fig, ax = plt.subplots(figsize=figsize)
    unique_labels = sorted(set(labels))

    for seg_id in unique_labels:
        mask = labels == seg_id
        color = "#cccccc" if seg_id < 0 else _get_color(seg_id)
        name = segment_names.get(seg_id, f"Seg {seg_id}") if segment_names else f"Seg {seg_id}"
        label_str = "Noise" if seg_id < 0 else f"[{seg_id}] {name}"
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=color, s=5, alpha=0.5, label=label_str, linewidths=0,
        )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP Dim 1")
    ax.set_ylabel("UMAP Dim 2")
    ax.legend(markerscale=3, fontsize=8, loc="best",
              bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def plot_segment_heatmap(
    df: pd.DataFrame,
    feature_cols: list,
    segment_col: str = "segment_id",
    segment_names: Optional[dict] = None,
    title: str = "Segment Feature Profiles (Z-scores)",
    figsize: tuple = (14, 7),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Z-score heatmap showing how each segment differs from the population."""
    from sklearn.preprocessing import StandardScaler

    avail = [c for c in feature_cols if c in df.columns]
    seg_matrix = df.groupby(segment_col)[avail].mean()

    if segment_names:
        seg_matrix.index = [segment_names.get(i, f"Seg {i}") for i in seg_matrix.index]

    seg_matrix_z = pd.DataFrame(
        StandardScaler().fit_transform(seg_matrix),
        index=seg_matrix.index,
        columns=avail,
    )

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        seg_matrix_z,
        annot=True, fmt=".1f",
        cmap="RdYlGn", center=0,
        ax=ax, linewidths=0.5,
        annot_kws={"size": 8},
        cbar_kws={"label": "Z-score vs population"},
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def plot_temporal_patterns(
    df: pd.DataFrame,
    segment_col: str = "segment_id",
    segment_names: Optional[dict] = None,
    figsize: tuple = (12, 5),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Bar chart of data usage by time-of-day for each segment."""
    time_cols = {
        "Morning\n06-12": "morning_data_ratio",
        "Afternoon\n12-18": "afternoon_data_ratio",
        "Evening\n18-22": "evening_data_ratio",
        "Night\n22-06": "night_data_ratio",
    }
    available_time = {k: v for k, v in time_cols.items() if v in df.columns}
    if not available_time:
        return None

    seg_time = df.groupby(segment_col)[[v for v in available_time.values()]].mean()
    if segment_names:
        seg_time.index = [segment_names.get(i, f"Seg {i}") for i in seg_time.index]
    seg_time.columns = list(available_time.keys())

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(seg_time))
    width = 0.2
    for i, (col, color) in enumerate(zip(seg_time.columns, ["#f39c12", "#3498db", "#e74c3c", "#9b59b6"])):
        ax.bar(x + i * width, seg_time[col], width, label=col,
               color=color, alpha=0.85, edgecolor="white")

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(seg_time.index, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Avg Data Usage Ratio")
    ax.set_title("Data Usage by Time of Day per Segment")
    ax.legend(fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def plot_segment_size_bars(
    df: pd.DataFrame,
    segment_col: str = "segment_id",
    segment_names: Optional[dict] = None,
    value_col: Optional[str] = None,
    title: str = "Subscribers per Segment",
    figsize: tuple = (9, 5),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Horizontal bar chart of segment sizes or any aggregate metric."""
    if value_col:
        seg_vals = df.groupby(segment_col)[value_col].mean().sort_values()
    else:
        seg_vals = df[segment_col].value_counts().sort_values()

    if segment_names:
        seg_vals.index = [segment_names.get(i, f"Seg {i}") for i in seg_vals.index]

    fig, ax = plt.subplots(figsize=figsize)
    colors = [_get_color(i) for i in range(len(seg_vals))]
    ax.barh(seg_vals.index, seg_vals.values, color=colors, edgecolor="white", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(value_col or "Count")
    for i, v in enumerate(seg_vals.values):
        ax.text(v + max(seg_vals) * 0.01, i, f"{v:,.0f}", va="center", fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig
