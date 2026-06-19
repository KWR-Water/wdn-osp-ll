"""Visualizations of the final OSP result (Pareto front, parallel coords, HV bar)."""

from __future__ import annotations

from typing import Sequence

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..sensor_placement_optimization.results import OSPResult


# ---------------------------------------------------------------------------
# 1. Pareto front (2-objective scatter)
# ---------------------------------------------------------------------------

def plot_pareto_front(
    results: Sequence[OSPResult],
    labels: Sequence[str],
    colors: Sequence,
    *,
    x_obj: int = 0,
    y_obj: int = 1,
    ax: matplotlib.axes.Axes | None = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Scatter the final Pareto front for each result.

    Parameters
    ----------
    x_obj, y_obj:
        Indices into ``best_f`` columns (and ``config["objectives"]`` names)
        to use for the x- and y-axes respectively.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    for result, label, color in zip(results, labels, colors):
        f = result.best_f  # (n_pareto, n_obj) or (n_obj,)
        if f.ndim == 1:
            f = f[np.newaxis, :]
        ax.scatter(f[:, x_obj], f[:, y_obj], label=label, color=color, s=40, zorder=3)

    # Axis labels from first result's config
    first_cfg = next(iter(results)).config
    obj_names = first_cfg.get("objectives", [])
    ax.set_xlabel(obj_names[x_obj] if x_obj < len(obj_names) else f"Objective {x_obj}")
    ax.set_ylabel(obj_names[y_obj] if y_obj < len(obj_names) else f"Objective {y_obj}")
    ax.set_title("Final Pareto Front")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    if standalone:
        fig.tight_layout()  # type: ignore[union-attr]
    return fig, ax  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 2. Parallel coordinates of objectives
# ---------------------------------------------------------------------------

def plot_parallel_coordinates(
    results: Sequence[OSPResult],
    labels: Sequence[str],
    colors: Sequence,
    *,
    ax: matplotlib.axes.Axes | None = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Parallel-coordinates plot of normalised objective values.

    Each solution on the combined Pareto fronts becomes one polyline.
    Values are normalised per objective using the metric bounds from the
    config (``min_val`` = ideal, ``max_val`` = nadir), so every axis spans
    [0, 1] where 0 is best.

    Tick annotations at the bottom/top of each axis show the raw bounds.
    """
    # Build a combined dataframe
    rows: list[dict] = []
    obj_names: list[str] = []

    for result, label in zip(results, labels):
        f = result.best_f
        if f.ndim == 1:
            f = f[np.newaxis, :]
        if not obj_names:
            obj_names = list(result.config.get("objectives", [
                f"obj_{i}" for i in range(f.shape[1])
            ]))
        for sol_f in f:
            row: dict[str, object] = {name: float(v) for name, v in zip(obj_names, sol_f)}
            row["_label"] = label
            rows.append(row)

    df = pd.DataFrame(rows)

    # Compute per-column bounds for normalisation and annotation
    bounds: dict[str, tuple[float, float]] = {}
    for col in obj_names:
        col_min = df[col].min()
        col_max = df[col].max()
        span = col_max - col_min
        if span > 0:
            df[col] = (df[col] - col_min) / span
        else:
            df[col] = 0.0
        bounds[col] = (col_min, col_max)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    color_map = {label: color for label, color in zip(labels, colors)}
    label_colors = [color_map[lbl] for lbl in df["_label"]]

    pd.plotting.parallel_coordinates(
        frame=df,
        class_column="_label",
        cols=obj_names,
        color=label_colors,
        ax=ax,
        alpha=0.5,
    )

    # Annotate raw bounds at bottom and top of each axis
    for i, col in enumerate(obj_names):
        lo, hi = bounds[col]
        ax.text(i, -0.04, f"{lo:.2g}", ha="center", va="top",
                fontsize=7, transform=ax.get_xaxis_transform())
        ax.text(i, 1.04, f"{hi:.2g}", ha="center", va="bottom",
                fontsize=7, transform=ax.get_xaxis_transform())

    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Parallel Coordinates — Objectives (normalised)")
    ax.grid(True, linestyle="--", alpha=0.4)

    if standalone:
        fig.tight_layout()  # type: ignore[union-attr]
    return fig, ax  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 3. Final hypervolume bar chart
# ---------------------------------------------------------------------------

def plot_hypervolume_bar(
    results: Sequence[OSPResult],
    labels: Sequence[str],
    colors: Sequence,
    *,
    ax: matplotlib.axes.Axes | None = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Bar chart of the best hypervolume achieved (max over islands, last gen).

    Each bar represents the peak HV across all islands at the final recorded
    generation.  The individual island values are shown as a strip on top
    to reveal the spread.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    x_pos = np.arange(len(labels))
    bar_heights: list[float] = []
    island_hvs: list[np.ndarray] = []

    for result in results:
        if result.history.hypervolumes is None:
            bar_heights.append(0.0)
            island_hvs.append(np.array([0.0]))
            continue
        final = result.history.hypervolumes[-1, :]
        bar_heights.append(float(final.max()))
        island_hvs.append(final)

    ax.bar(x_pos, bar_heights, color=colors, alpha=0.7)

    # Scatter individual island values
    for i, hvs in enumerate(island_hvs):
        ax.scatter(
            np.full(len(hvs), x_pos[i]),
            hvs,
            color=colors[i],
            edgecolors="black",
            linewidths=0.6,
            s=30,
            zorder=4,
        )

    for i, h in enumerate(bar_heights):
        ax.text(x_pos[i], h, f"{h:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Hypervolume")
    ax.set_title("Final Hypervolume (max over islands)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    if standalone:
        fig.tight_layout()  # type: ignore[union-attr]
    return fig, ax  # type: ignore[return-value]
