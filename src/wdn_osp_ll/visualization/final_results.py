"""Visualizations of the final OSP result (Pareto front, parallel coords, HV bar)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from ..sensor_placement_optimization.problems.integrated_dd_leak_localization import DDLeakLocOSPProblem
from ..sensor_placement_optimization.results import OSPResult


# ---------------------------------------------------------------------------
# 1a. n_sensors vs MSA Pareto front (domain-specific)
# ---------------------------------------------------------------------------

def plot_pareto_front_nsensors_msa(
    results: Sequence[OSPResult],
    labels: Sequence[str],
    colors: Sequence,
    *,
    ax: matplotlib.axes.Axes | None = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Pareto front with n_sensors on x and mean_penalized_search_area on y.

    Reference lines
    ---------------
    - Perfect leak localization: horizontal at MSA = 1 (one-node search area).
    - Resolution:  y = n_nodes / n_sensors — the expected search area if sensors
      perfectly partition the network.  Expressed as "1/n_sensors" of the network
      and visible as such in the y-axis percentage annotations.

    X-axis is capped at ``min(sens_u_limit, n_nodes // 4)`` (25 % of the network),
    which also serves as the n_sensors nadir for hypervolume calculations.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 6))
    else:
        fig = ax.get_figure()

    # Locate objective columns in best_f
    first_cfg = next(iter(results)).config

    problem = DDLeakLocOSPProblem.from_config(config=first_cfg["problem"])
    n_nodes = problem.leak_analysis.n_nodes
    n_junctions = problem.leak_analysis.n_junctions
    x_max = min(problem.sens_u_limit, n_nodes // 4)
    x_min = problem.sens_l_limit

    
    obj_names = problem.objectives
    ns_idx = obj_names.index("n_sensors")
    msa_idx = obj_names.index("mean_penalized_search_area")

    # --- scatter Pareto fronts ---
    all_msa: list[float] = []
    for result, label, color in zip(results, labels, colors):
        f = result.best_f
        if f.ndim == 1:
            f = f[np.newaxis, :]
        mask = f[:, ns_idx] <= x_max
        pts = f[mask]
        ax.scatter(pts[:, ns_idx], pts[:, msa_idx],
                   label=label, color=color, s=40, zorder=3)
        all_msa.extend(pts[:, msa_idx].tolist())

    # --- reference lines ---
    x_ref = np.linspace(x_min, x_max, 300)

    ax.axhline(1.0, color="green", linestyle="--", linewidth=1.5,
               label="Perfect leak localization (MSA = 1)", zorder=2)
    ax.plot(x_ref, n_junctions / x_ref, color="gray", linestyle=":", linewidth=1.5,
            label=r"Resolution  (MSA = $n_{junctions}$ / $n_{sensors}$)", zorder=2)

    # --- axis limits ---
    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    y_data_max = max(all_msa) if all_msa else float(n_nodes)
    ax.set_ylim(0.5, y_data_max * 1.1)

    # --- x ticks: one per integer sensor count, labelled with network coverage ---
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{int(v)}\n({100 * v / n_junctions:.0f}%)" if v == int(v) else ""
    ))
    ax.set_xlabel("Number of Sensors\n(% of sensor junctions)")

    # --- y ticks: sensible integers, labelled with fraction of network ---
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8, min_n_ticks=4))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{int(v)}\n({100 * v / n_nodes:.0f}%)" if v >= 1 else ""
    ))
    ax.set_ylabel("Mean Penalized Search Area\n(% of network nodes)")

    ax.set_title("Pareto Front: Number of Sensors vs. Search Area")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    if standalone:
        fig.tight_layout()  # type: ignore[union-attr]
    return fig, ax  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 1b. Pareto front (2-objective scatter, generic)
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
    obj_names = first_cfg["problem"].get("objectives", [])
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
    """Bar chart of the final hypervolume of the combined Pareto front.

    Each bar represents the hypervolume of the merged Pareto front across all
    islands (``OSPResult.final_hypervolume``).  The individual island values
    at the last generation are shown as a strip to reveal the spread.
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
        bar_heights.append(result.final_hypervolume or 0.0)
        if result.history.hypervolumes is not None:
            island_hvs.append(result.history.hypervolumes[-1, :])
        else:
            island_hvs.append(np.array([0.0]))

    ax.bar(x_pos, bar_heights, color=colors, alpha=0.7)

    # Scatter individual island values as context
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
    ax.set_title("Final Hypervolume (combined Pareto front)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    if standalone:
        fig.tight_layout()  # type: ignore[union-attr]
    return fig, ax  # type: ignore[return-value]
