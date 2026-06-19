"""Visualizations of the per-generation optimisation history (OSPRunHistory)."""

from __future__ import annotations

from typing import Sequence

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt

from ..sensor_placement_optimization.results import OSPResult


def plot_hypervolume_trajectory(
    results: Sequence[OSPResult],
    labels: Sequence[str],
    colors: Sequence,
    *,
    ax: matplotlib.axes.Axes | None = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Hypervolume per island across generations, one line style per result.

    Each island is drawn with the same color but reduced opacity so the
    envelope of the archipelago is visible.  Only the first island's line
    gets a legend entry.

    Returns
    -------
    fig, ax
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    for result, label, color in zip(results, labels, colors):
        h = result.history
        if h.hypervolumes is None:
            continue
        n_islands = h.hypervolumes.shape[1]
        for i in range(n_islands):
            ax.plot(
                h.generations,
                h.hypervolumes[:, i],
                color=color,
                linewidth=1.8,
                alpha=0.45,
                label=label if i == 0 else "",
            )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume")
    ax.set_title("Hypervolume Trajectory")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    if standalone:
        fig.tight_layout()  # type: ignore[union-attr]
    return fig, ax  # type: ignore[return-value]


def plot_fitness_trajectory(
    results: Sequence[OSPResult],
    labels: Sequence[str],
    colors: Sequence,
    *,
    obj_idx: int = 0,
    ax: matplotlib.axes.Axes | None = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Mean population fitness for one objective across generations.

    Uses ``history.fitness`` (shape: n_steps × n_islands × pop_size × n_obj).
    The mean is taken over the population for each island; each island gets its
    own line.

    Parameters
    ----------
    obj_idx:
        Which objective column to plot (0-based).
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    for result, label, color in zip(results, labels, colors):
        h = result.history
        # fitness: (n_steps, n_islands, pop_size, n_obj)
        n_islands = h.fitness.shape[1]
        for i in range(n_islands):
            mean_f = h.fitness[:, i, :, obj_idx].mean(axis=1)
            ax.plot(
                h.generations,
                mean_f,
                color=color,
                linewidth=1.8,
                alpha=0.45,
                label=label if i == 0 else "",
            )

    obj_names = []
    for result in results:
        objs = result.config.get("objectives", [])
        if 0 <= obj_idx < len(objs):
            obj_names.append(objs[obj_idx])
    obj_label = obj_names[0] if obj_names else f"Objective {obj_idx}"

    ax.set_xlabel("Generation")
    ax.set_ylabel(f"Mean {obj_label}")
    ax.set_title(f"Fitness Trajectory — {obj_label}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    if standalone:
        fig.tight_layout()  # type: ignore[union-attr]
    return fig, ax  # type: ignore[return-value]
