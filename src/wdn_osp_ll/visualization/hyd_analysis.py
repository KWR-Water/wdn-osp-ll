"""Plots that summarise a :class:`WDNLeakAnalysis` and its leak signatures.

These mirror the exploratory figures used to sanity-check a hydraulic-analysis
config: the leak signature matrices (raw and node-scaled) across a few emitter
coefficients, and the distribution of the simulated leak magnitudes relative to
the network demand.
"""

import numpy as np
import matplotlib.pyplot as plt

from wdn_osp_ll.hyd_analysis.data_structures import WDNLeakAnalysis
from wdn_osp_ll.hyd_analysis.lsm_manipulation.operations import (
    scale_nodes_by_scenario_mean,
)


def _leak_magnitudes_by_junction(analysis: WDNLeakAnalysis) -> np.ndarray:
    """Reshape ``leak_magnitudes`` ``(1, n_scenarios)`` to ``(n_junctions,
    n_coefficients)`` so each row holds one junction's magnitudes across the
    emitter coefficients. Relies on the scenario ordering (junction slowest,
    coefficient fastest) used when the scenarios are simulated."""
    n_junctions = analysis.n_junctions
    n_coefficients = len(analysis.emitter_coefficients)
    return analysis.leak_magnitudes.reshape(n_junctions, n_coefficients)


def plot_lsms(
    analysis: WDNLeakAnalysis,
    emitter_coefficients: np.ndarray | None = None,
) -> None:
    """Show the leak signature matrices (Sensitivity Matrix and Chi-squared), 
    raw and node-scaled, for a few emitter coefficients.

    One column per emitter coefficient; the top row is the raw sensitivity
    matrix and the bottom row is the same matrix scaled per node by its mean
    across scenarios. ``emitter_coefficients`` selects which coefficients to
    show (defaults to every other one available).
    """
    if emitter_coefficients is None:
        emitter_coefficients = analysis.emitter_coefficients[::2]

    fig, axes = plt.subplots(
        nrows=4,  # raw sensitivity matrix and its node-scaled version and same for chi-squared
        ncols=len(emitter_coefficients),
        figsize=(12, 10),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for col, k in enumerate(emitter_coefficients):
        scenarios = [(j, float(k)) for j in analysis.junctions]
        raw_sm = analysis.select_scenarios(scenarios).sensitivity_matrix
        scaled_sm = scale_nodes_by_scenario_mean(raw_sm)
        raw_chi = analysis.select_scenarios(scenarios).chi_squared_matrix
        scaled_chi = scale_nodes_by_scenario_mean(raw_chi)

        for row, (lsm, label_front, label_back) in enumerate([
                (raw_sm, "Sens. M.", ""),
                (scaled_sm, "Sens. M.", " - scaled"),
                (raw_chi, "Chi-sq. M.", ""),
                (scaled_chi, "Chi-sq. M.", " - scaled"),
        ]):
            ax = axes[row, col]
            ax.imshow(lsm.data, cmap="viridis", aspect="auto")
            ax.set_title(f"{label_front} - EC:{round(float(k), 3)}{label_back}")
            if row == 3:
                ax.set_xlabel("Leaky junction index")
            if col == 0:
                ax.set_ylabel("Node index")

    fig.tight_layout()
    plt.show()


def plot_leak_magnitudes(analysis: WDNLeakAnalysis) -> None:
    """Box-plot the absolute leak magnitudes per junction (one box per junction,
    over the emitter coefficients) against the nominal demands and a 2.5%-of-
    total-demand reference line."""
    magnitudes = _leak_magnitudes_by_junction(analysis)
    n_junctions = analysis.n_junctions
    total_demand = analysis.total_demand()
    junction_x = np.arange(1, n_junctions + 1)

    plt.figure(figsize=(5, 2))
    plt.boxplot(magnitudes.T)
    plt.scatter(x=junction_x, y=analysis.demands(), label="nominal demand")
    plt.hlines(
        y=0.025 * total_demand,
        xmin=1,
        xmax=n_junctions,
        color="red",
        label="2.5%",
    )
    plt.title("Leak magnitudes and nominal demands in absolute value")
    plt.xlabel("Junction index")
    plt.ylabel("Flowrate [L/s]")
    plt.legend()
    plt.show()

    print(f"total demand: {total_demand} L/s")
    print(f"total demand * 2.5%: {0.025 * total_demand} L/s")


def plot_relative_leak_magnitudes(analysis: WDNLeakAnalysis) -> None:
    """Box-plot the leak magnitudes per junction as a percentage of the total
    network demand, with a 2.5% reference line."""
    magnitudes = _leak_magnitudes_by_junction(analysis)
    n_junctions = analysis.n_junctions
    relative = 100 * magnitudes / analysis.total_demand()

    plt.figure(figsize=(5, 2))
    plt.boxplot(relative.T)
    plt.hlines(y=2.5, xmin=1, xmax=n_junctions, color="red", label="2.5%")
    plt.title("Leak magnitudes relative to nominal network demand")
    plt.xlabel("Junction index")
    plt.ylabel("Flowrate / total demand [%]")
    plt.legend()
    plt.show()
