"""OSP optimisation runner.

Entry point: ``load_or_compute(config)`` — returns a cached ``OSPResult`` if
one exists for that config, otherwise runs the archipelago and caches the
result.

Config shape
------------
::

    problem:                       # forwarded to DDLeakLocOSPProblem.from_config
        leak_analysis: ...
        dd_leak_localization: ...
        metrics: [...]
        objectives: [...]

    algorithm:
        n_islands: 5               # default 5
        population_size: 100       # default 100
        generations: 500           # total generations to evolve
        report_every: 10           # print a progress line every N generations
        # name: nsga2              # optional override; inferred from n_obj by default
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pygmo

from .problems.integrated_dd_leak_localization import DDLeakLocOSPProblem
from .results import OSPResult, OSPRunHistory


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hypervolume(fv: np.ndarray, ref_point: np.ndarray | list[float]) -> float:
    """Hypervolume of ``fv`` strictly dominated by ``ref_point``."""
    fv = fv[np.all(fv < ref_point, axis=1)]
    if fv.shape[0] < 2:
        return 0.0
    return float(pygmo.hypervolume(fv).compute(ref_point))


def _filtered_hypervolume(pop: Any, ref_point: np.ndarray) -> float:
    """Hypervolume of the non-dominated front strictly dominated by ref_point."""
    fv = pop.get_f()
    ndf, *_ = pygmo.fast_non_dominated_sorting(fv)
    return _hypervolume(fv[ndf[0]], ref_point)


def _make_algorithm(name: str, rep_gen: int) -> pygmo.algorithm:
    if name == "sga":
        return pygmo.algorithm(pygmo.sga(gen=rep_gen))
    if name == "nsga2":
        return pygmo.algorithm(pygmo.nsga2(gen=rep_gen))
    raise ValueError(f"Unknown algorithm name: {name!r}. Use 'sga' or 'nsga2'.")


def _print_progress(gen: int, tracked: str, best: float, median: float) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] generation {gen:>6d} | best {tracked}: {best:.6f} | median: {median:.6f}")


# ---------------------------------------------------------------------------
# Core run
# ---------------------------------------------------------------------------

def run_optimisation(config: dict) -> OSPResult:
    """Build the problem from ``config``, evolve an archipelago, return the result.

    This is the raw compute path — no caching.  Prefer ``load_or_compute``.
    """
    prob_cfg = config["problem"]
    alg_cfg = config.get("algorithm", {})

    n_islands: int = alg_cfg.get("n_islands", 5)
    pop_size: int = alg_cfg.get("population_size", 100)
    generations: int = alg_cfg.get("generations", 500)
    rep_gen: int = alg_cfg.get("report_every", 10)

    # Build the UDP (user-defined problem)
    problem = DDLeakLocOSPProblem.from_config(prob_cfg)
    pyprob = pygmo.problem(problem)
    n_obj = problem.get_nobj()
    is_multi = n_obj > 1

    algo_name: str = alg_cfg.get("name", "nsga2" if is_multi else "sga")
    pyalgo = _make_algorithm(algo_name, rep_gen)

    ref_point = problem.get_nadir_point() if is_multi else None

    # Reduced reference point: cap the n_sensors dimension at 25% of the network.
    # Computed here so the _snapshot closure can capture it.
    reduced_ref: np.ndarray | None = None
    _ns_idx: int | None = None
    _x_max: int | None = None
    if is_multi and "n_sensors" in problem.objectives:
        _ns_idx = problem.objectives.index("n_sensors")
        _x_max = min(problem.sens_u_limit, problem.leak_analysis.n_nodes // 4)
        reduced_ref = np.array(ref_point, dtype=float)
        reduced_ref[_ns_idx] = float(_x_max + 1)

    # Build the archipelago
    archi = pygmo.archipelago(
        n=n_islands, algo=pyalgo, prob=problem, pop_size=pop_size
    )

    # Allocate history arrays
    n_steps = generations // rep_gen + 1
    gen_values = np.full(n_steps, np.nan)
    fitness_hist = np.full((n_steps, n_islands, pop_size, n_obj), np.nan)
    decision_hist = np.full((n_steps, n_islands, pop_size, pyprob.get_nx()), np.nan)
    hv_hist = np.full((n_steps, n_islands), np.nan) if is_multi else None
    reduced_hv_hist = np.full((n_steps, n_islands), np.nan) if reduced_ref is not None else None

    tracked = "hypervolume" if is_multi else "fitness"

    def _snapshot(step: int) -> tuple[float, float]:
        gen_values[step] = step * rep_gen
        fitness_hist[step] = np.array([isl.get_population().get_f() for isl in archi])
        decision_hist[step] = np.array([isl.get_population().get_x() for isl in archi])

        if is_multi:
            hvs = [_filtered_hypervolume(isl.get_population(), ref_point) for isl in archi]
            assert hv_hist is not None
            hv_hist[step] = hvs
            if reduced_ref is not None:
                assert reduced_hv_hist is not None
                reduced_hv_hist[step] = [
                    _filtered_hypervolume(isl.get_population(), reduced_ref) for isl in archi
                ]
            return float(np.max(hvs)), float(np.median(hvs))
        else:
            champ_fs = [champ[0] for champ in archi.get_champions_f()]
            return float(np.min(champ_fs)), float(np.median(champ_fs))

    # Step 0: initial population
    print("Starting optimisation...")
    best_v, median_v = _snapshot(0)
    _print_progress(0, tracked, best_v, median_v)

    # Evolve
    for step, gen_start in enumerate(range(0, generations, rep_gen), start=1):
        archi.evolve()
        archi.wait()
        gen_values[step] = gen_start + rep_gen
        best_v, median_v = _snapshot(step)
        _print_progress(gen_start + rep_gen, tracked, best_v, median_v)

    print("Optimisation complete.")

    # Extract the best solution(s)
    if is_multi:
        all_f = np.vstack([isl.get_population().get_f() for isl in archi])
        all_x = np.vstack([isl.get_population().get_x() for isl in archi])
        ndf, *_ = pygmo.fast_non_dominated_sorting(all_f)
        best_x = all_x[ndf[0]]
        best_f = all_f[ndf[0]]
        assert ref_point is not None
        final_hv = _hypervolume(best_f, ref_point)

        reduced_hv: float | None = None
        resolution_hv: float | None = None
        if reduced_ref is not None:
            reduced_hv = _hypervolume(best_f, reduced_ref)
            if "mean_penalized_search_area" in problem.objectives:
                assert _ns_idx is not None and _x_max is not None
                msa_idx = problem.objectives.index("mean_penalized_search_area")
                ns_range = np.arange(problem.sens_l_limit, _x_max + 1, dtype=float)
                res_f = np.zeros((len(ns_range), n_obj))
                res_f[:, _ns_idx] = ns_range
                res_f[:, msa_idx] = problem.leak_analysis.n_junctions / ns_range
                # all other objectives stay at 0.0 (ideal) so points are dominated by ref
                resolution_hv = _hypervolume(res_f, reduced_ref)
    else:
        champ_xs = archi.get_champions_x()
        champ_fs = archi.get_champions_f()
        best_idx = int(np.argmin([f[0] for f in champ_fs]))
        best_x = champ_xs[best_idx]
        best_f = champ_fs[best_idx]
        final_hv = None
        reduced_hv = None
        resolution_hv = None

    history = OSPRunHistory(
        generations=gen_values,
        fitness=fitness_hist,
        decisions=decision_hist,
        hypervolumes=hv_hist,
        reduced_hypervolumes=reduced_hv_hist,
    )

    config_hash = OSPResult.hash_config(config)
    return OSPResult(
        best_x=best_x,
        best_f=best_f,
        is_pareto=is_multi,
        config=config,
        config_hash=config_hash,
        history=history,
        final_hypervolume=final_hv,
        reduced_hypervolume=reduced_hv,
        resolution_hypervolume=resolution_hv,
    )


# ---------------------------------------------------------------------------
# Cached entry point
# ---------------------------------------------------------------------------

def load_or_compute(config: dict, *, force_recompute: bool = False) -> OSPResult:
    """Return a cached ``OSPResult`` for ``config``, or run and cache it.

    Parameters
    ----------
    config:
        Full optimisation config dict (``problem`` + ``algorithm`` subtrees).
    force_recompute:
        Ignore any existing cache and re-run the optimisation.
    """
    return OSPResult.load_or_compute(
        config,
        compute=lambda: run_optimisation(config),
        force_recompute=force_recompute,
    )
