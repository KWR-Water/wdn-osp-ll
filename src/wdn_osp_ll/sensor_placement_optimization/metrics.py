"""OSP metric definitions and registry.

A ``Metric`` is a named scalar with theoretical bounds and an evaluation
function.  Metrics are registered by name; at problem construction time the
string list from the config is resolved to concrete ``Metric`` instances via
``resolve_metric(name, problem)`` — which lets problem-dependent bounds (e.g.
the range of valid sensor counts) be computed once and reused.

Adding a new metric
-------------------
::

    @register_metric("my_metric")
    def _(problem: "DDLeakLocOSPProblem") -> Metric:
        def fn(ctx: MetricContext) -> float:
            return ...
        return Metric("my_metric", fn, min_val=0.0, max_val=1.0)

``min_val`` is the theoretical best (ideal point component).
``max_val`` is the theoretical worst (nadir point component).
Since pygmo minimises, ``get_nadir_point()`` returns ``[m.max_val ...]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from .problems.integrated_dd_leak_localization import DDLeakLocOSPProblem


# ---------------------------------------------------------------------------
# Context passed to every metric function
# ---------------------------------------------------------------------------

@dataclass
class MetricContext:
    """All intermediate results produced during one fitness evaluation.

    ``lsm_observed`` is the raw ``(n_nodes, n_scenarios)`` array returned by
    ``observe()``: sensor rows contain the signature values, all other rows are
    NaN.  ``search_area`` is ``(n_nodes, n_scenarios)`` bool — ``True`` means
    the node is included in the search area for that scenario.

    ``problem`` gives metric functions access to the full problem state
    (``leak_analysis``, ``lsm``, sensor limits, possible locations, etc.).
    """

    sensors_pyidxs: np.ndarray          # (n_sensors,) selected sensor py-indices
    lsm_observed: np.ndarray            # (n_nodes, n_scenarios), NaN at non-sensor rows
    search_area: np.ndarray             # (n_nodes, n_scenarios) bool
    problem: "DDLeakLocOSPProblem"


# ---------------------------------------------------------------------------
# Metric descriptor
# ---------------------------------------------------------------------------

@dataclass
class Metric:
    """Named scalar with evaluation function and theoretical bounds.

    ``min_val`` / ``max_val`` define the range of the metric:
    - ``min_val``: theoretical best value (ideal point component).
    - ``max_val``: theoretical worst value (nadir point component).

    Since pygmo minimises every objective, the nadir point for an objective
    is simply ``max_val`` of its metric.  Normalisation to ``[0, 1]`` is
    ``(value - min_val) / (max_val - min_val)``.
    """

    name: str
    fn: Callable[[MetricContext], float]
    min_val: float
    max_val: float

    def __call__(self, ctx: MetricContext) -> float:
        return self.fn(ctx)

    def normalize(self, value: float) -> float:
        span = self.max_val - self.min_val
        return (value - self.min_val) / span if span > 0 else 0.0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# name  →  factory(problem) → Metric
_REGISTRY: dict[str, Callable[["DDLeakLocOSPProblem"], Metric]] = {}


def register_metric(name: str):
    """Decorator that registers a metric factory under ``name``."""
    def decorator(factory: Callable[["DDLeakLocOSPProblem"], Metric]):
        _REGISTRY[name] = factory
        return factory
    return decorator


def resolve_metric(name: str, problem: "DDLeakLocOSPProblem") -> Metric:
    """Look up ``name`` in the registry and instantiate it for ``problem``.

    Raises ``ValueError`` for unknown names.
    """
    if name not in _REGISTRY:
        available = sorted(_REGISTRY)
        raise ValueError(
            f"Unknown metric {name!r}. Available metrics: {available}"
        )
    return _REGISTRY[name](problem)


# ---------------------------------------------------------------------------
# Built-in metrics
# ---------------------------------------------------------------------------

@register_metric("mean_penalized_search_area")
def _(problem: "DDLeakLocOSPProblem") -> Metric:
    """Mean penalized search area across all scenarios.

    For each scenario the leak is at a known junction.  If that junction is
    inside the search area the cost is the search-area size (number of nodes
    flagged).  If it is not — i.e. the localizer missed the leak — the cost is
    ``n_nodes`` (worst-case penalty).  The metric is the average cost over all
    scenarios, so it ranges from ``1`` (one-node search area, leak always found)
    to ``n_nodes`` (leak never found).
    """
    id_to_pyidx = {nid: i for i, nid in enumerate(problem.leak_analysis.node_ids)}
    leak_node_pyidxs = np.array([
        id_to_pyidx[s[0]] for s in problem.leak_analysis.leak_scenarios
    ])
    n_scenarios = len(problem.leak_analysis.leak_scenarios)
    n_nodes = problem.leak_analysis.n_nodes

    def fn(ctx: MetricContext) -> float:
        scenario_idxs = np.arange(n_scenarios)
        leak_found = ctx.search_area[leak_node_pyidxs, scenario_idxs]   # (n_scenarios,) bool
        search_area_sizes = ctx.search_area.sum(axis=0)                  # (n_scenarios,) int
        costs = np.where(leak_found, search_area_sizes, n_nodes)
        return float(costs.mean())

    return Metric("mean_penalized_search_area", fn, min_val=1.0, max_val=float(n_nodes))


@register_metric("n_sensors")
def _(problem: "DDLeakLocOSPProblem") -> Metric:
    """Number of sensors placed (only meaningful when n_sensors is a free DV)."""
    def fn(ctx: MetricContext) -> float:
        return float(len(ctx.sensors_pyidxs))
    return Metric(
        "n_sensors",
        fn,
        min_val=float(problem.sens_l_limit),
        max_val=float(problem.sens_u_limit),
    )
