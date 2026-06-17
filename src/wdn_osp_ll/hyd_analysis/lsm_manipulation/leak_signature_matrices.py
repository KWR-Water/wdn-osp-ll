"""Leak signature matrices (LSMs).

An LSM is a labelled 2-D array: rows are network nodes, columns are leak
scenarios. The labels are part of the object because every manipulation
(observing through a sensor set, scaling, mapping back to a pressure field)
needs them.

The base :class:`LeakSignatureMatrix` is a pure container: it owns the data,
the row/column labels, the per-column ``scale_factors`` (identity until a
scaling operation records one) and the ``nominal_pressures`` needed to map a
reconstruction back to a physical pressure field. ``to_pressure_field`` is the
only behaviour that differs per matrix type and is left to the subclass.

Stateless manipulations live in :mod:`.operations`; they rebuild new instances
through :meth:`LeakSignatureMatrix._replace`, which preserves the concrete
subclass and every attribute except the ones being changed.
"""

import copy

import numpy as np


class LeakSignatureMatrix:
    """Labelled (``n_nodes``, ``n_scenarios``) matrix plus its inversion data.

    Parameters
    ----------
    data:
        The matrix itself, ``(n_nodes, n_scenarios)``.
    node_ids:
        Row labels: node EPANET IDs, ordered as ``data``'s rows.
    leak_scenarios:
        Column labels: ``(junction_id, emitter_coefficient)`` pairs.
    nominal_pressures:
        Nominal pressure at every node, ``(n_nodes,)``; used by subclasses to
        map a reconstructed matrix back to a pressure field.
    scale_factors:
        Factor already divided out of ``data``, kept in a shape that broadcasts
        against it (e.g. ``(n_nodes, 1)`` for per-node scaling, ``(1,
        n_scenarios)`` for per-scenario). Defaults to the scalar ``1.0`` (an
        unscaled matrix). A scaling operation records here what it removed so
        :meth:`_unscale` can put it back before inversion.
    """

    def __init__(
        self,
        data: np.ndarray,
        node_ids: list[str],
        leak_scenarios: list[tuple[str, float]],
        nominal_pressures: np.ndarray,
        scale_factors: np.ndarray | float | None = None,
    ):
        self.data = data
        self.node_ids = node_ids
        self.leak_scenarios = leak_scenarios
        self.nominal_pressures = nominal_pressures
        self.scale_factors = 1.0 if scale_factors is None else scale_factors

    # -- shape convenience --------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return self.data.shape[0]

    @property
    def n_scenarios(self) -> int:
        return self.data.shape[1]

    # -- functional update --------------------------------------------------
    def _replace(self, **changes) -> "LeakSignatureMatrix":
        """Return a copy of this matrix with the named attributes swapped,
        keeping the concrete subclass and everything else untouched."""
        new = copy.copy(self)
        for key, value in changes.items():
            setattr(new, key, value)
        return new
    
    def _scale(self, factors: np.ndarray) -> "LeakSignatureMatrix":
        self._replace(
            data=self.data / factors,
            scale_factors=self.scale_factors * factors,
        )
        return self

    # -- inversion ----------------------------------------------------------
    def _unscale(self, matrix: np.ndarray) -> np.ndarray:
        """Re-apply, to a reconstructed matrix, the scaling that was divided out
        of ``data`` — the first half of inverting any scaled LSM. ``scale_factors``
        is stored pre-broadcast, so this is a plain multiply."""
        return matrix * self.scale_factors

    def to_pressure_field(self, reconstructed: np.ndarray) -> np.ndarray:
        """Map a reconstructed matrix back to the pressure field that would have
        produced it. Subclass-specific: the base class has no signature law."""
        raise NotImplementedError


class SensitivityMatrix(LeakSignatureMatrix):
    """LSM whose entries are the nominal-minus-leak pressure drop.

    ``data[i, c] == nominal_pressures[i] - pressure_under_scenario_c[i]``.
    """

    @classmethod
    def from_analysis(cls, analysis) -> "SensitivityMatrix":
        """Build the sensitivity matrix over every scenario of a
        :class:`WDNLeakAnalysis`.

        To restrict to a subset of scenarios, reduce the analysis first with
        ``analysis.select_scenarios(...)`` and build from the result. The few
        small arrays the matrix needs are copied out, so it is self-contained
        and does not pin the heavy analysis object.
        """
        sensitivity = analysis.pressures[:, None] - analysis.pressures_per_leak_scenario
        return cls(
            data=sensitivity.copy(),
            node_ids=list(analysis.node_ids),
            leak_scenarios=list(analysis.leak_scenarios),
            nominal_pressures=analysis.pressures.copy(),
        )

    def to_pressure_field(self, reconstructed: np.ndarray) -> np.ndarray:
        """Leak pressure field that produced ``reconstructed``: undo any column
        scaling, then ``p = nominal - sensitivity``."""
        return self.nominal_pressures[:, None] - self._unscale(reconstructed)
