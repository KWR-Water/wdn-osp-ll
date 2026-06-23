from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..base import LSMReconstructionMethod

if TYPE_CHECKING:
    from wdn_osp_ll.hyd_analysis.data_structures import WDNBaseAnalysis


class BoatwrightDistanceMetric(IntEnum):
    LENGTH = 0
    RESISTANCE = 1


def _idw_reconstruct(observed_lsm: np.ndarray, dist: np.ndarray, exponent: int) -> np.ndarray:
    """Inverse distance weighted reconstruction of non-sensor LSM rows."""
    sensor_mask = ~np.all(np.isnan(observed_lsm), axis=1)
    sensors = np.where(sensor_mask)[0]
    non_sensors = np.where(~sensor_mask)[0]

    reconstructed = observed_lsm.copy()
    for j in non_sensors:
        weights = 1.0 / (dist[sensors, j] ** exponent)  # shape (n_sensors,)
        weighted_sum = (observed_lsm[sensors] * weights[:, np.newaxis]).sum(axis=0)
        reconstructed[j] = weighted_sum / weights.sum()

    return reconstructed


class BoatwrightLSMReconstruction(LSMReconstructionMethod):
    """Inverse-distance-weighted LSM reconstruction (Boatwright et al.).

    Each non-sensor node is estimated as a weighted average of all sensor
    nodes, with weights equal to 1/d^n where d is the shortest-path distance
    and n is the exponent.

    Parameters
    ----------
    metric:
        ``"length"`` or ``"resistance"``.  If *None*, becomes an integer
        decision variable (bounds [0, 1]).
    directed:
        Whether to use the directed graph.  If *None*, becomes an integer
        decision variable (bounds [0, 1]).
    exponent:
        Integer in [1, 5].  If *None*, becomes an integer decision variable
        (bounds [1, 5]).
    """

    LABEL = "boatwright"

    LB_EXPONENT = 1
    UB_EXPONENT = 5

    def __init__(
        self,
        wdn_base_analysis: "WDNBaseAnalysis",
        metric: Optional[str] = None,
        directed: Optional[bool] = None,
        exponent: Optional[int] = None,
    ) -> None:
        super().__init__(wdn_base_analysis)

        if metric is not None and metric.upper() not in BoatwrightDistanceMetric.__members__:
            raise ValueError(
                f"Unknown metric {metric!r}. "
                f"Valid options: {[m.name.lower() for m in BoatwrightDistanceMetric]}"
            )
        if exponent is not None and not (
            isinstance(exponent, int) and self.LB_EXPONENT <= exponent <= self.UB_EXPONENT
        ):
            raise ValueError(
                f"exponent must be an integer in [{self.LB_EXPONENT}, {self.UB_EXPONENT}], got {exponent!r}"
            )

        self._metric = BoatwrightDistanceMetric[metric.upper()] if metric is not None else None
        self._directed = directed
        self._exponent = exponent

    def _get_distances(self, metric: BoatwrightDistanceMetric, directed: bool) -> np.ndarray:
        method_name = f"shortest_path_{metric.name.lower()}_{'directed' if directed else 'undirected'}"
        return getattr(self.wdn_base_analysis, method_name)

    def _resolve(self, x: np.ndarray) -> tuple[BoatwrightDistanceMetric, bool, int]:
        idx = 0
        if self._metric is None:
            metric = BoatwrightDistanceMetric(int(x[idx]))
            idx += 1
        else:
            metric = self._metric
        if self._directed is None:
            directed = bool(int(x[idx]))
            idx += 1
        else:
            directed = self._directed
        if self._exponent is None:
            exponent = int(x[idx])
        else:
            exponent = self._exponent
        return metric, directed, exponent

    def _execute(self, observed_lsm: np.ndarray, x: np.ndarray) -> np.ndarray:
        metric, directed, exponent = self._resolve(x)
        dist = self._get_distances(metric, directed)
        return _idw_reconstruct(observed_lsm, dist, exponent)

    @property
    def nix(self) -> int:
        return (self._metric is None) + (self._directed is None) + (self._exponent is None)

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lbs, ubs = [], []
        if self._metric is None:
            lbs.append(0)
            ubs.append(1)  # 0=length, 1=resistance
        if self._directed is None:
            lbs.append(0)
            ubs.append(1)  # 0=undirected, 1=directed
        if self._exponent is None:
            lbs.append(self.LB_EXPONENT)
            ubs.append(self.UB_EXPONENT)
        return np.array(lbs), np.array(ubs)
