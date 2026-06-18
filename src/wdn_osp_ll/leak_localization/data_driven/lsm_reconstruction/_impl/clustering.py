from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..base import LSMReconstructionMethod

if TYPE_CHECKING:
    from wdn_osp_ll.hyd_analysis.data_structures import WDNBaseAnalysis


class DistanceMetric(IntEnum):
    TOPOLOGICAL = 0
    LENGTH = 1
    RESISTANCE = 2


class ClusteringLSMReconstruction(LSMReconstructionMethod):
    """
    Reconstructs the full LSM by assigning each non-sensor node the observed
    value of its nearest sensor, measured by a graph distance from wdn_base_analysis.

    Sensor nodes are identified as rows of observed_lsm that are not all-NaN.

    If metric or directed is None it becomes an integer decision variable
    (bounds [0, 2] and [0, 1] respectively); integer vars follow continuous vars
    in the decision vector per the base-class contract.
    """

    LABEL = "clustering"

    def __init__(
        self,
        wdn_base_analysis: "WDNBaseAnalysis",
        metric: Optional[str] = None,
        directed: Optional[bool] = None,
    ) -> None:
        super().__init__(wdn_base_analysis)

        if metric is not None and metric.upper() not in DistanceMetric.__members__:
            raise ValueError(
                f"Unknown metric {metric!r}. "
                f"Valid options: {[m.name.lower() for m in DistanceMetric]}"
            )

        self._metric = DistanceMetric[metric.upper()] if metric is not None else None
        self._directed = directed

    def _get_distances(self, metric: DistanceMetric, directed: bool) -> np.ndarray:
        method_name = f"shortest_path_{metric.name.lower()}_{'directed' if directed else 'undirected'}"
        return getattr(self.wdn_base_analysis, method_name)()

    def _resolve(self, x: np.ndarray) -> tuple[DistanceMetric, bool]:
        # x contains only integer vars (ncx == 0); metric index first, directed flag second
        idx = 0
        if self._metric is None:
            metric = DistanceMetric(int(x[idx]))
            idx += 1
        else:
            metric = self._metric
        if self._directed is None:
            directed = bool(int(x[idx]))
        else:
            directed = self._directed
        return metric, directed

    def _execute(self, observed_lsm: np.ndarray, x: np.ndarray) -> np.ndarray:
        metric, directed = self._resolve(x)
        dist_matrix = self._get_distances(metric, directed)

        # Non-sensor rows are fully NaN; sensor rows have finite values
        sensor_mask = ~np.all(np.isnan(observed_lsm), axis=1)
        sensors = np.where(sensor_mask)[0]

        reconstructed = observed_lsm.copy()
        for j in np.where(~sensor_mask)[0]:
            closest = sensors[np.argmin(dist_matrix[j, sensors])]
            reconstructed[j] = observed_lsm[closest]

        return reconstructed

    @property
    def nix(self) -> int:
        return (self._metric is None) + (self._directed is None)

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        # Layout: [cx... (none), ix...]; metric index first, directed flag second
        lbs, ubs = [], []
        if self._metric is None:
            lbs.append(0)
            ubs.append(2)  # 0=topological, 1=length, 2=resistance
        if self._directed is None:
            lbs.append(0)
            ubs.append(1)  # 0=undirected, 1=directed
        return np.array(lbs), np.array(ubs)
