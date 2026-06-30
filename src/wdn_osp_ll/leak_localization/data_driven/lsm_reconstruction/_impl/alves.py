from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from ..base import LSMReconstructionMethod

if TYPE_CHECKING:
    from wdn_osp_ll.hyd_analysis.data_structures import WDNBaseAnalysis


class AlvesLSMReconstruction(LSMReconstructionMethod):
    """Inverse-distance-weighted LSM reconstruction (Alves et al.).

    Like Boatwright but the metric is always resistance and there is no
    exponent (weights = 1/d).

    Parameters
    ----------
    directed:
        Whether to use the directed graph.  If *None*, becomes an integer
        decision variable (bounds [0, 1]).
    """

    LABEL = "alves"

    def __init__(
        self,
        wdn_base_analysis: "WDNBaseAnalysis",
        directed: Optional[bool] = None,
        sources_as_sensors: bool = False,
    ) -> None:
        super().__init__(wdn_base_analysis)
        self._directed = directed
        self._sources_as_sensors = sources_as_sensors

    def _get_resistances(self, directed: bool) -> tuple[np.ndarray, np.ndarray]:
        
        method_name = f"shortest_path_resistance_{'directed' if directed else 'undirected'}"
        resis = getattr(self.wdn_base_analysis, method_name)

        method_name = f"common_resistance_{'directed' if directed else 'undirected'}"
        common_resis = getattr(self.wdn_base_analysis, method_name)

        return resis, common_resis

    def _resolve(self, x: np.ndarray) -> bool:
        if self._directed is None:
            return bool(int(x[0]))
        return self._directed

    def _execute(self, observed_lsm: np.ndarray, x: np.ndarray) -> np.ndarray:
        directed = self._resolve(x)
        
        resistances, common_resistances = self._get_resistances(directed)

        observed_mask = ~np.all(np.isnan(observed_lsm), axis=1)

        if self._sources_as_sensors:
            anchors = np.where(observed_mask)[0]
        else:
            source_mask = np.zeros(observed_lsm.shape[0], dtype=bool)
            source_mask[self.wdn_base_analysis.source_pyidxs] = True
            anchors = np.where(observed_mask & ~source_mask)[0]

        reconstructed = observed_lsm.copy()
        for j in np.where(~observed_mask)[0]:
            r = resistances[j, anchors]
            inf_mask = np.isinf(r)
            if inf_mask.any():
                r = r.copy()
                r[inf_mask] = resistances[anchors[inf_mask], j]
            
            # eq between 13 and 14
            inv_r = 1.0 / r
            g_bar = inv_r / inv_r.sum() if inv_r.sum() != 0 else np.ones_like(inv_r)

            cr = common_resistances[j, anchors]
            inf_mask = np.isinf(cr)
            if inf_mask.any():
                cr = cr.copy()
                cr[inf_mask] = common_resistances[anchors[inf_mask], j]

            # eq 13
            g = cr * g_bar

            # eq 14
            l = g / g.sum()

            # (kind of) eq 18
            reconstructed[j] = np.sum(observed_lsm[anchors] * l[:, np.newaxis], axis=0)

        return reconstructed

    @property
    def nix(self) -> int:
        return self._directed is None

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self._directed is None:
            return np.array([0]), np.array([1])
        return np.array([]), np.array([])
