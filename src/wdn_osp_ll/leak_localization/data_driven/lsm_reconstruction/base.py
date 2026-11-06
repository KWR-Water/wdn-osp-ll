from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from wdn_osp_ll.hyd_analysis.data_structures import WDNBaseAnalysis


class LSMReconstructionMethod(ABC):

    LABEL = ""

    def __init__(self, wdn_base_analysis: "WDNBaseAnalysis") -> None:
        self.wdn_base_analysis = wdn_base_analysis

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self._execute(*args, **kwds)

    @abstractmethod
    def _execute(self, observed_lsm: np.ndarray, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def nix(self) -> int:
        return 0

    @property
    def ncx(self) -> int:
        return self.n_x-self.nix

    @property
    def n_x(self) -> int:
        nx = len(self.get_bounds()[0])
        assert nx == len(self.get_bounds()[1]), "The decision vector bounds' length don't match."
        return nx

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.array([]), np.array([]))

    @classmethod
    def from_config(cls, config: dict, wdn_base_analysis: "WDNBaseAnalysis") -> "LSMReconstructionMethod":
        return cls(wdn_base_analysis, **config)