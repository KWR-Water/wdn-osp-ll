from abc import ABC, abstractmethod
from typing import Any

import numpy as np

class DDLeakLocMethod(ABC):

    LABEL = ""

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self._localize(*args, **kwds)

    @abstractmethod
    def _localize(self, observations: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Method to localize a leak, i.e., define a search area (vector of bool 
        indicatin wheter to include the node or not).

        It requires some observations and a decision vector with the setting of 
        the localization method itself.
        """
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
    def from_config(cls, _config: dict) -> "DDLeakLocMethod":
        raise NotImplementedError
