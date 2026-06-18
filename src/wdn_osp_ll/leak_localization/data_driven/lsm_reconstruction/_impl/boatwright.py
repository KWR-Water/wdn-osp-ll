import numpy as np

from ..base import LSMReconstructionMethod


class BoatwrightLSMReconstruction(LSMReconstructionMethod):
    """
    LSM reconstruction based on Boatwright et al.
    Estimates the full LSM from a subset of observed sensor rows.
    """

    LABEL = "boatwright"

    def _execute(self, observed_lsm: np.ndarray, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: dict) -> "BoatwrightLSMReconstruction":
        return cls()
