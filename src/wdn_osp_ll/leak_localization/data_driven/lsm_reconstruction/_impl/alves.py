import numpy as np

from ..base import LSMReconstructionMethod


class AlvesLSMReconstruction(LSMReconstructionMethod):
    """
    LSM reconstruction based on Alves et al.
    Uses a projection-based approach to reconstruct the full LSM.
    """

    LABEL = "alves"

    def _execute(self, observed_lsm: np.ndarray, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: dict) -> "AlvesLSMReconstruction":
        return cls()
