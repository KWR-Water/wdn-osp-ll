from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..core import DDLeakLocMethod
from ..lsm_reconstruction.base import LSMReconstructionMethod
from ..lsm_reconstruction.services import get_lsm_reconstruction_method
from ..search_area_definition.base import SearchAreaDefinitionMethod
from ..search_area_definition.services import get_search_area_definition_method

if TYPE_CHECKING:
    from wdn_osp_ll.hyd_analysis.data_structures import WDNBaseAnalysis

class TwoStepsDDLeakLocMethod(DDLeakLocMethod):

    LABEL = "two_steps"
    
    def __init__(
            self,
            lsm_reconstructor: LSMReconstructionMethod,
            search_area_definer: SearchAreaDefinitionMethod
        ) -> None:
        super().__init__()

        self.lsm_reconstructor = lsm_reconstructor
        self.search_area_definer = search_area_definer

    def _localize(self, observations: np.ndarray, x: np.ndarray) -> np.ndarray:
        
        lsm_obs = observations.copy()

        x_reco, x_sadef = self._split_x(x)

        # Implement the two steps methodology
        lsm_reco = self.lsm_reconstructor(lsm_obs, x_reco)

        search_area = self.search_area_definer(lsm_reco, x_sadef)

        return search_area
    
    @property
    def nix(self) -> int:
        return self.lsm_reconstructor.nix+self.search_area_definer.nix
    
    @property
    def ncx(self) -> int:
        return self.n_x-self.nix
    
    @property
    def n_x(self) -> int:
        nx = len(self.get_bounds()[0])
        assert nx == len(self.get_bounds()[1]), "The decision vector bounds' length don't match."
        return nx
    
    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lb_reco, ub_reco = self.lsm_reconstructor.get_bounds()
        lb_sadef, ub_sadef = self.search_area_definer.get_bounds()

        lower_bound = np.concat([
            lb_reco[:self.lsm_reconstructor.ncx],
            lb_sadef[:self.search_area_definer.ncx],
            lb_reco[self.lsm_reconstructor.ncx:],
            lb_sadef[self.search_area_definer.ncx:]
        ])

        upper_bound = np.concat([
            ub_reco[:self.lsm_reconstructor.ncx],
            ub_sadef[:self.search_area_definer.ncx],
            ub_reco[self.lsm_reconstructor.ncx:],
            ub_sadef[self.search_area_definer.ncx:]
        ])

        return (lower_bound, upper_bound)
    
    def _split_x(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Split the decision vector between the reconstructor and the definer.
        The generic implementations is:
            ``[dv_cx(reco), dv_cx(sadef), dv_ix(reco), dv_ix(sadef),]``
        where cx are continuous decision variables and ix integers
        """
        nix_reco = self.lsm_reconstructor.nix
        ncx_reco = self.lsm_reconstructor.ncx

        nix_sadef = self.search_area_definer.nix
        ncx_sadef = self.search_area_definer.ncx

        assert nix_reco+ncx_reco+nix_sadef+ncx_sadef == len(x), "Not matching lengths in decison vector for localization action"

        ncx_all = ncx_reco+ncx_sadef

        cx_reco = x[:ncx_reco]
        cx_sadef = x[ncx_reco:ncx_all]

        ix_reco = x[ncx_all:ncx_all+nix_reco]
        ix_sadef = x[ncx_all+nix_reco:]

        x_reco = np.concat([cx_reco, ix_reco])
        x_sadef = np.concat([cx_sadef, ix_sadef])
        
        return x_reco, x_sadef 
    
    @classmethod
    def from_config(cls, config: dict, wdn_base_analysis: "WDNBaseAnalysis") -> "TwoStepsDDLeakLocMethod":

        reco_method = get_lsm_reconstruction_method(
            config["reconstruction_method"], wdn_base_analysis
        )

        sadef_method = get_search_area_definition_method(config["search_area_definition_method"])

        return cls(
            lsm_reconstructor=reco_method,
            search_area_definer=sadef_method,
        )