"""Data-driven leak localization OSP problem (pygmo-compatible).

The problem is parameterised by three composable modules:

- **LSM** (:class:`LeakSignatureMatrix` subclass): the pre-computed leak
  signature matrix derived from a :class:`WDNLeakAnalysis`.
- **ReconstructionMethod**: how to fill in the unobserved LSM rows from the
  sensor readings (stores name + args; implementation added later).
- **SearchAreaDefinitionMethod**: how to turn a reconstructed pressure field
  into a search area (stores name + args; implementation added later).

Use :meth:`DDLeakLocOSPProblem.from_config` to build from the ``problem:``
subtree of a YAML optimisation config.
"""

from __future__ import annotations

from typing import Self

import numpy as np

from ...hyd_analysis.data_structures import WDNLeakAnalysis
from ...hyd_analysis.lsm_manipulation.operations import observe
from ...hyd_analysis.lsm_manipulation.leak_signature_matrices import LeakSignatureMatrix

from ...leak_localization.data_driven import DDLeakLocMethod, get_dd_leak_loc_method

# ---------------------------------------------------------------------------
# OSP problem
# ---------------------------------------------------------------------------

class DDLeakLocOSPProblem:
    """Pygmo-compatible data-driven leak localisation OSP problem.

    Parameters
    ----------
    leak_analysis:
        Pre-computed leak analysis on the WDN (a :class:`WDNLeakAnalysis`)
    lsm:
        Pre-computed leak signature matrix (any :class:`LeakSignatureMatrix`
        subclass).
    reconstruction_method:
        Descriptor for the LSM reconstruction method.
    search_area_method:
        Descriptor for the search-area definition method.
    metrics:
        Names of all metrics computed at each evaluation.  Results are
        returned as a ``dict[str, float]`` by :meth:`evaluate`.
    objectives:
        Subset of ``metrics`` returned to pygmo as the fitness vector
        (length == ``get_nobj()``).  Order matters: pygmo minimises each
        component in this order.
    possible_sensor_locations:
        Junction EPANET IDs that are valid sensor placements.  Defaults to
        all junctions present in the LSM.
    """

    def __init__(
        self,
        leak_analysis: WDNLeakAnalysis,
        lsm: LeakSignatureMatrix,
        leak_localizer: DDLeakLocMethod,
        n_sensors: int | None,
        metrics: list[str],
        objectives: list[str],
        excluded_junctions: set[str] | None = None,
    ) -> None:
        self.leak_analysis = leak_analysis
        self.lsm = lsm
        self.leak_localizer = leak_localizer
        self.n_sensors = n_sensors
        self.metrics = list(metrics)
        self.objectives = list(objectives)

        if excluded_junctions is None:
            self.possible_sensor_locations = leak_analysis.junction_ids
            self.possible_sensor_locations_pyidxs = np.array(leak_analysis.junction_pyidxs)
        else:
            excluded_junctions = set(excluded_junctions)
            psl = {
                i: j
                for i, j in zip(leak_analysis.junction_pyidxs, leak_analysis.junction_pyidxs)
                if j not in excluded_junctions
            }
            self.possible_sensor_locations = list(psl.values())
            self.possible_sensor_locations_pyidxs = np.array(psl.keys())

        self._validate()
        self._init_sensor_dvs()

    def _validate(self) -> Self:
        node_set = set(self.lsm.node_ids)
        unknown = [s for s in self.possible_sensor_locations if s not in node_set]
        if unknown:
            raise ValueError(
                f"possible_sensor_locations contains IDs not in the LSM: {unknown}"
            )
        
        non_metric = [o for o in self.objectives if o not in self.metrics]
        if non_metric:
            raise ValueError(
                f"objectives must be a subset of metrics; unknown: {non_metric}"
            )
        return self
        
    def _init_sensor_dvs(self) -> Self:
        """
        Initialize the lower and upper bound of the decison variables regarding
        the optimal sensor placement problem.
        For a fixed number of sensors, this is a 
        """
        # Limits on the number of installable sensor (not the bound on the 
        # decision variable, which depends on how we formulate them):
        # Max 1 sensor every 2 junctions
        # Min 1 sensor per source, or at least 2 otherwise it doesn't make sense
        self.sens_u_limit = self.leak_analysis.n_junctions // 2
        self.sens_l_limit = max(self.leak_analysis.n_sources, 2)

        # Dimension of the sensor decison space
        self.sens_dim = len(self.possible_sensor_locations)
        assert self.sens_l_limit < self.sens_dim , \
            "Review the excluded junctions, because there are not enough junctions to cover every network's source."+\
            f"Min n. of sensors: {self.sens_l_limit}, requested n. of sensors: {self.sens_dim}"
         
        if self.n_sensors is not None and not (self.sens_l_limit <= self.n_sensors <= self.sens_u_limit):
            raise ValueError(f"Invalid number of sensors, n: {self.n_sensors}. Must be between {self.sens_l_limit} and {self.sens_u_limit}")

        if self.n_sensors is None:
            # The number of sensors is also a decision variable along with the location
            dv_lb = 0
            dv_ub = 1
            dv_dim = self.sens_dim

        else:
            # The number of sensors is fixed and only the location is a decision v.
            dv_lb = 0
            dv_ub = self.sens_dim-1
            dv_dim = self.n_sensors

        self._sens_dvs_lb = np.full(shape=(dv_dim,), fill_value=dv_lb)
        self._sens_dvs_ub = np.full(shape=(dv_dim,), fill_value=dv_ub)
        self._sens_dvs_nix = dv_dim

        return self
    
    def _get_sensors_pyidxs_from_x(self, x: np.ndarray) -> np.ndarray:
        # The sensor locations are always the last values in the decision vector
        # because they are discrete.
        if self.n_sensors is None:
            sens_loc_start_idx = -self.sens_dim
            sens_loc_mask = x[sens_loc_start_idx:].astype(bool)
            return self.possible_sensor_locations_pyidxs[sens_loc_mask]

        else:
            sens_loc_start_idx = -self.n_sensors
            sens_loc_idx = np.unique(x[sens_loc_start_idx:].astype(int))
            return self.possible_sensor_locations_pyidxs[sens_loc_idx]
        
    def compute_metrics(self, x: np.ndarray) -> dict[str, float]:
        sensors_pyidxs = self._get_sensors_pyidxs_from_x(x)
        
        lsm_observed = observe(lsm=self.lsm, sensor_idxs=sensors_pyidxs)

        search_area = self.leak_localizer(lsm_observed, x[:-self._sens_dvs_nix])

        # calculate metrics
        # TODO: impl
        metrics = {}
        return metrics

    # -- pygmo mandatory interface --------------------------------------------

    def fitness(self, x: np.ndarray) -> list[float]:
        metrics = self.compute_metrics(x)

        # Constraint check: this decision vector has installed too many sensors:
        if self.n_sensors is None and (
            metrics["n_sensors"] < self.sens_l_limit or metrics["n_sensors"] > self.sens_u_limit
        ):
            return self.get_nadir_point()

        return [metrics[obj] for obj in self.objectives]

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        #------------- LL METHOD DECISION VARIABLES  ---------------#
        ll_dvs_lb, ll_dvs_ub = self.leak_localizer.get_bounds()

        #------------- SENSOR PLACE DECISION VARIABLES -------------#
        # These are all and always discrete, which means that for pygmo convention
        # they always go at the end (no need to re-shuffle).
        lower_bound = np.append(ll_dvs_lb, self._sens_dvs_lb)
        upper_bound = np.append(ll_dvs_ub, self._sens_dvs_ub)
        
        return (lower_bound, upper_bound)

    def get_nix(self) -> int:
        return self.leak_localizer.nix + self._sens_dvs_nix

    def get_nobj(self) -> int:
        return len(self.objectives)

    # -- pygmo optional interface ---------------------------------------------

    def get_name(self) -> str:
        return "Data-Driven Leak Localisation OSP"

    def get_extra_info(self) -> str:
        return "\n".join([
            f"  LSM type              : {type(self.lsm).__name__}",
            f"  n_nodes / n_scenarios : {self.lsm.n_nodes} / {self.lsm.n_scenarios}",
            f"  Leak localization     : {self.leak_localizer.LABEL}",
            f"  Objectives            : {self.objectives}",
            f"  Metrics               : {self.metrics}",
        ])

    # -- factory --------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict) -> "DDLeakLocOSPProblem":
        """Build from the ``problem:`` subtree of an optimisation YAML config.

        Expected shape::

            leak_analysis:
                config: path/to/hyd_analysis.yaml
                emitter_coefficient_filter: null   # null → all scenarios
                signature_matrix: sensitivity      # sensitivity | chi_squared

            leak_localization:
                reconstruction_method:
                    name: idw
                    args:
                        idw_exponent: 2.0
                search_area_definition_method:
                    name: relative
                    args:
                        threshold: 0.1

            possible_sensor_locations: null        # null → all junctions

            metrics:
                - mean_search_area
                - n_sensors

            objectives:
                - mean_search_area                 # subset of metrics
        """
        la_cfg = config["leak_analysis"]

        # Load (or reuse cached) hydraulic analysis
        analysis = WDNLeakAnalysis.load_or_compute(la_cfg["config"])

        # Optionally restrict to a single emitter coefficient
        ec_filter = la_cfg.get("emitter_coefficient_filter")
        if ec_filter is not None:
            ec_filter = float(ec_filter)
            wanted = [s for s in analysis.leak_scenarios if s[1] == ec_filter]
            if not wanted:
                raise ValueError(
                    f"emitter_coefficient_filter={ec_filter} not found in "
                    f"analysis scenarios.  Available: "
                    f"{sorted({s[1] for s in analysis.leak_scenarios})}"
                )
            analysis = analysis.select_scenarios(wanted)

        # Build the LSM
        lsm_type = la_cfg.get("signature_matrix", "sensitivity").lower()
        if lsm_type == "sensitivity":
            lsm = analysis.sensitivity_matrix
        elif lsm_type in ("chi_squared", "chi-squared", "chi2"):
            lsm = analysis.chi_squared_matrix
        else:
            raise ValueError(
                f"Unknown signature_matrix: {lsm_type!r}. "
                "Use 'sensitivity' or 'chi_squared'."
            )

        ll_cfg = config["dd_leak_localization"]
        ll_method = get_dd_leak_loc_method(config=ll_cfg, wdn_base_analysis=analysis) # TODO: restrict to base class
        
        # Restrict the optimal sensor placement problem 

        # Sensor locations: null → all junctions from the analysis
        excluded_junctions = config.get("excluded_junctions", None)

        # N. of sensors, fix or optimized? null -> optmized
        n_sensors = config.get("n_sensors", None)

        return cls(
            leak_analysis=analysis,
            lsm=lsm,
            leak_localizer=ll_method,
            metrics=list(config["metrics"]),
            objectives=list(config["objectives"]),
            excluded_junctions=excluded_junctions,
            n_sensors=n_sensors,
        )
