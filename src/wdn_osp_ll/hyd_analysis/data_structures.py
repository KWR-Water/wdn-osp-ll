"""Hydraulic-analysis data structures.

``WDNBaseAnalysis`` caches the static + nominal-condition EPANET data needed by
the OSP algorithms; ``WDNLeakAnalysis`` extends it with the per-leak-scenario
simulation results.

Both classes are picklable: the heavy state lives in plain attributes (numpy
arrays + a ``networkx.DiGraph``) and never holds an open EPANET handle. The lazy
cache of derived quantities is included in pickle copies (needed by pygmo) but
excluded from the ``dump`` / ``load`` disk format, where it is rebuilt on demand.

Conventions (asserted on load): the network uses SI units with LPS flows, has no
tanks, contains only pipes, and uses the Hazen-Williams head-loss formula. As a
consequence no unit conversion is needed: pressures are in metres, flows in L/s.
"""

import os
import copy
import pickle
import hashlib
import pathlib

import yaml
import numpy as np
import networkx as nx
from epyt import epanet

from wdn_osp_ll.hyd_analysis.lsm_manipulation.leak_signature_matrices import (
    ChiSquaredMatrix,
    SensitivityMatrix,
)

# --- graph attribute keys --------------------------------------------------
# node attributes
PY_INDEX = "py_index"   # 0-based python index; EPANET index is py_index + 1
ELEVATION = "elevation"
DEMAND = "demand"       # requested demand
OUTFLOW = "outflow"     # delivered demand
# pipe (edge) attributes
ID = "id"               # EPANET pipe ID (the edge itself is keyed by node IDs)
LENGTH = "length"
DIAMETER = "diameter"
ROUGHNESS = "roughness"
FLOW = "flow"           # absolute flow at MNF (edge orientation carries the sign)

# --- config keys -----------------------------------------------------------
INP_FILEPATH = "inp_filepath"
SETTINGS = "settings"
EMITTER_COEFFICIENTS = "emitter_coefficients"
DEMAND_MODEL = "demand_model"

# The networks are snapshot simulations (duration == 0): a single time step that
# represents the minimum-night-flow operating point.
_MNF = 0


def _open_network(inp_filepath: str) -> epanet:
    """Open an EPANET network without spamming messages/warnings."""
    return epanet(inp_filepath, display_msg=False, display_warnings=False)


class WDNBaseAnalysis:
    """Static topology + nominal-condition EPANET data used by the OSP algorithms.

    Stored state:

    - ``inp_filepath``: the ``.inp`` file this analysis was built from.
    - ``graph``: a ``networkx.DiGraph`` whose
        - nodes (junctions and sources, keyed by their EPANET ID) carry
          ``py_index``, ``elevation``, ``demand`` (requested) and ``outflow``
          (delivered);
        - edges (pipes, keyed by their EPANET ID) carry ``py_index``, ``length``,
          ``diameter``, ``roughness`` and the absolute ``flow`` at MNF. Edge 
          orientation follows the EPANET orientation flipped by the sign of the 
          MNF flow, so it always points in the actual flow direction.
    - ``pressures``: nominal pressure at every node (``n_nodes``,), metres.
    - ``junction_ids``: list of junction EPANET IDs.

    If ``inp_filepath`` is ``None`` the instance is left empty (used by the
    derived class, which builds the EPANET object itself). If a path is given,
    the nominal simulation is run, the data above is cached and the EPANET handle
    is closed. ``demand_model`` is an optional dict forwarded verbatim to
    ``setDemandModel`` (keys ``code``, ``pmin``, ``preq``, ``pexp``).
    """

    def __init__(self, inp_filepath: str | None = None, demand_model: dict | None = None):
        self._init_cache()
        if inp_filepath is None:
            self._init_empty()
        else:
            ensystem = self._build(inp_filepath, demand_model)
            ensystem.closeNetwork()

    # -- construction -------------------------------------------------------
    def _init_empty(self) -> None:
        self.inp_filepath = None
        self.graph = None
        self.pressures = None
        self.junction_ids = None

    def _build(self, inp_filepath: str, demand_model: dict | None = None) -> epanet:
        """Run the nominal simulation, cache the static + nominal data and return
        the (still open) EPANET object so a caller can keep using it."""
        ensystem = _open_network(inp_filepath)

        assert ensystem.getUnits().Units_SI_Metric > 0 and ensystem.getFlowUnits() == "LPS", \
            "Network must use SI units with LPS flows."
        assert ensystem.getNodeTankCount() == 0, \
            "Network contains tanks and our algorithms don't handle them."
        assert set(ensystem.getLinkType()) <= {"PIPE"}, \
            "Network contains non-pipe links; only pipes are supported."
        assert ensystem.getOptionsHeadLossFormula() == "HW", \
            "Head-loss formula must be Hazen-Williams (HW)."

        if demand_model is not None:
            ensystem.setDemandModel(**demand_model)

        self.inp_filepath = inp_filepath
        self.junction_ids = ensystem.getNodeJunctionNameID()

        results = ensystem.getComputedHydraulicTimeSeries()
        self.pressures = np.asarray(results.Pressure[_MNF, :], dtype=float)
        outflows = np.asarray(results.Demand[_MNF, :], dtype=float)
        flows = np.asarray(results.Flow[_MNF, :], dtype=float)

        self.graph = self._build_graph(ensystem, outflows, flows)
        return ensystem

    @staticmethod
    def _build_graph(ensystem: epanet, outflows: np.ndarray, flows: np.ndarray) -> nx.DiGraph:
        node_ids = ensystem.getNodeNameID()
        elevations = np.asarray(ensystem.getNodeElevations(), dtype=float)
        # requested demand: sum across demand categories
        requested = np.sum(
            [np.asarray(v, dtype=float) for v in ensystem.getNodeBaseDemands().values()],
            axis=0,
        )

        graph = nx.DiGraph()
        for py_idx, node_id in enumerate(node_ids):
            graph.add_node(
                node_id,
                **{
                    PY_INDEX: py_idx,
                    ELEVATION: float(elevations[py_idx]),
                    DEMAND: float(requested[py_idx]),
                    OUTFLOW: float(outflows[py_idx]),
                },
            )

        link_ids = ensystem.getLinkNameID()
        nodes_index = np.asarray(ensystem.getLinkNodesIndex())  # (n_links, 2), EPANET indices
        lengths = np.asarray(ensystem.getLinkLength(), dtype=float)
        diameters = np.asarray(ensystem.getLinkDiameter(), dtype=float)
        roughnesses = np.asarray(ensystem.getLinkRoughnessCoeff(), dtype=float)

        for link_py_idx, link_id in enumerate(link_ids):
            start_en, end_en = nodes_index[link_py_idx]
            flow = float(flows[link_py_idx])
            # Orient the edge in the actual flow direction: EPANET start->end when
            # the flow is positive, reversed when negative.
            if flow < 0:
                start_en, end_en = end_en, start_en
            graph.add_edge(
                node_ids[start_en - 1],
                node_ids[end_en - 1],
                **{
                    ID: link_id,
                    PY_INDEX: link_py_idx,
                    LENGTH: float(lengths[link_py_idx]),
                    DIAMETER: float(diameters[link_py_idx]),
                    ROUGHNESS: float(roughnesses[link_py_idx]),
                    FLOW: abs(flow),
                },
            )
        return graph

    # -- lazy cache ---------------------------------------------------------
    def _init_cache(self) -> None:
        self._cache: dict = {}

    def _cached(self, key: str, compute):
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

    # -- convenience --------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return self.graph.number_of_nodes() if self.graph is not None else 0

    @property
    def n_junctions(self) -> int:
        return len(self.junction_ids) if self.junction_ids is not None else 0

    @property
    def n_sources(self) -> int:
        return self.n_nodes - self.n_junctions

    @property
    def node_ids(self) -> list[str]:
        """Node EPANET IDs ordered by ``py_index`` (matching the rows of the
        node-indexed arrays such as ``pressures``)."""
        def _ordered():
            assert self.graph is not None
            ids: list[str | None] = [None] * self.n_nodes
            for node_id, py_idx in self.graph.nodes(data=PY_INDEX):
                ids[py_idx] = node_id
            return ids
        return self._cached("node_ids", _ordered)
    
    # @property
    # junction_ids(self) -> list[str]: Already a property

    @property
    def source_ids(self) -> list[str]:
        """Source EPANET IDs"""
        assert self.junction_ids is not None
        junc_ids = set(self.junction_ids)
        return [n for n in self.node_ids if n not in junc_ids]

    @property
    def node_pyidxs(self) -> list[int]:
        assert self.graph is not None
        return [self.graph.nodes(data=PY_INDEX)[n] for n in self.node_ids]
    
    @property
    def junction_pyidxs(self) -> list[int]:
        assert self.graph is not None
        assert self.junction_ids is not None
        junc_ids = set(self.junction_ids)
        return [self.graph.nodes(data=PY_INDEX)[n] for n in junc_ids]
    
    @property
    def source_pyidxs(self) -> list[int]:
        junc_pyidxs = set(self.junction_pyidxs)
        return [n for n in self.node_pyidxs if n not in junc_pyidxs]

    def demands(self) -> np.ndarray:
        """Requested demand at every junction, ordered as ``self.junctions``."""
        assert self.graph is not None
        assert self.junction_ids is not None
        return np.array([self.graph.nodes[j][DEMAND] for j in self.junction_ids])

    def total_demand(self) -> float:
        return self._cached("total_demand", lambda: float(self.demands().sum()))

    # -- shortest-path distance matrices (mock implementations) ---------------

    def shortest_path_topological_undirected(self) -> np.ndarray:
        return np.zeros((self.n_nodes, self.n_nodes))

    def shortest_path_topological_directed(self) -> np.ndarray:
        return np.zeros((self.n_nodes, self.n_nodes))

    def shortest_path_length_undirected(self) -> np.ndarray:
        return np.zeros((self.n_nodes, self.n_nodes))

    def shortest_path_length_directed(self) -> np.ndarray:
        return np.zeros((self.n_nodes, self.n_nodes))

    def shortest_path_resistance_undirected(self) -> np.ndarray:
        return np.zeros((self.n_nodes, self.n_nodes))

    def shortest_path_resistance_directed(self) -> np.ndarray:
        return np.zeros((self.n_nodes, self.n_nodes))


class WDNLeakAnalysis(WDNBaseAnalysis):
    """Extends :class:`WDNBaseAnalysis` with the leak-scenario simulations.

    A leak scenario applies an emitter coefficient to a single junction. Every
    (junction, emitter coefficient) combination is simulated, giving
    ``n_leak_scenarios == n_junctions * len(emitter_coefficients)``.

    Stored state (in addition to the base class):

    - ``pressures_per_leak_scenario``: (``n_nodes``, ``n_leak_scenarios``).
    - ``outflows_per_leak_scenario``: (``n_nodes``, ``n_leak_scenarios``).
    - ``leak_magnitudes``: (1, ``n_leak_scenarios``) emitter outflow, L/s.
    - ``leak_scenarios``: list of (junction ID, emitter coefficient) pairs, one
      per column of the matrices above.
    """

    def __init__(self, config_filepath: str):
        super().__init__()  # empty base init
        self.config_filepath = config_filepath

        config = self._load_config(config_filepath)
        inp_filepath = self._resolve_inp_filepath(config_filepath, config[INP_FILEPATH])

        settings = config.get(SETTINGS, {}) or {}
        self.emitter_coefficients = self._build_emitter_coefficients(
            settings.get(EMITTER_COEFFICIENTS, {})
        )
        demand_model = settings.get(DEMAND_MODEL)

        # Build the base data and keep the EPANET handle for the leak sweep.
        ensystem = self._build(inp_filepath, demand_model)
        try:
            self._run_leak_scenarios(ensystem)
        finally:
            ensystem.closeNetwork()

    # -- config -------------------------------------------------------------
    @staticmethod
    def _load_config(config_filepath: str) -> dict:
        with open(config_filepath, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _resolve_inp_filepath(config_filepath: str, inp_filepath: str) -> str:
        """Resolve the .inp path stored in the config relative to the config's
        parent directory."""
        config_dir = os.path.dirname(os.path.abspath(config_filepath))
        return os.path.normpath(os.path.join(config_dir, inp_filepath))

    @staticmethod
    def _build_emitter_coefficients(spec: dict) -> np.ndarray:
        return np.arange(spec.get("start", 1), spec.get("end", 15), spec.get("step", 1))

    # -- leak simulations ---------------------------------------------------
    def _run_leak_scenarios(self, ensystem: epanet) -> None:
        junction_ids: list[str] = ensystem.getNodeJunctionNameID()
        junction_enidxs = ensystem.getNodeJunctionIndex()
        coefficients = self.emitter_coefficients
        emitter_exponent = ensystem.getOptionsEmitterExponent()

        n_scenarios = len(junction_ids) * len(coefficients)
        self.pressures_per_leak_scenario = np.zeros((self.n_nodes, n_scenarios))
        self.outflows_per_leak_scenario = np.zeros((self.n_nodes, n_scenarios))
        self.leak_magnitudes = np.zeros((1, n_scenarios))
        self.leak_scenarios: list[tuple[str, float]] = []

        col = 0
        for j_id, j_enidx in zip(junction_ids, junction_enidxs):
            j_py_idx = j_enidx - 1  # row of this junction in the node-indexed arrays
            for k in coefficients:
                ensystem.setNodeEmitterCoeff(j_enidx, k)
                results = ensystem.getComputedHydraulicTimeSeries()

                pressures = np.asarray(results.Pressure[_MNF, :], dtype=float)
                self.pressures_per_leak_scenario[:, col] = pressures
                self.outflows_per_leak_scenario[:, col] = results.Demand[_MNF, :]
                # Emitter law: q = k * p^gamma (no flow when the pressure is <= 0).
                leak_pressure = max(float(pressures[j_py_idx]), 0.0)
                self.leak_magnitudes[0, col] = k * (leak_pressure ** emitter_exponent)
                self.leak_scenarios.append((j_id, float(k)))

                ensystem.setNodeEmitterCoeff(j_enidx, 0)  # reset the leak
                col += 1

    # -- persistence --------------------------------------------------------
    _CACHE_DIR = pathlib.Path.home() / ".wdn-osp-ll" / "cache" / "hyd_analysis"

    @classmethod
    def _cache_path(cls, config_filepath: str) -> pathlib.Path:
        """``~/.wdn-osp-ll/cache/hyd_analysis/<config_stem>.pkl``."""
        cls._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        stem = pathlib.Path(config_filepath).resolve().stem
        return cls._CACHE_DIR / (stem + ".pkl")

    @staticmethod
    def _config_hash(config_filepath: str) -> str:
        raw = pathlib.Path(config_filepath).resolve().read_bytes()
        return hashlib.sha256(raw).hexdigest()

    def dump(self) -> pathlib.Path:
        """Pickle this analysis (without the lazy cache) to the user cache dir.

        The config file's SHA-256 is stored alongside the state so ``load``
        can detect stale data if the config is edited without renaming it.
        Returns the path written.
        """
        payload = {
            "state": {k: v for k, v in self.__dict__.items() if k != "_cache"},
            "config_hash": self._config_hash(self.config_filepath),
        }
        path = self._cache_path(self.config_filepath)
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, config_filepath: str) -> "WDNLeakAnalysis":
        """Load a previously dumped analysis from the user cache dir.

        Raises ``ValueError`` if the config file has changed since the cache
        was written — edit the config name or pass ``force_recompute=True``
        to ``load_or_compute``.
        """
        path = cls._cache_path(config_filepath)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        stored_hash = payload.get("config_hash")
        if stored_hash is not None and stored_hash != cls._config_hash(config_filepath):
            raise ValueError(
                f"Cached analysis for '{pathlib.Path(config_filepath).stem}' is stale: "
                "the config file changed since the cache was written. "
                "Rename the config or call load_or_compute(..., force_recompute=True)."
            )
        obj = cls.__new__(cls)
        obj.__dict__.update(payload["state"])
        obj._init_cache()
        return obj

    @classmethod
    def load_or_compute(
        cls, config_filepath: str, *, force_recompute: bool = False
    ) -> "WDNLeakAnalysis":
        """Return a cached analysis if one exists and is fresh, otherwise compute and cache it.

        Parameters
        ----------
        config_filepath:
            Path to the ``.yaml`` config (can be relative to CWD).
        force_recompute:
            Ignore any existing cache and re-run the EPANET simulations.
        """
        path = cls._cache_path(config_filepath)
        if not force_recompute and path.exists():
            try:
                return cls.load(config_filepath)
            except ValueError:
                pass  # stale cache — fall through to recompute
        obj = cls(config_filepath)
        obj.dump()
        return obj

    # -- scenario selection -------------------------------------------------
    def select_scenarios(
        self, leak_scenarios: list[tuple[str, float]]
    ) -> "WDNLeakAnalysis":
        """Return a copy restricted to ``leak_scenarios`` (a subset of
        ``self.leak_scenarios``, e.g. a single emitter coefficient).

        Only the per-scenario columns are reduced; the static topology and
        nominal data are shared with the original. The result is a full
        :class:`WDNLeakAnalysis`, so any operation defined on the original works
        on it unchanged. Scenario order follows ``self.leak_scenarios``.
        """
        wanted = set(leak_scenarios)
        cols = [i for i, s in enumerate(self.leak_scenarios) if s in wanted]

        reduced = copy.copy(self)
        reduced._init_cache()  # don't share the lazy cache with the original
        reduced.pressures_per_leak_scenario = self.pressures_per_leak_scenario[:, cols].copy()
        reduced.outflows_per_leak_scenario = self.outflows_per_leak_scenario[:, cols].copy()
        reduced.leak_magnitudes = self.leak_magnitudes[:, cols].copy()
        reduced.leak_scenarios = [self.leak_scenarios[i] for i in cols]
        return reduced

    # -- convenience --------------------------------------------------------
    @property
    def n_leak_scenarios(self) -> int:
        return len(self.leak_scenarios)

    @property
    def sensitivity_matrix(self) -> SensitivityMatrix:
        """Nominal-minus-leak pressure drop over all leak scenarios, as a
        labelled :class:`SensitivityMatrix` (``n_nodes`` x ``n_leak_scenarios``).

        For a scenario subset, reduce first with
        ``self.select_scenarios(...).sensitivity_matrix``; this property is the
        all-scenarios case.
        """
        return self._cached(
            "sensitivity_matrix",
            lambda: SensitivityMatrix.from_analysis(self),
        )

    @property
    def chi_squared_matrix(self) -> ChiSquaredMatrix:
        """Chi-squared normalised pressure drop over all leak scenarios, as a
        labelled :class:`ChiSquaredMatrix` (``n_nodes`` x ``n_leak_scenarios``).

        For a scenario subset, reduce first with
        ``self.select_scenarios(...).chi_squared_matrix``; this property is the
        all-scenarios case.
        """
        return self._cached(
            "chi_squared_matrix",
            lambda: ChiSquaredMatrix.from_analysis(self),
        )

