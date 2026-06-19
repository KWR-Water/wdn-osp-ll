"""Optimisation result data structures and persistence.

``OSPRunHistory`` holds the full population state at every recorded generation
(useful for convergence plots).  ``OSPResult`` wraps the final answer together
with the config that produced it and provides the same dump/load/load_or_compute
cache interface as ``WDNLeakAnalysis``.

Cache location: ``~/.wdn-osp-ll/cache/opt/<config_hash[:16]>.pkl``
The hash is over ``json.dumps(config, sort_keys=True)`` so changing any field
in the config dict produces a different cache entry.
"""

from __future__ import annotations

import json
import hashlib
import pathlib
import pickle
from dataclasses import dataclass
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@dataclass
class OSPRunHistory:
    """Population snapshots recorded during the archipelago run.

    Shapes
    ------
    generations      : (n_steps,)
    fitness          : (n_steps, n_islands, pop_size, n_obj)
    decisions        : (n_steps, n_islands, pop_size, n_dv)
    hypervolumes     : (n_steps, n_islands) for multi-objective, else None
    """

    generations: np.ndarray
    fitness: np.ndarray
    decisions: np.ndarray
    hypervolumes: np.ndarray | None


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class OSPResult:
    """Final result of an OSP optimisation run.

    For single-objective problems ``best_x`` is (n_dv,) and ``best_f`` is
    (n_obj,).  For multi-objective problems they are the Pareto front:
    ``best_x`` is (n_pareto, n_dv) and ``best_f`` is (n_pareto, n_obj).
    ``is_pareto`` tells you which case you are in.
    """

    best_x: np.ndarray
    best_f: np.ndarray
    is_pareto: bool

    config: dict
    config_hash: str

    history: OSPRunHistory

    # -- persistence --------------------------------------------------------

    _CACHE_DIR = pathlib.Path.home() / ".wdn-osp-ll" / "cache" / "opt"

    @staticmethod
    def hash_config(config: dict) -> str:
        raw = json.dumps(config, sort_keys=True, default=str).encode()
        return hashlib.sha256(raw).hexdigest()

    @classmethod
    def _cache_path(cls, config_hash: str) -> pathlib.Path:
        cls._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return cls._CACHE_DIR / f"{config_hash[:16]}.pkl"

    def dump(self) -> pathlib.Path:
        """Pickle this result to the user cache dir. Returns the path written."""
        path = self._cache_path(self.config_hash)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, config: dict) -> "OSPResult":
        """Load a previously cached result for ``config``.

        Raises ``FileNotFoundError`` if no cache exists, ``ValueError`` if the
        stored hash does not match (should never happen, but guards against
        truncated writes).
        """
        config_hash = cls.hash_config(config)
        path = cls._cache_path(config_hash)
        if not path.exists():
            raise FileNotFoundError(f"No cached result for config hash {config_hash[:16]}")
        with open(path, "rb") as f:
            obj: OSPResult = pickle.load(f)
        if obj.config_hash != config_hash:
            raise ValueError("Cached result hash mismatch — cache may be corrupt.")
        return obj

    @classmethod
    def load_or_compute(
        cls,
        config: dict,
        compute: Callable[[], "OSPResult"],
        *,
        force_recompute: bool = False,
    ) -> "OSPResult":
        """Return a cached result if one exists, otherwise call ``compute()``.

        Parameters
        ----------
        config:
            The full optimisation config dict (used to derive the cache key).
        compute:
            Zero-argument callable that runs the optimisation and returns an
            ``OSPResult``.  Only called on a cache miss or when
            ``force_recompute=True``.
        force_recompute:
            Ignore any existing cache and always re-run.
        """
        if not force_recompute:
            try:
                return cls.load(config)
            except (FileNotFoundError, ValueError):
                pass
        result = compute()
        result.dump()
        return result
