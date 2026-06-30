"""WDN graph schema, construction, and topology-based distance matrices.

This module owns the ``networkx.DiGraph`` attribute-key constants and the
``build_graph`` function so that ``data_structures.py`` can import from here
without a circular dependency.
"""

import numpy as np
import networkx as nx
from epyt import epanet

# --- graph attribute keys --------------------------------------------------
# node attributes
PY_INDEX = "py_index"   # 0-based python index; EPANET index is py_index + 1
ELEVATION = "elevation"
DEMAND = "demand"       # requested demand
OUTFLOW = "outflow"     # delivered demand
# pipe (edge) attributes
ID = "id"               # EPANET pipe ID (the edge itself is keyed by node IDs)
LENGTH = "length"
DIAMETER = "diameter"   # stored in mm, as read from EPANET
ROUGHNESS = "roughness"
FLOW = "flow"           # absolute flow at MNF (edge orientation carries the sign)


def build_graph(
    ensystem: epanet, outflows: np.ndarray, flows: np.ndarray
) -> nx.DiGraph:
    """Build a ``DiGraph`` from a live EPANET handle and pre-computed flow arrays.

    Nodes carry ``PY_INDEX``, ``ELEVATION``, ``DEMAND``, and ``OUTFLOW``.
    Edges carry ``ID``, ``PY_INDEX``, ``LENGTH``, ``DIAMETER``, ``ROUGHNESS``,
    and ``FLOW`` (absolute). Edge orientation follows the actual flow direction
    at MNF (EPANET's start→end for positive flow, reversed for negative).
    """
    node_ids = ensystem.getNodeNameID()
    elevations = np.asarray(ensystem.getNodeElevations(), dtype=float)
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


# --- shortest-path helpers -------------------------------------------------

def _hw_resistance_weight(u, v, data: dict) -> float:
    """Edge weight callable for Hazen-Williams hydraulic resistance (metres).

    Diameter is stored in mm; the HW formula requires metres.
    """
    d_m = data[DIAMETER] / 1000.0
    return (data[LENGTH] * 10.67) / ((data[ROUGHNESS] ** 1.852) * (d_m ** 4.8704))


def shortest_path_matrix(
    graph: nx.DiGraph,
    *,
    directed: bool,
    weight,
) -> np.ndarray:
    """All-pairs shortest-path distances as an ``(n_nodes, n_nodes)`` matrix.

    Parameters
    ----------
    graph:
        The ``WDNBaseAnalysis.graph`` DiGraph.
    directed:
        Use the directed graph as-is when ``True``; convert to undirected when
        ``False``.
    weight:
        Passed to ``nx.all_pairs_dijkstra_path_length``. ``None`` → hop count,
        a string → that edge-attribute name, a callable ``(u, v, data) → float``
        → computed on the fly.
    """
    n = graph.number_of_nodes()
    distances = np.full((n, n), np.inf)
    np.fill_diagonal(distances, 0.0)

    g: nx.Graph = graph if directed else graph.to_undirected()

    for source, targets in nx.all_pairs_dijkstra_path_length(g, weight=weight):
        src_idx = graph.nodes[source][PY_INDEX]
        for target, dist in targets.items():
            tgt_idx = graph.nodes[target][PY_INDEX]
            distances[src_idx, tgt_idx] = dist

    return distances

def common_path_matrix(
    graph: nx.DiGraph,
    *,
    directed: bool,
    weight,
    source_node_ids: list[str],
) -> np.ndarray:
    """Common-path resistance matrix: entry [j, k] is the weight of the shared
    path segment between node j and node k on their respective shortest paths
    to j's nearest source.

    Only junction rows/columns are written; source entries stay at 0.

    Parameters
    ----------
    graph:
        The ``WDNBaseAnalysis.graph`` DiGraph.
    directed:
        Use the directed graph when ``True``; undirected otherwise.
    weight:
        Same semantics as ``shortest_path_matrix``.
    source_node_ids:
        EPANET IDs of the source nodes (reservoirs).
    """
    n = graph.number_of_nodes()
    result = np.zeros((n, n))

    g: nx.Graph = graph if directed else graph.to_undirected()

    all_pairs: dict = dict(nx.all_pairs_dijkstra(g, weight=weight))

    source_set = set(source_node_ids)
    junction_ids = [nid for nid in graph.nodes if nid not in source_set]

    for j in junction_ids:
        j_idx = graph.nodes[j][PY_INDEX]

        # Find nearest source using paths FROM the source (src→j), which is
        # the flow direction and the only valid direction in directed graphs.
        closest_source = min(
            source_set,
            key=lambda s: all_pairs[s][0].get(j, np.inf),
        )
        src_dists, src_paths = all_pairs[closest_source]
        src_to_j = src_paths.get(j)
        if src_to_j is None:
            continue  # no source can reach j

        for k in junction_ids:
            k_idx = graph.nodes[k][PY_INDEX]
            src_to_k = src_paths.get(k)
            if src_to_k is None:
                continue  # same source can't reach k → zero common path

            # Common prefix of src→j and src→k.
            common_len = 0
            for a, b in zip(src_to_j, src_to_k):
                if a == b:
                    common_len += 1
                else:
                    break

            if common_len == 0:
                continue

            # Last node in the common prefix is the branching point.
            branching_point = src_to_j[common_len - 1]
            result[j_idx, k_idx] = src_dists.get(branching_point, 0.0)

    return result
