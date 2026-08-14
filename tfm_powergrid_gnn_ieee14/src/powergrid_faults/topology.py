from __future__ import annotations

import networkx as nx
import numpy as np
import torch

# DIgSILENT/PowerFactory IEEE-14 representation used by the public model:
# 16 transmission-line objects + 5 transformer objects. Bus numbers are 1-based here.
# The two first entries are parallel circuits between buses 1 and 2.
FAULT_LINES_1BASED = [
    (1, 2), (1, 2), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (4, 5),
    (6, 11), (6, 12), (6, 13), (9, 10), (9, 14), (10, 11), (12, 13), (13, 14),
]

TRANSFORMERS_1BASED = [(4, 7), (4, 9), (5, 6), (7, 8), (7, 9)]

LINE_NAMES = [
    "1-2 (circuit 1)", "1-2 (circuit 2)", "1-5", "2-3", "2-4", "2-5",
    "3-4", "4-5", "6-11", "6-12", "6-13", "9-10", "9-14", "10-11",
    "12-13", "13-14",
]

GENERATOR_BUSES_1BASED = [1, 2, 3, 6, 8]


def case14_powerfactory_graph():
    """PowerFactory-style IEEE-14 topology.

    Returns a NetworkX MultiGraph, a bidirectional PyG edge_index, and metadata.
    MultiGraph is intentional because buses 1-2 have two parallel line circuits.
    """
    g = nx.MultiGraph()
    g.add_nodes_from(range(14))

    physical_edges = []
    for i, (u1, v1) in enumerate(FAULT_LINES_1BASED):
        u, v = u1 - 1, v1 - 1
        g.add_edge(u, v, kind="line", line_id=i, name=LINE_NAMES[i])
        physical_edges.append((u, v, "line", i))
    for i, (u1, v1) in enumerate(TRANSFORMERS_1BASED):
        u, v = u1 - 1, v1 - 1
        g.add_edge(u, v, kind="transformer", transformer_id=i)
        physical_edges.append((u, v, "transformer", i))

    directed = []
    for u, v, _, _ in physical_edges:
        directed.extend([(u, v), (v, u)])
    edge_index = torch.tensor(np.asarray(directed).T, dtype=torch.long)

    meta = {
        "fault_lines": [(u - 1, v - 1) for u, v in FAULT_LINES_1BASED],
        "line_names": LINE_NAMES,
        "transformers": [(u - 1, v - 1) for u, v in TRANSFORMERS_1BASED],
        "generator_buses": [b - 1 for b in GENERATOR_BUSES_1BASED],
    }
    return g, edge_index, meta


def simple_graph(g: nx.MultiGraph) -> nx.Graph:
    out = nx.Graph()
    out.add_nodes_from(g.nodes)
    out.add_edges_from((u, v) for u, v in g.edges())
    return out


def topological_positions(g: nx.Graph | nx.MultiGraph):
    sg = simple_graph(g) if isinstance(g, nx.MultiGraph) else g
    pos = nx.kamada_kawai_layout(sg)
    return {int(k): (float(v[0]), float(v[1])) for k, v in pos.items()}
