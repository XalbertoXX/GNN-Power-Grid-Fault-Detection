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

# Function to create a simple undirected graph from a MultiGraph by removing parallel edges. 
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

# Function to create a simple undirected graph from a MultiGraph by removing parallel edges. This is useful for certain graph algorithms that require a simple graph representation.
def simple_graph(g: nx.MultiGraph) -> nx.Graph:
    out = nx.Graph()
    out.add_nodes_from(g.nodes)
    out.add_edges_from((u, v) for u, v in g.edges())
    return out

# The `topological_positions` function computes the positions of nodes in a graph using the Kamada-Kawai layout algorithm.
def topological_positions(g: nx.Graph | nx.MultiGraph):
    sg = simple_graph(g) if isinstance(g, nx.MultiGraph) else g
    pos = nx.kamada_kawai_layout(sg)
    return {int(k): (float(v[0]), float(v[1])) for k, v in pos.items()}

# The function generates a new edge index that maintains the same number of buses, edges, and degree distribution as the original graph while shuffling the connections. 
# It ensures that the resulting graph remains connected and does not introduce excessive multiplicities beyond what is present in the original topology. 
def degree_preserving_shuffled_edge_index(
    seed: int = 20260821,
    n_swaps: int = 250,
):
    """Create a connected shuffled IEEE-14 topology for ablation.

    The shuffled graph preserves:
    - the 14 buses,
    - the number of physical edge objects,
    - the degree of every bus,
    - graph connectivity.

    Only the wiring is changed. Parallel edges are allowed because the original
    PowerFactory-style graph also contains a parallel 1-2 circuit.

    Returns
    -------
    edge_index : torch.LongTensor
        Bidirectional PyG edge index with the same number of directed edges
        as the real topology.
    metadata : dict
        Reproducibility information about the shuffle.
    """
    from collections import Counter

    rng = np.random.default_rng(seed)

    # Treat transmission lines and transformers identically for message passing:
    # the ST-GAT currently consumes only connectivity, not edge attributes.
    original = [
        tuple(sorted((u - 1, v - 1)))
        for u, v in (FAULT_LINES_1BASED + TRANSFORMERS_1BASED)
    ]
    edges = list(original)

    def connected(edge_list):
        g = nx.MultiGraph()
        g.add_nodes_from(range(14))
        g.add_edges_from(edge_list)
        return nx.is_connected(g)

    successful = 0
    attempts = 0
    max_attempts = max(10000, n_swaps * 200)

    while successful < n_swaps and attempts < max_attempts:
        attempts += 1
        i, j = rng.choice(len(edges), size=2, replace=False)
        a, b = edges[i]
        c, d = edges[j]

        # Four distinct endpoints make a clean degree-preserving double-edge swap.
        if len({a, b, c, d}) < 4:
            continue

        if rng.random() < 0.5:
            cand1 = tuple(sorted((a, d)))
            cand2 = tuple(sorted((c, b)))
        else:
            cand1 = tuple(sorted((a, c)))
            cand2 = tuple(sorted((b, d)))

        if cand1[0] == cand1[1] or cand2[0] == cand2[1]:
            continue

        old_pair = sorted([edges[i], edges[j]])
        new_pair = sorted([cand1, cand2])
        if old_pair == new_pair:
            continue

        proposal = list(edges)
        proposal[i] = cand1
        proposal[j] = cand2

        # Do not create multiplicities larger than those already reasonable for
        # this MultiGraph (the real topology contains a double circuit).
        if max(Counter(proposal).values()) > 2:
            continue

        if not connected(proposal):
            continue

        edges = proposal
        successful += 1

    if successful < n_swaps:
        raise RuntimeError(
            f"Could only perform {successful}/{n_swaps} valid topology swaps"
        )

    directed = []
    for u, v in edges:
        directed.extend([(u, v), (v, u)])
    edge_index = torch.tensor(np.asarray(directed).T, dtype=torch.long)

    g_real = nx.MultiGraph()
    g_real.add_nodes_from(range(14))
    g_real.add_edges_from(original)

    g_shuffled = nx.MultiGraph()
    g_shuffled.add_nodes_from(range(14))
    g_shuffled.add_edges_from(edges)

    original_counter = Counter(original)
    shuffled_counter = Counter(edges)
    overlap = sum(
        min(original_counter[e], shuffled_counter[e])
        for e in set(original_counter) | set(shuffled_counter)
    )

    metadata = {
        "shuffle_seed": int(seed),
        "requested_swaps": int(n_swaps),
        "successful_swaps": int(successful),
        "n_buses": 14,
        "n_undirected_edge_objects": len(edges),
        "n_directed_edges": int(edge_index.shape[1]),
        "connected": bool(nx.is_connected(g_shuffled)),
        "degree_sequence_real": [int(g_real.degree(n)) for n in range(14)],
        "degree_sequence_shuffled": [int(g_shuffled.degree(n)) for n in range(14)],
        "edge_object_overlap_with_real": int(overlap),
        "edge_object_overlap_fraction": float(overlap / len(original)),
        "real_edges_0based": [list(e) for e in original],
        "shuffled_edges_0based": [list(e) for e in edges],
    }
    return edge_index, metadata
