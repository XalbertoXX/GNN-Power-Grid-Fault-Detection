import torch

def build_power_grid_graph():
    """ grafo simulando la red eléctrica"""
    edges = torch.tensor([
        [0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 9]
    ], dtype=torch.long)
    
    edge_index = torch.cat([edges, edges[[1, 0]]], dim=1)
    # (nodos, secuencia_temporal, caracteristicas_V_I_F)
    node_features = torch.randn((10, 5, 3)) 
    return node_features, edge_index