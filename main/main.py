import torch
from models import ST_GAT, BaselineCNN, BaselineLSTM
from dataset import build_power_grid_graph
from visualization import plot_grid_topology, plot_comparative_metrics

def main():
    print("1. Cargando topología de la red")
    node_features, edge_index = build_power_grid_graph()
    
    print("2. Topologías y métricas comparativas")
    plot_grid_topology(edge_index)
    plot_comparative_metrics()
    
    print("3. Inicializando modelo ST-GNN")
    modelo_gnn = ST_GAT(in_channels=3, hidden_channels=32, num_classes=2)
    modelo_gnn.eval()
    
    print("4. Ejecutando inferencia (Spatio-Temporal GNN)...")
    with torch.no_grad():
        out = modelo_gnn(node_features, edge_index)
        prediction = out.argmax(dim=1).item()
        estado = 'Fallo Crítico Localizado' if prediction == 1 else 'Operación Nominal'
        print(f"-> Diagnóstico de la red (ST-GNN): {estado}")

if __name__ == '__main__':
    main()