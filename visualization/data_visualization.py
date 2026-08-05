import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def plot_grid_topology(edge_index):
    G = nx.Graph()
    edges = edge_index.t().numpy()
    G.add_edges_from(edges)
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    
    nx.draw_networkx_nodes(G, pos, node_color='#3498db', node_size=900, edgecolors='#2c3e50', linewidths=2)
    nx.draw_networkx_edges(G, pos, width=2.5, alpha=0.6, edge_color='#7f8c8d')
    nx.draw_networkx_labels(G, pos, font_size=14, font_color='white', font_weight='bold')
    
    plt.title("Topología de la Red Eléctrica de Transporte", fontsize=16, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_comparative_metrics():
    modelos = ['CNN 1D', 'LSTM', 'ST-GNN (GAT)']
    accuracy = [0.81, 0.86, 0.96]
    f1_score = [0.78, 0.85, 0.95]
    
    x = np.arange(len(modelos))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(9, 6))
    rects1 = ax.bar(x - width/2, accuracy, width, label='Accuracy', color='#2ecc71', edgecolor='black')
    rects2 = ax.bar(x + width/2, f1_score, width, label='F1-Score', color='#9b59b6', edgecolor='black')
    
    ax.set_ylabel('Rendimiento (0.0 - 1.0)', fontsize=12)
    ax.set_title('Estudio Comparativo: Precisión en Detección', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(modelos, fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
