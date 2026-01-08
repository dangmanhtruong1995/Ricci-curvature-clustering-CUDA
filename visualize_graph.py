"""
Graph Cluster Visualization

Reads a graph file with the format:
- Line 1: "N M" (N nodes, M edges)
- Next M lines: "src dst" (edges)
- Next N lines: cluster assignment for each node (node 0, 1, 2, ...)

Usage: python visualize_graph.py <graph_file> [output_image]
"""

import sys
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import numpy as np


def read_graph_file(filepath):
    """Read graph and cluster assignments from file."""
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')
    
    # Parse header
    n, m = map(int, lines[0].split())
    
    # Parse edges
    edges = []
    for i in range(1, m + 1):
        src, dst = map(int, lines[i].split())
        edges.append((src, dst))
    
    # Parse cluster assignments
    clusters = []
    for i in range(m + 1, m + 1 + n):
        clusters.append(int(lines[i]))
    
    return n, m, edges, clusters


def visualize_graph(n, edges, clusters, output_file=None):
    """Visualize the graph with nodes colored by cluster."""
    # Create graph
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    
    # Get unique clusters and create color mapping
    unique_clusters = sorted(set(clusters))
    num_clusters = len(unique_clusters)
    
    # Create colormap (using Set1/Set3 to avoid gray which matches edge color)
    if num_clusters <= 9:
        cmap = cm.get_cmap('Set1')
    elif num_clusters <= 12:
        cmap = cm.get_cmap('Set3')
    else:
        cmap = cm.get_cmap('turbo')
    
    # Map cluster IDs to colors
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    node_colors = [cmap(cluster_to_idx[clusters[node]] / max(num_clusters - 1, 1)) 
                   for node in range(n)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Compute layout
    if n < 100:
        pos = nx.spring_layout(G, k=2/np.sqrt(n) if n > 1 else 1, iterations=50, seed=42)
    else:
        pos = nx.spring_layout(G, k=1/np.sqrt(n), iterations=30, seed=42)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', ax=ax)
    
    # Draw nodes
    node_size = max(300 - n * 2, 20)  # Smaller nodes for larger graphs
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                           node_size=node_size, ax=ax)
    
    # Draw labels for smaller graphs
    if n <= 50:
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    
    # Create legend
    legend_elements = []
    for cluster_id in unique_clusters:
        color = cmap(cluster_to_idx[cluster_id] / max(num_clusters - 1, 1))
        count = clusters.count(cluster_id)
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=color, markersize=10,
                      label=f'Cluster {cluster_id} ({count} nodes)')
        )
    
    # Position legend outside plot if many clusters
    if num_clusters <= 10:
        ax.legend(handles=legend_elements, loc='best', fontsize=9)
    else:
        ax.legend(handles=legend_elements, loc='center left', 
                 bbox_to_anchor=(1, 0.5), fontsize=8)
    
    ax.set_title(f'Graph Visualization\n{n} nodes, {len(edges)} edges, {num_clusters} clusters',
                fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_file}")
    else:
        plt.savefig('graph_visualization.png', dpi=150, bbox_inches='tight')
        print("Saved visualization to graph_visualization.png")
    
    plt.close()
    return fig


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_graph.py <graph_file> [output_image]")
        print("\nFile format:")
        print("  Line 1: N M (number of nodes and edges)")
        print("  Next M lines: src dst (edge definitions)")
        print("  Next N lines: cluster assignment for each node")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Read and visualize
    n, m, edges, clusters = read_graph_file(input_file)
    print(f"Loaded graph: {n} nodes, {m} edges, {len(set(clusters))} clusters")
    
    visualize_graph(n, edges, clusters, output_file)


if __name__ == "__main__":
    main()
