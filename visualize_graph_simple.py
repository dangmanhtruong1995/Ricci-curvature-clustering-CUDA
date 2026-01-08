"""
Simple Graph Visualization

Reads a graph file with the format:
- Line 1: "N M" (N nodes, M edges)
- Next M lines: "src dst" (edges)

Usage: python visualize_graph_simple.py <graph_file> [output_image]
"""

import sys
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def read_graph_file(filepath):
    """Read graph from file."""
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')
    
    # Parse header
    n, m = map(int, lines[0].split())
    
    # Parse edges
    edges = []
    for i in range(1, m + 1):
        src, dst = map(int, lines[i].split())
        edges.append((src, dst))
    
    return n, m, edges


def visualize_graph(n, edges, output_file=None):
    """Visualize the graph."""
    # Create graph
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Compute layout
    if n < 100:
        pos = nx.spring_layout(G, k=2/np.sqrt(n) if n > 1 else 1, iterations=50, seed=42)
    else:
        pos = nx.spring_layout(G, k=1/np.sqrt(n), iterations=30, seed=42)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color='gray', ax=ax)
    
    # Draw nodes
    node_size = max(300 - n * 2, 20)  # Smaller nodes for larger graphs
    nx.draw_networkx_nodes(G, pos, node_color='steelblue', 
                           node_size=node_size, alpha=0.8, ax=ax)
    
    # Draw labels for smaller graphs
    if n <= 50:
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    
    ax.set_title(f'Graph Visualization\n{n} nodes, {len(edges)} edges', fontsize=14)
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
        print("Usage: python visualize_graph_simple.py <graph_file> [output_image]")
        print("\nFile format:")
        print("  Line 1: N M (number of nodes and edges)")
        print("  Next M lines: src dst (edge definitions)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Read and visualize
    n, m, edges = read_graph_file(input_file)
    print(f"Loaded graph: {n} nodes, {m} edges")
    
    visualize_graph(n, edges, output_file)


if __name__ == "__main__":
    main()
