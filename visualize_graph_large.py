"""
Graph Cluster Visualization - Large Scale Version

Reads a graph file with the format:
- Line 1: "N M" (N nodes, M edges)
- Next M lines: "src dst" (edges)
- Next N lines: cluster assignment for each node (node 0, 1, 2, ...)

Usage: python visualize_graph_large.py <graph_file> [output_image]
"""

import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from collections import defaultdict


def read_graph_file(filepath):
    """Read graph and cluster assignments from file."""
    print(f"Reading {filepath}...")
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')
    
    # Parse header
    n, m = map(int, lines[0].split())
    print(f"  Header: {n} nodes, {m} edges")
    
    # Parse edges
    edges = []
    for i in range(1, m + 1):
        src, dst = map(int, lines[i].split())
        edges.append((src, dst))
    
    # Parse cluster assignments
    clusters = []
    for i in range(m + 1, m + 1 + n):
        clusters.append(int(lines[i]))
    
    print(f"  Loaded {len(edges)} edges and {len(clusters)} cluster assignments")
    return n, m, edges, clusters


def compute_cluster_layout(n, edges, clusters):
    """
    Compute layout based on cluster structure.
    Much faster than spring layout for large graphs.
    
    Strategy:
    1. Place cluster centers in a circle (or grid for many clusters)
    2. Place nodes around their cluster center with small random offset
    """
    unique_clusters = sorted(set(clusters))
    num_clusters = len(unique_clusters)
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    
    # Count nodes per cluster
    cluster_sizes = defaultdict(int)
    for c in clusters:
        cluster_sizes[c] += 1
    
    # Compute cluster center positions
    cluster_centers = {}
    if num_clusters <= 20:
        # Arrange in a circle
        for i, c in enumerate(unique_clusters):
            angle = 2 * np.pi * i / num_clusters
            radius = 10
            cluster_centers[c] = (radius * np.cos(angle), radius * np.sin(angle))
    else:
        # Arrange in a grid
        grid_size = int(np.ceil(np.sqrt(num_clusters)))
        for i, c in enumerate(unique_clusters):
            row = i // grid_size
            col = i % grid_size
            cluster_centers[c] = (col * 3, row * 3)
    
    # Compute node positions
    np.random.seed(42)
    pos = {}
    
    # Track how many nodes placed in each cluster (for spiral placement)
    cluster_node_count = defaultdict(int)
    
    for node in range(n):
        cluster = clusters[node]
        cx, cy = cluster_centers[cluster]
        
        # Place nodes in a spiral pattern around cluster center
        idx = cluster_node_count[cluster]
        cluster_node_count[cluster] += 1
        
        # Spiral parameters
        max_radius = 1.0 + 0.5 * np.log10(cluster_sizes[cluster] + 1)
        angle = idx * 0.5  # Golden angle-ish
        radius = max_radius * np.sqrt(idx / max(cluster_sizes[cluster], 1))
        
        # Add small random jitter
        jitter = 0.1
        x = cx + radius * np.cos(angle) + np.random.uniform(-jitter, jitter)
        y = cy + radius * np.sin(angle) + np.random.uniform(-jitter, jitter)
        
        pos[node] = (x, y)
    
    return pos, cluster_centers


def visualize_graph_large(n, edges, clusters, output_file=None, 
                          show_edges=True, sample_edges=True, max_edges=50000):
    """
    Visualize large graphs efficiently.
    
    For large graphs:
    - Uses cluster-based layout (O(N) instead of O(N²))
    - Samples edges if too many
    - Uses scatter plot instead of networkx drawing
    """
    print("Computing layout...")
    pos, cluster_centers = compute_cluster_layout(n, edges, clusters)
    
    # Get unique clusters and create color mapping
    unique_clusters = sorted(set(clusters))
    num_clusters = len(unique_clusters)
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    
    print(f"  {num_clusters} clusters found")
    
    # Create colormap
    if num_clusters <= 10:
        cmap = cm.get_cmap('tab10')
    elif num_clusters <= 20:
        cmap = cm.get_cmap('tab20')
    else:
        cmap = cm.get_cmap('turbo')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Draw edges (sampled if too many)
    if show_edges:
        edges_to_draw = edges
        if sample_edges and len(edges) > max_edges:
            print(f"  Sampling {max_edges} edges from {len(edges)}...")
            np.random.seed(42)
            indices = np.random.choice(len(edges), max_edges, replace=False)
            edges_to_draw = [edges[i] for i in indices]
        
        print(f"  Drawing {len(edges_to_draw)} edges...")
        
        # Separate intra-cluster and inter-cluster edges
        intra_edges_x = []
        intra_edges_y = []
        inter_edges_x = []
        inter_edges_y = []
        
        for src, dst in edges_to_draw:
            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            
            if clusters[src] == clusters[dst]:
                intra_edges_x.extend([x0, x1, None])
                intra_edges_y.extend([y0, y1, None])
            else:
                inter_edges_x.extend([x0, x1, None])
                inter_edges_y.extend([y0, y1, None])
        
        # Draw inter-cluster edges first (red, more visible)
        if inter_edges_x:
            ax.plot(inter_edges_x, inter_edges_y, 'r-', alpha=0.3, linewidth=0.5, 
                   label=f'Inter-cluster edges')
        
        # Draw intra-cluster edges (gray, less visible)
        if intra_edges_x:
            ax.plot(intra_edges_x, intra_edges_y, 'gray', alpha=0.1, linewidth=0.3)
    
    # Draw nodes using scatter (much faster than networkx)
    print("  Drawing nodes...")
    
    # Group nodes by cluster for efficient drawing
    for cluster_id in unique_clusters:
        node_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
        xs = [pos[i][0] for i in node_indices]
        ys = [pos[i][1] for i in node_indices]
        
        color = cmap(cluster_to_idx[cluster_id] / max(num_clusters - 1, 1))
        
        # Adjust point size based on number of nodes
        if n < 1000:
            size = 50
        elif n < 10000:
            size = 10
        elif n < 100000:
            size = 2
        else:
            size = 0.5
        
        ax.scatter(xs, ys, c=[color], s=size, alpha=0.7, 
                  label=f'Cluster {cluster_id} ({len(node_indices)} nodes)')
    
    # Draw cluster centers with labels
    for cluster_id, (cx, cy) in cluster_centers.items():
        count = clusters.count(cluster_id)
        ax.annotate(f'{cluster_id}', (cx, cy), fontsize=8, ha='center', va='center',
                   bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
    
    # Legend (only show if not too many clusters)
    if num_clusters <= 15:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    
    # Title and formatting
    ax.set_title(f'Graph Visualization\n{n:,} nodes, {len(edges):,} edges, {num_clusters} clusters',
                fontsize=14)
    ax.axis('equal')
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save
    if output_file:
        output_path = output_file
    else:
        output_path = 'graph_visualization.png'
    
    print(f"  Saving to {output_path}...")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Done!")
    
    plt.close()
    return fig


def print_cluster_stats(n, edges, clusters):
    """Print statistics about the clustering."""
    unique_clusters = sorted(set(clusters))
    num_clusters = len(unique_clusters)
    
    print(f"\n=== Cluster Statistics ===")
    print(f"Total nodes: {n:,}")
    print(f"Total edges: {len(edges):,}")
    print(f"Number of clusters: {num_clusters}")
    
    # Cluster sizes
    cluster_sizes = defaultdict(int)
    for c in clusters:
        cluster_sizes[c] += 1
    
    sizes = list(cluster_sizes.values())
    print(f"\nCluster sizes:")
    print(f"  Min: {min(sizes)}")
    print(f"  Max: {max(sizes)}")
    print(f"  Mean: {np.mean(sizes):.1f}")
    print(f"  Median: {np.median(sizes):.1f}")
    
    # Edge statistics
    intra_edges = 0
    inter_edges = 0
    for src, dst in edges:
        if clusters[src] == clusters[dst]:
            intra_edges += 1
        else:
            inter_edges += 1
    
    print(f"\nEdge breakdown:")
    print(f"  Intra-cluster: {intra_edges:,} ({100*intra_edges/len(edges):.1f}%)")
    print(f"  Inter-cluster: {inter_edges:,} ({100*inter_edges/len(edges):.1f}%)")
    
    # Show individual cluster sizes if not too many
    if num_clusters <= 20:
        print(f"\nIndividual cluster sizes:")
        for c in unique_clusters:
            print(f"  Cluster {c}: {cluster_sizes[c]:,} nodes")
    else:
        print(f"\nTop 10 largest clusters:")
        sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: -x[1])[:10]
        for c, size in sorted_clusters:
            print(f"  Cluster {c}: {size:,} nodes")


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_graph_large.py <graph_file> [output_image]")
        print("\nOptions:")
        print("  --no-edges     Don't draw edges (faster for very large graphs)")
        print("  --stats-only   Only print statistics, don't visualize")
        print("\nFile format:")
        print("  Line 1: N M (number of nodes and edges)")
        print("  Next M lines: src dst (edge definitions)")
        print("  Next N lines: cluster assignment for each node")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = None
    show_edges = True
    stats_only = False
    
    # Parse arguments
    for arg in sys.argv[2:]:
        if arg == '--no-edges':
            show_edges = False
        elif arg == '--stats-only':
            stats_only = True
        elif not arg.startswith('--'):
            output_file = arg
    
    # Read graph
    n, m, edges, clusters = read_graph_file(input_file)
    
    # Print statistics
    print_cluster_stats(n, edges, clusters)
    
    # Visualize if requested
    if not stats_only:
        print("\n=== Generating Visualization ===")
        visualize_graph_large(n, edges, clusters, output_file, show_edges=show_edges)


if __name__ == "__main__":
    main()
