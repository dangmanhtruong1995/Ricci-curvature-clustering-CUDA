"""
Graph Cluster Visualization - Memory Efficient Version
Optimized for 100k+ nodes and millions of edges
"""

import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import colorsys
import gc


def read_graph_file(filepath):
    """Memory-efficient graph reading."""
    print(f"Reading {filepath}...")
    
    with open(filepath, 'r') as f:
        header = f.readline().strip()
        n, m = map(int, header.split())
        print(f"  Header: {n} nodes, {m} edges")
        
        # Read edges
        edges = []
        for _ in range(m):
            line = f.readline().strip()
            src, dst = map(int, line.split())
            edges.append((src, dst))
        
        # Read clusters
        clusters = []
        for _ in range(n):
            clusters.append(int(f.readline().strip()))
    
    return n, m, edges, clusters


def generate_distinct_colors(n_colors):
    colors = []
    golden_ratio = 0.618033988749895
    h = 0.0
    for i in range(n_colors):
        h = (h + golden_ratio) % 1.0
        s = 0.65 + 0.35 * ((i % 3) / 2)
        v = 0.75 + 0.25 * ((i % 5) / 4)
        rgb = colorsys.hsv_to_rgb(h, s, v)
        colors.append(rgb)
    return colors


def compute_cluster_layout(n, clusters, spacing_factor=1.5):
    """Compute layout without needing edges."""
    unique_clusters = sorted(set(clusters))
    num_clusters = len(unique_clusters)
    
    # Count nodes per cluster
    cluster_sizes = defaultdict(int)
    cluster_nodes = defaultdict(list)
    for i, c in enumerate(clusters):
        cluster_sizes[c] += 1
        cluster_nodes[c].append(i)
    
    # Compute radii
    cluster_radii = {c: np.sqrt(cluster_sizes[c]) * 0.1 for c in unique_clusters}
    max_radius = max(cluster_radii.values())
    
    # Compute cluster centers
    cluster_centers = {}
    if num_clusters <= 20:
        circle_radius = max(max_radius * num_clusters * spacing_factor / (2 * np.pi), 15)
        for i, c in enumerate(unique_clusters):
            angle = 2 * np.pi * i / num_clusters
            cluster_centers[c] = (circle_radius * np.cos(angle), circle_radius * np.sin(angle))
    else:
        grid_size = int(np.ceil(np.sqrt(num_clusters)))
        grid_spacing = max(2 * max_radius * spacing_factor, 4)
        for i, c in enumerate(unique_clusters):
            row, col = i // grid_size, i % grid_size
            cluster_centers[c] = (col * grid_spacing, row * grid_spacing)
    
    # Compute node positions using numpy arrays (more memory efficient)
    print("  Computing node positions...")
    pos_x = np.zeros(n, dtype=np.float32)
    pos_y = np.zeros(n, dtype=np.float32)
    
    np.random.seed(42)
    golden_angle = np.pi * (3 - np.sqrt(5))
    
    for cluster_id in unique_clusters:
        nodes = cluster_nodes[cluster_id]
        cx, cy = cluster_centers[cluster_id]
        n_nodes = len(nodes)
        radius = cluster_radii[cluster_id]
        
        for idx, node in enumerate(nodes):
            r = radius * np.sqrt(idx / n_nodes) if n_nodes > 1 else 0
            theta = idx * golden_angle
            jitter = radius * 0.02
            pos_x[node] = cx + r * np.cos(theta) + np.random.uniform(-jitter, jitter)
            pos_y[node] = cy + r * np.sin(theta) + np.random.uniform(-jitter, jitter)
    
    return pos_x, pos_y, cluster_centers, cluster_radii, cluster_nodes


def sample_edges(edges, clusters, max_per_type=50000):
    """Sample edges efficiently."""
    print("  Sampling edges...")
    
    intra_indices = []
    inter_indices = []
    
    for i, (src, dst) in enumerate(edges):
        if clusters[src] == clusters[dst]:
            intra_indices.append(i)
        else:
            inter_indices.append(i)
    
    n_intra = len(intra_indices)
    n_inter = len(inter_indices)
    
    print(f"    Total intra-cluster: {n_intra:,}")
    print(f"    Total inter-cluster: {n_inter:,}")
    
    # Sample
    np.random.seed(42)
    if n_intra > max_per_type:
        intra_indices = list(np.random.choice(intra_indices, max_per_type, replace=False))
    if n_inter > max_per_type:
        inter_indices = list(np.random.choice(inter_indices, max_per_type, replace=False))
    
    sampled_intra = [edges[i] for i in intra_indices]
    sampled_inter = [edges[i] for i in inter_indices]
    
    print(f"    Sampled intra: {len(sampled_intra):,}")
    print(f"    Sampled inter: {len(sampled_inter):,}")
    
    return sampled_intra, sampled_inter, n_intra, n_inter


def visualize_graph(n, edges, clusters, output_file=None, max_edges=50000, spacing_factor=1.5):
    """Memory-efficient visualization."""
    
    # Compute layout
    print("Computing layout...")
    pos_x, pos_y, cluster_centers, cluster_radii, cluster_nodes = compute_cluster_layout(
        n, clusters, spacing_factor
    )
    
    unique_clusters = sorted(set(clusters))
    num_clusters = len(unique_clusters)
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    colors = generate_distinct_colors(num_clusters)
    
    # Sample edges
    sampled_intra, sampled_inter, total_intra, total_inter = sample_edges(
        edges, clusters, max_edges
    )
    
    # Free original edges from memory
    del edges
    gc.collect()
    
    # Create figure
    print("Creating figure...")
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.set_facecolor('#fafafa')
    
    # Draw inter-cluster edges (red) - draw first so they're behind
    print(f"  Drawing {len(sampled_inter)} inter-cluster edges...")
    if sampled_inter:
        # Build line segments efficiently
        n_edges = len(sampled_inter)
        inter_x = np.empty(n_edges * 3, dtype=np.float32)
        inter_y = np.empty(n_edges * 3, dtype=np.float32)
        
        for i, (src, dst) in enumerate(sampled_inter):
            idx = i * 3
            inter_x[idx] = pos_x[src]
            inter_x[idx + 1] = pos_x[dst]
            inter_x[idx + 2] = np.nan
            inter_y[idx] = pos_y[src]
            inter_y[idx + 1] = pos_y[dst]
            inter_y[idx + 2] = np.nan
        
        label = f'Inter-cluster ({len(sampled_inter):,}/{total_inter:,})'
        ax.plot(inter_x, inter_y, color='#dd5555', alpha=0.35, linewidth=0.5, label=label)
        
        del inter_x, inter_y, sampled_inter
        gc.collect()
    
    # Draw intra-cluster edges (gray)
    print(f"  Drawing {len(sampled_intra)} intra-cluster edges...")
    if sampled_intra:
        n_edges = len(sampled_intra)
        intra_x = np.empty(n_edges * 3, dtype=np.float32)
        intra_y = np.empty(n_edges * 3, dtype=np.float32)
        
        for i, (src, dst) in enumerate(sampled_intra):
            idx = i * 3
            intra_x[idx] = pos_x[src]
            intra_x[idx + 1] = pos_x[dst]
            intra_x[idx + 2] = np.nan
            intra_y[idx] = pos_y[src]
            intra_y[idx + 1] = pos_y[dst]
            intra_y[idx + 2] = np.nan
        
        label = f'Intra-cluster ({len(sampled_intra):,}/{total_intra:,})'
        ax.plot(intra_x, intra_y, color='#555555', alpha=0.25, linewidth=0.4, label=label)
        
        del intra_x, intra_y, sampled_intra
        gc.collect()
    
    # Draw nodes by cluster
    print("  Drawing nodes...")
    size = 30 if n < 1000 else (8 if n < 10000 else (2 if n < 100000 else 0.5))
    
    for cluster_id in unique_clusters:
        nodes = cluster_nodes[cluster_id]
        xs = pos_x[nodes]
        ys = pos_y[nodes]
        color = colors[cluster_to_idx[cluster_id]]
        ax.scatter(xs, ys, c=[color], s=size, alpha=0.8, edgecolors='none')
    
    # Cluster labels
    for cluster_id, (cx, cy) in cluster_centers.items():
        ax.annotate(f'{cluster_id}', (cx, cy), fontsize=7, ha='center', va='center',
                    bbox=dict(boxstyle='circle,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.9))
    
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title(f'Graph Visualization\n{n:,} nodes, {total_intra + total_inter:,} edges, {num_clusters} clusters',
                 fontsize=14, fontweight='bold')
    ax.axis('equal')
    ax.axis('off')
    plt.tight_layout()
    
    # Save
    output_path = output_file or 'graph_output.png'
    print(f"  Saving to {output_path}...")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    gc.collect()
    print("  Done!")


def print_stats(n, edges, clusters):
    unique_clusters = sorted(set(clusters))
    print(f"\n=== Statistics ===")
    print(f"Nodes: {n:,}, Edges: {len(edges):,}, Clusters: {len(unique_clusters)}")
    
    # Count without creating new lists
    intra = sum(1 for s, d in edges if clusters[s] == clusters[d])
    inter = len(edges) - intra
    print(f"Intra-cluster: {intra:,} ({100*intra/len(edges):.1f}%)")
    print(f"Inter-cluster: {inter:,} ({100*inter/len(edges):.1f}%)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_large.py <graph_file> [output.png] [--spacing=X] [--max-edges=N]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = None
    spacing = 1.8
    max_edges = 50000
    
    for arg in sys.argv[2:]:
        if arg.startswith('--spacing='):
            spacing = float(arg.split('=')[1])
        elif arg.startswith('--max-edges='):
            max_edges = int(arg.split('=')[1])
        elif not arg.startswith('--'):
            output_file = arg
    
    n, m, edges, clusters = read_graph_file(input_file)
    print_stats(n, edges, clusters)
    print("\n=== Generating Visualization ===")
    visualize_graph(n, edges, clusters, output_file, max_edges=max_edges, spacing_factor=spacing)


if __name__ == "__main__":
    main()
