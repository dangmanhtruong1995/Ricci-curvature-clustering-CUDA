"""
Graph Cluster Visualization - Improved Version
Shows ALL edges (both intra and inter-cluster)
"""

import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import colorsys


def read_graph_file(filepath):
    print(f"Reading {filepath}...")
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')
    
    n, m = map(int, lines[0].split())
    print(f"  Header: {n} nodes, {m} edges")
    
    edges = []
    for i in range(1, m + 1):
        src, dst = map(int, lines[i].split())
        edges.append((src, dst))
    
    clusters = []
    for i in range(m + 1, m + 1 + n):
        clusters.append(int(lines[i]))
    
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


def compute_cluster_layout(n, edges, clusters, spacing_factor=1.5):
    unique_clusters = sorted(set(clusters))
    num_clusters = len(unique_clusters)
    
    cluster_sizes = defaultdict(int)
    cluster_nodes = defaultdict(list)
    for i, c in enumerate(clusters):
        cluster_sizes[c] += 1
        cluster_nodes[c].append(i)
    
    cluster_radii = {c: np.sqrt(cluster_sizes[c]) * 0.1 for c in unique_clusters}
    max_radius = max(cluster_radii.values())
    
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
    
    np.random.seed(42)
    pos = {}
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
            x = cx + r * np.cos(theta) + np.random.uniform(-jitter, jitter)
            y = cy + r * np.sin(theta) + np.random.uniform(-jitter, jitter)
            pos[node] = (x, y)
    
    return pos, cluster_centers, cluster_radii


def visualize_graph(n, edges, clusters, output_file=None, max_edges=50000, spacing_factor=1.5):
    print("Computing layout...")
    pos, cluster_centers, cluster_radii = compute_cluster_layout(n, edges, clusters, spacing_factor)
    
    unique_clusters = sorted(set(clusters))
    num_clusters = len(unique_clusters)
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    colors = generate_distinct_colors(num_clusters)
    
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.set_facecolor('#fafafa')
    
    # Separate edges
    intra_edges = [(s, d) for s, d in edges if clusters[s] == clusters[d]]
    inter_edges = [(s, d) for s, d in edges if clusters[s] != clusters[d]]
    
    # Sample if needed
    sampled_intra = len(intra_edges) > max_edges
    sampled_inter = len(inter_edges) > max_edges
    
    if sampled_intra:
        np.random.seed(42)
        idx = np.random.choice(len(intra_edges), max_edges, replace=False)
        intra_edges = [intra_edges[i] for i in idx]
    
    if sampled_inter:
        np.random.seed(43)
        idx = np.random.choice(len(inter_edges), max_edges, replace=False)
        inter_edges = [inter_edges[i] for i in idx]
    
    # Draw INTER-cluster edges first (red)
    print(f"  Drawing {len(inter_edges)} inter-cluster edges...")
    inter_x, inter_y = [], []
    for src, dst in inter_edges:
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        inter_x.extend([x0, x1, None])
        inter_y.extend([y0, y1, None])
    
    if inter_x:
        label = f'Inter-cluster ({len(inter_edges):,}' + (' sampled)' if sampled_inter else ')')
        ax.plot(inter_x, inter_y, color='#dd5555', alpha=0.35, linewidth=0.5, label=label)
    
    # Draw INTRA-cluster edges (gray)
    print(f"  Drawing {len(intra_edges)} intra-cluster edges...")
    intra_x, intra_y = [], []
    for src, dst in intra_edges:
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        intra_x.extend([x0, x1, None])
        intra_y.extend([y0, y1, None])
    
    if intra_x:
        label = f'Intra-cluster ({len(intra_edges):,}' + (' sampled)' if sampled_intra else ')')
        ax.plot(intra_x, intra_y, color='#555555', alpha=0.25, linewidth=0.4, label=label)
    
    # Draw nodes
    print("  Drawing nodes...")
    size = 30 if n < 1000 else (8 if n < 10000 else (1.5 if n < 100000 else 0.3))
    
    for cluster_id in unique_clusters:
        node_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
        xs = [pos[i][0] for i in node_indices]
        ys = [pos[i][1] for i in node_indices]
        color = colors[cluster_to_idx[cluster_id]]
        ax.scatter(xs, ys, c=[color], s=size, alpha=0.8, edgecolors='none')
    
    # Cluster labels
    for cluster_id, (cx, cy) in cluster_centers.items():
        ax.annotate(f'{cluster_id}', (cx, cy), fontsize=7, ha='center', va='center',
                    bbox=dict(boxstyle='circle,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.9))
    
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title(f'Graph Visualization\n{n:,} nodes, {len(edges):,} edges, {num_clusters} clusters', fontsize=14, fontweight='bold')
    ax.axis('equal')
    ax.axis('off')
    plt.tight_layout()
    
    output_path = output_file or 'graph_improved.png'
    print(f"  Saving to {output_path}...")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Done!")


def print_stats(n, edges, clusters):
    unique_clusters = sorted(set(clusters))
    print(f"\n=== Statistics ===")
    print(f"Nodes: {n:,}, Edges: {len(edges):,}, Clusters: {len(unique_clusters)}")
    
    intra = sum(1 for s, d in edges if clusters[s] == clusters[d])
    inter = len(edges) - intra
    print(f"Intra-cluster: {intra:,} ({100*intra/len(edges):.1f}%)")
    print(f"Inter-cluster: {inter:,} ({100*inter/len(edges):.1f}%)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_graph_improved.py <graph_file> [output.png] [--spacing=X]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = None
    spacing = 1.5
    
    for arg in sys.argv[2:]:
        if arg.startswith('--spacing='):
            spacing = float(arg.split('=')[1])
        elif not arg.startswith('--'):
            output_file = arg
    
    n, m, edges, clusters = read_graph_file(input_file)
    print_stats(n, edges, clusters)
    print("\n=== Generating Visualization ===")
    visualize_graph(n, edges, clusters, output_file, spacing_factor=spacing)


if __name__ == "__main__":
    main()
