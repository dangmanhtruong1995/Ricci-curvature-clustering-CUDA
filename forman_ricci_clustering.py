#!/usr/bin/env python
"""
Forman-Ricci Curvature Clustering

This script implements the Ricci flow clustering algorithm from the CUDA implementation,
but uses the GraphRicciCurvature library for Forman-Ricci curvature calculation.

The algorithm:
1. Generate a Stochastic Block Model (SBM) graph
2. Perform Ricci flow iterations:
   - Compute Forman-Ricci curvature for all edges
   - Update edge weights: w_new = (1 - step_size * curvature) * w_old
   - Normalize weights
3. Find communities by cutting edges above different thresholds
4. Select the best partition based on modularity
5. Evaluate using NMI against ground truth
"""

import numpy as np
import networkx as nx
from GraphRicciCurvature.FormanRicci import FormanRicci
from collections import deque
import time
from sklearn.metrics import normalized_mutual_info_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration (matching CUDA defaults)
# =============================================================================

# Graph parameters
N_NODE = 5000
NODES_PER_CLUSTER = 2500
N_CLUSTERS = 2
P_IN = 0.5
P_OUT = 0.01

# Algorithm parameters
N_ITERATION = 10
MAX_CURVATURE = 1000.0
MIN_WEIGHT = 1e-6
STEP_SCALE = 1.1  # νt = 1 / (1.1 × max|κ|)
QUANTILE_Q = 0.999  # q = 0.999 for cutoff
DELTA_STEP = 0.25  # δ = 0.25 for uniform spacing

# Random seed for reproducibility
SEED = 42


def create_sbm_graph(n_nodes, n_clusters, nodes_per_cluster, p_in, p_out, seed=None):
    """
    Create a Stochastic Block Model graph.
    
    Parameters
    ----------
    n_nodes : int
        Total number of nodes
    n_clusters : int
        Number of clusters
    nodes_per_cluster : int
        Number of nodes per cluster
    p_in : float
        Intra-cluster edge probability
    p_out : float
        Inter-cluster edge probability
    seed : int, optional
        Random seed
        
    Returns
    -------
    G : networkx.Graph
        The generated graph with 'weight' attribute on edges
    ground_truth : dict
        Mapping from node to true cluster label
    """
    if seed is not None:
        np.random.seed(seed)
    
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    
    # Assign ground truth clusters
    ground_truth = {}
    for node in range(n_nodes):
        ground_truth[node] = node // nodes_per_cluster
    
    n_intra = 0
    n_inter = 0
    
    # Generate edges
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            cluster_i = ground_truth[i]
            cluster_j = ground_truth[j]
            
            if cluster_i == cluster_j:
                prob = p_in
            else:
                prob = p_out
            
            if np.random.random() <= prob:
                G.add_edge(i, j, weight=1.0)
                if cluster_i == cluster_j:
                    n_intra += 1
                else:
                    n_inter += 1
    
    print(f"Number of intra-cluster connections: {n_intra}")
    print(f"Number of inter-cluster connections: {n_inter}")
    print(f"Total edges: {G.number_of_edges()}")
    
    return G, ground_truth


def compute_forman_ricci_curvature(G):
    """
    Compute Forman-Ricci curvature for all edges using GraphRicciCurvature library.
    
    Parameters
    ----------
    G : networkx.Graph
        Input graph with edge weights
        
    Returns
    -------
    curvatures : dict
        Dictionary mapping (u, v) edges to their Forman-Ricci curvature
    """
    # Create FormanRicci object
    frc = FormanRicci(G, verbose="ERROR")
    
    # Compute curvature
    frc.compute_ricci_curvature()
    
    # Extract curvatures
    curvatures = {}
    for u, v, data in frc.G.edges(data=True):
        curv = data.get('formanCurvature', 0.0)
        # Clamp curvature
        curv = max(-MAX_CURVATURE, min(MAX_CURVATURE, curv))
        curvatures[(u, v)] = curv
        curvatures[(v, u)] = curv
    
    return curvatures


def update_weights(G, curvatures, step_size):
    """
    Update edge weights based on Ricci curvature (Ricci flow step).
    
    w_new = (1 - step_size * curvature) * w_old
    
    Parameters
    ----------
    G : networkx.Graph
        Input graph
    curvatures : dict
        Edge curvatures
    step_size : float
        Adaptive step size
    """
    for u, v in G.edges():
        curv = curvatures.get((u, v), 0.0)
        w_old = G[u][v]['weight']
        w_new = (1 - step_size * curv) * w_old
        w_new = max(w_new, MIN_WEIGHT)
        G[u][v]['weight'] = w_new


def normalize_weights(G, n_edges):
    """
    Normalize edge weights to maintain total weight = n_edges.
    
    Parameters
    ----------
    G : networkx.Graph
        Input graph
    n_edges : int
        Number of undirected edges
    """
    total_weight = sum(data['weight'] for _, _, data in G.edges(data=True))
    if total_weight > 0:
        scale = n_edges / total_weight
        for u, v in G.edges():
            G[u][v]['weight'] *= scale


def find_connected_components(G, threshold):
    """
    Find connected components by only traversing edges with weight < threshold.
    
    Parameters
    ----------
    G : networkx.Graph
        Input graph
    threshold : float
        Weight threshold (edges with weight >= threshold are cut)
        
    Returns
    -------
    labels : dict
        Node to component label mapping
    n_components : int
        Number of connected components
    """
    labels = {node: -1 for node in G.nodes()}
    n_components = 0
    
    for start_node in G.nodes():
        if labels[start_node] != -1:
            continue
        
        # BFS from this node
        queue = deque([start_node])
        labels[start_node] = n_components
        
        while queue:
            node = queue.popleft()
            for neighbor in G.neighbors(node):
                if labels[neighbor] == -1:
                    weight = G[node][neighbor]['weight']
                    if weight < threshold:
                        labels[neighbor] = n_components
                        queue.append(neighbor)
        
        n_components += 1
    
    return labels, n_components


def calculate_modularity(G, labels):
    """
    Calculate modularity for a given partition.
    
    Q = (1/2m) * Σ[A_ij - k_i*k_j/(2m)] * δ(c_i, c_j)
    
    Parameters
    ----------
    G : networkx.Graph
        Input graph
    labels : dict
        Node to community label mapping
        
    Returns
    -------
    modularity : float
        Modularity score
    """
    m = G.number_of_edges()
    if m == 0:
        return 0.0
    
    modularity = 0.0
    degrees = dict(G.degree())
    
    for u in G.nodes():
        k_u = degrees[u]
        for v in G.neighbors(u):
            k_v = degrees[v]
            if labels[u] == labels[v]:
                modularity += 1.0 - (k_u * k_v) / (2.0 * m)
            else:
                modularity += 0.0 - (k_u * k_v) / (2.0 * m)
    
    modularity /= (2.0 * m)
    return modularity


def calculate_nmi(pred_labels, true_labels):
    """
    Calculate Normalized Mutual Information between predicted and true labels.
    
    Parameters
    ----------
    pred_labels : dict
        Predicted node labels
    true_labels : dict
        Ground truth node labels
        
    Returns
    -------
    nmi : float
        NMI score
    """
    nodes = sorted(pred_labels.keys())
    y_pred = [pred_labels[n] for n in nodes]
    y_true = [true_labels[n] for n in nodes]
    return normalized_mutual_info_score(y_true, y_pred)


def ricci_flow_clustering(G, ground_truth, n_iterations=N_ITERATION):
    """
    Main Ricci flow clustering algorithm.
    
    Parameters
    ----------
    G : networkx.Graph
        Input graph
    ground_truth : dict
        Ground truth cluster labels
    n_iterations : int
        Number of Ricci flow iterations
        
    Returns
    -------
    best_labels : dict
        Best community assignment
    best_modularity : float
        Best modularity score
    best_nmi : float
        NMI of best partition
    """
    n_edges = G.number_of_edges()
    n_nodes = G.number_of_nodes()
    n_true_clusters = len(set(ground_truth.values()))
    
    print("\n" + "=" * 70)
    print("Starting Ricci Flow")
    print("=" * 70)
    
    start_time = time.time()
    
    # Ricci flow iterations
    for iteration in range(n_iterations):
        iter_start = time.time()
        
        # Step 1: Compute Forman-Ricci curvature
        curvatures = compute_forman_ricci_curvature(G)
        
        # Get max absolute curvature for adaptive step size
        max_curv = max(abs(c) for c in curvatures.values()) if curvatures else 1.0
        max_curv = max(max_curv, 1e-6)  # Avoid division by zero
        
        # Adaptive step size: νt = 1 / (1.1 × max|κ|)
        step_size = 1.0 / (STEP_SCALE * max_curv)
        
        # Step 2: Update weights
        update_weights(G, curvatures, step_size)
        
        # Step 3: Normalize weights
        normalize_weights(G, n_edges)
        
        iter_time = time.time() - iter_start
        
        # Get weight statistics
        weights = [data['weight'] for _, _, data in G.edges(data=True)]
        print(f"Iteration {iteration + 1}/{n_iterations}: "
              f"max_curv={max_curv:.4f}, step={step_size:.6f}, "
              f"weight range=[{min(weights):.6f}, {max(weights):.6f}], "
              f"time={iter_time:.2f}s")
    
    ricci_flow_time = time.time() - start_time
    print(f"\nRicci flow completed in {ricci_flow_time:.2f}s")
    
    # ==========================================================================
    # Threshold search phase
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Starting Threshold Search")
    print("=" * 70)
    
    threshold_start = time.time()
    
    # Get sorted weights
    sorted_weights = sorted(data['weight'] for _, _, data in G.edges(data=True))
    min_w = sorted_weights[0]
    
    # Get quantile weight
    q_idx = int(QUANTILE_Q * len(sorted_weights))
    if q_idx >= len(sorted_weights):
        q_idx = len(sorted_weights) - 1
    w_quantile = sorted_weights[q_idx]
    
    print(f"\nWeight statistics:")
    print(f"  Min weight: {min_w:.8f}")
    print(f"  Max weight: {sorted_weights[-1]:.8f}")
    print(f"  Quantile ({QUANTILE_Q}): {w_quantile:.6f}")
    
    # Build cutoff list
    cutoff_list = []
    
    # Add unique weights above quantile
    seen = set()
    for w in sorted_weights:
        if w > w_quantile and w not in seen:
            cutoff_list.append(w)
            seen.add(w)
    
    # Add uniform spacing below quantile
    w_stop = 1.1 * min_w
    t = w_quantile - DELTA_STEP
    while t >= w_stop:
        cutoff_list.append(t)
        t -= DELTA_STEP
    
    # Sort cutoffs in descending order
    cutoff_list = sorted(set(cutoff_list), reverse=True)
    
    print(f"Number of cutoff points to try: {len(cutoff_list)}")
    
    # Search for best threshold
    best_modularity = -1.0
    best_threshold = 0.0
    best_n_communities = 0
    best_labels = None
    best_nmi = 0.0
    
    print(f"\n{'Threshold':<15} {'Communities':<12} {'Modularity':<12} {'NMI':<10}")
    print("-" * 55)
    
    results = []
    
    for threshold in cutoff_list:
        labels, n_communities = find_connected_components(G, threshold)
        
        # Skip invalid partitions
        if n_communities < 2 or n_communities > n_nodes // 2:
            continue
        
        modularity = calculate_modularity(G, labels)
        nmi = calculate_nmi(labels, ground_truth)
        
        results.append({
            'threshold': threshold,
            'n_communities': n_communities,
            'modularity': modularity,
            'nmi': nmi,
            'labels': labels.copy()
        })
        
        marker = '*' if modularity > best_modularity else ' '
        
        if n_communities <= 30:
            print(f"{threshold:<15.6e} {n_communities:<12} {modularity:<12.6f} {nmi:<10.4f} {marker}")
        
        if modularity > best_modularity:
            best_modularity = modularity
            best_threshold = threshold
            best_n_communities = n_communities
            best_labels = labels.copy()
            best_nmi = nmi
    
    threshold_time = time.time() - threshold_start
    total_time = time.time() - start_time
    
    print(f"\nRicci flow time:       {ricci_flow_time:.3f}s")
    print(f"Threshold search time: {threshold_time:.3f}s")
    print(f"TOTAL TIME:            {total_time:.3f}s")
    
    # Print final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Best threshold: {best_threshold:.6e}")
    print(f"Communities found: {best_n_communities} (ground truth: {n_true_clusters})")
    print(f"Modularity: {best_modularity:.6f}")
    print(f"NMI: {best_nmi:.6f}")
    
    # Print community sizes
    if best_labels:
        community_sizes = {}
        for node, label in best_labels.items():
            community_sizes[label] = community_sizes.get(label, 0) + 1
        
        print(f"\nCommunity sizes:")
        for c in sorted(community_sizes.keys())[:15]:
            print(f"  Community {c}: {community_sizes[c]} nodes")
        if len(community_sizes) > 15:
            print(f"  ... ({len(community_sizes) - 15} more communities)")
    
    return best_labels, best_modularity, best_nmi


def main():
    """Main function to run the Forman-Ricci clustering experiment."""
    
    print("\n" + "=" * 70)
    print("Graph Configuration")
    print("=" * 70)
    print(f"N_NODE:             {N_NODE}")
    print(f"N_CLUSTERS:         {N_CLUSTERS}")
    print(f"NODES_PER_CLUSTER:  {NODES_PER_CLUSTER}")
    print(f"P_IN:               {P_IN:.4f}")
    print(f"P_OUT:              {P_OUT:.6f}")
    print(f"N_ITERATION:        {N_ITERATION}")
    
    # Calculate expected intra/inter ratio
    n_per_c = N_NODE / N_CLUSTERS
    ratio = (P_IN * (n_per_c - 1)) / (P_OUT * (N_NODE - n_per_c))
    print(f"Intra/Inter ratio:  {ratio:.4f}")
    print("=" * 70)
    
    # Create SBM graph
    print("\nGenerating SBM graph...")
    G, ground_truth = create_sbm_graph(
        n_nodes=N_NODE,
        n_clusters=N_CLUSTERS,
        nodes_per_cluster=NODES_PER_CLUSTER,
        p_in=P_IN,
        p_out=P_OUT,
        seed=SEED
    )
    
    # Run Ricci flow clustering
    best_labels, best_modularity, best_nmi = ricci_flow_clustering(
        G, ground_truth, n_iterations=N_ITERATION
    )
    
    return G, ground_truth, best_labels


if __name__ == "__main__":
    G, ground_truth, best_labels = main()
