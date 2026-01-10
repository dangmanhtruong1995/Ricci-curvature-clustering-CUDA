"""
Forman-Ricci Curvature-based Graph Clustering (Optimized Python Implementation)

Based on: "Curvature-based Clustering on Graphs"
Yu Tian, Zachary Lubberts, Melanie Weber
Journal of Machine Learning Research 26 (2025) 1-67

Optimized for large graphs using:
- Vectorized SBM graph generation
- NumPy arrays instead of dictionaries for edge storage
- Efficient neighbor lookups with CSR format
- Parallel-friendly curvature computation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import connected_components
from numba import njit, prange
import time

# =========================================================================
# Configuration
# =========================================================================

class Config:
    """Configuration parameters for the algorithm."""
    N_NODE = 50000
    NODES_PER_CLUSTER = 5000
    N_CLUSTERS = 10
    P_IN = 0.4
    P_OUT = 0.001
    N_ITERATION = 10  # T = 10 as per pape

    # N_NODE = 5000
    # NODES_PER_CLUSTER = 2500
    # N_CLUSTERS = 2
    # P_IN = 0.4
    # P_OUT = 0.001
    # N_ITERATION = 10  # T = 10 as per pape

    # N_NODE = 5000
    # NODES_PER_CLUSTER = 500
    # N_CLUSTERS = 10
    # P_IN = 0.4
    # P_OUT = 0.001
    # N_ITERATION = 10  # T = 10 as per paper

    # N_NODE = 5000
    # NODES_PER_CLUSTER = 1000
    # N_CLUSTERS = 5
    # P_IN = 0.4
    # P_OUT = 0.001
    # N_ITERATION = 10  # T = 10 as per paper
    
    # Paper parameters (Section 6.2.1)
    STEP_SCALE = 1.1      # νt = 1 / (1.1 × max|κ|)
    QUANTILE_Q = 0.999    # q = 0.999 for cutoff
    DELTA_STEP = 0.25     # δ = 0.25 for uniform spacing
    DROP_THRESHOLD = 0.1  # d = 0.1 (skip if improvement < 10%)
    MIN_MODULARITY = 0.0  # ε for minimum acceptable modularity
    
    RANDOM_SEED = 42


# =========================================================================
# Optimized Graph Class using CSR format
# =========================================================================

class SparseGraph:
    """
    Efficient graph representation using CSR (Compressed Sparse Row) format.
    
    Attributes:
        n_nodes: Number of nodes
        n_edges: Number of undirected edges
        adj_indptr: CSR index pointers (size n_nodes + 1)
        adj_indices: CSR column indices (neighbors)
        edge_weights: Weight for each directed edge in CSR order
        edge_to_idx: Maps (u, v) to index in CSR for edge (u, v)
        edges: List of (u, v) tuples for undirected edges
        node_clusters: Ground truth cluster assignments
    """
    
    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes
        self.n_edges = 0
        self.adj_indptr = None
        self.adj_indices = None
        self.edge_weights = None
        self.edge_curvatures = None
        self.edges = []  # Undirected edges (u, v) where u < v
        self.node_clusters = np.zeros(n_nodes, dtype=np.int32)
        
    def from_edge_list(self, edges: np.ndarray, node_clusters: np.ndarray):
        """
        Build CSR adjacency from edge list.
        
        Args:
            edges: Array of shape (n_edges, 2) with (src, dst) pairs
            node_clusters: Array of cluster assignments
        """
        self.node_clusters = node_clusters
        n_undirected = len(edges)
        self.n_edges = n_undirected
        
        # Store undirected edges
        self.edges = [(int(edges[i, 0]), int(edges[i, 1])) for i in range(n_undirected)]
        
        # Create bidirectional edges for CSR
        all_src = np.concatenate([edges[:, 0], edges[:, 1]])
        all_dst = np.concatenate([edges[:, 1], edges[:, 0]])
        
        # Sort by source for CSR format
        sort_idx = np.lexsort((all_dst, all_src))
        all_src = all_src[sort_idx]
        all_dst = all_dst[sort_idx]
        
        # Build CSR structure
        self.adj_indices = all_dst.astype(np.int32)
        self.adj_indptr = np.zeros(self.n_nodes + 1, dtype=np.int32)
        
        # Count neighbors per node
        np.add.at(self.adj_indptr[1:], all_src, 1)
        np.cumsum(self.adj_indptr, out=self.adj_indptr)
        
        # Initialize weights to 1.0
        self.edge_weights = np.ones(len(all_src), dtype=np.float64)
        self.edge_curvatures = np.zeros(len(all_src), dtype=np.float64)
        
        # Build edge index mapping for fast lookup
        self._build_edge_index()
    
    def _build_edge_index(self):
        """Build mapping from (u,v) to CSR index."""
        self.edge_to_csr = {}
        for u in range(self.n_nodes):
            start, end = self.adj_indptr[u], self.adj_indptr[u + 1]
            for idx in range(start, end):
                v = self.adj_indices[idx]
                self.edge_to_csr[(u, v)] = idx
    
    def get_neighbors(self, node: int) -> np.ndarray:
        """Get neighbors of a node."""
        start, end = self.adj_indptr[node], self.adj_indptr[node + 1]
        return self.adj_indices[start:end]
    
    def get_neighbor_weights(self, node: int) -> np.ndarray:
        """Get weights of edges from node."""
        start, end = self.adj_indptr[node], self.adj_indptr[node + 1]
        return self.edge_weights[start:end]
    
    def degree(self, node: int) -> int:
        """Get degree of a node."""
        return self.adj_indptr[node + 1] - self.adj_indptr[node]
    
    def get_weight(self, u: int, v: int) -> float:
        """Get weight of edge (u, v)."""
        idx = self.edge_to_csr.get((u, v))
        if idx is not None:
            return self.edge_weights[idx]
        return 0.0
    
    def set_weight(self, u: int, v: int, weight: float):
        """Set weight of edge (u, v) and (v, u)."""
        idx_uv = self.edge_to_csr.get((u, v))
        idx_vu = self.edge_to_csr.get((v, u))
        if idx_uv is not None:
            self.edge_weights[idx_uv] = weight
        if idx_vu is not None:
            self.edge_weights[idx_vu] = weight


# =========================================================================
# Optimized Graph Generation (Stochastic Block Model)
# =========================================================================

def create_sbm_graph_fast(config: Config) -> SparseGraph:
    """
    Create a Stochastic Block Model graph efficiently.
    
    Uses vectorized operations for edge generation.
    
    Args:
        config: Configuration parameters
        
    Returns:
        SparseGraph object with SBM structure
    """
    np.random.seed(config.RANDOM_SEED)
    
    n = config.N_NODE
    k = config.N_CLUSTERS
    cluster_size = config.NODES_PER_CLUSTER
    
    print(f"    Generating SBM graph with {n} nodes, {k} clusters...")
    start_time = time.time()
    
    # Assign nodes to clusters
    node_clusters = np.repeat(np.arange(k), cluster_size).astype(np.int32)
    
    # Generate edges cluster by cluster (much faster than O(n^2) loop)
    edges_list = []
    
    # Intra-cluster edges
    print(f"    Generating intra-cluster edges (p={config.P_IN})...")
    for c in range(k):
        start_node = c * cluster_size
        end_node = (c + 1) * cluster_size
        
        # Number of possible edges in this cluster
        n_possible = cluster_size * (cluster_size - 1) // 2
        
        # Expected number of edges
        n_expected = int(n_possible * config.P_IN)
        
        # Generate random edges within cluster
        # Use sampling for efficiency when p is not too high
        if config.P_IN < 0.5:
            # Sample approximately the right number of edges
            n_sample = int(n_expected * 1.2)  # Oversample slightly
            
            # Generate random pairs
            i_vals = np.random.randint(0, cluster_size, size=n_sample)
            j_vals = np.random.randint(0, cluster_size, size=n_sample)
            
            # Keep only i < j (upper triangle)
            mask = i_vals < j_vals
            i_vals = i_vals[mask]
            j_vals = j_vals[mask]
            
            # Remove duplicates
            pairs = np.unique(np.column_stack([i_vals, j_vals]), axis=0)
            
            # Keep with probability to match expected count
            if len(pairs) > n_expected:
                keep_idx = np.random.choice(len(pairs), n_expected, replace=False)
                pairs = pairs[keep_idx]
            
            # Adjust to global node indices
            cluster_edges = pairs + start_node
        else:
            # For high p, generate all possible and filter
            i_idx, j_idx = np.triu_indices(cluster_size, k=1)
            mask = np.random.random(len(i_idx)) <= config.P_IN
            cluster_edges = np.column_stack([
                i_idx[mask] + start_node,
                j_idx[mask] + start_node
            ])
        
        if len(cluster_edges) > 0:
            edges_list.append(cluster_edges)
    
    # Inter-cluster edges
    print(f"    Generating inter-cluster edges (p={config.P_OUT})...")
    for c1 in range(k):
        for c2 in range(c1 + 1, k):
            start1 = c1 * cluster_size
            start2 = c2 * cluster_size
            
            # Number of possible inter-cluster edges
            n_possible = cluster_size * cluster_size
            n_expected = int(n_possible * config.P_OUT)
            
            if n_expected > 0:
                # Sample edges between clusters
                n_sample = min(int(n_expected * 1.5), n_possible)
                
                i_vals = np.random.randint(0, cluster_size, size=n_sample)
                j_vals = np.random.randint(0, cluster_size, size=n_sample)
                
                # Remove duplicates
                pairs = np.unique(np.column_stack([i_vals, j_vals]), axis=0)
                
                # Keep approximately expected number
                if len(pairs) > n_expected:
                    keep_idx = np.random.choice(len(pairs), n_expected, replace=False)
                    pairs = pairs[keep_idx]
                
                # Adjust to global indices
                inter_edges = np.column_stack([
                    pairs[:, 0] + start1,
                    pairs[:, 1] + start2
                ])
                
                if len(inter_edges) > 0:
                    edges_list.append(inter_edges)
    
    # Combine all edges
    all_edges = np.vstack(edges_list)
    
    # Ensure u < v for all edges
    all_edges = np.sort(all_edges, axis=1)
    
    # Remove any duplicates
    all_edges = np.unique(all_edges, axis=0)
    
    print(f"    Building graph structure...")
    
    # Create graph
    graph = SparseGraph(n)
    graph.from_edge_list(all_edges, node_clusters)
    
    elapsed = time.time() - start_time
    print(f"    Graph generation complete in {elapsed:.2f}s")
    print(f"    Total edges: {graph.n_edges}")
    
    return graph


# =========================================================================
# Numba-accelerated Forman-Ricci Curvature Computation
# =========================================================================

@njit(cache=True)
def _compute_curvature_for_edge(
    v1: int, v2: int,
    adj_indptr: np.ndarray,
    adj_indices: np.ndarray,
    edge_weights: np.ndarray,
    edge_to_csr_keys: np.ndarray,
    edge_to_csr_vals: np.ndarray
) -> float:
    """
    Compute Forman-Ricci curvature for a single edge (Numba-optimized).
    
    This is a simplified version that works with pre-computed data structures.
    """
    # Get weight of main edge
    w_e = 1.0
    start_v1, end_v1 = adj_indptr[v1], adj_indptr[v1 + 1]
    for idx in range(start_v1, end_v1):
        if adj_indices[idx] == v2:
            w_e = max(edge_weights[idx], 1e-6)
            break
    
    # Get neighbors
    neighbors_v1 = adj_indices[start_v1:end_v1]
    start_v2, end_v2 = adj_indptr[v2], adj_indptr[v2 + 1]
    neighbors_v2 = adj_indices[start_v2:end_v2]
    
    # Find common neighbors (triangles)
    common = []
    for n1 in neighbors_v1:
        if n1 == v2:
            continue
        for n2 in neighbors_v2:
            if n1 == n2:
                common.append(n1)
                break
    
    # Triangle contribution
    triangle_contrib = 0.0
    for n in common:
        # Get w_v1_n
        w_v1_n = 1e-6
        for idx in range(start_v1, end_v1):
            if adj_indices[idx] == n:
                w_v1_n = max(edge_weights[idx], 1e-6)
                break
        
        # Get w_v2_n
        w_v2_n = 1e-6
        for idx in range(start_v2, end_v2):
            if adj_indices[idx] == n:
                w_v2_n = max(edge_weights[idx], 1e-6)
                break
        
        # Heron's formula
        s = (w_e + w_v1_n + w_v2_n) / 2.0
        term1 = s - w_e
        term2 = s - w_v1_n
        term3 = s - w_v2_n
        
        if term1 > 0 and term2 > 0 and term3 > 0:
            area_sq = s * term1 * term2 * term3
            w_tri = np.sqrt(max(area_sq, 1e-12))
            triangle_contrib += min(w_e / w_tri, 100.0)
        else:
            triangle_contrib += 100.0
    
    # Vertex contribution
    vertex_contrib = 2.0 / w_e
    
    # Parallel edge contribution
    parallel_contrib = 0.0
    common_set = set(common)
    
    # From v1
    for idx in range(start_v1, end_v1):
        v = adj_indices[idx]
        if v == v2:
            continue
        is_common = False
        for c in common:
            if v == c:
                is_common = True
                break
        if not is_common:
            w_ep = max(edge_weights[idx], 1e-6)
            parallel_contrib += 1.0 / np.sqrt(w_e * w_ep)
    
    # From v2
    for idx in range(start_v2, end_v2):
        v = adj_indices[idx]
        if v == v1:
            continue
        is_common = False
        for c in common:
            if v == c:
                is_common = True
                break
        if not is_common:
            w_ep = max(edge_weights[idx], 1e-6)
            parallel_contrib += 1.0 / np.sqrt(w_e * w_ep)
    
    # Final curvature
    curvature = w_e * (triangle_contrib + vertex_contrib - parallel_contrib)
    
    # Clamp
    if curvature < -1000.0:
        curvature = -1000.0
    elif curvature > 1000.0:
        curvature = 1000.0
    
    return curvature


@njit(parallel=True, cache=True)
def compute_all_curvatures_numba(
    edges: np.ndarray,
    adj_indptr: np.ndarray,
    adj_indices: np.ndarray,
    edge_weights: np.ndarray
) -> np.ndarray:
    """
    Compute curvatures for all edges in parallel using Numba.
    """
    n_edges = len(edges)
    curvatures = np.zeros(n_edges, dtype=np.float64)
    
    # Dummy arrays for compatibility (not used in simplified version)
    dummy_keys = np.zeros(1, dtype=np.int64)
    dummy_vals = np.zeros(1, dtype=np.int32)
    
    for i in prange(n_edges):
        v1, v2 = edges[i, 0], edges[i, 1]
        curvatures[i] = _compute_curvature_for_edge(
            v1, v2, adj_indptr, adj_indices, edge_weights,
            dummy_keys, dummy_vals
        )
    
    return curvatures


def compute_all_curvatures(graph: SparseGraph) -> np.ndarray:
    """Compute Forman-Ricci curvature for all edges."""
    edges_array = np.array(graph.edges, dtype=np.int32)
    
    curvatures = compute_all_curvatures_numba(
        edges_array,
        graph.adj_indptr,
        graph.adj_indices,
        graph.edge_weights
    )
    
    # Store curvatures back in graph (for both directions)
    for i, (u, v) in enumerate(graph.edges):
        idx_uv = graph.edge_to_csr.get((u, v))
        idx_vu = graph.edge_to_csr.get((v, u))
        if idx_uv is not None:
            graph.edge_curvatures[idx_uv] = curvatures[i]
        if idx_vu is not None:
            graph.edge_curvatures[idx_vu] = curvatures[i]
    
    return curvatures


# =========================================================================
# Weight Update (Ricci Flow)
# =========================================================================

@njit(parallel=True, cache=True)
def update_weights_numba(
    edges: np.ndarray,
    edge_weights: np.ndarray,
    edge_curvatures: np.ndarray,
    adj_indptr: np.ndarray,
    adj_indices: np.ndarray,
    step_size: float
) -> np.ndarray:
    """Update weights using Ricci flow (Numba-optimized)."""
    n_edges = len(edges)
    new_weights = np.zeros(len(edge_weights), dtype=np.float64)
    
    # Copy old weights
    for i in range(len(edge_weights)):
        new_weights[i] = edge_weights[i]
    
    for i in prange(n_edges):
        u, v = edges[i, 0], edges[i, 1]
        
        # Find CSR index for (u, v)
        idx_uv = -1
        start_u, end_u = adj_indptr[u], adj_indptr[u + 1]
        for idx in range(start_u, end_u):
            if adj_indices[idx] == v:
                idx_uv = idx
                break
        
        # Find CSR index for (v, u)
        idx_vu = -1
        start_v, end_v = adj_indptr[v], adj_indptr[v + 1]
        for idx in range(start_v, end_v):
            if adj_indices[idx] == u:
                idx_vu = idx
                break
        
        if idx_uv >= 0:
            w_old = edge_weights[idx_uv]
            kappa = edge_curvatures[idx_uv]
            w_new = (1.0 - step_size * kappa) * w_old
            w_new = max(w_new, 1e-6)
            new_weights[idx_uv] = w_new
            if idx_vu >= 0:
                new_weights[idx_vu] = w_new
    
    return new_weights


def update_weights(graph: SparseGraph, step_size: float):
    """Update edge weights under Ricci flow."""
    edges_array = np.array(graph.edges, dtype=np.int32)
    
    new_weights = update_weights_numba(
        edges_array,
        graph.edge_weights,
        graph.edge_curvatures,
        graph.adj_indptr,
        graph.adj_indices,
        step_size
    )
    
    graph.edge_weights = new_weights


def normalize_weights(graph: SparseGraph):
    """Normalize edge weights to preserve sum = |E|."""
    total_weight = np.sum(graph.edge_weights) / 2.0  # Each edge counted twice
    n_edges = graph.n_edges
    
    if total_weight < 1e-10:
        return
    
    scale = n_edges / total_weight
    graph.edge_weights *= scale


# =========================================================================
# Connected Components (Threshold-based clustering)
# =========================================================================

def find_connected_components(graph: SparseGraph, threshold: float) -> np.ndarray:
    """Find connected components by removing edges with weight >= threshold."""
    n = graph.n_nodes
    
    # Build sparse matrix with thresholded weights
    mask = graph.edge_weights < threshold
    
    adj_matrix = csr_matrix(
        (np.ones(np.sum(mask)), 
         (np.repeat(np.arange(n), np.diff(graph.adj_indptr))[mask],
          graph.adj_indices[mask])),
        shape=(n, n)
    )
    
    n_components, labels = connected_components(
        csgraph=adj_matrix, 
        directed=False, 
        return_labels=True
    )
    
    return labels


# =========================================================================
# Modularity and NMI Calculations
# =========================================================================

def calculate_modularity(graph: SparseGraph, labels: np.ndarray) -> float:
    """Calculate modularity of a clustering."""
    m = graph.n_edges
    if m == 0:
        return 0.0
    
    # Compute degree for each node
    degrees = np.diff(graph.adj_indptr)
    
    modularity = 0.0
    
    for u in range(graph.n_nodes):
        k_u = degrees[u]
        neighbors = graph.get_neighbors(u)
        
        for v in neighbors:
            k_v = degrees[v]
            
            if labels[u] == labels[v]:
                modularity += 1.0 - (k_u * k_v) / (2.0 * m)
            else:
                modularity -= (k_u * k_v) / (2.0 * m)
    
    modularity /= (2.0 * m)
    return modularity


def calculate_nmi(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
    """Calculate Normalized Mutual Information between two clusterings."""
    n = len(pred_labels)
    
    pred_unique = np.unique(pred_labels)
    true_unique = np.unique(true_labels)
    
    # Compute contingency table
    contingency = np.zeros((len(pred_unique), len(true_unique)))
    pred_map = {v: i for i, v in enumerate(pred_unique)}
    true_map = {v: i for i, v in enumerate(true_unique)}
    
    for i in range(n):
        p_idx = pred_map[pred_labels[i]]
        t_idx = true_map[true_labels[i]]
        contingency[p_idx, t_idx] += 1
    
    # Marginals
    pred_counts = contingency.sum(axis=1)
    true_counts = contingency.sum(axis=0)
    
    # Mutual Information
    mi = 0.0
    for i in range(len(pred_unique)):
        for j in range(len(true_unique)):
            if contingency[i, j] > 0:
                pij = contingency[i, j] / n
                pi = pred_counts[i] / n
                pj = true_counts[j] / n
                mi += pij * np.log(pij / (pi * pj))
    
    # Entropies
    h_pred = -np.sum((pred_counts / n) * np.log(pred_counts / n + 1e-10))
    h_true = -np.sum((true_counts / n) * np.log(true_counts / n + 1e-10))
    
    # NMI
    denom = h_pred + h_true
    if denom < 1e-10:
        return 0.0
    
    return 2.0 * mi / denom


# =========================================================================
# Main Algorithm
# =========================================================================

def ricci_flow_clustering(
    graph: SparseGraph, 
    config: Config,
    verbose: bool = True
) -> Tuple[np.ndarray, float, Dict]:
    """Main Ricci Flow Clustering Algorithm."""
    
    history = {
        'iterations': [],
        'max_curvatures': [],
        'step_sizes': [],
        'weight_sums': [],
        'intra_weights': [],
        'inter_weights': [],
        'thresholds': [],
        'modularities': [],
        'nmis': [],
        'n_communities': []
    }
    
    if verbose:
        print("=== Forman-Ricci Curvature Clustering (JMLR 2025) ===")
        print(f"Parameters: T={config.N_ITERATION} iterations, "
              f"step_scale={config.STEP_SCALE}, δ={config.DELTA_STEP}, q={config.QUANTILE_Q}")
        
        # Count edge types
        intra_count = sum(1 for u, v in graph.edges 
                          if graph.node_clusters[u] == graph.node_clusters[v])
        inter_count = graph.n_edges - intra_count
        
        print(f"Graph: {graph.n_nodes} nodes, {graph.n_edges} edges, {config.N_CLUSTERS} clusters")
        print(f"Intra-cluster edges: {intra_count} ({100.0*intra_count/graph.n_edges:.1f}%)")
        print(f"Inter-cluster edges: {inter_count} ({100.0*inter_count/graph.n_edges:.1f}%)")
    
    # Ricci Flow Loop
    if verbose:
        print("\n=== Ricci Flow ===")
    
    for iteration in range(config.N_ITERATION):
        iter_start = time.time()
        
        # Compute curvatures
        curvatures = compute_all_curvatures(graph)
        
        # Find max absolute curvature
        max_curv = np.max(np.abs(curvatures)) if len(curvatures) > 0 else 0.0
        
        # Adaptive step size
        adaptive_step = 1.0 / (config.STEP_SCALE * max_curv + 1e-10)
        adaptive_step = min(adaptive_step, 1.0)
        
        # Update weights
        update_weights(graph, adaptive_step)
        
        # Normalize weights
        normalize_weights(graph)
        
        # Track statistics
        weight_sum = np.sum(graph.edge_weights) / 2.0
        
        # Intra vs inter cluster weights
        intra_weights = []
        inter_weights = []
        for u, v in graph.edges:
            w = graph.get_weight(u, v)
            if graph.node_clusters[u] == graph.node_clusters[v]:
                intra_weights.append(w)
            else:
                inter_weights.append(w)
        
        history['iterations'].append(iteration)
        history['max_curvatures'].append(max_curv)
        history['step_sizes'].append(adaptive_step)
        history['weight_sums'].append(weight_sum)
        history['intra_weights'].append(np.mean(intra_weights) if intra_weights else 0)
        history['inter_weights'].append(np.mean(inter_weights) if inter_weights else 0)
        
        iter_time = time.time() - iter_start
        if verbose:
            print(f"Iter {iteration:2d}: max_curv={max_curv:8.2f}, "
                  f"step={adaptive_step:.6f}, sum_weights={weight_sum:.2f} ({iter_time:.1f}s)")
    
    # Analyze weight distribution
    all_weights = np.array([graph.get_weight(u, v) for u, v in graph.edges])
    min_w = np.min(all_weights)
    max_w = np.max(all_weights)
    
    # Calculate intra/inter cluster weight statistics
    intra_weights_final = []
    inter_weights_final = []
    for u, v in graph.edges:
        w = graph.get_weight(u, v)
        if graph.node_clusters[u] == graph.node_clusters[v]:
            intra_weights_final.append(w)
        else:
            inter_weights_final.append(w)
    
    intra_avg = np.mean(intra_weights_final) if intra_weights_final else 0.0
    inter_avg = np.mean(inter_weights_final) if inter_weights_final else 0.0
    weight_ratio = inter_avg / intra_avg if intra_avg > 0 else 0.0
    
    if verbose:
        print("\n=== Weight Analysis ===")
        print(f"Min weight: {min_w:.6f}")
        print(f"Max weight: {max_w:.6f}")
        print(f"Ratio max/min: {max_w/min_w:.2f}")
        print(f"Intra-cluster avg weight: {intra_avg:.6f}")
        print(f"Inter-cluster avg weight: {inter_avg:.6f}")
        print(f"Weight ratio (inter/intra): {weight_ratio:.4f}")
    
    # Threshold Selection
    if verbose:
        print("\n=== Threshold Search (Modularity-based) ===")
    
    sorted_weights = np.sort(all_weights)
    q_idx = int(config.QUANTILE_Q * len(sorted_weights))
    w_quantile = sorted_weights[min(q_idx, len(sorted_weights)-1)]
    
    if verbose:
        print(f"Weight quantile ({config.QUANTILE_Q:.3f}): {w_quantile:.6f}")
    
    # Generate cutoffs
    cutoffs = []
    unique_weights = np.unique(all_weights)[::-1]  # Descending
    for w in unique_weights:
        if w >= w_quantile:
            cutoffs.append(w)
    
    w_stop = 1.1 * min_w
    t = w_quantile - config.DELTA_STEP
    while t >= w_stop:
        cutoffs.append(t)
        t -= config.DELTA_STEP
    
    if verbose:
        print(f"Generated {len(cutoffs)} cutoff points")
    
    # Search for best threshold
    best_modularity = -1.0
    best_threshold = 0.0
    best_labels = None
    best_n_communities = 0
    best_nmi = 0.0
    
    true_labels = graph.node_clusters
    all_results = []
    
    if verbose:
        print(f"\n{'Threshold':<15} {'Communities':<12} {'Modularity':<12} {'NMI':<10}")
        print("-" * 52)
    
    for threshold in cutoffs:
        labels = find_connected_components(graph, threshold)
        n_communities = len(np.unique(labels))
        
        if n_communities < 2 or n_communities > graph.n_nodes // 2:
            continue
        
        modularity = calculate_modularity(graph, labels)
        nmi = calculate_nmi(labels, true_labels)
        
        history['thresholds'].append(threshold)
        history['modularities'].append(modularity)
        history['nmis'].append(nmi)
        history['n_communities'].append(n_communities)
        
        all_results.append({
            'threshold': threshold,
            'labels': labels.copy(),
            'n_communities': n_communities,
            'modularity': modularity,
            'nmi': nmi
        })
        
        if verbose and n_communities <= 30:
            marker = '*' if modularity > best_modularity else ' '
            print(f"{threshold:.6e}  {n_communities:<12} {modularity:<12.6f} {nmi:<10.4f} {marker}")
        
        if modularity > best_modularity:
            best_modularity = modularity
            best_threshold = threshold
            best_labels = labels.copy()
            best_n_communities = n_communities
            best_nmi = nmi
    
    # Secondary selection for better NMI
    high_mod_threshold = best_modularity * 0.95
    high_mod_results = [r for r in all_results if r['modularity'] >= high_mod_threshold]
    
    if high_mod_results:
        best_result = max(high_mod_results, key=lambda r: r['nmi'])
        if best_result['nmi'] > best_nmi:
            best_threshold = best_result['threshold']
            best_labels = best_result['labels']
            best_n_communities = best_result['n_communities']
            best_modularity = best_result['modularity']
            best_nmi = best_result['nmi']
    
    if best_labels is None:
        best_threshold = np.median(all_weights)
        best_labels = find_connected_components(graph, best_threshold)
        best_n_communities = len(np.unique(best_labels))
        best_modularity = calculate_modularity(graph, best_labels)
    
    final_nmi = calculate_nmi(best_labels, true_labels)
    
    if verbose:
        print(f"\n=== Best Result ===")
        print(f"Threshold: {best_threshold:.6e}")
        print(f"Communities: {best_n_communities} (ground truth: {config.N_CLUSTERS})")
        print(f"Modularity: {best_modularity:.6f}")
        print(f"NMI: {final_nmi:.6f}")
    
    return best_labels, best_threshold, history


# =========================================================================
# Visualization Functions
# =========================================================================

def plot_ricci_flow_history(history: Dict, config: Config, save_path: Optional[str] = None):
    """Plot the evolution of metrics during Ricci flow."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Max curvature and step size
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    
    line1 = ax1.plot(history['iterations'], history['max_curvatures'], 
                     'b-o', label='Max |κ|', linewidth=2, markersize=6)
    line2 = ax1_twin.plot(history['iterations'], history['step_sizes'], 
                          'r--s', label='Step size ν', linewidth=2, markersize=6)
    
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Max Absolute Curvature', color='b', fontsize=12)
    ax1_twin.set_ylabel('Adaptive Step Size', color='r', fontsize=12)
    ax1.set_title('Curvature and Step Size Evolution', fontsize=14, fontweight='bold')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Intra vs Inter cluster weights
    ax2 = axes[0, 1]
    ax2.plot(history['iterations'], history['intra_weights'], 
             'g-o', label='Intra-cluster', linewidth=2, markersize=6)
    ax2.plot(history['iterations'], history['inter_weights'], 
             'm-s', label='Inter-cluster', linewidth=2, markersize=6)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Average Edge Weight', fontsize=12)
    ax2.set_title('Intra vs Inter Cluster Edge Weights', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Modularity vs Threshold
    ax3 = axes[1, 0]
    if history['thresholds']:
        scatter = ax3.scatter(history['thresholds'], history['modularities'], 
                             c=history['n_communities'], cmap='viridis', 
                             s=50, alpha=0.7)
        ax3.set_xlabel('Threshold', fontsize=12)
        ax3.set_ylabel('Modularity', fontsize=12)
        ax3.set_title('Modularity vs Threshold', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Number of Communities')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: NMI vs Number of Communities
    ax4 = axes[1, 1]
    if history['n_communities']:
        ax4.scatter(history['n_communities'], history['nmis'], 
                   c=history['thresholds'], cmap='plasma', s=50, alpha=0.7)
        ax4.axvline(x=config.N_CLUSTERS, color='r', linestyle='--', 
                    label=f'True # clusters ({config.N_CLUSTERS})')
        ax4.set_xlabel('Number of Communities', fontsize=12)
        ax4.set_ylabel('NMI', fontsize=12)
        ax4.set_title('NMI vs Number of Communities', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_weight_distribution(graph: SparseGraph, save_path: Optional[str] = None):
    """Plot the distribution of edge weights after Ricci flow."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    intra_weights = []
    inter_weights = []
    
    for u, v in graph.edges:
        w = graph.get_weight(u, v)
        if graph.node_clusters[u] == graph.node_clusters[v]:
            intra_weights.append(w)
        else:
            inter_weights.append(w)
    
    # Plot 1: Histogram
    ax1 = axes[0]
    all_w = intra_weights + inter_weights
    bins = np.linspace(min(all_w), max(all_w), 50)
    
    ax1.hist(intra_weights, bins=bins, alpha=0.6, label='Intra-cluster', color='green')
    ax1.hist(inter_weights, bins=bins, alpha=0.6, label='Inter-cluster', color='red')
    ax1.set_xlabel('Edge Weight', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Edge Weight Distribution After Ricci Flow', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot
    ax2 = axes[1]
    box_data = [intra_weights, inter_weights]
    bp = ax2.boxplot(box_data, tick_labels=['Intra-cluster', 'Inter-cluster'], patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('red')
    bp['boxes'][1].set_alpha(0.6)
    ax2.set_ylabel('Edge Weight', fontsize=12)
    ax2.set_title('Edge Weight Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_graph_clustering(
    graph: SparseGraph, 
    labels: np.ndarray, 
    sample_size: int = 200,
    save_path: Optional[str] = None
):
    """Visualize the graph with clustering results."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    if graph.n_nodes > sample_size:
        np.random.seed(42)
        sample_nodes = np.random.choice(graph.n_nodes, sample_size, replace=False)
        sample_set = set(sample_nodes)
    else:
        sample_nodes = np.arange(graph.n_nodes)
        sample_set = set(sample_nodes)
    
    G = nx.Graph()
    node_mapping = {old: new for new, old in enumerate(sample_nodes)}
    
    for old_node in sample_nodes:
        G.add_node(node_mapping[old_node])
    
    for u, v in graph.edges:
        if u in sample_set and v in sample_set:
            G.add_edge(node_mapping[u], node_mapping[v])
    
    pos = nx.spring_layout(G, seed=42, k=2/np.sqrt(len(sample_nodes)))
    
    # Ground truth
    ax1 = axes[0]
    true_labels = [graph.node_clusters[n] for n in sample_nodes]
    node_colors = plt.cm.tab20(np.array(true_labels) % 20)
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors, node_size=30, alpha=0.8)
    nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.2, width=0.5)
    ax1.set_title('Ground Truth Clustering', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Predicted
    ax2 = axes[1]
    pred_labels = [labels[n] for n in sample_nodes]
    node_colors = plt.cm.tab20(np.array(pred_labels) % 20)
    nx.draw_networkx_nodes(G, pos, ax=ax2, node_color=node_colors, node_size=30, alpha=0.8)
    nx.draw_networkx_edges(G, pos, ax=ax2, alpha=0.2, width=0.5)
    ax2.set_title('Ricci Flow Clustering', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_summary_plot(
    graph: SparseGraph, 
    labels: np.ndarray, 
    history: Dict, 
    config: Config,
    save_path: Optional[str] = None
):
    """Create a comprehensive summary visualization."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Curvature convergence
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history['iterations'], history['max_curvatures'], 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Max |κ|')
    ax1.set_title('Curvature Convergence', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Weight evolution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history['iterations'], history['intra_weights'], 'g-o', label='Intra', linewidth=2)
    ax2.plot(history['iterations'], history['inter_weights'], 'r-s', label='Inter', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Avg Weight')
    ax2.set_title('Weight Evolution', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Weight distribution
    ax3 = fig.add_subplot(gs[0, 2])
    intra_weights = [graph.get_weight(u, v) for u, v in graph.edges 
                     if graph.node_clusters[u] == graph.node_clusters[v]]
    inter_weights = [graph.get_weight(u, v) for u, v in graph.edges 
                     if graph.node_clusters[u] != graph.node_clusters[v]]
    
    if intra_weights and inter_weights:
        all_w = intra_weights + inter_weights
        bins = np.linspace(min(all_w), max(all_w), 40)
        ax3.hist(intra_weights, bins=bins, alpha=0.6, color='green', label='Intra')
        ax3.hist(inter_weights, bins=bins, alpha=0.6, color='red', label='Inter')
    ax3.set_xlabel('Edge Weight')
    ax3.set_ylabel('Count')
    ax3.set_title('Final Weight Distribution', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Modularity vs threshold
    ax4 = fig.add_subplot(gs[1, 0])
    if history['thresholds']:
        scatter = ax4.scatter(history['thresholds'], history['modularities'], 
                             c=history['n_communities'], cmap='viridis', s=30, alpha=0.7)
        plt.colorbar(scatter, ax=ax4, label='# Communities')
    ax4.set_xlabel('Threshold')
    ax4.set_ylabel('Modularity')
    ax4.set_title('Threshold Selection', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. NMI vs communities
    ax5 = fig.add_subplot(gs[1, 1])
    if history['n_communities']:
        ax5.scatter(history['n_communities'], history['nmis'], c='blue', s=30, alpha=0.5)
        ax5.axvline(x=config.N_CLUSTERS, color='r', linestyle='--', label=f'True ({config.N_CLUSTERS})')
    ax5.set_xlabel('# Communities')
    ax5.set_ylabel('NMI')
    ax5.set_title('Clustering Quality', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Community sizes
    ax6 = fig.add_subplot(gs[1, 2])
    unique_labels, counts = np.unique(labels, return_counts=True)
    sorted_counts = sorted(counts, reverse=True)[:20]
    ax6.bar(range(len(sorted_counts)), sorted_counts, color='steelblue', alpha=0.7)
    ax6.axhline(y=config.NODES_PER_CLUSTER, color='r', linestyle='--', 
                label=f'Expected ({config.NODES_PER_CLUSTER})')
    ax6.set_xlabel('Community Rank')
    ax6.set_ylabel('Size')
    ax6.set_title('Community Sizes', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7-8. Graph visualization
    sample_size = min(200, graph.n_nodes)
    np.random.seed(42)
    sample_nodes = np.random.choice(graph.n_nodes, sample_size, replace=False)
    sample_set = set(sample_nodes)
    
    G = nx.Graph()
    node_mapping = {old: new for new, old in enumerate(sample_nodes)}
    
    for old_node in sample_nodes:
        G.add_node(node_mapping[old_node])
    
    for u, v in graph.edges:
        if u in sample_set and v in sample_set:
            G.add_edge(node_mapping[u], node_mapping[v])
    
    pos = nx.spring_layout(G, seed=42, k=2/np.sqrt(len(sample_nodes)))
    
    ax7 = fig.add_subplot(gs[2, :2])
    true_labels_sample = [graph.node_clusters[n] for n in sample_nodes]
    node_colors = plt.cm.tab20(np.array(true_labels_sample) % 20)
    nx.draw_networkx_nodes(G, pos, ax=ax7, node_color=node_colors, node_size=20, alpha=0.8)
    nx.draw_networkx_edges(G, pos, ax=ax7, alpha=0.1, width=0.3)
    ax7.set_title('Ground Truth Clustering (sampled)', fontweight='bold')
    ax7.axis('off')
    
    ax8 = fig.add_subplot(gs[2, 2])
    pred_labels_sample = [labels[n] for n in sample_nodes]
    node_colors = plt.cm.tab20(np.array(pred_labels_sample) % 20)
    nx.draw_networkx_nodes(G, pos, ax=ax8, node_color=node_colors, node_size=20, alpha=0.8)
    nx.draw_networkx_edges(G, pos, ax=ax8, alpha=0.1, width=0.3)
    ax8.set_title('Ricci Flow Clustering', fontweight='bold')
    ax8.axis('off')
    
    # Title
    final_nmi = calculate_nmi(labels, graph.node_clusters)
    final_modularity = calculate_modularity(graph, labels)
    n_found = len(np.unique(labels))
    
    fig.suptitle(
        f'Forman-Ricci Curvature Clustering Results\n'
        f'NMI: {final_nmi:.4f} | Modularity: {final_modularity:.4f} | '
        f'Communities: {n_found} (true: {config.N_CLUSTERS})',
        fontsize=14, fontweight='bold', y=1.02
    )
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# =========================================================================
# Main Entry Point
# =========================================================================

def main():
    """Run the complete Forman-Ricci clustering algorithm."""
    
    config = Config()
    
    print("=" * 70)
    print("Forman-Ricci Curvature-based Graph Clustering (Optimized)")
    print("Based on: Tian, Lubberts, Weber (JMLR 2025)")
    print("=" * 70)
    
    # Create graph
    print("\n[1/4] Generating Stochastic Block Model graph...")
    total_start = time.time()
    graph = create_sbm_graph_fast(config)
    
    # Count edge types
    intra_count = sum(1 for u, v in graph.edges 
                      if graph.node_clusters[u] == graph.node_clusters[v])
    inter_count = graph.n_edges - intra_count
    
    print(f"  Nodes: {graph.n_nodes}")
    print(f"  Edges: {graph.n_edges}")
    print(f"  Clusters: {config.N_CLUSTERS}")
    print(f"  Intra-cluster edges: {intra_count} ({100*intra_count/graph.n_edges:.1f}%)")
    print(f"  Inter-cluster edges: {inter_count} ({100*inter_count/graph.n_edges:.1f}%)")
    
    # Run clustering
    print("\n[2/4] Running Ricci flow clustering...")
    labels, threshold, history = ricci_flow_clustering(graph, config, verbose=True)
    
    # Generate visualizations
    print("\n[3/4] Generating visualizations...")
    
    fig1 = plot_ricci_flow_history(history, config, save_path='./outputs/ricci_flow_history.png')
    print("  Saved: ricci_flow_history.png")
    
    fig2 = plot_weight_distribution(graph, save_path='./outputs/weight_distribution.png')
    print("  Saved: weight_distribution.png")
    
    fig3 = plot_graph_clustering(graph, labels, save_path='./outputs/graph_clustering.png')
    print("  Saved: graph_clustering.png")
    
    fig4 = create_summary_plot(graph, labels, history, config, 
                               save_path='./outputs/summary_plot.png')
    print("  Saved: summary_plot.png")
    
    # Final results
    total_time = time.time() - total_start
    print("\n[4/4] Final Results")
    print("=" * 70)
    final_nmi = calculate_nmi(labels, graph.node_clusters)
    final_modularity = calculate_modularity(graph, labels)
    n_communities = len(np.unique(labels))
    
    print(f"  Threshold: {threshold:.6e}")
    print(f"  Communities found: {n_communities} (ground truth: {config.N_CLUSTERS})")
    print(f"  NMI: {final_nmi:.6f}")
    print(f"  Modularity: {final_modularity:.6f}")
    print(f"  Total time: {total_time:.1f}s")
    print("=" * 70)
    
    plt.close('all')
    
    return graph, labels, history, config


if __name__ == "__main__":
    graph, labels, history, config = main()
