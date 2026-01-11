# Ricci-Curvature-Clustering-CUDA

A GPU-accelerated implementation of Forman-Ricci curvature-based graph clustering in CUDA.

## Introduction

In recent years, there has been increasing attention on geometry-aware models for clustering in order to leverage intrinsic data structures and their geometric characterizations. One of the most well-known methods is Ricci curvature, which is a discretized version of Ricci flow. This method takes advantage of the fact that nodes within a cluster usually have high Ricci curvature, while nodes which connect between clusters (called bridges) usually have low Ricci curvature. Curvature-based thresholding removes low-curvature edges, revealing community structure through the remaining connected components; an associated discrete Ricci flow can further refine the detection.

While these methods have been well-known, until now there has not been any implementation of Ricci curvature-based clustering on GPU, which severely limits their application to large-scale graphs. This repository provides an implementation of Forman-Ricci curvature-based clustering on CUDA.

> **Note:** There are two variants of Ricci curvature: Ollivier-Ricci curvature and Forman-Ricci curvature. This implementation focuses on Forman-Ricci curvature, with Ollivier-Ricci left as possible future work.

## Graph Generation

The code uses a Stochastic Block Model (SBM) to generate input graphs. SBM defines two parameters:
- **P_in**: Probability that two nodes within the same cluster are connected
- **P_out**: Probability that two nodes in different clusters are connected

## Usage

### 1. Set Hyperparameters

Configure the parameters in the `#define` section of the source file:
```c
#define N_NODE 1000
#define NODES_PER_CLUSTER 500
#define N_CLUSTERS 2
#define P_IN 0.12
#define P_OUT 0.01
```

### 2. Compile and Run
```bash
nvcc -arch=native ricci_clustering_cuda.cu -o ricci_clustering_cuda
./ricci_clustering_cuda
```

### 3. Sample Output
```
=== Graph Configuration ===
N_NODE:             1000
N_CLUSTERS:         2
NODES_PER_CLUSTER:  500
P_IN:               0.1200
P_OUT:              0.010000
N_EDGES_MAX:        38928
N_ITERATION:        10
Intra/Inter ratio:  11.9760
===============================

Number of undirected edges: 32399
Verification PASSED: All edges match!
Iter 0: max_curv=160.69, step=0.005657
Iteration 0: sum_weights = 100903.9190
...

=== FINAL RESULTS ===
Threshold: 5.605037e-01
Communities found: 2 (ground truth: 2)
Modularity: 0.853216
NMI: 1.000000

Community sizes:
  Community 0: 500 nodes
  Community 1: 500 nodes

Saved graph to graph_output.txt
```

### 4. Visualization

The output graph and detected clusters are saved to `graph_output.txt`. Use one of the provided visualization scripts:
```bash
python visualize_graph.py
# or
python visualize_graph_large.py
# or
python visualize_graph_improved.py
# or
python visualize_graph_large_optimized.py
```

## Results

**Hardware:** Experiments were performed on a PC with 64GB RAM and an NVIDIA GeForce RTX 4090 (24GB VRAM).

| Nodes | Clusters | Edges | P_in | P_out | Iterations | Avg Time (s) |
|------:|:--------:|------:|-----:|------:|:----------:|-------------:|
| 5,000 | 2 | 3,186,679 | 0.50 | 0.01 | 10 | 7.03 |
| 50,000 | 2 | 25,625,893 | 0.04 | 0.001 | 10 | 74.39 |
| 100,000 | 2 | 102,498,762 | 0.04 | 0.001 | 10 | 625.46 |

## References

1. Y. Tian, Z. Lubberts, and M. Weber, "Curvature-based clustering on graphs," *J. Mach. Learn. Res.*, vol. 26, no. 52, pp. 1–67, 2025.

2. C.-C. Ni, Y.-Y. Lin, F. Luo, and J. Gao, "Community detection on networks with Ricci flow," *Sci. Rep.*, vol. 9, no. 1, pp. 1–12, 2019.

3. A. Samal, R. P. Sreejith, J. Gu, et al., "Comparative analysis of two discretizations of Ricci curvature for complex networks," *Sci. Rep.*, vol. 8, 8650, 2018.

4. [GraphRicciCurvature](https://github.com/saibalmars/GraphRicciCurvature) — Python implementation of Ricci curvature for NetworkX graphs.

## License

MIT License.