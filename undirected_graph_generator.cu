/**
 * undirected_graph_generator.cu
 * 
 * Generates undirected random graphs and stores them in the 3-array format:
 *   - v_adj_list:   Concatenation of all adjacency lists (size |E|)
 *   - v_adj_begin:  Offset where each vertex's adjacency list begins (size |V|)
 *   - v_adj_length: Length of each vertex's adjacency list / degree (size |V|)
 * 
 * Example:
 *   Graph:  0 -- 1
 *           |    |
 *           2 -- 3
 * 
 *   v_adj_list:   [1, 2, 0, 3, 0, 3, 1, 2]
 *                  ^^^^  ^^^^  ^^^^  ^^^^
 *                  v=0   v=1   v=2   v=3
 * 
 *   v_adj_begin:  [0, 2, 4, 6]
 *                  ^  ^  ^  ^
 *                  |  |  |  └─ vertex 3's neighbors start at index 6
 *                  |  |  └──── vertex 2's neighbors start at index 4
 *                  |  └─────── vertex 1's neighbors start at index 2
 *                  └────────── vertex 0's neighbors start at index 0
 * 
 *   v_adj_length: [2, 2, 2, 2]   (all vertices have degree 2)
 * 
 * Usage:
 *   // Get neighbors of vertex v:
 *   int start = v_adj_begin[v];
 *   int len   = v_adj_length[v];
 *   for (int i = 0; i < len; i++) {
 *       int neighbor = v_adj_list[start + i];
 *   }
 * 
 * Compile:
 *   nvcc -o graph_gen undirected_graph_generator.cu -lcurand
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <math.h>

// ============================================================================
// Configuration
// ============================================================================

#define N_NODES 10000        // Number of vertices |V|
#define N_EDGES_TARGET 50000 // Target number of unique undirected edges
#define BLOCK_SIZE 256

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// Step 0: Initialize CURAND states
// ============================================================================

__global__ void init_curand_states(curandState *states, int n, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// ============================================================================
// Step 1: Generate Random Edges (Canonical Form: src < dst)
// ============================================================================

__global__ void generate_random_edges_canonical(
    int *edge_src,
    int *edge_dst,
    int n_edges,
    int n_nodes,
    curandState *states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_edges) return;
    
    curandState localState = states[idx];
    
    // Generate two random node indices using uniform distribution
    // curand_uniform returns (0.0, 1.0]
    // We want integers in [0, n_nodes - 1]
    float rand1 = curand_uniform(&localState);
    float rand2 = curand_uniform(&localState);
    
    int a = (int)truncf(rand1 * (n_nodes - 1 + 0.999999f));
    int b = (int)truncf(rand2 * (n_nodes - 1 + 0.999999f));
    
    // Avoid self-loops
    if (a == b) {
        b = (b + 1) % n_nodes;
    }
    
    // Canonical ordering: ensure src < dst
    if (a > b) {
        int tmp = a;
        a = b;
        b = tmp;
    }
    
    edge_src[idx] = a;
    edge_dst[idx] = b;
    
    states[idx] = localState;
}

// ============================================================================
// Step 2: Duplicate Edges for Undirected Representation
// ============================================================================

__global__ void duplicate_edges_undirected(
    const int *edge_src_in,
    const int *edge_dst_in,
    int *edge_src_out,
    int *edge_dst_out,
    int n_edges_canonical
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_edges_canonical) return;
    
    int a = edge_src_in[idx];
    int b = edge_dst_in[idx];
    
    // Forward direction: a → b
    edge_src_out[2 * idx] = a;
    edge_dst_out[2 * idx] = b;
    
    // Backward direction: b → a
    edge_src_out[2 * idx + 1] = b;
    edge_dst_out[2 * idx + 1] = a;
}

// ============================================================================
// Step 3: Sort Edges by Source (using Thrust)
// ============================================================================

void sort_edges_by_source(int *d_edge_src, int *d_edge_dst, int n_edges) {
    thrust::device_ptr<int> src_ptr(d_edge_src);
    thrust::device_ptr<int> dst_ptr(d_edge_dst);
    thrust::sort_by_key(src_ptr, src_ptr + n_edges, dst_ptr);
}

// ============================================================================
// Step 4: Build 3-Array Format (v_adj_list, v_adj_begin, v_adj_length)
// ============================================================================

/**
 * Count edges per node → this becomes v_adj_length (degree array)
 */
__global__ void count_edges_per_node(
    const int *edge_src,
    int *v_adj_length,  // Output: degree of each vertex
    int n_edges
) {
    // Grid-stride loop for large graphs
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < n_edges; 
         idx += blockDim.x * gridDim.x) {
        
        int src = edge_src[idx];
        atomicAdd(&v_adj_length[src], 1);
    }
}

/**
 * Build the 3-array graph format from sorted edges.
 * 
 * After sorting, edge_dst IS the adjacency list (v_adj_list).
 * We just need to compute v_adj_begin from v_adj_length.
 */
void build_3array_format(
    const int *d_edge_src,      // Input: sorted edge sources
    const int *d_edge_dst,      // Input: sorted edge destinations
    int *d_v_adj_list,          // Output: concatenated adjacency lists
    int *d_v_adj_begin,         // Output: starting offset for each vertex
    int *d_v_adj_length,        // Output: degree of each vertex
    int n_nodes,
    int n_edges
) {
    int n_blocks_edges = (n_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int n_blocks_nodes = (n_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Initialize v_adj_length to zeros
    CUDA_CHECK(cudaMemset(d_v_adj_length, 0, n_nodes * sizeof(int)));
    
    // Step 4a: Count edges per node → v_adj_length (degrees)
    count_edges_per_node<<<n_blocks_edges, BLOCK_SIZE>>>(
        d_edge_src, d_v_adj_length, n_edges
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Step 4b: Exclusive prefix sum of v_adj_length → v_adj_begin
    // v_adj_begin[i] = sum of v_adj_length[0..i-1]
    thrust::device_ptr<int> length_ptr(d_v_adj_length);
    thrust::device_ptr<int> begin_ptr(d_v_adj_begin);
    thrust::exclusive_scan(length_ptr, length_ptr + n_nodes, begin_ptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Step 4c: Copy edge_dst → v_adj_list
    // Since edges are sorted by source, edge_dst is already the adjacency list!
    CUDA_CHECK(cudaMemcpy(d_v_adj_list, d_edge_dst, n_edges * sizeof(int), cudaMemcpyDeviceToDevice));
}

// ============================================================================
// Utility: Print graph in 3-array format
// ============================================================================

void print_graph_3array(
    int *d_v_adj_list,
    int *d_v_adj_begin,
    int *d_v_adj_length,
    int n_nodes,
    int n_edges
) {
    // Copy to host
    int *h_v_adj_list   = (int*)malloc(n_edges * sizeof(int));
    int *h_v_adj_begin  = (int*)malloc(n_nodes * sizeof(int));
    int *h_v_adj_length = (int*)malloc(n_nodes * sizeof(int));
    
    CUDA_CHECK(cudaMemcpy(h_v_adj_list, d_v_adj_list, n_edges * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v_adj_begin, d_v_adj_begin, n_nodes * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v_adj_length, d_v_adj_length, n_nodes * sizeof(int), cudaMemcpyDeviceToHost));
    
    printf("\n==================== Graph Statistics ====================\n");
    printf("Number of vertices |V|: %d\n", n_nodes);
    printf("Number of directed edges |E|: %d\n", n_edges);
    printf("Number of undirected edges: %d\n", n_edges / 2);
    printf("Average degree: %.2f\n", (float)n_edges / n_nodes);
    
    // Compute degree statistics
    int min_deg = h_v_adj_length[0], max_deg = h_v_adj_length[0];
    long long sum_deg = 0;
    int isolated = 0;
    for (int i = 0; i < n_nodes; i++) {
        int deg = h_v_adj_length[i];
        if (deg < min_deg) min_deg = deg;
        if (deg > max_deg) max_deg = deg;
        sum_deg += deg;
        if (deg == 0) isolated++;
    }
    printf("Min degree: %d\n", min_deg);
    printf("Max degree: %d\n", max_deg);
    printf("Isolated vertices (degree 0): %d\n", isolated);
    printf("Sum of degrees: %lld (should equal |E|=%d)\n", sum_deg, n_edges);
    
    // Print the 3 arrays for first few vertices
    printf("\n==================== 3-Array Format ====================\n");
    
    int show_nodes = (n_nodes < 10) ? n_nodes : 10;
    int show_edges = (n_edges < 30) ? n_edges : 30;
    
    printf("\nv_adj_length (degrees) [first %d of %d]:\n  [", show_nodes, n_nodes);
    for (int i = 0; i < show_nodes; i++) {
        printf("%d", h_v_adj_length[i]);
        if (i < show_nodes - 1) printf(", ");
    }
    if (n_nodes > show_nodes) printf(", ...");
    printf("]\n");
    
    printf("\nv_adj_begin (offsets) [first %d of %d]:\n  [", show_nodes, n_nodes);
    for (int i = 0; i < show_nodes; i++) {
        printf("%d", h_v_adj_begin[i]);
        if (i < show_nodes - 1) printf(", ");
    }
    if (n_nodes > show_nodes) printf(", ...");
    printf("]\n");
    
    printf("\nv_adj_list (neighbors) [first %d of %d]:\n  [", show_edges, n_edges);
    for (int i = 0; i < show_edges; i++) {
        printf("%d", h_v_adj_list[i]);
        if (i < show_edges - 1) printf(", ");
    }
    if (n_edges > show_edges) printf(", ...");
    printf("]\n");
    
    // Print adjacency list view for first few vertices
    printf("\n==================== Adjacency List View ====================\n");
    for (int v = 0; v < show_nodes && v < n_nodes; v++) {
        int start = h_v_adj_begin[v];
        int len = h_v_adj_length[v];
        
        printf("Vertex %d (degree=%d, start=%d): [", v, len, start);
        for (int i = 0; i < len && i < 10; i++) {
            printf("%d", h_v_adj_list[start + i]);
            if (i < len - 1 && i < 9) printf(", ");
        }
        if (len > 10) printf(", ...");
        printf("]\n");
    }
    
    // Verify undirected property
    printf("\n==================== Verifying Undirected Property ====================\n");
    int errors = 0;
    for (int v = 0; v < 5 && v < n_nodes; v++) {
        int start = h_v_adj_begin[v];
        int len = h_v_adj_length[v];
        
        for (int i = 0; i < len && i < 3; i++) {
            int neighbor = h_v_adj_list[start + i];
            
            // Check if neighbor has v in its adjacency list
            int n_start = h_v_adj_begin[neighbor];
            int n_len = h_v_adj_length[neighbor];
            
            int found = 0;
            for (int j = 0; j < n_len; j++) {
                if (h_v_adj_list[n_start + j] == v) {
                    found = 1;
                    break;
                }
            }
            
            printf("Edge (%d → %d): reverse (%d → %d) %s\n",
                   v, neighbor, neighbor, v,
                   found ? "EXISTS ✓" : "MISSING ✗");
            
            if (!found) errors++;
        }
    }
    if (errors == 0) {
        printf("All checked edges are bidirectional ✓\n");
    }
    
    free(h_v_adj_list);
    free(h_v_adj_begin);
    free(h_v_adj_length);
}

// ============================================================================
// Demo: How to use the 3-array format
// ============================================================================

/**
 * Example kernel showing how to iterate over neighbors using 3-array format.
 * This computes the sum of neighbor IDs for each vertex (just a demo).
 */
__global__ void demo_neighbor_iteration(
    const int *v_adj_list,
    const int *v_adj_begin,
    const int *v_adj_length,
    long long *neighbor_sum,  // Output: sum of neighbor IDs per vertex
    int n_nodes
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_nodes) return;
    
    int start = v_adj_begin[v];
    int len = v_adj_length[v];
    
    long long sum = 0;
    for (int i = 0; i < len; i++) {
        int neighbor = v_adj_list[start + i];
        sum += neighbor;
    }
    
    neighbor_sum[v] = sum;
}

/**
 * Forman-Ricci curvature kernel using 3-array format.
 * For each edge (u, v): F(e) = 4 - degree(u) - degree(v)
 * 
 * Note: This iterates over vertices and their edges, computing curvature
 * for each directed edge. Each undirected edge is processed twice.
 */
__global__ void forman_ricci_1d_kernel(
    const int *v_adj_list,
    const int *v_adj_begin,
    const int *v_adj_length,
    float *edge_curvature,  // Output: curvature for each edge in v_adj_list order
    int n_nodes
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n_nodes) return;
    
    int start = v_adj_begin[u];
    int deg_u = v_adj_length[u];
    
    for (int i = 0; i < deg_u; i++) {
        int v = v_adj_list[start + i];
        int deg_v = v_adj_length[v];
        
        // Forman-Ricci (1D, unweighted)
        edge_curvature[start + i] = 4.0f - deg_u - deg_v;
    }
}

// ============================================================================
// Main: Complete Pipeline
// ============================================================================

int main() {
    printf("============================================================\n");
    printf("  Undirected Random Graph Generator (CUDA)\n");
    printf("  Output: 3-Array Format (v_adj_list, v_adj_begin, v_adj_length)\n");
    printf("============================================================\n\n");
    
    // Configuration
    int n_nodes = N_NODES;
    int n_edges_canonical = N_EDGES_TARGET;
    int n_edges_directed = 2 * n_edges_canonical;  // After duplication
    
    printf("Configuration:\n");
    printf("  |V| = %d vertices\n", n_nodes);
    printf("  Target undirected edges: %d\n", n_edges_canonical);
    printf("  |E| = %d directed edges (after duplication)\n", n_edges_directed);
    
    // ========================================================================
    // Allocate memory
    // ========================================================================
    
    // Temporary arrays for edge generation
    int *d_edge_src_canonical, *d_edge_dst_canonical;
    CUDA_CHECK(cudaMalloc(&d_edge_src_canonical, n_edges_canonical * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edge_dst_canonical, n_edges_canonical * sizeof(int)));
    
    int *d_edge_src, *d_edge_dst;
    CUDA_CHECK(cudaMalloc(&d_edge_src, n_edges_directed * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edge_dst, n_edges_directed * sizeof(int)));
    
    // Final 3-array format
    int *d_v_adj_list, *d_v_adj_begin, *d_v_adj_length;
    CUDA_CHECK(cudaMalloc(&d_v_adj_list, n_edges_directed * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_v_adj_begin, n_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_v_adj_length, n_nodes * sizeof(int)));
    
    // CURAND states
    curandState *d_states;
    CUDA_CHECK(cudaMalloc(&d_states, n_edges_canonical * sizeof(curandState)));
    
    // ========================================================================
    // Pipeline
    // ========================================================================
    
    printf("\n--- Pipeline ---\n");
    
    // Step 0: Initialize random states
    printf("Step 0: Initializing CURAND states...\n");
    int n_blocks = (n_edges_canonical + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_curand_states<<<n_blocks, BLOCK_SIZE>>>(d_states, n_edges_canonical, time(NULL));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Step 1: Generate canonical edges (src < dst)
    printf("Step 1: Generating %d canonical edges...\n", n_edges_canonical);
    generate_random_edges_canonical<<<n_blocks, BLOCK_SIZE>>>(
        d_edge_src_canonical, d_edge_dst_canonical,
        n_edges_canonical, n_nodes, d_states
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Step 2: Duplicate edges for undirected graph
    printf("Step 2: Duplicating edges (x2 for undirected)...\n");
    duplicate_edges_undirected<<<n_blocks, BLOCK_SIZE>>>(
        d_edge_src_canonical, d_edge_dst_canonical,
        d_edge_src, d_edge_dst,
        n_edges_canonical
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Free canonical arrays
    CUDA_CHECK(cudaFree(d_edge_src_canonical));
    CUDA_CHECK(cudaFree(d_edge_dst_canonical));
    CUDA_CHECK(cudaFree(d_states));
    
    // Step 3: Sort edges by source vertex
    printf("Step 3: Sorting edges by source (thrust::sort_by_key)...\n");
    sort_edges_by_source(d_edge_src, d_edge_dst, n_edges_directed);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Step 4: Build 3-array format
    printf("Step 4: Building 3-array format...\n");
    build_3array_format(
        d_edge_src, d_edge_dst,
        d_v_adj_list, d_v_adj_begin, d_v_adj_length,
        n_nodes, n_edges_directed
    );
    
    // Free edge arrays (now we only need the 3-array format)
    CUDA_CHECK(cudaFree(d_edge_src));
    CUDA_CHECK(cudaFree(d_edge_dst));
    
    // ========================================================================
    // Print results
    // ========================================================================
    
    print_graph_3array(d_v_adj_list, d_v_adj_begin, d_v_adj_length,
                       n_nodes, n_edges_directed);
    
    // ========================================================================
    // Demo: Forman-Ricci curvature
    // ========================================================================
    
    printf("\n==================== Forman-Ricci Curvature Demo ====================\n");
    
    float *d_edge_curvature;
    CUDA_CHECK(cudaMalloc(&d_edge_curvature, n_edges_directed * sizeof(float)));
    
    n_blocks = (n_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
    forman_ricci_1d_kernel<<<n_blocks, BLOCK_SIZE>>>(
        d_v_adj_list, d_v_adj_begin, d_v_adj_length,
        d_edge_curvature, n_nodes
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy curvature to host and show statistics
    float *h_curvature = (float*)malloc(n_edges_directed * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_curvature, d_edge_curvature, n_edges_directed * sizeof(float), cudaMemcpyDeviceToHost));
    
    float min_curv = h_curvature[0], max_curv = h_curvature[0];
    double sum_curv = 0;
    int positive = 0, negative = 0, zero = 0;
    for (int i = 0; i < n_edges_directed; i++) {
        float c = h_curvature[i];
        if (c < min_curv) min_curv = c;
        if (c > max_curv) max_curv = c;
        sum_curv += c;
        if (c > 0) positive++;
        else if (c < 0) negative++;
        else zero++;
    }
    
    printf("Curvature statistics:\n");
    printf("  Min: %.2f\n", min_curv);
    printf("  Max: %.2f\n", max_curv);
    printf("  Mean: %.2f\n", sum_curv / n_edges_directed);
    printf("  Positive: %d (%.1f%%)\n", positive, 100.0 * positive / n_edges_directed);
    printf("  Zero: %d (%.1f%%)\n", zero, 100.0 * zero / n_edges_directed);
    printf("  Negative: %d (%.1f%%)\n", negative, 100.0 * negative / n_edges_directed);
    
    // Show curvature for first few edges
    int *h_v_adj_list = (int*)malloc(n_edges_directed * sizeof(int));
    int *h_v_adj_begin = (int*)malloc(n_nodes * sizeof(int));
    int *h_v_adj_length = (int*)malloc(n_nodes * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_v_adj_list, d_v_adj_list, n_edges_directed * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v_adj_begin, d_v_adj_begin, n_nodes * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v_adj_length, d_v_adj_length, n_nodes * sizeof(int), cudaMemcpyDeviceToHost));
    
    printf("\nFirst few edges with curvature:\n");
    for (int v = 0; v < 3 && v < n_nodes; v++) {
        int start = h_v_adj_begin[v];
        int len = h_v_adj_length[v];
        for (int i = 0; i < len && i < 3; i++) {
            int neighbor = h_v_adj_list[start + i];
            float curv = h_curvature[start + i];
            printf("  Edge (%d, %d): deg(%d)=%d, deg(%d)=%d, F(e) = 4 - %d - %d = %.0f\n",
                   v, neighbor, v, len, neighbor, h_v_adj_length[neighbor],
                   len, h_v_adj_length[neighbor], curv);
        }
    }
    
    // ========================================================================
    // Cleanup
    // ========================================================================
    
    free(h_curvature);
    free(h_v_adj_list);
    free(h_v_adj_begin);
    free(h_v_adj_length);
    
    CUDA_CHECK(cudaFree(d_v_adj_list));
    CUDA_CHECK(cudaFree(d_v_adj_begin));
    CUDA_CHECK(cudaFree(d_v_adj_length));
    CUDA_CHECK(cudaFree(d_edge_curvature));
    
    printf("\n============================================================\n");
    printf("  Done! Graph ready in 3-array format.\n");
    printf("============================================================\n");
    
    return 0;
}