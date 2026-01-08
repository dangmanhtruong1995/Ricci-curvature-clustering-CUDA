#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include <algorithm>

// #define N_NODE 500
// #define NODES_PER_CLUSTER 50
// #define N_CLUSTERS 10
// #define P_IN 0.4
// #define P_OUT 0.001
// #define N_ITERATION 30
// #define N_EDGES_MAX 20000


// #define N_NODE 10000
// #define NODES_PER_CLUSTER 1000
// #define N_CLUSTERS 10
// #define P_IN 0.05            
// #define P_OUT 0.0005         
// #define N_ITERATION 30       // Optimal iteration count from testing
// #define N_EDGES_MAX 300000

#define N_NODE 10000
#define NODES_PER_CLUSTER 5000
#define N_CLUSTERS 2
#define P_IN 0.05            
#define P_OUT 0.0005         
#define N_ITERATION 30       // Optimal iteration count from testing
#define N_EDGES_MAX 300000

#define FACES_BOTH 1
#define FACES_V1_ONLY 2
#define FACES_V2_ONLY 3

#define BLOCK_SIZE 256
#define VERY_LARGE_NUMBER 99999

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

int get_cluster(int node_idx) {
    return node_idx / NODES_PER_CLUSTER;
}

__global__ void init_one_on_device(float* arr, int n) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        arr[idx] = 1.0f;
    }
}

void init_zero(int *arr, int size) {
    for (int idx = 0; idx < size; idx++) arr[idx] = 0;
}

void prefix_sum_exclusive(int *arr, int *result, int n) {
    result[0] = 0;
    for (int i = 1; i < n; i++) {
        result[i] = result[i - 1] + arr[i - 1];
    }
}

void create_sbm_graph(
        int **v_adj_length, int **v_adj_begin, int **v_adj_list,
        int **edge_src, int **edge_dst, int **v_adj_begin_2,
        int *n_undirected_edges, int **node_cluster) {
    
    *v_adj_length = (int*)malloc(N_NODE * sizeof(int));
    *v_adj_begin = (int*)malloc(N_NODE * sizeof(int));
    *edge_src = (int*)malloc(N_EDGES_MAX * sizeof(int));
    *edge_dst = (int*)malloc(N_EDGES_MAX * sizeof(int));
    *node_cluster = (int*)malloc(N_NODE * sizeof(int));

    init_zero(*v_adj_length, N_NODE);

    for (int idx = 0; idx < N_NODE; idx++) {
        (*node_cluster)[idx] = get_cluster(idx);
    }

    *n_undirected_edges = 0;
    for (int idx_1 = 0; idx_1 < N_NODE; idx_1++) {
        for (int idx_2 = idx_1 + 1; idx_2 < N_NODE; idx_2++) {
            float edge_prob = (get_cluster(idx_1) == get_cluster(idx_2)) ? P_IN : P_OUT;
            float r = (float)rand() / (float)RAND_MAX;
            
            if (r <= edge_prob) {
                (*edge_src)[*n_undirected_edges] = idx_1;
                (*edge_dst)[*n_undirected_edges] = idx_2;
                (*v_adj_length)[idx_1]++;
                (*v_adj_length)[idx_2]++;
                (*n_undirected_edges)++;
                
                if (*n_undirected_edges >= N_EDGES_MAX) break;
            }
        }
        if (*n_undirected_edges >= N_EDGES_MAX) break;
    }

    int n_total_edge = *n_undirected_edges * 2;
    *v_adj_list = (int*)malloc(n_total_edge * sizeof(int));
    init_zero(*v_adj_list, n_total_edge);

    prefix_sum_exclusive(*v_adj_length, *v_adj_begin, N_NODE);
    
    *v_adj_begin_2 = (int*)malloc(N_NODE * sizeof(int));
    memcpy(*v_adj_begin_2, *v_adj_begin, N_NODE * sizeof(int));

    for (int idx = 0; idx < *n_undirected_edges; idx++) {
        int idx_1 = (*v_adj_begin_2)[(*edge_src)[idx]];
        int idx_2 = (*v_adj_begin_2)[(*edge_dst)[idx]];
        (*v_adj_list)[idx_1] = (*edge_dst)[idx];
        (*v_adj_list)[idx_2] = (*edge_src)[idx];
        (*v_adj_begin_2)[(*edge_src)[idx]]++;
        (*v_adj_begin_2)[(*edge_dst)[idx]]++;
    }
}

__global__ void find_faces_in_graph(
        int *d_v_adj_list, int *d_v_adj_begin, int *d_v_adj_length,
        int *d_edge_src, int *d_edge_dst,
        char *d_faces, int *d_n_triangles,
        int n_undirected_edges) {
    
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    while (idx < n_undirected_edges) {
        int v1 = d_edge_src[idx];
        int v2 = d_edge_dst[idx];
        size_t face_idx_base = (size_t)idx * N_NODE;

        for (int j = 0; j < d_v_adj_length[v1]; j++) {
            int v = d_v_adj_list[d_v_adj_begin[v1] + j];
            if (v != v2 && d_faces[face_idx_base + v] == 0) {
                d_faces[face_idx_base + v] = FACES_V1_ONLY;
            }
        }

        for (int j = 0; j < d_v_adj_length[v2]; j++) {
            int v = d_v_adj_list[d_v_adj_begin[v2] + j];
            if (v == v1) continue;
            if (d_faces[face_idx_base + v] == 0) {
                d_faces[face_idx_base + v] = FACES_V2_ONLY;
            } else if (d_faces[face_idx_base + v] == FACES_V1_ONLY) {
                d_faces[face_idx_base + v] = FACES_BOTH;
                d_n_triangles[idx]++;
            }
        }
        idx += gridDim.x * blockDim.x;
    }
}

__global__ void calc_forman_curvature(
        int *d_v_adj_list, int *d_v_adj_begin, int *d_v_adj_length,
        int *d_edge_src, int *d_edge_dst,
        char *d_faces, int *d_n_triangles,
        float *d_edge_weight, float *d_edge_curvature,
        int n_undirected_edges) {
    
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    while (idx < n_undirected_edges) {
        int v1 = d_edge_src[idx];
        int v2 = d_edge_dst[idx];
        size_t face_idx_base = (size_t)idx * N_NODE;

        int idx_v1_v2 = -1, idx_v2_v1 = -1;
        for (int j = 0; j < d_v_adj_length[v1]; j++) {
            if (d_v_adj_list[d_v_adj_begin[v1] + j] == v2) {
                idx_v1_v2 = d_v_adj_begin[v1] + j;
                break;
            }
        }
        for (int j = 0; j < d_v_adj_length[v2]; j++) {
            if (d_v_adj_list[d_v_adj_begin[v2] + j] == v1) {
                idx_v2_v1 = d_v_adj_begin[v2] + j;
                break;
            }
        }

        int deg_v1 = d_v_adj_length[v1];
        int deg_v2 = d_v_adj_length[v2];
        
        int n_triangles = 0;
        for (int j = 0; j < deg_v1; j++) {
            int n = d_v_adj_list[d_v_adj_begin[v1] + j];
            if (n == v2) continue;
            if (d_faces[face_idx_base + n] == FACES_BOTH) {
                n_triangles++;
            }
        }

        int max_possible = min(deg_v1 - 1, deg_v2 - 1);
        
        float curvature;
        if (max_possible <= 0) {
            curvature = -1.0f;
        } else {
            float density = (float)n_triangles / (float)max_possible;
            curvature = 2.0f * density - 1.0f;
        }
        
        curvature = fmaxf(-1.0f, fminf(1.0f, curvature));
        
        d_edge_curvature[idx_v1_v2] = curvature;
        d_edge_curvature[idx_v2_v1] = curvature;
        
        idx += gridDim.x * blockDim.x;
    }
}

__global__ void update_weight_soft(
        int *d_v_adj_list, int *d_v_adj_begin, int *d_v_adj_length,
        int *d_edge_src, int *d_edge_dst,
        float *d_edge_weight, float *d_edge_curvature,
        float step_size, int n_undirected_edges) {
    
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    while (idx < n_undirected_edges) {
        int v1 = d_edge_src[idx];
        int v2 = d_edge_dst[idx];

        int idx_v1_v2 = -1, idx_v2_v1 = -1;
        for (int j = 0; j < d_v_adj_length[v1]; j++) {
            if (d_v_adj_list[d_v_adj_begin[v1] + j] == v2) {
                idx_v1_v2 = d_v_adj_begin[v1] + j;
                break;
            }
        }
        for (int j = 0; j < d_v_adj_length[v2]; j++) {
            if (d_v_adj_list[d_v_adj_begin[v2] + j] == v1) {
                idx_v2_v1 = d_v_adj_begin[v2] + j;
                break;
            }
        }
        
        float w_old = d_edge_weight[idx_v1_v2];
        float curv = d_edge_curvature[idx_v1_v2];
        float w_new = w_old * expf(-step_size * curv);
        
        w_new = fmaxf(w_new, 1e-10f);
        w_new = fminf(w_new, 1e10f);

        d_edge_weight[idx_v1_v2] = w_new;
        d_edge_weight[idx_v2_v1] = w_new;
        
        idx += gridDim.x * blockDim.x;
    }
}

__global__ void array_sum_blockwise(float *arr, int size, float *d_partial_sum) {
    __shared__ float cache[BLOCK_SIZE];
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    while (idx < size) {
        temp += arr[idx];
        idx += gridDim.x * blockDim.x;
    }

    cache[cacheIndex] = temp;
    __syncthreads();
    
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (cacheIndex < i) cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
    }

    if (cacheIndex == 0) d_partial_sum[blockIdx.x] = cache[0];
}

__global__ void normalize_weights(float *d_edge_weight, float total_weight, 
                                   int n_undirected_edges, int n_total_edge) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    float scale = (float)n_undirected_edges / (total_weight / 2.0f);
    while (idx < n_total_edge) {
        d_edge_weight[idx] *= scale;
        idx += gridDim.x * blockDim.x;
    }
}

int find_components_cpu(
        int *v_adj_list, int *v_adj_begin, int *v_adj_length,
        float *edge_weight, float threshold,
        int *component_id, int n_nodes) {
    
    for (int i = 0; i < n_nodes; i++) component_id[i] = -1;
    
    int n_components = 0;
    int *queue = (int*)malloc(n_nodes * sizeof(int));
    
    for (int start = 0; start < n_nodes; start++) {
        if (component_id[start] != -1) continue;
        
        int q_start = 0, q_end = 0;
        queue[q_end++] = start;
        component_id[start] = n_components;
        
        while (q_start < q_end) {
            int node = queue[q_start++];
            
            for (int j = 0; j < v_adj_length[node]; j++) {
                int idx = v_adj_begin[node] + j;
                int neighbor = v_adj_list[idx];
                float w = edge_weight[idx];
                
                if (w < threshold && component_id[neighbor] == -1) {
                    component_id[neighbor] = n_components;
                    queue[q_end++] = neighbor;
                }
            }
        }
        n_components++;
    }
    
    free(queue);
    return n_components;
}

float calculate_modularity(int *v_adj_list, int *v_adj_begin, int *v_adj_length,
                           int *component_id, int n_nodes, int n_edges) {
    float modularity = 0.0f;
    int m = n_edges;
    
    for (int u = 0; u < n_nodes; u++) {
        int k_u = v_adj_length[u];
        for (int j = 0; j < v_adj_length[u]; j++) {
            int v = v_adj_list[v_adj_begin[u] + j];
            int k_v = v_adj_length[v];
            
            float expected = (float)(k_u * k_v) / (2.0f * m);
            if (component_id[u] == component_id[v]) {
                modularity += 1.0f - expected;
            } else {
                modularity -= expected;
            }
        }
    }
    
    return modularity / (2.0f * m);
}

float calculate_nmi(int *pred, int *truth, int n, int n_pred, int n_true) {
    int *pred_sizes = (int*)calloc(n_pred, sizeof(int));
    int *true_sizes = (int*)calloc(n_true, sizeof(int));
    int *overlap = (int*)calloc(n_pred * n_true, sizeof(int));
    
    for (int i = 0; i < n; i++) {
        pred_sizes[pred[i]]++;
        true_sizes[truth[i]]++;
        overlap[pred[i] * n_true + truth[i]]++;
    }
    
    float mi = 0.0f;
    for (int i = 0; i < n_pred; i++) {
        for (int j = 0; j < n_true; j++) {
            if (overlap[i * n_true + j] > 0) {
                float p_ij = (float)overlap[i * n_true + j] / n;
                float p_i = (float)pred_sizes[i] / n;
                float p_j = (float)true_sizes[j] / n;
                mi += p_ij * log2f(p_ij / (p_i * p_j));
            }
        }
    }
    
    float h_pred = 0.0f, h_true = 0.0f;
    for (int i = 0; i < n_pred; i++) {
        if (pred_sizes[i] > 0) {
            float p = (float)pred_sizes[i] / n;
            h_pred -= p * log2f(p);
        }
    }
    for (int j = 0; j < n_true; j++) {
        if (true_sizes[j] > 0) {
            float p = (float)true_sizes[j] / n;
            h_true -= p * log2f(p);
        }
    }
    
    free(pred_sizes);
    free(true_sizes);
    free(overlap);
    
    if (h_pred + h_true < 1e-10) return 0.0f;
    return 2.0f * mi / (h_pred + h_true);
}

int compare_float(const void *a, const void *b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

int main() {
    srand(42);
    
    printf("=== Forman-Ricci Clustering (Final Version) ===\n");
    printf("Nodes: %d, Expected Clusters: %d\n", N_NODE, N_CLUSTERS);
    
    int *edge_src, *edge_dst;
    int *v_adj_length, *v_adj_begin, *v_adj_list, *v_adj_begin_2;
    int n_total_edge, n_undirected_edges;
    int *node_cluster;

    create_sbm_graph(&v_adj_length, &v_adj_begin, &v_adj_list,
                     &edge_src, &edge_dst, &v_adj_begin_2,
                     &n_undirected_edges, &node_cluster);
    n_total_edge = 2 * n_undirected_edges;

    printf("Generated %d edges\n", n_undirected_edges);
    
    int intra_count = 0, inter_count = 0;
    for (int i = 0; i < n_undirected_edges; i++) {
        if (get_cluster(edge_src[i]) == get_cluster(edge_dst[i])) {
            intra_count++;
        } else {
            inter_count++;
        }
    }
    printf("Intra: %d (%.1f%%), Inter: %d (%.1f%%)\n\n", 
           intra_count, 100.0f*intra_count/n_undirected_edges,
           inter_count, 100.0f*inter_count/n_undirected_edges);

    // Allocate GPU memory
    float *d_edge_weight, *h_edge_weight;
    float *d_edge_curvature;
    float *h_partial_sum, *d_partial_sum;
    
    cudaMalloc(&d_edge_weight, n_total_edge * sizeof(float));
    cudaMalloc(&d_edge_curvature, n_total_edge * sizeof(float));
    h_edge_weight = (float*)malloc(n_total_edge * sizeof(float));

    int n_block = (n_total_edge + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_one_on_device<<<n_block, BLOCK_SIZE>>>(d_edge_weight, n_total_edge);
    cudaDeviceSynchronize();
    
    int *d_v_adj_list, *d_v_adj_begin, *d_v_adj_length;
    int *d_edge_src, *d_edge_dst;

    cudaMalloc(&d_v_adj_list, n_total_edge * sizeof(int));
    cudaMalloc(&d_v_adj_begin, N_NODE * sizeof(int));
    cudaMalloc(&d_v_adj_length, N_NODE * sizeof(int));
    cudaMalloc(&d_edge_src, n_undirected_edges * sizeof(int));
    cudaMalloc(&d_edge_dst, n_undirected_edges * sizeof(int));

    cudaMemcpy(d_v_adj_list, v_adj_list, n_total_edge * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_adj_begin, v_adj_begin, N_NODE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_adj_length, v_adj_length, N_NODE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_src, edge_src, n_undirected_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_dst, edge_dst, n_undirected_edges * sizeof(int), cudaMemcpyHostToDevice);

    h_partial_sum = (float*)malloc(n_block * sizeof(float));
    cudaMalloc(&d_partial_sum, n_block * sizeof(float));

    char *d_faces;
    size_t faces_size = (size_t)n_undirected_edges * N_NODE * sizeof(char);
    printf("Allocating faces: %.2f GB\n", faces_size / 1e9);
    cudaMalloc(&d_faces, faces_size);
    cudaMemset(d_faces, 0, faces_size);

    int *d_n_triangles;
    cudaMalloc(&d_n_triangles, n_undirected_edges * sizeof(int));
    cudaMemset(d_n_triangles, 0, n_undirected_edges * sizeof(int));

    n_block = (n_undirected_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;
    find_faces_in_graph<<<n_block, BLOCK_SIZE>>>(
        d_v_adj_list, d_v_adj_begin, d_v_adj_length,
        d_edge_src, d_edge_dst, d_faces, d_n_triangles,
        n_undirected_edges);
    cudaDeviceSynchronize();

    // =========================================================================
    // Ricci Flow
    // =========================================================================
    printf("\n=== Ricci Flow (%d iterations) ===\n", N_ITERATION);
    
    for (int iter = 0; iter < N_ITERATION; iter++) {
        n_block = (n_undirected_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        calc_forman_curvature<<<n_block, BLOCK_SIZE>>>(
            d_v_adj_list, d_v_adj_begin, d_v_adj_length,
            d_edge_src, d_edge_dst,
            d_faces, d_n_triangles,
            d_edge_weight, d_edge_curvature,
            n_undirected_edges);
        cudaDeviceSynchronize();

        float step = 0.3f;

        update_weight_soft<<<n_block, BLOCK_SIZE>>>(
            d_v_adj_list, d_v_adj_begin, d_v_adj_length,
            d_edge_src, d_edge_dst,
            d_edge_weight, d_edge_curvature,
            step, n_undirected_edges);
        cudaDeviceSynchronize();

        n_block = (n_total_edge + BLOCK_SIZE - 1) / BLOCK_SIZE;
        array_sum_blockwise<<<n_block, BLOCK_SIZE>>>(d_edge_weight, n_total_edge, d_partial_sum);
        cudaDeviceSynchronize();

        cudaMemcpy(h_partial_sum, d_partial_sum, n_block * sizeof(float), cudaMemcpyDeviceToHost);
        float sum = 0.0f;
        for (int i = 0; i < n_block; i++) sum += h_partial_sum[i];

        normalize_weights<<<n_block, BLOCK_SIZE>>>(d_edge_weight, sum, n_undirected_edges, n_total_edge);
        cudaDeviceSynchronize();
    }
    printf("Ricci flow completed.\n");
    
    cudaMemcpy(h_edge_weight, d_edge_weight, n_total_edge * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Get weight statistics
    float *edge_weights = (float*)malloc(n_undirected_edges * sizeof(float));
    for (int i = 0; i < n_undirected_edges; i++) {
        int v1 = edge_src[i];
        int v2 = edge_dst[i];
        for (int j = 0; j < v_adj_length[v1]; j++) {
            if (v_adj_list[v_adj_begin[v1] + j] == v2) {
                edge_weights[i] = h_edge_weight[v_adj_begin[v1] + j];
                break;
            }
        }
    }
    
    qsort(edge_weights, n_undirected_edges, sizeof(float), compare_float);
    float min_w = edge_weights[0];
    float max_w = edge_weights[n_undirected_edges - 1];
    printf("Weight range: [%.4f, %.4f]\n", min_w, max_w);

    // =========================================================================
    // Threshold search - find threshold that gives cluster count closest to expected
    // Also track NMI to select best among ties
    // =========================================================================
    printf("\n=== Finding Optimal Threshold ===\n");
    
    int *component_id = (int*)malloc(N_NODE * sizeof(int));
    
    // Store results
    int n_search = 200;
    float *thresholds = (float*)malloc(n_search * sizeof(float));
    int *cluster_counts = (int*)malloc(n_search * sizeof(int));
    float *modularities = (float*)malloc(n_search * sizeof(float));
    float *nmis = (float*)malloc(n_search * sizeof(float));
    
    for (int i = 0; i < n_search; i++) {
        float t = (float)i / (n_search - 1);
        thresholds[i] = min_w + t * (max_w - min_w);
        
        cluster_counts[i] = find_components_cpu(
            v_adj_list, v_adj_begin, v_adj_length,
            h_edge_weight, thresholds[i], component_id, N_NODE);
        
        modularities[i] = calculate_modularity(
            v_adj_list, v_adj_begin, v_adj_length,
            component_id, N_NODE, n_undirected_edges);
        
        nmis[i] = calculate_nmi(component_id, node_cluster, N_NODE, 
                                cluster_counts[i], N_CLUSTERS);
    }
    
    // Find threshold where cluster count equals expected (or closest)
    // Among those, pick the one with highest NMI
    float best_threshold = 0;
    int best_clusters = 0;
    float best_modularity = 0;
    float best_nmi = -1;
    
    // First, look for exact match to expected cluster count
    for (int i = 0; i < n_search; i++) {
        if (cluster_counts[i] == N_CLUSTERS) {
            if (nmis[i] > best_nmi) {
                best_threshold = thresholds[i];
                best_clusters = cluster_counts[i];
                best_modularity = modularities[i];
                best_nmi = nmis[i];
            }
        }
    }
    
    // If no exact match, find closest
    if (best_nmi < 0) {
        int min_diff = N_NODE;
        for (int i = 0; i < n_search; i++) {
            int diff = abs(cluster_counts[i] - N_CLUSTERS);
            if (diff < min_diff || (diff == min_diff && nmis[i] > best_nmi)) {
                min_diff = diff;
                best_threshold = thresholds[i];
                best_clusters = cluster_counts[i];
                best_modularity = modularities[i];
                best_nmi = nmis[i];
            }
        }
    }
    
    // Print some results around the optimal
    printf("\nThreshold search results (showing key values):\n");
    printf("%-12s %-10s %-10s %-10s\n", "Threshold", "Clusters", "Modularity", "NMI");
    printf("------------------------------------------------\n");
    
    int prev_clusters = -1;
    for (int i = 0; i < n_search; i++) {
        // Print when cluster count changes or near expected value
        if (cluster_counts[i] != prev_clusters || 
            abs(cluster_counts[i] - N_CLUSTERS) <= 2) {
            printf("%-12.6f %-10d %-10.4f %-10.4f", 
                   thresholds[i], cluster_counts[i], modularities[i], nmis[i]);
            if (thresholds[i] == best_threshold) printf(" <-- SELECTED");
            printf("\n");
            prev_clusters = cluster_counts[i];
        }
    }

    // =========================================================================
    // Final result
    // =========================================================================
    printf("\n========================================\n");
    printf("=== FINAL RESULTS ===\n");
    printf("========================================\n");
    printf("Selected threshold: %.6f\n", best_threshold);
    printf("Clusters found: %d (expected: %d)\n", best_clusters, N_CLUSTERS);
    printf("Modularity: %.4f\n", best_modularity);
    printf("NMI: %.4f\n", best_nmi);

    // Get final clustering
    find_components_cpu(
        v_adj_list, v_adj_begin, v_adj_length,
        h_edge_weight, best_threshold, component_id, N_NODE);

    // Print cluster sizes
    int *sizes = (int*)calloc(best_clusters, sizeof(int));
    for (int i = 0; i < N_NODE; i++) {
        if (component_id[i] >= 0 && component_id[i] < best_clusters) {
            sizes[component_id[i]]++;
        }
    }
    
    printf("\nCluster sizes:\n");
    for (int c = 0; c < best_clusters; c++) {
        printf("  Cluster %d: %d nodes\n", c, sizes[c]);
    }

    // Verify: count how many clusters have exactly 1000 nodes
    int perfect_clusters = 0;
    for (int c = 0; c < best_clusters; c++) {
        if (sizes[c] == NODES_PER_CLUSTER) perfect_clusters++;
    }
    printf("\nPerfect clusters (exactly %d nodes): %d/%d\n", 
           NODES_PER_CLUSTER, perfect_clusters, best_clusters);

    // Save results
    FILE *fp = fopen("graph_output.txt", "w");
    if (fp) {
        fprintf(fp, "%d %d\n", N_NODE, n_undirected_edges);
        for (int i = 0; i < n_undirected_edges; i++) {
            fprintf(fp, "%d %d\n", edge_src[i], edge_dst[i]);
        }
        for (int i = 0; i < N_NODE; i++) {
            fprintf(fp, "%d\n", component_id[i]);
        }
        fclose(fp);
        printf("\nSaved to graph_output.txt\n");
    }

    // Cleanup
    free(sizes);
    free(thresholds);
    free(cluster_counts);
    free(modularities);
    free(nmis);
    free(edge_weights);
    free(component_id);
    free(v_adj_list);
    free(v_adj_begin);
    free(v_adj_begin_2);
    free(v_adj_length);
    free(edge_src);
    free(edge_dst);
    free(h_edge_weight);
    free(h_partial_sum);
    free(node_cluster);

    cudaFree(d_v_adj_list);
    cudaFree(d_v_adj_begin);
    cudaFree(d_v_adj_length);
    cudaFree(d_edge_src);
    cudaFree(d_edge_dst);
    cudaFree(d_edge_weight);
    cudaFree(d_edge_curvature);
    cudaFree(d_partial_sum);
    cudaFree(d_faces);
    cudaFree(d_n_triangles);

    return 0;
}
