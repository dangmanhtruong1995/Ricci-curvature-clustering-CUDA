#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <assert.h>

#define N_NODE 1000
#define NODES_PER_CLUSTER 100
#define N_CLUSTERS (N_NODE / NODES_PER_CLUSTER)
// #define N_EDGES_MAX 10000
#define N_EDGES_MAX 50000
#define STEP_SIZE 0.1
// #define STEP_SIZE 0.001
#define N_EDGES_TARGET 3000
// #define N_ITERATION 57
#define N_ITERATION 10

// SBM probabilities
#define P_IN 0.3
#define P_OUT 0.01

#define BLOCK_SIZE 256
#define VERY_LARGE_NUMBER 99999.0

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


// =========================================================================
// Utility functions
// =========================================================================
int get_cluster(int node_idx) {
    return node_idx / NODES_PER_CLUSTER;
}

double get_edge_probability(int node_1, int node_2) {
    int cluster_1 = get_cluster(node_1);
    int cluster_2 = get_cluster(node_2);
    
    if (cluster_1 == cluster_2) {
        return P_IN;
    } else {
        return P_OUT;
    }
}

__global__ void init_one_on_device(double* arr, int n) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < n; 
         idx += blockDim.x * gridDim.x) {
        arr[idx] = 1.0;
    }
}

void init_zero(int *arr, int size){
    int idx;
    for (idx=0; idx<size; idx++){
        arr[idx] = 0;
    }
}

void prefix_sum_exclusive(int *arr, int *result, int n) {
    result[0] = 0;
    for (int i = 1; i < n; i++) {
        result[i] = result[i - 1] + arr[i - 1];
    }
}

// =========================================================================
// Sorting on GPU (double precision)
// =========================================================================
__global__ void bitonicSortGPU(double* arr, int j, int k)
{
    unsigned int i, ij;

    i = threadIdx.x + blockDim.x * blockIdx.x;

    ij = i ^ j;

    if (ij > i)
    {
        if ((i & k) == 0)
        {
            if (arr[i] > arr[ij])
            {
                double temp = arr[i];
                arr[i] = arr[ij];
                arr[ij] = temp;
            }
        }
        else
        {
            if (arr[i] < arr[ij])
            {
                double temp = arr[i];
                arr[i] = arr[ij];
                arr[ij] = temp;
            }
        }
    }
}

unsigned int next_power_of_2(unsigned int n) {
    if (n == 0) return 1;
    
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

__global__ void copy_with_pad(double *src, double *dst, unsigned int arr_size, unsigned int next_pow){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < next_pow){
        if (idx < arr_size){
            dst[idx] = src[idx];
        }
        else {
            dst[idx] = VERY_LARGE_NUMBER;
        }

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void copy_back_from_padded_array(double *src, double *dst, unsigned int arr_size){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < arr_size){
        dst[idx] = src[idx];
        idx += blockDim.x * gridDim.x;
    }
}

// =========================================================================
// Graph operations
// =========================================================================
void create_sbm_graph(
        int **v_adj_length,
        int **v_adj_begin,
        int **v_adj_list,
        int **edge_src,
        int **edge_dst,
        int **v_adj_begin_2,
        int *n_undirected_edges,
        int **node_cluster
    ) {
    int idx_1, idx_2, idx;
    double random_number;
    double edge_prob;
    int n_total_edge = 0;

    *v_adj_length = (int*)malloc(N_NODE * sizeof(int));
    *v_adj_begin = (int*)malloc(N_NODE * sizeof(int));
    *edge_src = (int*)malloc(N_EDGES_MAX * sizeof(int));
    *edge_dst = (int*)malloc(N_EDGES_MAX * sizeof(int));
    *node_cluster = (int*)malloc(N_NODE * sizeof(int));

    init_zero(*v_adj_length, N_NODE);
    init_zero(*v_adj_begin, N_NODE);
    init_zero(*edge_src, N_EDGES_MAX);
    init_zero(*edge_dst, N_EDGES_MAX);

    for (idx = 0; idx < N_NODE; idx++) {
        (*node_cluster)[idx] = get_cluster(idx);
    }

    *n_undirected_edges = 0;
    for (idx_1 = 0; idx_1 < N_NODE; idx_1++) {
        for (idx_2 = idx_1 + 1; idx_2 < N_NODE; idx_2++) {
            edge_prob = get_edge_probability(idx_1, idx_2);
            
            random_number = (double)rand() / (double)(RAND_MAX);
            
            if (random_number <= edge_prob) {
                (*edge_src)[*n_undirected_edges] = idx_1;
                (*edge_dst)[*n_undirected_edges] = idx_2;

                (*v_adj_length)[idx_1]++;
                (*v_adj_length)[idx_2]++;

                (*n_undirected_edges)++;
            }
            
            if (*n_undirected_edges >= N_EDGES_MAX) {
                break;
            }
        }
        if (*n_undirected_edges >= N_EDGES_MAX) {
            break;
        }
    }

    n_total_edge = *n_undirected_edges * 2;
    *v_adj_list = (int*)malloc(n_total_edge * sizeof(int));
    init_zero(*v_adj_list, n_total_edge);

    prefix_sum_exclusive(*v_adj_length, *v_adj_begin, N_NODE);
    
    *v_adj_begin_2 = (int*)malloc(N_NODE * sizeof(int));
    memcpy(*v_adj_begin_2, *v_adj_begin, N_NODE * sizeof(int));

    for (idx = 0; idx < *n_undirected_edges; idx++) {
        idx_1 = (*v_adj_begin_2)[(*edge_src)[idx]];
        idx_2 = (*v_adj_begin_2)[(*edge_dst)[idx]];

        (*v_adj_list)[idx_1] = (*edge_dst)[idx];
        (*v_adj_list)[idx_2] = (*edge_src)[idx];

        (*v_adj_begin_2)[(*edge_src)[idx]]++;
        (*v_adj_begin_2)[(*edge_dst)[idx]]++;
    }
}

int verify_graph(int *edge_src, int *edge_dst, int n_undirected_edges,
                 int *v_adj_list, int *v_adj_begin, int *v_adj_length) {
    int errors = 0;
    
    for (int i = 0; i < n_undirected_edges; i++) {
        int src = edge_src[i];
        int dst = edge_dst[i];
        
        int found_forward = 0;
        for (int j = 0; j < v_adj_length[src]; j++) {
            if (v_adj_list[v_adj_begin[src] + j] == dst) {
                found_forward = 1;
                break;
            }
        }
        if (!found_forward) {
            printf("ERROR: Edge %d->%d not found in adjacency list\n", src, dst);
            errors++;
        }
        
        int found_backward = 0;
        for (int j = 0; j < v_adj_length[dst]; j++) {
            if (v_adj_list[v_adj_begin[dst] + j] == src) {
                found_backward = 1;
                break;
            }
        }
        if (!found_backward) {
            printf("ERROR: Edge %d->%d not found in adjacency list\n", dst, src);
            errors++;
        }
    }
    
    int total_adj_entries = 0;
    for (int i = 0; i < N_NODE; i++) {
        total_adj_entries += v_adj_length[i];
    }
    
    if (total_adj_entries != n_undirected_edges * 2) {
        printf("ERROR: Edge count mismatch. Expected %d, got %d\n", 
               n_undirected_edges * 2, total_adj_entries);
        errors++;
    }
    
    if (errors == 0) {
        printf("Verification PASSED: All edges match!\n");
    } else {
        printf("Verification FAILED: %d errors found\n", errors);
    }
    
    return errors;
}

// =========================================================================
// Forman-Ricci curvature calculation on GPU (double precision)
// =========================================================================
__global__ void calc_forman_ricci_curvature(
        int *d_v_adj_list, int *d_v_adj_begin, int *d_v_adj_length,
        int *d_edge_src, int *d_edge_dst,
        double *d_edge_weight, double *d_edge_curvature,
        int n_undirected_edges
        ){    
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    int j;
    int v1;
    int v2;
    int v;
    int idx_v1_v2_adj_list;
    int idx_v2_v1_adj_list;
    double w_e, w_v1, w_v2, ev1_sum, ev2_sum;

    while (idx < n_undirected_edges){
        v1 = d_edge_src[idx];
        v2 = d_edge_dst[idx];

        for (j=0; j<d_v_adj_length[v1]; j++){
            v = d_v_adj_list[d_v_adj_begin[v1] + j];
            if (v == v2){
                idx_v1_v2_adj_list = d_v_adj_begin[v1] + j;
                break;
            }
        }

        idx_v2_v1_adj_list = -1;
        for (j = 0; j < d_v_adj_length[v2]; j++) {
            if (d_v_adj_list[d_v_adj_begin[v2] + j] == v1) {
                idx_v2_v1_adj_list = d_v_adj_begin[v2] + j;
                break;
            }
        }

        w_e = d_edge_weight[idx_v1_v2_adj_list];

        w_v1 = 1.0;
        w_v2 = 1.0;
        ev1_sum = 0.0;
        ev2_sum = 0.0;

        for (j=0; j<d_v_adj_length[v1]; j++){
            v = d_v_adj_list[d_v_adj_begin[v1] + j];
            if (v == v2){
                continue;
            }
            ev1_sum += (w_v1 / sqrt(w_e * d_edge_weight[d_v_adj_begin[v1] + j]));
        }

        for (j=0; j<d_v_adj_length[v2]; j++){
            v = d_v_adj_list[d_v_adj_begin[v2] + j];
            if (v == v1){
                continue;
            }
            ev2_sum += (w_v2 / sqrt(w_e * d_edge_weight[d_v_adj_begin[v2] + j]));
        }

        d_edge_curvature[idx_v1_v2_adj_list] = w_e * (w_v1 / w_e + w_v2 / w_e - (ev1_sum + ev2_sum));
        d_edge_curvature[idx_v2_v1_adj_list] = d_edge_curvature[idx_v1_v2_adj_list];

        idx += gridDim.x*blockDim.x;
    }
}

__global__ void update_weight(
        int *d_v_adj_list, int *d_v_adj_begin, int *d_v_adj_length,
        int *d_edge_src, int *d_edge_dst,
        double *d_edge_weight, double *d_edge_curvature,
        int n_undirected_edges
        ){    
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int j;
    int v1;
    int v2;
    int v;
    int idx_v1_v2_adj_list;
    int idx_v2_v1_adj_list;
    double w_new;

    while (idx < n_undirected_edges){
        v1 = d_edge_src[idx];
        v2 = d_edge_dst[idx];

        for (j=0; j<d_v_adj_length[v1]; j++){
            v = d_v_adj_list[d_v_adj_begin[v1] + j];
            if (v == v2){
                idx_v1_v2_adj_list = d_v_adj_begin[v1] + j;
                break;
            }
        }

        idx_v2_v1_adj_list = -1;
        for (j = 0; j < d_v_adj_length[v2]; j++) {
            if (d_v_adj_list[d_v_adj_begin[v2] + j] == v1) {
                idx_v2_v1_adj_list = d_v_adj_begin[v2] + j;
                break;
            }
        }
        
        w_new = (1.0 - STEP_SIZE * d_edge_curvature[idx_v1_v2_adj_list]) * d_edge_weight[idx_v1_v2_adj_list];
        
        // Clamp to prevent negative weights
        // if (w_new < 0.0001) w_new = 0.0001;
        // if (w_new < 1e-30) w_new = 1e-30;  // Much lower floor
        if (w_new < 1e-10) w_new = 1e-10;

        d_edge_weight[idx_v1_v2_adj_list] = w_new;
        d_edge_weight[idx_v2_v1_adj_list] = w_new;
        
        idx += gridDim.x*blockDim.x;
    }        
}

__global__ void array_sum_blockwise(
        double *arr,
        int size,
        double *d_partial_sum
){
    __shared__ double cache[BLOCK_SIZE];
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int cacheIndex = threadIdx.x;

    double temp = 0.0;
	
    while (idx < size) {
        temp += arr[idx];
        idx += gridDim.x * blockDim.x;
    }

    cache[cacheIndex] = temp;
    __syncthreads();
	
    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0){
        d_partial_sum[blockIdx.x] = cache[0];
    }
}

__global__ void normalize_weights(
        double *d_edge_weight,
        double total_weight,
        int n_total_edge
) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    while (idx < n_total_edge) {
        d_edge_weight[idx] = d_edge_weight[idx] / total_weight;
        idx += gridDim.x * blockDim.x;
    }
}

__global__ void clamp_weights_after_norm(double *d_edge_weight, int n_total_edge) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    while (idx < n_total_edge) {
        if (d_edge_weight[idx] < 1e-10) d_edge_weight[idx] = 1e-10;
        idx += gridDim.x * blockDim.x;
    }
}

// =========================================================================
// CPU Reference Implementation (double precision)
// =========================================================================
int find_connected_components_cpu_reference(
    int *v_adj_list, int *v_adj_begin, int *v_adj_length,
    double *edge_weight, double threshold,
    int *component_id,
    int n_nodes
) {
    for (int i = 0; i < n_nodes; i++) {
        component_id[i] = -1;
    }
    
    int n_components = 0;
    int *queue = (int*)malloc(n_nodes * sizeof(int));
    
    for (int start = 0; start < n_nodes; start++) {
        if (component_id[start] != -1) continue;
        
        int queue_start = 0, queue_end = 0;
        queue[queue_end++] = start;
        component_id[start] = n_components;
        
        while (queue_start < queue_end) {
            int node = queue[queue_start++];
            
            for (int j = 0; j < v_adj_length[node]; j++) {
                int idx = v_adj_begin[node] + j;
                int neighbor = v_adj_list[idx];
                double w = edge_weight[idx];
                
                // if (w >= threshold && component_id[neighbor] == -1){
                // if (w <= threshold && component_id[neighbor] == -1){
                if (w < threshold && component_id[neighbor] == -1){
                    component_id[neighbor] = n_components;
                    queue[queue_end++] = neighbor;
                }
            }
        }
        
        n_components++;
    }
    
    free(queue);
    return n_components;
}

double calculate_modularity_cpu_reference(
    int *v_adj_list, int *v_adj_begin, int *v_adj_length,
    int *component_id,
    int n_nodes, int n_edges
) {
    int m = n_edges;
    double modularity = 0.0;
    
    for (int u = 0; u < n_nodes; u++) {
        int k_u = v_adj_length[u];
        
        for (int j = 0; j < v_adj_length[u]; j++) {
            int v = v_adj_list[v_adj_begin[u] + j];
            int k_v = v_adj_length[v];
            
            if (component_id[u] == component_id[v]) {
                modularity += 1.0 - (double)(k_u * k_v) / (2.0 * m);
            } else {
                modularity += 0.0 - (double)(k_u * k_v) / (2.0 * m);
            }
        }
    }
    
    modularity /= (2.0 * m);
    return modularity;
}


int main(){
    srand(time(NULL));
    int n_block;
    int *edge_src, *edge_dst;
    int idx, iter, i, j, k;
    double edge_weight_sum;

    int *v_adj_length, *v_adj_begin, *v_adj_list;  
    int *v_adj_begin_2;
    double *h_partial_sum, *d_partial_sum;
    int n_total_edge, n_undirected_edges;

    int *node_cluster;
    create_sbm_graph(
        &v_adj_length, &v_adj_begin, &v_adj_list,
        &edge_src, &edge_dst, &v_adj_begin_2,
        &n_undirected_edges, &node_cluster);
    n_total_edge = 2*n_undirected_edges;

    printf("Generated %d undirected edges (%d directed entries)\n", n_undirected_edges, n_total_edge);

    // Verify the graph
    verify_graph(edge_src, edge_dst, n_undirected_edges,
             v_adj_list, v_adj_begin, v_adj_length);

    // Ricci curvature (double precision)
    double *d_edge_weight, *h_edge_weight;
    double *d_edge_curvature, *h_edge_curvature;
    
    cudaMalloc(&d_edge_weight, n_total_edge * sizeof(double));
    cudaMalloc(&d_edge_curvature, n_total_edge * sizeof(double));
    h_edge_weight = (double*)malloc(n_total_edge * sizeof(double));
    h_edge_curvature = (double*)malloc(n_total_edge * sizeof(double));

    n_block = (n_total_edge+BLOCK_SIZE-1)/BLOCK_SIZE; 
    init_one_on_device<<<n_block, BLOCK_SIZE>>>(d_edge_weight, n_total_edge);
    
    // =========================================================================
    // Allocate device memory and copy graph data to GPU
    // =========================================================================
    int *d_v_adj_list, *d_v_adj_begin, *d_v_adj_length;
    int *d_edge_src, *d_edge_dst;

    cudaMalloc(&d_v_adj_list, n_total_edge * sizeof(int));
    cudaMalloc(&d_v_adj_begin, N_NODE * sizeof(int));
    cudaMalloc(&d_v_adj_length, N_NODE * sizeof(int));
    cudaMalloc(&d_edge_src, n_undirected_edges * sizeof(int));
    cudaMalloc(&d_edge_dst, n_undirected_edges * sizeof(int));
    cudaCheckErrors("cudaMalloc failed");

    cudaMemcpy(d_v_adj_list, v_adj_list, n_total_edge * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_adj_begin, v_adj_begin, N_NODE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_adj_length, v_adj_length, N_NODE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_src, edge_src, n_undirected_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_dst, edge_dst, n_undirected_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy failed");
    
    n_block = (n_total_edge+BLOCK_SIZE-1)/BLOCK_SIZE;    
    h_partial_sum = (double*)malloc(n_block * sizeof(double));
    cudaMalloc(&d_partial_sum, n_block * sizeof(double));
    cudaCheckErrors("cudaMalloc for d_partial_sum failed");

    // Calculate Ricci curvature, update then normalize weights
    for (iter=0; iter<N_ITERATION; iter++){
        n_block = (n_undirected_edges+BLOCK_SIZE-1)/BLOCK_SIZE; 
        calc_forman_ricci_curvature<<<n_block, BLOCK_SIZE>>>(
            d_v_adj_list, d_v_adj_begin, d_v_adj_length,
            d_edge_src, d_edge_dst,
            d_edge_weight, d_edge_curvature,
            n_undirected_edges);
        cudaCheckErrors("Kernel launch failed");
        cudaDeviceSynchronize();

        cudaMemcpy(h_edge_curvature, d_edge_curvature, n_total_edge*sizeof(double), cudaMemcpyDeviceToHost);

        // DEBUG: Check weights AFTER update, BEFORE normalization
        if (iter == N_ITERATION - 1) {  // Only on last iteration
            cudaMemcpy(h_edge_weight, d_edge_weight, n_total_edge * sizeof(double), cudaMemcpyDeviceToHost);
            
            double debug_min = 1e30, debug_max = -1e30, debug_sum = 0;
            int zeros_count = 0;
            int at_clamp_count = 0;
            
            for (int i = 0; i < n_total_edge; i++) {
                if (h_edge_weight[i] < debug_min) debug_min = h_edge_weight[i];
                if (h_edge_weight[i] > debug_max) debug_max = h_edge_weight[i];
                debug_sum += h_edge_weight[i];
                if (h_edge_weight[i] == 0.0) zeros_count++;
                if (h_edge_weight[i] <= 1e-10) at_clamp_count++;
            }
            
            printf("\n=== DEBUG: After update_weight, BEFORE normalization ===\n");
            printf("Min weight: %.20e\n", debug_min);
            printf("Max weight: %.20e\n", debug_max);
            printf("Sum: %.20e\n", debug_sum);
            printf("Zeros: %d / %d\n", zeros_count, n_total_edge);
            printf("At/below clamp (1e-10): %d / %d\n", at_clamp_count, n_total_edge);
        }

        update_weight<<<n_block, BLOCK_SIZE>>>(
            d_v_adj_list, d_v_adj_begin, d_v_adj_length,
            d_edge_src, d_edge_dst,
            d_edge_weight, d_edge_curvature,
            n_undirected_edges);
        cudaCheckErrors("Kernel launch for update_weight failed");
        cudaDeviceSynchronize();
        
        n_block = (n_total_edge+BLOCK_SIZE-1)/BLOCK_SIZE;
        array_sum_blockwise<<<n_block, BLOCK_SIZE>>>(
            d_edge_weight,
            n_total_edge,
            d_partial_sum
        );
        cudaCheckErrors("Kernel launch for array_sum_blockwise failed");
        cudaDeviceSynchronize();

        cudaMemcpy(h_partial_sum, d_partial_sum, n_block*sizeof(double), cudaMemcpyDeviceToHost);
        edge_weight_sum = 0.0;
        for (idx=0; idx<n_block; idx++) {
            edge_weight_sum += h_partial_sum[idx];
        }

        n_block = (n_total_edge + BLOCK_SIZE - 1) / BLOCK_SIZE;
        normalize_weights<<<n_block, BLOCK_SIZE>>>(
            d_edge_weight, edge_weight_sum, n_total_edge);
        cudaDeviceSynchronize();

        clamp_weights_after_norm<<<n_block, BLOCK_SIZE>>>(d_edge_weight, n_total_edge);
        cudaDeviceSynchronize();

        // DEBUG: Check weights AFTER normalization
        if (iter == N_ITERATION - 1) {
            cudaMemcpy(h_edge_weight, d_edge_weight, n_total_edge * sizeof(double), cudaMemcpyDeviceToHost);
            
            double debug_min = 1e30, debug_max = -1e30;
            int zeros_count = 0;
            
            for (int i = 0; i < n_total_edge; i++) {
                if (h_edge_weight[i] < debug_min) debug_min = h_edge_weight[i];
                if (h_edge_weight[i] > debug_max) debug_max = h_edge_weight[i];
                if (h_edge_weight[i] == 0.0) zeros_count++;
            }
            
            printf("\n=== DEBUG: AFTER normalization ===\n");
            printf("Min weight: %.20e\n", debug_min);
            printf("Max weight: %.20e\n", debug_max);
            printf("Normalization divisor was: %.20e\n", edge_weight_sum);
            printf("Zeros: %d / %d\n", zeros_count, n_total_edge);
        }

        printf("Iteration %d: sum_weights = %.4f\n", iter, edge_weight_sum);
    }

    // Copy final weights back
    cudaMemcpy(h_edge_weight, d_edge_weight, n_total_edge * sizeof(double), cudaMemcpyDeviceToHost);

    // Analyze weight distribution
    double min_w = 1e30, max_w = -1e30;
    int min_idx = 0, max_idx = 0;

    for (int i = 0; i < n_undirected_edges; i++) {
        int v1 = edge_src[i];
        int v2 = edge_dst[i];
        
        int idx_adj = -1;
        for (int j = 0; j < v_adj_length[v1]; j++) {
            if (v_adj_list[v_adj_begin[v1] + j] == v2) {
                idx_adj = v_adj_begin[v1] + j;
                break;
            }
        }
        
        double w = h_edge_weight[idx_adj];
        if (w < min_w) { min_w = w; min_idx = i; }
        if (w > max_w) { max_w = w; max_idx = i; }
    }

    // Intra vs Inter cluster weight analysis
    double intra_sum = 0.0, inter_sum = 0.0;
    int intra_count = 0, inter_count = 0;

    for (int i = 0; i < n_undirected_edges; i++) {
        int v1 = edge_src[i];
        int v2 = edge_dst[i];
        
        double w = 0.0;
        for (int j = 0; j < v_adj_length[v1]; j++) {
            if (v_adj_list[v_adj_begin[v1] + j] == v2) {
                w = h_edge_weight[v_adj_begin[v1] + j];
                break;
            }
        }
        
        if (node_cluster[v1] == node_cluster[v2]) {
            intra_sum += w;
            intra_count++;
        } else {
            inter_sum += w;
            inter_count++;
        }
    }

    printf("\n=== Intra vs Inter Cluster Weights ===\n");
    printf("Intra-cluster: count=%d, avg_weight=%.16f\n", intra_count, intra_sum/intra_count);
    printf("Inter-cluster: count=%d, avg_weight=%.16f\n", inter_count, inter_sum/inter_count);
    printf("Ratio (intra/inter): %.4f\n", (intra_sum/intra_count)/(inter_sum/inter_count));

    // Sort the weights in ascending order
    unsigned int next_pow = next_power_of_2(n_total_edge);
    double *h_edge_weight_padded, *d_edge_weight_padded;
    h_edge_weight_padded = (double*)malloc(next_pow * sizeof(double));
    cudaMalloc((void**)&d_edge_weight_padded, next_pow * sizeof(double));

    n_block = (next_pow + BLOCK_SIZE - 1) / BLOCK_SIZE;    
    copy_with_pad<<<n_block, BLOCK_SIZE>>>(d_edge_weight, d_edge_weight_padded, n_total_edge, next_pow);
    cudaDeviceSynchronize();

    for (k = 2; k <= (int)next_pow; k <<= 1)
    {
        for (j = k >> 1; j > 0; j = j >> 1)
        {            
            bitonicSortGPU<<<n_block, BLOCK_SIZE>>>(d_edge_weight_padded, j, k);
            cudaDeviceSynchronize();
        }
    }   
    
    // Copy sorted weights back to host
    double *h_sorted_weights = (double*)malloc(n_total_edge * sizeof(double));
    cudaMemcpy(h_sorted_weights, d_edge_weight_padded, n_total_edge * sizeof(double), cudaMemcpyDeviceToHost);

    printf("\n=== Weight Analysis After %d Iterations ===\n", N_ITERATION);
    printf("Min weight: %.16f at edge (%d, %d)\n", 
        min_w, edge_src[min_idx], edge_dst[min_idx]);
    printf("Max weight: %.16f at edge (%d, %d)\n", 
        max_w, edge_src[max_idx], edge_dst[max_idx]);
    printf("Ratio max/min: %.4f\n", max_w / min_w);

    // Edge count verification
    intra_count = 0;
    inter_count = 0;
    for (int i = 0; i < n_undirected_edges; i++) {
        if (get_cluster(edge_src[i]) == get_cluster(edge_dst[i])) {
            intra_count++;
        } else {
            inter_count++;
        }
    }
    printf("Intra-cluster edges: %d (%.1f%%)\n", intra_count, 100.0*intra_count/n_undirected_edges);
    printf("Inter-cluster edges: %d (%.1f%%)\n", inter_count, 100.0*inter_count/n_undirected_edges);

    // =========================================================================
    // Cutoff Search (EXTREME resolution with double precision)
    // =========================================================================
    // printf("\n=== Cutoff Search (EXTREME resolution - DOUBLE PRECISION) ===\n");
    // printf("%-26s %-12s %-12s\n", "Threshold", "Communities", "Modularity");

    // int *component_id = (int*)malloc(N_NODE * sizeof(int));
    // double best_modularity = -1.0;
    // int best_communities = 0;
    // double best_threshold = 0.0;

    // // 10000 steps at tiny increments
    // int last_communities = -1;
    // for (int i = 0; i <= 10000; i++) {
    //     double multiplier = 1.0 + i * 0.00000001;  // 0.000001% increments
    //     double threshold = min_w * multiplier;
        
    //     int n_communities = find_connected_components_cpu_reference(
    //         v_adj_list, v_adj_begin, v_adj_length,
    //         h_edge_weight, threshold, component_id, N_NODE);
        
    //     double modularity = calculate_modularity_cpu_reference(
    //         v_adj_list, v_adj_begin, v_adj_length,
    //         component_id, N_NODE, n_undirected_edges);
        
    //     // Only print when community count changes
    //     if (n_communities != last_communities) {
    //         printf("%.24f %-12d %-12.6f", threshold, n_communities, modularity);
    //         if (modularity > best_modularity) {
    //             best_modularity = modularity;
    //             best_communities = n_communities;
    //             best_threshold = threshold;
    //             printf(" *");
    //         }
    //         printf("\n");
    //         last_communities = n_communities;
    //     }
    // }

    // printf("\nBest: threshold=%.24f → %d communities (modularity: %.4f)\n",
    //     best_threshold, best_communities, best_modularity);
    // printf("Ground truth: %d clusters\n", N_CLUSTERS);

    // free(component_id);

    // Change the threshold search:
    // printf("\n=== Cutoff Search (from MAX to MIN) ===\n");
    // printf("%-26s %-12s %-12s\n", "Threshold", "Communities", "Modularity");

    // int *component_id = (int*)malloc(N_NODE * sizeof(int));
    // double best_modularity = -1.0;
    // int best_communities = 0;
    // double best_threshold = 0.0;

    // int last_communities = -1;

    // // Search from max_w DOWN to min_w
    // for (int i = 0; i <= 10000; i++) {
    //     double fraction = 1.0 - i * 0.0001;  // 1.0 → 0.0
    //     double threshold = min_w + (max_w - min_w) * fraction;  // max_w → min_w
        
    //     int n_communities = find_connected_components_cpu_reference(
    //         v_adj_list, v_adj_begin, v_adj_length,
    //         h_edge_weight, threshold, component_id, N_NODE);
        
    //     double modularity = calculate_modularity_cpu_reference(
    //         v_adj_list, v_adj_begin, v_adj_length,
    //         component_id, N_NODE, n_undirected_edges);
        
    //     if (n_communities != last_communities) {
    //         printf("%.16e %-12d %-12.6f", threshold, n_communities, modularity);
    //         if (modularity > best_modularity) {
    //             best_modularity = modularity;
    //             best_communities = n_communities;
    //             best_threshold = threshold;
    //             printf(" *");
    //         }
    //         printf("\n");
    //         last_communities = n_communities;
    //     }
    // }

    // printf("\nBest: threshold=%.16e → %d communities (modularity: %.4f)\n",
    //     best_threshold, best_communities, best_modularity);
    // printf("Ground truth: %d clusters\n", N_CLUSTERS);

    // =========================================================================
    // Cutoff Search (JMLR 2025 paper method with drop threshold)
    // =========================================================================
    printf("\n=== Cutoff Search (JMLR 2025 method) ===\n");
    printf("%-26s %-12s %-12s %-12s\n", "Threshold", "Communities", "Modularity", "RelChange");

    int *component_id = (int*)malloc(N_NODE * sizeof(int));
    double best_modularity = -1.0;
    int best_communities = 0;
    double best_threshold = 0.0;

    // JMLR 2025 hyperparameters
    double drop_threshold = 0.1;   // d: reject if modularity drops more than 10%
    double tolerance = 1e-4;       // ε: minimum acceptable modularity

    double prev_modularity = -1.0;
    int last_communities = -1;

    // Search from max_w DOWN to min_w
    for (int i = 0; i <= 10000; i++) {
        double fraction = 1.0 - i * 0.0001;  // 1.0 → 0.0
        double threshold = min_w + (max_w - min_w) * fraction;  // max_w → min_w
        
        int n_communities = find_connected_components_cpu_reference(
            v_adj_list, v_adj_begin, v_adj_length,
            h_edge_weight, threshold, component_id, N_NODE);
        
        double modularity = calculate_modularity_cpu_reference(
            v_adj_list, v_adj_begin, v_adj_length,
            component_id, N_NODE, n_undirected_edges);
        
        // Calculate relative change from previous modularity
        double relative_change = 0.0;
        if (prev_modularity > 0) {
            relative_change = (modularity - prev_modularity) / fabs(modularity);
        }
        
        // Only print when community count changes
        if (n_communities != last_communities) {
            printf("%.16e %-12d %-12.6f %-12.4f", threshold, n_communities, modularity, relative_change);
            
            // JMLR 2025 selection criteria:
            // 1. Modularity must be better than current best
            // 2. Modularity must not be dropping too fast (relative_change > -drop_threshold)
            // 3. Modularity must exceed minimum tolerance
            int is_candidate = (modularity > best_modularity) && 
                            (prev_modularity < 0 || relative_change > -drop_threshold) &&
                            (modularity > tolerance);
            
            if (is_candidate) {
                best_modularity = modularity;
                best_communities = n_communities;
                best_threshold = threshold;
                printf(" *");
            }
            printf("\n");
            
            last_communities = n_communities;
            prev_modularity = modularity;
        }
    }

    printf("\n=== Results ===\n");
    if (best_modularity > tolerance) {
        printf("Best: threshold=%.16e → %d communities (modularity: %.4f)\n",
            best_threshold, best_communities, best_modularity);
    } else {
        printf("No valid clustering found (best modularity %.4f below tolerance %.4f)\n",
            best_modularity, tolerance);
    }
    printf("Ground truth: %d clusters\n", N_CLUSTERS);

    free(component_id);

    // =========================================================================
    // Cleanup
    // =========================================================================
    cudaFree(d_v_adj_list);
    cudaFree(d_v_adj_begin);
    cudaFree(d_v_adj_length);
    cudaFree(d_edge_src);
    cudaFree(d_edge_dst);
    cudaFree(d_edge_weight);
    cudaFree(d_edge_curvature);
    cudaFree(d_partial_sum); 
    cudaFree(d_edge_weight_padded);

    free(v_adj_length);
    free(v_adj_begin);
    free(v_adj_list);
    free(v_adj_begin_2);
    free(edge_src);
    free(edge_dst);
    free(h_edge_weight);
    free(h_edge_curvature);
    free(h_partial_sum);
    free(h_edge_weight_padded);
    free(h_sorted_weights);
    free(node_cluster);

    return 0;
}
