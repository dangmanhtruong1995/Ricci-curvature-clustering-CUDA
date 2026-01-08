#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <assert.h>
#include <queue>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/fill.h>

// #define N_NODE 1000000
// #define MAX_RAND_OUT_DEGREE 200

// #define N_NODE 1000
// #define NODES_PER_CLUSTER 100
// #define N_CLUSTERS (N_NODE / NODES_PER_CLUSTER)
// #define N_EDGES_MAX 10000
// // #define N_EDGES_MAX 25000
// // #define N_EDGES_MAX 50000
// // #define N_EDGES_MAX 100000
// // #define N_EDGES_MAX 500000  // Maximum edges to allocate
// // #define STEP_SIZE 0.0001

// // #define STEP_SIZE 0.1f
// #define STEP_SIZE 0.1f

// #define N_EDGES_TARGET 3000 // Target number of unique undirected edges
// // #define N_ITERATION 10 // Number of iterations in calculating Ricci curvature
// #define N_ITERATION 10
// // #define N_ITERATION 100
// // #define N_ITERATION 1 // For debugging purpose
// // #define N_ITERATION 10        // More iterations
// // #define N_ITERATION 12 

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
// #define P_IN 0.02         // ~20 intra-cluster edges per node
// #define P_OUT 0.00002     // ~0.2 inter-cluster edges per node (10x lower)
// #define N_ITERATION 120
// #define N_EDGES_MAX 200000  // Should be enough for ~110K edges

// LARGE SCALE (fixed parameters)
#define N_NODE 10000
#define NODES_PER_CLUSTER 1000
#define N_CLUSTERS 10
#define P_IN 0.05            // ~50 intra-cluster edges per node, good for triangles
#define P_OUT 0.0005         // ~4.5 inter-cluster edges per node (ratio 100:1)
#define N_ITERATION 80
#define N_EDGES_MAX 300000

// SBM probabilities
// #define P_IN 0.4    // Probability of edge within same cluster
// #define P_OUT 0.001  // Probability of edge between different clusters

// For faces calculation (1: face, 2: v1_nbr, 3: v2_nbr)
#define FACES_BOTH 1
#define FACES_V1_ONLY 2
#define FACES_V2_ONLY 3


#define BLOCK_SIZE 256 // CUDA maximum is 1024
#define VERY_LARGE_NUMBER 99999
#define VWARP_SIZE 16 // Size of virtual warp

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

float get_edge_probability(int node_1, int node_2) {
    int cluster_1 = get_cluster(node_1);
    int cluster_2 = get_cluster(node_2);
    
    if (cluster_1 == cluster_2) {
        return P_IN;
    } else {
        return P_OUT;
    }
}

__global__ void init_one_on_device(float* arr, int n) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < n; 
         idx += blockDim.x * gridDim.x) {
        arr[idx] = 1;
    }
}

__global__ void init_on_device(int* arr, int n, int value) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < n; 
         idx += blockDim.x * gridDim.x) {
        arr[idx] = value;
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

unsigned int next_power_of_2(unsigned int n) {
    // https://graphics.stanford.edu/%7Eseander/bithacks.html#RoundUpPowerOf2
    // The code works for 32 bit data types
    if (n == 0) return 1;
    
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

__global__ void copy_with_pad(float *src, float *dst, unsigned int arr_size, unsigned int next_pow){
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

__global__ void copy_back_from_padded_array(float *src, float *dst, unsigned int arr_size){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < arr_size){
        dst[idx] = src[idx];
        idx += blockDim.x * gridDim.x;
    }
}


// =========================================================================
// Sorting on GPU
// =========================================================================
__global__ void bitonicSortGPU(float* arr, int j, int k)
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
                float temp = arr[i];
                arr[i] = arr[ij];
                arr[ij] = temp;
            }
        }
        else
        {
            if (arr[i] < arr[ij])
            {
                float temp = arr[i];
                arr[i] = arr[ij];
                arr[ij] = temp;
            }
        }
    }
}

// =========================================================================
// Prefix sum calculation
// =========================================================================
__global__ void prefix_sum(const int *A, int *sums, int *block_sums, size_t ds){
    __shared__ int sdata[32]; // Each warp has 32 threads, and each block has a maximum of 32 warps
    size_t idx = threadIdx.x+blockDim.x*blockIdx.x; // create typical 1D thread index from built-in variables
    int tid = threadIdx.x;
    int lane = threadIdx.x % warpSize;
    int warpID = threadIdx.x / warpSize;
    int temp=0;
    int val1 = 0;
    int val2 = 0;
    int val3 = 0;
    unsigned mask = 0xFFFFFFFFU; 

    // Add bounds check at the very beginning
    // if (idx >= ds) return;

    val1 = A[idx];    
    // Sum within each warp
    for (int offset=1; offset <= warpSize/2; offset<<=1){
        temp = __shfl_up_sync(mask, val1, offset);        
        if (lane >= offset) {
            val1 += temp;
        }
    }

    // Store the last element of the warp (which is also the total sum of all the warp elements in this case) to shared memory
    if (lane == 31){
        sdata[warpID] = val1;
    }
    __syncthreads(); // put warp results in shared mem

    // Next, we perform a prefix sum over the data in shared memory (so that later we can sum each of them directly 
    // with the corresponding warp)
    // Here only warp 0 is used
    if (warpID == 0){
        val2 = (tid < (blockDim.x /  warpSize)) ? sdata[lane] : 0; // Get data from shared memory to first warp
        for (int offset=1; offset <= warpSize/2; offset<<=1){
            temp = __shfl_up_sync(mask, val2, offset);            
            if (lane >= offset) {
                val2 += temp;
            }
        }
        sdata[lane] = val2;        
    }
    __syncthreads();    
    
    // Then, we sum all the threads in each warp with the corresponding entry in shared memory (which has been
    // prefix summed beforehand)    
    if (warpID > 0){
        temp = sdata[warpID-1];
        val3 = val1 + temp;            
    }
    else {
        val3 = val1;
    }

    // Next, save the partial block sums
    if ((warpID == (blockDim.x /  warpSize)-1) && (lane == 31)){
        block_sums[blockIdx.x] = val3;
    }
    // __syncthreads();      

    sums[idx] = val3;    
}

void prefix_sum_partial_blocks(int *block_sums, int n_block){
    int i;
    // float prefix_sum=0;
    for (i = 0; i < n_block; i++){        
        if (i > 0){
            block_sums[i] += block_sums[i-1];
        }
    }
}

__global__ void prefix_sum_final(const int *A, int *sums, int *block_sums, size_t ds){
    size_t idx = threadIdx.x+blockDim.x*blockIdx.x; // create typical 1D thread index from built-in variables   

    // Add bounds check at the very beginning
    // if (idx >= ds) return;

    if (blockIdx.x > 0){
        sums[idx] += block_sums[blockIdx.x-1];
    }
}

__global__ void shift_prefix_sum(int *arr, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n - 1) {
        // Shift everything right by one position
        arr[n - 1 - idx] = arr[n - 2 - idx];
    } else if (idx == n - 1) {
        arr[0] = 0;  // First element should always be 0
    }
}

__global__ void safe_shift(int *input, int *output, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (idx == 0) {
        output[0] = 0;
    } else if (idx < n) {
        output[idx] = input[idx - 1];
    }
}

__global__ void copy_back(int *input, int *output, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (idx < n){
        output[idx] = input[idx];
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
        int **node_cluster  // Output: cluster assignment for each node
    ) {
    int idx_1, idx_2, idx;
    float random_number;
    float edge_prob;
    int n_total_edge = 0;

    // Allocate memory
    *v_adj_length = (int*)malloc(N_NODE * sizeof(int));
    *v_adj_begin = (int*)malloc(N_NODE * sizeof(int));
    *edge_src = (int*)malloc(N_EDGES_MAX * sizeof(int));
    *edge_dst = (int*)malloc(N_EDGES_MAX * sizeof(int));
    *node_cluster = (int*)malloc(N_NODE * sizeof(int));

    init_zero(*v_adj_length, N_NODE);
    init_zero(*v_adj_begin, N_NODE);
    init_zero(*edge_src, N_EDGES_MAX);
    init_zero(*edge_dst, N_EDGES_MAX);

    // Assign clusters to nodes
    for (idx = 0; idx < N_NODE; idx++) {
        (*node_cluster)[idx] = get_cluster(idx);
    }

    // Generate edges based on SBM probabilities
    *n_undirected_edges = 0;
    for (idx_1 = 0; idx_1 < N_NODE; idx_1++) {
        for (idx_2 = idx_1 + 1; idx_2 < N_NODE; idx_2++) {
            // Get edge probability based on cluster membership
            edge_prob = get_edge_probability(idx_1, idx_2);
            
            random_number = (float)rand() / (float)(RAND_MAX);
            
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

    // Build adjacency list
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
    
    // Check 1: Every edge in edge_src/edge_dst exists in adjacency list
    for (int i = 0; i < n_undirected_edges; i++) {
        int src = edge_src[i];
        int dst = edge_dst[i];
        
        // Check src -> dst exists
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
        
        // Check dst -> src exists (undirected)
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
    
    // Check 2: Total edges match
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
// Forman-Ricci curvature calculation on GPU
// =========================================================================
__global__ void calc_forman_ricci_curvature(
        int *d_v_adj_list, int *d_v_adj_begin, int *d_v_adj_length,
        int *d_edge_src, int *d_edge_dst,
        float *d_edge_weight, float*d_edge_curvature,
        int n_undirected_edges
        ){    
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    // int tid = threadIdx.x;

    // int neighbor_idx;
    int j;
    int v1;
    int v2;
    int v;
    int idx_v1_v2_adj_list;
    int idx_v2_v1_adj_list;
    float w_e, w_v1, w_v2, ev1_sum, ev2_sum;

    while (idx < n_undirected_edges){
        // Inspecting idx-th edge
        v1 = d_edge_src[idx];
        v2 = d_edge_dst[idx];

        // Find edge (v1 -> v2) in v1's adjacency list
        for (j=0; j<d_v_adj_length[v1]; j++){
            v = d_v_adj_list[d_v_adj_begin[v1] + j];
            if (v == v2){
                // continue;
                idx_v1_v2_adj_list = d_v_adj_begin[v1] + j;
                break;
            }
        }

        // Find edge (v2 -> v1) in v2's adjacency list
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
            ev1_sum += (w_v1 / sqrtf(w_e*d_edge_weight[d_v_adj_begin[v1] + j]));
        }

        for (j=0; j<d_v_adj_length[v2]; j++){
            v = d_v_adj_list[d_v_adj_begin[v2] + j];
            if (v == v1){
                continue;
            }
            ev2_sum += (w_v2 / sqrtf(w_e*d_edge_weight[d_v_adj_begin[v2] + j]));
        }

        d_edge_curvature[idx_v1_v2_adj_list] = w_e * (w_v1 / w_e + w_v2 / w_e - (ev1_sum + ev2_sum));
        d_edge_curvature[idx_v2_v1_adj_list] = d_edge_curvature[idx_v1_v2_adj_list];
        
        idx += gridDim.x*blockDim.x;
    }
}


// For faces calculation (1: face, 2: v1_nbr, 3: v2_nbr)
__global__ void find_faces_in_graph(
        int *d_v_adj_list, int *d_v_adj_begin, int *d_v_adj_length,
        int *d_edge_src, int *d_edge_dst,
        char *d_faces,  // Changed from int* to char* (4x memory reduction)
        int *d_n_triangles,
        int n_undirected_edges
        ){    
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    int j;
    int v1;
    int v2;
    int v;    
    size_t face_idx_base;  // Changed to size_t for large indices

    while (idx < n_undirected_edges){
        // Inspecting idx-th edge
        v1 = d_edge_src[idx];
        v2 = d_edge_dst[idx];

        face_idx_base = (size_t)idx * N_NODE; 

        // Pass 1: Mark all v1 neighbors as V1_ONLY
        for (j=0; j<d_v_adj_length[v1]; j++){
            v = d_v_adj_list[d_v_adj_begin[v1] + j];
            if (v == v2){
                continue;                
            }
            if (d_faces[face_idx_base+v] == 0){
                d_faces[face_idx_base+v] = FACES_V1_ONLY;
            }             
        }

        // Pass 2: Check v2 neighbors
        for (j = 0; j < d_v_adj_length[v2]; j++) {
            v = d_v_adj_list[d_v_adj_begin[v2] + j];
            if (v == v1){
                continue;                
            }
            if (d_faces[face_idx_base+v] == 0){
                d_faces[face_idx_base+v] = FACES_V2_ONLY;                
            } else if (d_faces[face_idx_base+v] == FACES_V1_ONLY){
                d_faces[face_idx_base+v] = FACES_BOTH;
                d_n_triangles[idx]++;
            }
        }
        idx += gridDim.x*blockDim.x;
    }

}

__global__ void calc_augmented_forman_ricci_curvature(
        int *d_v_adj_list, int *d_v_adj_begin, int *d_v_adj_length,
        int *d_edge_src, int *d_edge_dst,
        char *d_faces,  // Changed from int* to char*
        int *d_n_triangles,
        float *d_edge_weight, float *d_edge_curvature,        
        int n_undirected_edges
        ){    
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    while (idx < n_undirected_edges){
        int v1 = d_edge_src[idx];
        int v2 = d_edge_dst[idx];
        size_t face_idx_base = (size_t)idx * N_NODE;  // Changed to size_t

        // Find edge indices
        int idx_v1_v2 = -1, idx_v2_v1 = -1;
        for (int j = 0; j < d_v_adj_length[v1]; j++){
            if (d_v_adj_list[d_v_adj_begin[v1] + j] == v2){
                idx_v1_v2 = d_v_adj_begin[v1] + j;
                break;
            }
        }
        for (int j = 0; j < d_v_adj_length[v2]; j++){
            if (d_v_adj_list[d_v_adj_begin[v2] + j] == v1){
                idx_v2_v1 = d_v_adj_begin[v2] + j;
                break;
            }
        }

        float w_e = d_edge_weight[idx_v1_v2];
        
        // Count triangles and parallel neighbors for unweighted formula
        // F(e) = #triangles + 2 - #parallel_edges
        // Edges within communities have MORE triangles -> positive curvature
        // Bridge edges have FEWER triangles -> negative curvature
        
        int n_triangles = 0;      // |face| - common neighbors forming triangles
        int n_parallel = 0;       // |prl_nbr| - non-shared neighbors
        
        // Count v1's neighbors
        for (int j = 0; j < d_v_adj_length[v1]; j++){
            int n = d_v_adj_list[d_v_adj_begin[v1] + j];
            if (n == v2) continue;
            
            if (d_faces[face_idx_base + n] == FACES_BOTH) {
                // This neighbor forms a triangle
                n_triangles++;
            } else if (d_faces[face_idx_base + n] == FACES_V1_ONLY) {
                // V1-only neighbor (parallel edge)
                n_parallel++;
            }
        }
        
        // Count v2-only neighbors (parallel edges)
        for (int j = 0; j < d_v_adj_length[v2]; j++){
            int n = d_v_adj_list[d_v_adj_begin[v2] + j];
            if (n == v1) continue;
            
            if (d_faces[face_idx_base + n] == FACES_V2_ONLY) {
                n_parallel++;
            }
        }
        
        // Forman-Ricci curvature (unweighted/simple version)
        // F(e) = #faces + 2 - #parallel_edges
        // Positive for edges within dense clusters (many triangles)
        // Negative for bridge edges (few triangles, many parallel edges)
        float curvature = (float)(n_triangles + 2 - n_parallel);
        
        d_edge_curvature[idx_v1_v2] = curvature;
        d_edge_curvature[idx_v2_v1] = curvature;
        
        idx += gridDim.x * blockDim.x;
    }
}

__global__ void update_weight(
        int *d_v_adj_list, int *d_v_adj_begin, int *d_v_adj_length,
        int *d_edge_src, int *d_edge_dst,
        float *d_edge_weight, float*d_edge_curvature,
        float adaptive_size,
        int n_undirected_edges
        ){    
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int j;
    int v1;
    int v2;
    int v;
    int idx_v1_v2_adj_list;
    int idx_v2_v1_adj_list;
    float w_new;

    while (idx < n_undirected_edges){
        // Inspecting idx-th edge
        v1 = d_edge_src[idx];
        v2 = d_edge_dst[idx];

        // Find edge (v1 -> v2) in v1's adjacency list
        for (j=0; j<d_v_adj_length[v1]; j++){
            v = d_v_adj_list[d_v_adj_begin[v1] + j];
            if (v == v2){
                // continue;
                idx_v1_v2_adj_list = d_v_adj_begin[v1] + j;
                break;
            }
        }

        // Find edge (v2 -> v1) in v2's adjacency list
        idx_v2_v1_adj_list = -1;
        for (j = 0; j < d_v_adj_length[v2]; j++) {
            if (d_v_adj_list[d_v_adj_begin[v2] + j] == v1) {
                idx_v2_v1_adj_list = d_v_adj_begin[v2] + j;
                break;
            }
        }        
        w_new = (1-adaptive_size*d_edge_curvature[idx_v1_v2_adj_list])*d_edge_weight[idx_v1_v2_adj_list];        
        
        // Clamp to prevent negative weights
        if (w_new < 0.0001f) w_new = 0.0001f;

        d_edge_weight[idx_v1_v2_adj_list] = w_new;
        d_edge_weight[idx_v2_v1_adj_list] = w_new;
        
        idx += gridDim.x*blockDim.x;
    }        
}

__global__ void array_sum_blockwise(
        float *arr,
        int size,
        float *d_partial_sum // Summation for all elements within the block only
){
    __shared__ float cache[BLOCK_SIZE];
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
	
	while (idx < size) {
		temp += arr[idx];
		idx += gridDim.x * blockDim.x;
	}

    // set the cache values
	cache[cacheIndex] = temp;
	
	// synchronize threads in this block
	__syncthreads();
	
	// for reductions, threadsPerBlock must be a power of 2
	// because of the following code
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
        float *d_edge_weight,
        float total_weight,
        int n_undirected_edges, // Number of UNDIRECTED edges
        int n_total_edge
) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    // Scale factor to preserve average weight
    // total_weight is sum of both directions, so divide by 2 for undirected sum
    float scale = (float)n_undirected_edges / (total_weight / 2.0f);

    while (idx < n_total_edge) {
        // d_edge_weight[idx] = d_edge_weight[idx] / total_weight;
        d_edge_weight[idx] = d_edge_weight[idx] * scale;
        idx += gridDim.x * blockDim.x;
    }
}

__global__ void clamp_weights_after_norm(float *d_edge_weight, int n_total_edge) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    while (idx < n_total_edge) {
        if (d_edge_weight[idx] < 1e-10) d_edge_weight[idx] = 1e-10;
        idx += gridDim.x * blockDim.x;
    }
}

__global__ void bfs_loop_using_virtual_warp_for_cc_labeling(
        int *d_v_adj_list, int *d_v_adj_begin, int *d_v_adj_length,         
        int *d_edge_src, int *d_edge_dst,
        float *d_edge_weight,
        int *d_dist_arr, int *d_still_running,
        int iteration_counter, int start_vertex,
        int *component_list, int component_idx,
        float threshold
    ){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    // int tid = threadIdx.x;
    int neighbor_idx;
    int j;
    int vWarp_id;
    int vWarp_offset;
    int vertex;
    // int temp;
    int n1;
    // float w;
    // int neighbor_index;

    while (idx < N_NODE){
        vWarp_id = idx / VWARP_SIZE;
        vWarp_offset = idx % VWARP_SIZE;
        
        for (vertex=vWarp_id*VWARP_SIZE; vertex<vWarp_id*VWARP_SIZE+VWARP_SIZE; vertex++){
            if ((vertex < N_NODE)&&(d_dist_arr[vertex] == iteration_counter)){
                // For each vertex to be processed, the virtual warps
                // divide the neighbors amongst themselves
                j = 0;
                while (((j+vWarp_offset) < d_v_adj_length[vertex])&&(j<d_v_adj_length[vertex])){
                    n1 = vWarp_offset + j;
                    neighbor_idx = d_v_adj_list[d_v_adj_begin[vertex] + n1];
                    if ((d_dist_arr[neighbor_idx] == VERY_LARGE_NUMBER) && (d_edge_weight[d_v_adj_begin[vertex] + n1] < threshold)) {
                        d_dist_arr[neighbor_idx] = iteration_counter + 1;
                        *d_still_running = 1;
                        component_list[neighbor_idx] = component_idx;
                    }
                    j += VWARP_SIZE;
                }
            }
        }
        idx += gridDim.x*blockDim.x;
    } 
} 

int bfs_loop_using_virtual_warp_for_cc_labeling_wrapper(
    int *d_v_adj_list, int *d_v_adj_begin, int *d_v_adj_length,         
    int *d_edge_src, int *d_edge_dst,
    float *d_edge_weight,
    int *h_component_list, int *d_component_list,
    int *d_dist_arr,
    int *d_still_running,
    float threshold){

    int idx;
    int n_block;

    // Find connected components using BFS
    int component_idx=0;
    // int *h_component_list, *d_component_list;
    // int *d_dist_arr;
    // int *h_dist_arr;
    int start_vertex = 0;
    int h_still_running = 1;
    // int *d_still_running;
    int still_has_cc=1;
    int iteration_counter;
    int zero = 0;

    // Initialize (needs to do this each time the function is called)
    n_block = (N_NODE+BLOCK_SIZE-1)/BLOCK_SIZE; 
    init_on_device<<<n_block, BLOCK_SIZE>>>(d_component_list, N_NODE, -1);    
    cudaDeviceSynchronize();
    cudaMemcpy(h_component_list, d_component_list, N_NODE * sizeof(int), cudaMemcpyDeviceToHost);    
    
    init_on_device<<<n_block, BLOCK_SIZE>>>(d_dist_arr, N_NODE, VERY_LARGE_NUMBER);
    cudaDeviceSynchronize();
    // cudaMemcpy(h_dist_arr, d_dist_arr, N_NODE * sizeof(int), cudaMemcpyDeviceToHost);    

    h_still_running = 1;
    cudaMemcpy(d_still_running, &h_still_running, sizeof(int), cudaMemcpyHostToDevice);

    while (1){
        still_has_cc = 0;
        
        for(idx = 0; idx < N_NODE; idx++){
            if (h_component_list[idx] < 0){
                still_has_cc = 1;
                start_vertex = idx; 
                // Do a BFS for this current component                
                // for(idx_1 = 0; idx_1 < N_NODE; idx_1++) h_dist_arr[idx_1] = VERY_LARGE_NUMBER;
                // start_vertex = idx; 
                // h_dist_arr[start_vertex] = 0;
                // cudaMemcpy(d_dist_arr, h_dist_arr, N_NODE * sizeof(int), cudaMemcpyHostToDevice);    

                init_on_device<<<n_block, BLOCK_SIZE>>>(d_dist_arr, N_NODE, VERY_LARGE_NUMBER);
                cudaDeviceSynchronize();
                cudaMemcpy(d_dist_arr + start_vertex, &zero, sizeof(int),
                    cudaMemcpyHostToDevice);
                // cudaMemcpy(&d_dist_arr[start_vertex], &zero, sizeof(int),
                //     cudaMemcpyHostToDevice);

                h_still_running = 1;
                iteration_counter = 0;
                h_component_list[start_vertex] = component_idx;
                cudaMemcpy(d_component_list, h_component_list, N_NODE * sizeof(int), cudaMemcpyHostToDevice);

                n_block = (N_NODE+BLOCK_SIZE-1)/BLOCK_SIZE;    
                while (h_still_running == 1){
                    h_still_running = 0;                    
                    cudaMemcpy(d_still_running, &h_still_running, sizeof(int), cudaMemcpyHostToDevice);

                    bfs_loop_using_virtual_warp_for_cc_labeling<<<n_block, BLOCK_SIZE>>>(
                        d_v_adj_list, d_v_adj_begin, d_v_adj_length, 
                        d_edge_src, d_edge_dst,
                        d_edge_weight,
                        d_dist_arr, d_still_running,
                        iteration_counter,  
                        start_vertex,
                        d_component_list, component_idx,
                        threshold);
                    cudaDeviceSynchronize();

                    cudaMemcpy(&h_still_running, d_still_running, sizeof(int), cudaMemcpyDeviceToHost);

                    iteration_counter++;
                }
                cudaMemcpy(h_component_list, d_component_list, N_NODE * sizeof(int), cudaMemcpyDeviceToHost);
                component_idx += 1; // Prepare for next component search
                continue;
            }
        }
        if (still_has_cc == 0){
            break; // No more component left
        }
    }

    // free(h_dist_arr);
    // cudaFree(d_dist_arr);
    // cudaFree(d_still_running);

    return component_idx; // At this point this stores the number of components found
}

// =========================================================================
// CPU Reference Implementation for Verification
// =========================================================================
void calc_forman_ricci_cpu(
    int *v_adj_list, int *v_adj_begin, int *v_adj_length,
    int *edge_src, int *edge_dst,
    float *edge_weight, float *edge_curvature,
    int n_undirected_edges
) {
    for (int idx = 0; idx < n_undirected_edges; idx++) {
        int v1 = edge_src[idx];
        int v2 = edge_dst[idx];

        // Find edge weight w_e
        float w_e = 1.0f;
        int idx_v1_v2 = -1;
        for (int j = 0; j < v_adj_length[v1]; j++) {
            if (v_adj_list[v_adj_begin[v1] + j] == v2) {
                idx_v1_v2 = v_adj_begin[v1] + j;
                w_e = edge_weight[idx_v1_v2];
                break;
            }
        }

        float w_v1 = 1.0f;
        float w_v2 = 1.0f;

        // Sum over neighbors of v1 (excluding v2)
        float ev1_sum = 0.0f;
        for (int j = 0; j < v_adj_length[v1]; j++) {
            int v = v_adj_list[v_adj_begin[v1] + j];
            if (v == v2) continue;
            float w_e_prime = edge_weight[v_adj_begin[v1] + j];
            ev1_sum += w_v1 / sqrtf(w_e * w_e_prime);
        }

        // Sum over neighbors of v2 (excluding v1)
        float ev2_sum = 0.0f;
        for (int j = 0; j < v_adj_length[v2]; j++) {
            int v = v_adj_list[v_adj_begin[v2] + j];
            if (v == v1) continue;
            float w_e_prime = edge_weight[v_adj_begin[v2] + j];
            ev2_sum += w_v2 / sqrtf(w_e * w_e_prime);
        }

        float curvature = w_e * (w_v1 / w_e + w_v2 / w_e - (ev1_sum + ev2_sum));
        edge_curvature[idx_v1_v2] = curvature;
    }
}

// Find connected components using BFS
// Returns number of communities and fills component_id array
int find_connected_components_cpu_reference(
    int *v_adj_list, int *v_adj_begin, int *v_adj_length,
    float *edge_weight, float threshold,
    int *component_id,  // Output: component ID for each node
    int n_nodes
) {
    // Initialize all nodes as unvisited
    for (int i = 0; i < n_nodes; i++) {
        component_id[i] = -1;
    }
    
    int n_components = 0;
    int *queue = (int*)malloc(n_nodes * sizeof(int));
    
    for (int start = 0; start < n_nodes; start++) {
        if (component_id[start] != -1) continue;  // Already visited
        
        // BFS from this node
        int queue_start = 0, queue_end = 0;
        queue[queue_end++] = start;
        component_id[start] = n_components;
        
        while (queue_start < queue_end) {
            int node = queue[queue_start++];
            
            // Visit neighbors through edges with weight <= threshold
            for (int j = 0; j < v_adj_length[node]; j++) {
                int idx = v_adj_begin[node] + j;
                int neighbor = v_adj_list[idx];
                float w = edge_weight[idx];
                
                // Only traverse edge if weight <= threshold (not a bridge)
                // if (w <= threshold && component_id[neighbor] == -1) {
                // if (w >= threshold && component_id[neighbor] == -1){
                if (w < threshold && component_id[neighbor] == -1) {
                // if (w > threshold && component_id[neighbor] == -1){
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

// Calculate modularity for a given partition
// Q = (1/2m) * Σ[A_ij - k_i*k_j/(2m)] * δ(c_i, c_j)
float calculate_modularity_cpu_reference(
    int *v_adj_list, int *v_adj_begin, int *v_adj_length,
    int *component_id,
    int n_nodes, int n_edges  // n_edges = number of undirected edges
) {
    int m = n_edges;  // Total undirected edges
    float modularity = 0.0f;
    
    // For each edge, check if both endpoints are in same community
    for (int u = 0; u < n_nodes; u++) {
        int k_u = v_adj_length[u];  // Degree of u
        
        for (int j = 0; j < v_adj_length[u]; j++) {
            int v = v_adj_list[v_adj_begin[u] + j];
            int k_v = v_adj_length[v];  // Degree of v
            
            // A_ij = 1 (edge exists)
            // δ(c_i, c_j) = 1 if same community
            if (component_id[u] == component_id[v]) {
                modularity += 1.0f - (float)(k_u * k_v) / (2.0f * m);
            } else {
                modularity += 0.0f - (float)(k_u * k_v) / (2.0f * m);
            }
        }
    }
    
    modularity /= (2.0f * m);
    return modularity;
}

// Find optimal threshold and communities using Ricci flow weights
void find_communities_ricci_cpu_reference(
    int *v_adj_list, int *v_adj_begin, int *v_adj_length,
    float *edge_weight,
    int *edge_src, int *edge_dst,
    int n_nodes, int n_undirected_edges
) {
    int *component_id = (int*)malloc(n_nodes * sizeof(int));
    
    // Collect all unique edge weights to use as thresholds
    float *thresholds = (float*)malloc(n_undirected_edges * sizeof(float));
    int n_thresholds = 0;
    
    for (int i = 0; i < n_undirected_edges; i++) {
        int v1 = edge_src[i];
        int v2 = edge_dst[i];
        
        // Find weight
        for (int j = 0; j < v_adj_length[v1]; j++) {
            if (v_adj_list[v_adj_begin[v1] + j] == v2) {
                thresholds[n_thresholds++] = edge_weight[v_adj_begin[v1] + j];
                break;
            }
        }
    }
    
    // Sort thresholds (simple bubble sort - replace with qsort for large arrays)
    for (int i = 0; i < n_thresholds - 1; i++) {
        for (int j = i + 1; j < n_thresholds; j++) {
            if (thresholds[i] > thresholds[j]) {
                float tmp = thresholds[i];
                thresholds[i] = thresholds[j];
                thresholds[j] = tmp;
            }
        }
    }
    
    // Try different thresholds and compute modularity
    float best_modularity = -1.0f;
    float best_threshold = 0.0f;
    int best_n_communities = 1;
    
    printf("\n=== Threshold Search (CPU Reference) ===\n");
    printf("%-15s %-12s %-12s\n", "Threshold", "Communities", "Modularity");
    printf("-------------------------------------------\n");
    
    // Sample thresholds (don't try all - too slow)
    int step = n_thresholds / 50;  // ~50 samples
    if (step < 1) step = 1;
    
    for (int i = 0; i < n_thresholds; i += step) {
        float threshold = thresholds[i];
        
        int n_communities = find_connected_components_cpu_reference(
            v_adj_list, v_adj_begin, v_adj_length,
            edge_weight, threshold,
            component_id, n_nodes
        );
        
        float modularity = calculate_modularity_cpu_reference(
            v_adj_list, v_adj_begin, v_adj_length,
            component_id, n_nodes, n_undirected_edges
        );
        
        printf("%-15.8f %-12d %-12.6f", threshold, n_communities, modularity);
        
        if (modularity > best_modularity) {
            best_modularity = modularity;
            best_threshold = threshold;
            best_n_communities = n_communities;
            printf(" *");
        }
        printf("\n");
    }
    
    // Final result with best threshold
    printf("\n=== Best Partition (CPU Reference) ===\n");
    printf("Threshold: %.8f\n", best_threshold);
    printf("Communities: %d\n", best_n_communities);
    printf("Modularity: %.6f\n", best_modularity);
    
    // Get final partition
    find_connected_components_cpu_reference(
        v_adj_list, v_adj_begin, v_adj_length,
        edge_weight, best_threshold,
        component_id, n_nodes
    );
    
    // Print community sizes
    int *community_sizes = (int*)calloc(best_n_communities, sizeof(int));
    for (int i = 0; i < n_nodes; i++) {
        if (component_id[i] >= 0 && component_id[i] < best_n_communities) {
            community_sizes[component_id[i]]++;
        }
    }
    
    printf("\nCommunity sizes:\n");
    for (int c = 0; c < best_n_communities && c < 20; c++) {
        printf("  Community %d: %d nodes\n", c, community_sizes[c]);
    }
    if (best_n_communities > 20) {
        printf("  ... (%d more communities)\n", best_n_communities - 20);
    }
    
    free(community_sizes);
    free(component_id);
    free(thresholds);
}


int main(){
    srand(time(NULL));
    // srand(42);
    int n_block;
    int *edge_src, *edge_dst;
    int idx, iter, i, j, k;
    // int idx_1, idx_2;    
    // float random_number;
    float edge_weight_sum;

    // Create random edges for an undirected graph
    // Then put it into adjacency list form (v_adj_list, v_adj_begin, v_adj_length)
    int *v_adj_length, *v_adj_begin, *v_adj_list;  
    int *v_adj_begin_2;
    float *h_partial_sum, *d_partial_sum;
    int n_total_edge, n_undirected_edges;
    int *node_cluster;

    create_sbm_graph(
        &v_adj_length, &v_adj_begin, &v_adj_list,
        &edge_src, &edge_dst, &v_adj_begin_2,
        &n_undirected_edges, &node_cluster);
    n_total_edge = 2*n_undirected_edges;

    // Print results to verify
    // printf("Generated %d undirected edges (%d directed entries)\n", n_undirected_edges, n_total_edge);

    // printf("\nAdjacency list:\n");
    // for (i = 0; i < N_NODE; i++) {
    //     printf("Node %d: ", i);
    //     for (j = 0; j < v_adj_length[i]; j++) {
    //         printf("%d ", v_adj_list[v_adj_begin[i] + j]);
    //     }
    //     printf("\n");
    // }

    // for (idx=0; idx<n_undirected_edges; idx++){
    //     printf("(%d, %d)\n", edge_src[idx], edge_dst[idx]);
    // }

    // Verify the graph
    verify_graph(edge_src, edge_dst, n_undirected_edges,
             v_adj_list, v_adj_begin, v_adj_length);

    // Ricci curvature
    float *d_edge_weight, *h_edge_weight;
    float *d_edge_curvature, *h_edge_curvature;
    
    cudaMalloc(&d_edge_weight, n_total_edge * sizeof(float));
    cudaMalloc(&d_edge_curvature, n_total_edge * sizeof(float));
    h_edge_weight = (float*)malloc(n_total_edge * sizeof(float));
    h_edge_curvature = (float*)malloc(n_total_edge * sizeof(float));

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
    cudaCheckErrors("Copy from v_adj_list to d_v_adj_list failed.");

    cudaMemcpy(d_v_adj_begin, v_adj_begin, N_NODE * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckErrors("Copy from v_adj_begin to d_v_adj_begin failed.");

    cudaMemcpy(d_v_adj_length, v_adj_length, N_NODE * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckErrors("Copy from v_adj_length to d_v_adj_length failed.");

    cudaMemcpy(d_edge_src, edge_src, n_undirected_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckErrors("Copy from edge_src to d_edge_src failed.");

    cudaMemcpy(d_edge_dst, edge_dst, n_undirected_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckErrors("Copy from edge_dst to d_edge_dst failed.");
    
    n_block = (n_total_edge+BLOCK_SIZE-1)/BLOCK_SIZE;    
    h_partial_sum = (float*)malloc( n_block*sizeof(float) );
    cudaMalloc(&d_partial_sum, n_block * sizeof(float));
    cudaCheckErrors("cudaMalloc for d_partial_sum failed");

    // Precompute faces, v1_nbr - face and v2_nbr - face for each edge
    // Using char instead of int: 4x memory reduction (5GB instead of 20GB)
    // Values: 0=unused, 1=FACES_BOTH, 2=FACES_V1_ONLY, 3=FACES_V2_ONLY
    char *d_faces;
    size_t faces_size = (size_t)n_undirected_edges * N_NODE * sizeof(char);
    printf("Allocating d_faces: %zu bytes (%.2f GB)\n", faces_size, faces_size / 1e9);
    cudaMalloc(&d_faces, faces_size);
    cudaCheckErrors("cudaMalloc for d_faces failed");
    cudaMemset(d_faces, 0, faces_size);
    cudaCheckErrors("cudaMemset for d_faces to zero failed");

    int *d_n_triangles;
    cudaMalloc(&d_n_triangles, n_undirected_edges*sizeof(int));
    cudaCheckErrors("cudaMalloc for d_n_triangles failed");
    cudaMemset(d_n_triangles, 0, n_undirected_edges* sizeof(int));
    cudaCheckErrors("cudaMemset for d_n_triangles to zero failed"); 

    n_block = (n_undirected_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;
    find_faces_in_graph<<<n_block, BLOCK_SIZE>>>(
        d_v_adj_list, d_v_adj_begin, d_v_adj_length,
        d_edge_src, d_edge_dst,
        d_faces,
        d_n_triangles,
        n_undirected_edges);
    cudaDeviceSynchronize();
    cudaCheckErrors("find_faces_in_graph failed");

    // // Optional: Print triangle statistics
    // int *h_n_triangles = (int*)malloc(n_undirected_edges * sizeof(int));
    // cudaMemcpy(h_n_triangles, d_n_triangles, n_undirected_edges * sizeof(int), cudaMemcpyDeviceToHost);
    
    // int total_triangles = 0;
    // int max_triangles = 0;
    // int edges_with_triangles = 0;
    // for (int i = 0; i < n_undirected_edges; i++) {
    //     total_triangles += h_n_triangles[i];
    //     if (h_n_triangles[i] > max_triangles) max_triangles = h_n_triangles[i];
    //     if (h_n_triangles[i] > 0) edges_with_triangles++;
    // }
    // printf("Triangle statistics:\n");
    // printf("  Total triangle-edge incidences: %d\n", total_triangles);
    // printf("  Unique triangles (approx): %d\n", total_triangles / 3);  // Each triangle counted 3 times
    // printf("  Edges with triangles: %d / %d (%.1f%%)\n", 
    //        edges_with_triangles, n_undirected_edges, 
    //        100.0 * edges_with_triangles / n_undirected_edges);
    // printf("  Max triangles per edge: %d\n", max_triangles);
    // free(h_n_triangles);

    // Calculate Ricci curvature, update then normalize weights
    for (iter=0; iter<N_ITERATION; iter++){
        // n_block = (n_undirected_edges+BLOCK_SIZE-1)/BLOCK_SIZE; 
        // calc_forman_ricci_curvature<<<n_block, BLOCK_SIZE>>>(
        //     d_v_adj_list, d_v_adj_begin, d_v_adj_length,
        //     d_edge_src, d_edge_dst,
        //     d_edge_weight, d_edge_curvature,
        //     n_undirected_edges);
        // cudaCheckErrors("Kernel launch failed");
        // cudaDeviceSynchronize();

        n_block = (n_undirected_edges + BLOCK_SIZE - 1) / BLOCK_SIZE; 
        calc_augmented_forman_ricci_curvature<<<n_block, BLOCK_SIZE>>>(
            d_v_adj_list, d_v_adj_begin, d_v_adj_length,
            d_edge_src, d_edge_dst,
            d_faces,
            d_n_triangles,
            d_edge_weight, d_edge_curvature,
            n_undirected_edges);
        cudaDeviceSynchronize();
        cudaCheckErrors("calc_augmented_forman_ricci_curvature failed");

        cudaMemcpy(h_edge_curvature, d_edge_curvature, n_total_edge*sizeof(float), cudaMemcpyDeviceToHost);
        cudaCheckErrors("Copy from d_edge_curvature to h_edge_curvature failed.");
        
        // After computing curvature, find max absolute value
        float max_curv = 0.0f;
        for (int i = 0; i < n_total_edge; i++) {
            float abs_curv = fabsf(h_edge_curvature[i]);
            if (abs_curv > max_curv) max_curv = abs_curv;
        }

        // Adaptive step size (as in paper)
        float adaptive_step = 1.0f / (1.1f * max_curv + 1e-10f);
        if (adaptive_step > 1.0f) adaptive_step = 1.0f;  // Cap at 1.0

        printf("Iter %d: max_curv=%.2f, step=%.6f\n", iter, max_curv, adaptive_step);

        // DEBUG: Check weights AFTER update, BEFORE normalization
        // if (iter == N_ITERATION - 1) {  // Only on last iteration
        //     cudaMemcpy(h_edge_weight, d_edge_weight, n_total_edge * sizeof(float), cudaMemcpyDeviceToHost);
            
        //     float debug_min = 1e30, debug_max = -1e30, debug_sum = 0;
        //     int zeros_count = 0;
        //     int at_clamp_count = 0;
            
        //     for (int i = 0; i < n_total_edge; i++) {
        //         if (h_edge_weight[i] < debug_min) debug_min = h_edge_weight[i];
        //         if (h_edge_weight[i] > debug_max) debug_max = h_edge_weight[i];
        //         debug_sum += h_edge_weight[i];
        //         if (h_edge_weight[i] == 0.0) zeros_count++;
        //         if (h_edge_weight[i] <= 1e-10) at_clamp_count++;
        //     }
            
        //     printf("\n=== DEBUG: After update_weight, BEFORE normalization ===\n");
        //     printf("Min weight: %.20e\n", debug_min);
        //     printf("Max weight: %.20e\n", debug_max);
        //     printf("Sum: %.20e\n", debug_sum);
        //     printf("Zeros: %d / %d\n", zeros_count, n_total_edge);
        //     printf("At/below clamp (1e-10): %d / %d\n", at_clamp_count, n_total_edge);
        // }

        update_weight<<<n_block, BLOCK_SIZE>>>(
            d_v_adj_list, d_v_adj_begin, d_v_adj_length,
            d_edge_src, d_edge_dst,
            d_edge_weight, d_edge_curvature,
            adaptive_step,   
            n_undirected_edges);
        cudaCheckErrors("Kernel launch for update_weight failed");
        cudaDeviceSynchronize();
        
        n_block = (n_total_edge+BLOCK_SIZE-1)/BLOCK_SIZE;
        array_sum_blockwise<<<n_block, BLOCK_SIZE>>>(
            d_edge_weight,
            n_total_edge,
            d_partial_sum // Summation for all elements within the block only
        );
        cudaCheckErrors("Kernel launch for array_sum_blockwise failed");
        cudaDeviceSynchronize();

        cudaMemcpy(h_partial_sum, d_partial_sum, n_block*sizeof(float), cudaMemcpyDeviceToHost );
        edge_weight_sum = 0.0;
        for (idx=0; idx<n_block; idx++) {
            edge_weight_sum += h_partial_sum[idx];
        }

        n_block = (n_total_edge + BLOCK_SIZE - 1) / BLOCK_SIZE;
        normalize_weights<<<n_block, BLOCK_SIZE>>>(
            d_edge_weight, edge_weight_sum, n_undirected_edges, n_total_edge);
        cudaDeviceSynchronize();

        clamp_weights_after_norm<<<n_block, BLOCK_SIZE>>>(d_edge_weight, n_total_edge);
        cudaDeviceSynchronize();

        // DEBUG: Check weights AFTER normalization
        // if (iter == N_ITERATION - 1) {
        //     cudaMemcpy(h_edge_weight, d_edge_weight, n_total_edge * sizeof(float), cudaMemcpyDeviceToHost);
            
        //     float debug_min = 1e30, debug_max = -1e30;
        //     int zeros_count = 0;
            
        //     for (int i = 0; i < n_total_edge; i++) {
        //         if (h_edge_weight[i] < debug_min) debug_min = h_edge_weight[i];
        //         if (h_edge_weight[i] > debug_max) debug_max = h_edge_weight[i];
        //         if (h_edge_weight[i] == 0.0) zeros_count++;
        //     }
            
        //     printf("\n=== DEBUG: AFTER normalization ===\n");
        //     printf("Min weight: %.20e\n", debug_min);
        //     printf("Max weight: %.20e\n", debug_max);
        //     printf("Normalization divisor was: %.20e\n", edge_weight_sum);
        //     printf("Zeros: %d / %d\n", zeros_count, n_total_edge);
        // }

        printf("Iteration %d: sum_weights = %.4f\n", iter, edge_weight_sum);

        // if (iter == 0) {
        //     cudaMemcpy(h_edge_curvature, d_edge_curvature, n_total_edge * sizeof(float), cudaMemcpyDeviceToHost);
            
        //     float curv_min = 1e30f, curv_max = -1e30f;
        //     float curv_sum = 0.0f;
        //     int pos_count = 0, neg_count = 0;
            
        //     float intra_curv_sum = 0.0f, inter_curv_sum = 0.0f;
        //     int intra_count = 0, inter_count = 0;
            
        //     for (int i = 0; i < n_undirected_edges; i++) {
        //         int s = edge_src[i];
        //         int d = edge_dst[i];
                
        //         // Find curvature (need to look up in adjacency)
        //         float curv = 0.0f;
        //         for (int j = 0; j < v_adj_length[s]; j++) {
        //             if (v_adj_list[v_adj_begin[s] + j] == d) {
        //                 curv = h_edge_curvature[v_adj_begin[s] + j];
        //                 break;
        //             }
        //         }
                
        //         if (curv < curv_min) curv_min = curv;
        //         if (curv > curv_max) curv_max = curv;
        //         curv_sum += curv;
        //         if (curv > 0) pos_count++; else neg_count++;
                
        //         if (node_cluster[s] == node_cluster[d]) {
        //             intra_curv_sum += curv;
        //             intra_count++;
        //         } else {
        //             inter_curv_sum += curv;
        //             inter_count++;
        //         }
        //     }
            
        //     printf("\n=== Curvature Statistics (Iter 0) ===\n");
        //     printf("Min curvature: %.4f\n", curv_min);
        //     printf("Max curvature: %.4f\n", curv_max);
        //     printf("Avg curvature: %.4f\n", curv_sum / n_undirected_edges);
        //     printf("Positive: %d, Negative: %d\n", pos_count, neg_count);
        //     printf("Intra-cluster avg curvature: %.4f\n", intra_curv_sum / intra_count);
        //     printf("Inter-cluster avg curvature: %.4f\n", inter_curv_sum / inter_count);
        //     printf("Expected: Intra > Inter (intra edges have more triangles)\n\n");
        // }
    }

    // Copy final weights back
    cudaMemcpy(h_edge_weight, d_edge_weight, n_total_edge * sizeof(float), cudaMemcpyDeviceToHost);

    // Analyze weight distribution
    float min_w = 1e9, max_w = -1e9;
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
        
        float w = h_edge_weight[idx_adj];
        if (w < min_w) { min_w = w; min_idx = i; }
        if (w > max_w) { max_w = w; max_idx = i; }
    }

    // Add this after computing weights, before sorting:
    float intra_sum = 0, inter_sum = 0;
    int intra_count = 0, inter_count = 0;

    for (int i = 0; i < n_undirected_edges; i++) {
        int v1 = edge_src[i];
        int v2 = edge_dst[i];
        
        // Find weight
        float w = 0;
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
    printf("Intra-cluster: count=%d, avg_weight=%.8f\n", intra_count, intra_sum/intra_count);
    printf("Inter-cluster: count=%d, avg_weight=%.8f\n", inter_count, inter_sum/inter_count);
    printf("Ratio (intra/inter): %.4f\n", (intra_sum/intra_count)/(inter_sum/inter_count));
    
    // Sort the weights in ascending order
    unsigned int next_pow = next_power_of_2(n_total_edge);
    float *h_edge_weight_padded, *d_edge_weight_padded;
    h_edge_weight_padded = (float*)malloc(next_pow * sizeof(float));
    cudaMalloc((void**)&d_edge_weight_padded, next_pow * sizeof(float));

    n_block = (next_pow + BLOCK_SIZE - 1) / BLOCK_SIZE;    
    copy_with_pad <<<n_block, BLOCK_SIZE>>>(d_edge_weight, d_edge_weight_padded, n_total_edge, next_pow);
    cudaDeviceSynchronize();

    for (k = 2; k <= next_pow; k <<= 1)
    {
        for (j = k >> 1; j > 0; j = j >> 1)
        {            
            bitonicSortGPU <<<n_block, BLOCK_SIZE>>>(d_edge_weight_padded, j, k);
            cudaDeviceSynchronize();
        }
    }   
    
    // Copy sorted weights back to host
    float *h_sorted_weights = (float*)malloc(n_total_edge * sizeof(float));
    cudaMemcpy(h_sorted_weights, d_edge_weight_padded, n_total_edge * sizeof(float), cudaMemcpyDeviceToHost);

    // Print sorted weights
    printf("\n=== Sorted Weights (ascending) ===\n");
    // printf("First 20 weights (smallest):\n");
    // for (i = 0; i < n_total_edge; i++) {
    //     printf("  [%d] %.8f\n", i, h_sorted_weights[i]);
    // }
    
    printf("\n=== Weight Analysis After %d Iterations ===\n", N_ITERATION);
    printf("Min weight: %.8f at edge (%d, %d)\n", 
        min_w, edge_src[min_idx], edge_dst[min_idx]);
    printf("Max weight: %.8f at edge (%d, %d)\n", 
        max_w, edge_src[max_idx], edge_dst[max_idx]);
    printf("Ratio max/min: %.4f\n", max_w / min_w);

    // Add this after graph generation to verify:
    // int intra_count = 0, inter_count = 0;
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
    // Threshold Search with Improved Stable Region Detection
    // =========================================================================
    printf("\n=== Cutoff Search (with modularity-based selection) ===\n");
    printf("%-15s %-12s %-12s %-12s\n", "Threshold", "Communities", "Modularity", "Status");

    // int component_idx=0;
    int *h_component_list, *d_component_list;
    int *d_dist_arr;
    int *d_still_running;

    float epsilon = 0.001f;       // Modularity stability threshold
    float min_modularity = 0.8f;  // Minimum acceptable modularity

    // Track the best stable region (longest consecutive same communities)
    int best_stable_length = 0;
    float best_stable_threshold = 0.0f;
    int best_stable_communities = 0;
    float best_stable_modularity = 0.0f;

    // Current stable region tracking
    int current_stable_length = 0;
    float current_stable_start = 0.0f;
    int prev_communities = -1;
    float prev_modularity = -1.0f;

    // Also track best modularity overall
    float best_modularity = -1.0f;
    float best_threshold = 0.0f;
    int best_communities = 0;

    cudaMalloc(&d_still_running, sizeof(int));
    h_component_list = (int*) malloc(N_NODE * sizeof(int));
    cudaMalloc(&d_component_list, N_NODE * sizeof(int));  
    cudaMalloc(&d_dist_arr, N_NODE * sizeof(int));

    for (int i = 0; i <= 200; i++) {
        // Use logarithmic spacing to better cover the range
        // This gives more resolution at lower thresholds where the action is
        float log_min = logf(min_w + 1e-10f);
        float log_max = logf(max_w + 1e-10f);
        float threshold = expf(log_min + (log_max - log_min) * i / 200.0f);
        
        // int n_communities = find_connected_components_cpu_reference(
        //     v_adj_list, v_adj_begin, v_adj_length,
        //     h_edge_weight, threshold, h_component_list, N_NODE);

        int n_communities = bfs_loop_using_virtual_warp_for_cc_labeling_wrapper(
            d_v_adj_list, d_v_adj_begin, d_v_adj_length,
            d_edge_src, d_edge_dst,
            d_edge_weight,
            h_component_list, d_component_list,
            d_dist_arr, 
            d_still_running,
            threshold
        );
        
        if (n_communities < 2 || n_communities > N_NODE / 2) {
            prev_communities = -1;
            current_stable_length = 0;
            continue;
        }
        
        float modularity = calculate_modularity_cpu_reference(
            v_adj_list, v_adj_begin, v_adj_length,
            h_component_list, N_NODE, n_undirected_edges);
        
        char status[64] = "";
        
        // Track stable regions (same number of communities with similar modularity)
        if (n_communities == prev_communities && 
            fabsf(modularity - prev_modularity) < epsilon) {
            current_stable_length++;
        } else {
            // New region started - check if previous was best
            if (current_stable_length > best_stable_length && 
                prev_modularity > min_modularity &&
                prev_communities > 2) {  // Don't want trivial solutions
                best_stable_length = current_stable_length;
                best_stable_threshold = current_stable_start;
                best_stable_communities = prev_communities;
                best_stable_modularity = prev_modularity;
            }
            // Start new region
            current_stable_length = 1;
            current_stable_start = threshold;
        }
        
        // Mark stable regions in output
        if (current_stable_length == 3) {
            sprintf(status, " ** STABLE START **");
        } else if (current_stable_length > 3 && current_stable_length % 10 == 0) {
            sprintf(status, " (stable x%d)", current_stable_length);
        }
        
        // Track best modularity (with penalty for too few communities)
        if (modularity > min_modularity && n_communities >= 3) {
            if (modularity > best_modularity) {
                best_modularity = modularity;
                best_threshold = threshold;
                best_communities = n_communities;
            }
        }
        
        // Print results for small community counts OR every 10th iteration for visibility
        if (n_communities <= 20 || i % 20 == 0) {
            printf("%.10e  %-12d %-12.6f %s\n", threshold, n_communities, modularity, status);
        }
        
        prev_communities = n_communities;
        prev_modularity = modularity;
    }

    // Check final region
    if (current_stable_length > best_stable_length && 
        prev_modularity > min_modularity &&
        prev_communities > 2) {
        best_stable_length = current_stable_length;
        best_stable_threshold = current_stable_start;
        best_stable_communities = prev_communities;
        best_stable_modularity = prev_modularity;
    }

    printf("\n=== Final Selection ===\n");
    printf("Longest stable region:\n");
    printf("  Threshold: %.10e\n", best_stable_threshold);
    printf("  Communities: %d\n", best_stable_communities);
    printf("  Modularity: %.6f\n", best_stable_modularity);
    printf("  Stability length: %d consecutive thresholds\n", best_stable_length);

    printf("\nHighest modularity:\n");
    printf("  Threshold: %.10e\n", best_threshold);
    printf("  Communities: %d\n", best_communities);
    printf("  Modularity: %.6f\n", best_modularity);

    printf("\nGround truth: %d clusters\n", N_CLUSTERS);

    // Use the stable region for final clustering
    printf("\n=== Using Stable Region Result ===\n");

    int n_communities = bfs_loop_using_virtual_warp_for_cc_labeling_wrapper(
            d_v_adj_list, d_v_adj_begin, d_v_adj_length,
            d_edge_src, d_edge_dst,
            d_edge_weight,
            h_component_list, d_component_list,
            d_dist_arr, 
            d_still_running,
            best_stable_threshold
        );
    printf("%d\n", n_communities);

    // =========================================================================
    // Save graph and clustering results to file
    // =========================================================================
    printf("\n=== Saving graph to file ===\n");

    FILE *fp = fopen("graph_output.txt", "w");
    if (fp == NULL) {
        printf("Error: Could not open file for writing\n");
    } else {
        // First line: N M
        fprintf(fp, "%d %d\n", N_NODE, n_undirected_edges);
        
        // Next M lines: src dst (edges)
        for (int i = 0; i < n_undirected_edges; i++) {
            fprintf(fp, "%d %d\n", edge_src[i], edge_dst[i]);
        }
        
        // Next N lines: component assignment for each node
        for (int i = 0; i < N_NODE; i++) {
            fprintf(fp, "%d\n", h_component_list[i]);
        }
        
        fclose(fp);
        printf("Graph saved to graph_output.txt\n");
        printf("  Nodes: %d\n", N_NODE);
        printf("  Edges: %d\n", n_undirected_edges);
        printf("  Communities: %d\n", best_stable_communities);
    }

    // find_connected_components_cpu_reference(
    //     v_adj_list, v_adj_begin, v_adj_length,
    //     h_edge_weight, best_stable_threshold, h_component_list, N_NODE);

    // free(h_component_list);
    
    // =========================================================================
    // Cleanup
    // =========================================================================

    // Device memory
    cudaFree(d_v_adj_list);
    cudaFree(d_v_adj_begin);
    cudaFree(d_v_adj_length);
    cudaFree(d_edge_src);
    cudaFree(d_edge_dst);
    cudaFree(d_edge_weight);
    cudaFree(d_edge_curvature);
    cudaFree(d_partial_sum);
    cudaFree(d_component_list);
    cudaFree(d_dist_arr);
    cudaFree(d_still_running);
    cudaFree(d_edge_weight_padded);
    cudaFree(d_faces);

    // Host memory
    free(v_adj_list);
    free(v_adj_begin);
    free(v_adj_begin_2);
    free(v_adj_length);
    free(edge_src);
    free(edge_dst);
    free(h_edge_weight);
    free(h_edge_curvature);
    free(h_partial_sum);
    free(h_component_list);
    free(h_sorted_weights);
    free(h_edge_weight_padded);  
}