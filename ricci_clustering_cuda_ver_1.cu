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

#define N_NODE 1000
#define NODES_PER_CLUSTER 100
#define N_CLUSTERS (N_NODE / NODES_PER_CLUSTER)
#define N_EDGES_MAX 10000
// #define N_EDGES_MAX 25000
// #define N_EDGES_MAX 50000
// #define N_EDGES_MAX 100000
// #define N_EDGES_MAX 500000  // Maximum edges to allocate
// #define STEP_SIZE 0.0001
#define STEP_SIZE 0.1f
#define N_EDGES_TARGET 3000 // Target number of unique undirected edges
#define N_ITERATION 10 // Number of iterations in calculating Ricci curvature
// #define N_ITERATION 100
// #define N_ITERATION 1 // For debugging purpose
// #define N_ITERATION 10        // More iterations
// #define N_ITERATION 12 

// SBM probabilities
#define P_IN 0.3    // Probability of edge within same cluster
#define P_OUT 0.01  // Probability of edge between different clusters

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

void create_random_undirected_graph(
        int **v_adj_length,   // Pointer to pointer
        int **v_adj_begin,
        int **v_adj_list,
        int **edge_src,
        int **edge_dst,
        int **v_adj_begin_2,
        int *n_undirected_edges  
    ){
    // Create random edges for an undirected graph
    // Then put it into adjacency list form (v_adj_list, v_adj_begin, v_adj_length)
    // int *v_adj_length, *v_adj_begin, *v_adj_list;  
    int idx_1, idx_2, idx;
    float random_number; 
    int n_total_edge=0;
    // int n_undirected_edges=0;

    *v_adj_length = (int*)malloc(N_NODE * sizeof(int));
    *v_adj_begin = (int*)malloc(N_NODE * sizeof(int));
    *edge_src = (int*)malloc(N_EDGES_TARGET * sizeof(int));
    *edge_dst = (int*)malloc(N_EDGES_TARGET * sizeof(int));

    init_zero(*v_adj_length, N_NODE);
    init_zero(*v_adj_begin, N_NODE);

    init_zero(*edge_src, N_EDGES_TARGET);
    init_zero(*edge_dst, N_EDGES_TARGET);

    n_total_edge=0;
    *n_undirected_edges=0;
    for (idx_1=0; idx_1<N_NODE; idx_1++){
        for (idx_2=idx_1+1; idx_2<N_NODE; idx_2++){
            random_number = (float)rand()/(float)(RAND_MAX); // Float between 0.0 and 1.0            
            if (random_number <= 0.5){
                (*edge_src)[*n_undirected_edges] = idx_1;
                (*edge_dst)[*n_undirected_edges] = idx_2;

                (*v_adj_length)[idx_1]++;
                (*v_adj_length)[idx_2]++;

                (*n_undirected_edges)++; 
            }
            if (*n_undirected_edges >= N_EDGES_TARGET){
                break;
            }
        }
        if (*n_undirected_edges >= N_EDGES_TARGET){
            break;
        }
    }
    n_total_edge = *n_undirected_edges * 2;
    *v_adj_list = (int*)malloc(n_total_edge * sizeof(int));
    init_zero(*v_adj_list, n_total_edge);

    prefix_sum_exclusive(*v_adj_length, *v_adj_begin, N_NODE);    
    *v_adj_begin_2 = (int*)malloc(N_NODE * sizeof(int));
    memcpy(*v_adj_begin_2, *v_adj_begin, N_NODE * sizeof(int));

    for (idx=0; idx<*n_undirected_edges; idx++){
        // a -> b first. Then b -> a
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
            ev1_sum += (w_v1 / sqrt(w_e*d_edge_weight[d_v_adj_begin[v1] + j]));
        }

        for (j=0; j<d_v_adj_length[v2]; j++){
            v = d_v_adj_list[d_v_adj_begin[v2] + j];
            if (v == v1){
                continue;
            }
            ev2_sum += (w_v2 / sqrt(w_e*d_edge_weight[d_v_adj_begin[v2] + j]));
        }

        d_edge_curvature[idx_v1_v2_adj_list] = w_e * (w_v1 / w_e + w_v2 / w_e - (ev1_sum + ev2_sum));
        d_edge_curvature[idx_v2_v1_adj_list] = d_edge_curvature[idx_v1_v2_adj_list];

        
        idx += gridDim.x*blockDim.x;
    }
}

__global__ void update_weight(
        int *d_v_adj_list, int *d_v_adj_begin, int *d_v_adj_length,
        int *d_edge_src, int *d_edge_dst,
        float *d_edge_weight, float*d_edge_curvature,
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
        w_new = (1-STEP_SIZE*d_edge_curvature[idx_v1_v2_adj_list])*d_edge_weight[idx_v1_v2_adj_list];
        // w_new = (1+STEP_SIZE*d_edge_curvature[idx_v1_v2_adj_list])*d_edge_weight[idx_v1_v2_adj_list];
        
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
        int n_total_edge
) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    while (idx < n_total_edge) {
        d_edge_weight[idx] = d_edge_weight[idx] / total_weight;
        idx += gridDim.x * blockDim.x;
    }
}

__global__ void bfs_loop_for_cc_labeling(
        int *d_v_adj_list, int *d_v_adj_begin, int *d_v_adj_length, 
        int *d_dist_arr, int *d_still_running, int start_vertex,
        int *component_list, int component_idx){    
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    // int tid = threadIdx.x;
    int neighbor_idx;
    int j;

    while (idx < N_NODE){
        for (j=0; j<d_v_adj_length[idx]; j++){
            neighbor_idx = d_v_adj_list[d_v_adj_begin[idx] + j];
            if (d_dist_arr[neighbor_idx] > (d_dist_arr[idx]+1)){
                d_dist_arr[neighbor_idx] = d_dist_arr[idx]+1;
                *d_still_running = 1;
                component_list[neighbor_idx] = component_idx;
                // atomicOr(d_still_running, 1); // Uncomment if you want guaranteed correctness
            }
        }
        idx += gridDim.x*blockDim.x;
    }
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
                if (w >= threshold && component_id[neighbor] == -1){
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
    int n_block;
    int *edge_src, *edge_dst;
    int idx, iter, i, j, k, idx_1;
    // int idx_1, idx_2;    
    // float random_number;
    float edge_weight_sum;

    // Create random edges for an undirected graph
    // Then put it into adjacency list form (v_adj_list, v_adj_begin, v_adj_length)
    int *v_adj_length, *v_adj_begin, *v_adj_list;  
    int *v_adj_begin_2;
    float *h_partial_sum, *d_partial_sum;
    // int *d_n_undirected_edges;
    int n_total_edge, n_undirected_edges;
    // cudaMalloc(&d_n_undirected_edges, sizeof(int));

    // create_random_undirected_graph(
    //     &v_adj_length, &v_adj_begin, &v_adj_list,
    //     &edge_src, &edge_dst, &v_adj_begin_2,
    //     &n_undirected_edges);    

    int *node_cluster;
    create_sbm_graph(
        &v_adj_length, &v_adj_begin, &v_adj_list,
        &edge_src, &edge_dst, &v_adj_begin_2,
        &n_undirected_edges, &node_cluster);
    n_total_edge = 2*n_undirected_edges;

    // Print results to verify
    printf("Generated %d undirected edges (%d directed entries)\n", n_undirected_edges, n_total_edge);

    printf("\nAdjacency list:\n");
    for (i = 0; i < N_NODE; i++) {
        printf("Node %d: ", i);
        for (j = 0; j < v_adj_length[i]; j++) {
            printf("%d ", v_adj_list[v_adj_begin[i] + j]);
        }
        printf("\n");
    }

    for (idx=0; idx<n_undirected_edges; idx++){
        printf("(%d, %d)\n", edge_src[idx], edge_dst[idx]);
    }

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

        cudaMemcpy(h_edge_curvature, d_edge_curvature, n_total_edge*sizeof(float), cudaMemcpyDeviceToHost);
        cudaCheckErrors("Copy from d_edge_curvature to h_edge_curvature failed.");
        
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
            d_edge_weight, edge_weight_sum, n_total_edge);
        cudaDeviceSynchronize();

        printf("Iteration %d: sum_weights = %.4f\n", iter, edge_weight_sum);
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

    // printf("\n=== Cutoff Search ===\n");
    // printf("%-10s %-15s %-12s %-12s\n", "Cutoff", "Threshold", "Communities", "Modularity");

    // float cutoffs[] = {0.05f, 0.10f, 0.15f, 0.20f, 0.25f, 0.30f, 0.35f, 0.40f,
    //                 0.45f, 0.50f, 0.55f, 0.60f, 0.65f, 0.70f, 0.75f, 0.80f};

    // int *component_id = (int*)malloc(N_NODE * sizeof(int));
    // float best_modularity = -1.0f;
    // int best_communities = 0;
    // float best_cutoff = 0.0f;

    // for (int c = 0; c < 16; c++) {
    //     int threshold_idx = (int)(cutoffs[c] * n_total_edge);
    //     float threshold = h_sorted_weights[threshold_idx];
        
    //     // Find connected components (keep edges with weight > threshold)
    //     int n_communities = find_connected_components_cpu_reference(
    //         v_adj_list, v_adj_begin, v_adj_length,
    //         h_edge_weight, threshold, component_id, N_NODE);
        
    //     float modularity = calculate_modularity_cpu_reference(
    //         v_adj_list, v_adj_begin, v_adj_length,
    //         component_id, N_NODE, n_undirected_edges);
        
    //     printf("%-10.0f%% %-15.8f %-12d %-12.6f", 
    //         cutoffs[c]*100, threshold, n_communities, modularity);
        
    //     if (modularity > best_modularity) {
    //         best_modularity = modularity;
    //         best_communities = n_communities;
    //         best_cutoff = cutoffs[c];
    //         printf(" *");
    //     }
    //     printf("\n");
    // }

    // printf("\nBest: %.0f%% cutoff → %d communities (modularity: %.4f)\n",
    //     best_cutoff*100, best_communities, best_modularity);
    // printf("Ground truth: %d clusters\n", N_CLUSTERS);


    // printf("\n=== Cutoff Search (linear threshold range) ===\n");
    // printf("%-15s %-12s %-12s\n", "Threshold", "Communities", "Modularity");

    // int *component_id = (int*)malloc(N_NODE * sizeof(int));
    // float best_modularity = -1.0f;
    // int best_communities = 0;
    // float best_threshold = 0.0f;

    // // Use actual min/max from analysis (not sorted array which has duplicates)
    // float min_thresh = min_w;
    // float max_thresh = max_w;

    // for (int i = 0; i <= 20; i++) {
    //     float threshold = min_thresh + (max_thresh - min_thresh) * i / 20.0f;
        
    //     int n_communities = find_connected_components_cpu_reference(
    //         v_adj_list, v_adj_begin, v_adj_length,
    //         h_edge_weight, threshold, component_id, N_NODE);
        
    //     float modularity = calculate_modularity_cpu_reference(
    //         v_adj_list, v_adj_begin, v_adj_length,
    //         component_id, N_NODE, n_undirected_edges);
        
    //     printf("%-15.8f %-12d %-12.6f", threshold, n_communities, modularity);
        
    //     if (modularity > best_modularity) {
    //         best_modularity = modularity;
    //         best_communities = n_communities;
    //         best_threshold = threshold;
    //         printf(" *");
    //     }
    //     printf("\n");
    // }

    // printf("\nBest: threshold=%.8f → %d communities (modularity: %.4f)\n",
    //     best_threshold, best_communities, best_modularity);
    // printf("Ground truth: %d clusters\n", N_CLUSTERS);

    // free(component_id);

    
    printf("\n=== Cutoff Search (maximum resolution) ===\n");
    printf("%-18s %-12s %-12s\n", "Threshold", "Communities", "Modularity");

    int *component_id = (int*)malloc(N_NODE * sizeof(int));
    float best_modularity = -1.0f;
    int best_communities = 0;
    float best_threshold = 0.0f;

    // 1.00000x to 1.00010x in tiny steps
    for (int i = 0; i <= 100; i++) {
        float multiplier = 1.0f + i * 0.000001f;  // 0.0001% increments
        float threshold = min_w * multiplier;
        
        int n_communities = find_connected_components_cpu_reference(
            v_adj_list, v_adj_begin, v_adj_length,
            h_edge_weight, threshold, component_id, N_NODE);
        
        float modularity = calculate_modularity_cpu_reference(
            v_adj_list, v_adj_begin, v_adj_length,
            component_id, N_NODE, n_undirected_edges);
        
        printf("%.16f %-12d %-12.6f", threshold, n_communities, modularity);
        
        if (modularity > best_modularity) {
            best_modularity = modularity;
            best_communities = n_communities;
            best_threshold = threshold;
            printf(" *");
        }
        printf("\n");
    }

    printf("\nBest: threshold=%.16f → %d communities (modularity: %.4f)\n",
        best_threshold, best_communities, best_modularity);
    printf("Ground truth: %d clusters\n", N_CLUSTERS);

    free(component_id);



    // n_block = (n_total_edge + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // copy_back_from_padded_array<<<n_block, BLOCK_SIZE>>> (d_edge_weight_padded, gpuArrbiton, size);

    // For each threshold in the 10%, 20%, 30%, etc. 90%, etc. of the weights
    // we now cut the graph and get the connected components
    // then calculate the modularity and get the best results

    

    // =========================================================================
    // Cleanup - Free device memory
    // =========================================================================
    cudaFree(d_v_adj_list);
    cudaFree(d_v_adj_begin);
    cudaFree(d_v_adj_length);
    cudaFree(d_edge_src);
    cudaFree(d_edge_dst);
    cudaFree(d_edge_weight);
    cudaFree(d_edge_curvature);
    cudaFree(d_partial_sum); 

    // =========================================================================
    // Cleanup - Free host memory
    // =========================================================================
    free(v_adj_length);
    free(v_adj_begin);
    free(v_adj_list);
    free(v_adj_begin_2);
    free(edge_src);
    free(edge_dst);
    free(h_edge_weight);
    free(h_edge_curvature);
    free(h_partial_sum);
   

}