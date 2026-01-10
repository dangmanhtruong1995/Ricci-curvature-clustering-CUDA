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

#include <algorithm>
#include <vector>


#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/fill.h>

// #define N_NODE 1000
// #define NODES_PER_CLUSTER 500
// #define N_CLUSTERS 2
// #define P_IN 0.12
// #define P_OUT 0.01

// #define N_NODE 1000
// #define NODES_PER_CLUSTER 500
// #define N_CLUSTERS 2
// #define P_IN 0.15
// #define P_OUT 0.01

// #define N_NODE 5000
// #define NODES_PER_CLUSTER 2500
// #define N_CLUSTERS 2
// #define P_IN 0.4
// #define P_OUT 0.01

// #define N_NODE 5000
// #define NODES_PER_CLUSTER 1000
// #define N_CLUSTERS 5
// #define P_IN 0.5
// #define P_OUT 0.01

// #define N_NODE 5000
// #define NODES_PER_CLUSTER 500
// #define N_CLUSTERS 10
// #define P_IN 0.7
// #define P_OUT 0.01

// #define N_NODE 50000
// #define NODES_PER_CLUSTER 10000
// #define N_CLUSTERS 5
// #define P_IN 0.5
// #define P_OUT 0.05

// #define N_NODE 10000
// #define N_CLUSTERS 10
// #define NODES_PER_CLUSTER 1000
// #define P_IN 0.4   
// #define P_OUT 0.005

// #define N_NODE 100000
// #define N_CLUSTERS 100
// #define NODES_PER_CLUSTER 1000
// #define P_IN 0.4      
// #define P_OUT 0.001

// #define N_NODE 300000
// #define N_CLUSTERS 300
// #define NODES_PER_CLUSTER 1000
// #define P_IN 0.4      
// #define P_OUT 0.0005

#define N_NODE 500000
#define N_CLUSTERS 500
#define NODES_PER_CLUSTER 1000
#define P_IN 0.9     
#define P_OUT 0.0005

// Ratio between expected intra-cluster connections and inter-cluster connections
// (P_in*(N/N_C-1))/(P_out*(N-N/N_C))


// 1. Calculate nodes per cluster (n)
#define N_PER_C ((double)N_NODE / N_CLUSTERS)

// 2. Expected internal edges: K * [n(n-1)/2] * P_in
#define EXPECTED_INTERNAL (N_CLUSTERS * (N_PER_C * (N_PER_C - 1.0) / 2.0) * P_IN)

// 3. Expected external edges: [K(K-1)/2] * n^2 * P_out
#define EXPECTED_EXTERNAL ((N_CLUSTERS * (N_CLUSTERS - 1.0) / 2.0) * (N_PER_C * N_PER_C) * P_OUT)

// 4. Final Max: (Internal + External) * 1.2 Safety Margin
#define N_EDGES_MAX ((int)((EXPECTED_INTERNAL + EXPECTED_EXTERNAL) * 1.2))

// #define N_EDGES_MAX 1100000

#define N_ITERATION 10

#define BLOCK_SIZE 256 // CUDA maximum is 1024
#define VERY_LARGE_NUMBER 99999
#define VWARP_SIZE 16 // Size of virtual warp

// Paper parameters (Section 6.2.1)
#define STEP_SCALE 1.1f      // νt = 1 / (1.1 × max|κ|)
#define QUANTILE_Q 0.999f    // q = 0.999 for cutoff
#define DELTA_STEP 0.25f     // δ = 0.25 for uniform spacing
// #define DROP_THRESHOLD 0.1f  // d = 0.1 (skip if improvement < 10%)
// #define MIN_MODULARITY 0.0f  // ε for minimum acceptable modularity

#define MIN_WEIGHT 1e-6
#define MAX_CURVATURE 1000.0
#define MIN_AREA 1e-12
#define MAX_TRIANGLE_CONTRIB 100.0

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
// Sorting on GPU
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
    // double prefix_sum=0;
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
    double random_number;
    double edge_prob;
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
            
            random_number = (double)rand() / (double)(RAND_MAX);
            
            if (random_number <= edge_prob) {
                (*edge_src)[*n_undirected_edges] = idx_1;
                (*edge_dst)[*n_undirected_edges] = idx_2;

                (*v_adj_length)[idx_1]++;
                (*v_adj_length)[idx_2]++;

                (*n_undirected_edges)++;
            }
            
            if ((*n_undirected_edges) >= N_EDGES_MAX) {
                break;
            }
        }
        if ((*n_undirected_edges) >= N_EDGES_MAX) {
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
__global__ void calc_augmented_forman_ricci_curvature(
        int *d_v_adj_list, int *d_v_adj_begin, int *d_v_adj_length,
        int *d_edge_src, int *d_edge_dst,
        double *d_edge_weight, double *d_edge_curvature,        
        int n_undirected_edges
        ){    
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    int v1;
    int v2;
    // int is_triangle;

    while (idx < n_undirected_edges){
        v1 = d_edge_src[idx];
        v2 = d_edge_dst[idx];        

        // Find edge weight and indices for (v1,v2)
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
 
        double w_e = fmax(d_edge_weight[idx_v1_v2], MIN_WEIGHT);     
        double triangle_contrib = 0.0f;        
        double sum_ve = 2.0f / w_e; // Vertex contribution 
        double sum_veeh = 0.0f; // Parallel edge contribution (non-triangle neighbors)
        
        int start_v1 = d_v_adj_begin[v1];
        int len_v1 = d_v_adj_length[v1];
        int end_v1 = start_v1 + len_v1;
        
        int start_v2 = d_v_adj_begin[v2];
        int len_v2 = d_v_adj_length[v2];
        int end_v2 = start_v2 + len_v2;

        int i1 = start_v1;
        int i2 = start_v2;

        while ((i1 < end_v1) && (i2 < end_v2)){
            int n1 = d_v_adj_list[i1];
            int n2 = d_v_adj_list[i2];

            // Skip v2 in v1's list and v1 in v2's list
            if (n1 == v2) { i1++; continue; }
            if (n2 == v1) { i2++; continue; }
            if (n1 == n2){
                // Common neighbor - triangle contribution
                double w1 = fmax(d_edge_weight[i1], MIN_WEIGHT);
                double w2 = fmax(d_edge_weight[i2], MIN_WEIGHT);

                double s = (w_e + w1 + w2) / 2.0;
                double term1 = s - w_e;
                double term2 = s - w1;
                double term3 = s - w2;
                double area_sq = fabs(s * term1 * term2 * term3);
                double w_tri = sqrt(fmax(area_sq, MIN_AREA));
                // triangle_contrib += fmin(w_e / w_tri, MAX_TRIANGLE_CONTRIB);
                triangle_contrib += w_e / w_tri;
                
                i1++; i2++;
            } else if (n1 < n2) {
                // n1 only in v1's neighborhood - parallel edge
                double w_ep = fmax(d_edge_weight[i1], MIN_WEIGHT);
                sum_veeh += 1.0 / sqrt(w_e * w_ep);
                i1++;
            } else {
                // n2 only in v2's neighborhood - parallel edge
                double w_ep = fmax(d_edge_weight[i2], MIN_WEIGHT);
                sum_veeh += 1.0 / sqrt(w_e * w_ep);
                i2++;
            }
        }

        // Remaining in v1's list are all non-common
        while (i1 < end_v1) {
            if (d_v_adj_list[i1] != v2) {
                double w_ep = fmax(d_edge_weight[i1], MIN_WEIGHT);
                sum_veeh += 1.0 / sqrt(w_e * w_ep);
            }
            i1++;
        }
        
        // Remaining in v2's list are all non-common
        while (i2 < end_v2) {
            if (d_v_adj_list[i2] != v1) {
                double w_ep = fmax(d_edge_weight[i2], MIN_WEIGHT);
                sum_veeh += 1.0 / sqrt(w_e * w_ep);
            }
            i2++;
        }

        // Final curvature
        double curvature = w_e * (triangle_contrib + sum_ve - sum_veeh);
        curvature = fmin(MAX_CURVATURE, fmax(-MAX_CURVATURE, curvature));

        d_edge_curvature[idx_v1_v2] = curvature;
        d_edge_curvature[idx_v2_v1] = curvature;
        
        idx += gridDim.x * blockDim.x;
    }
}


__global__ void update_weight(
        int *d_v_adj_list, int *d_v_adj_begin, int *d_v_adj_length,
        int *d_edge_src, int *d_edge_dst,
        double *d_edge_weight, double*d_edge_curvature,
        double adaptive_size,
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
        // Inspecting idx-th edge
        v1 = d_edge_src[idx];
        v2 = d_edge_dst[idx];

        idx_v1_v2_adj_list = -1;
        idx_v2_v1_adj_list = -1;

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

        if (idx_v1_v2_adj_list >= 0) {
            w_new = (1-adaptive_size*d_edge_curvature[idx_v1_v2_adj_list])*d_edge_weight[idx_v1_v2_adj_list];        
            
            // Clamp to prevent negative weights
            // if (w_new < 0.0001f) w_new = 0.0001f;
            w_new = fmax(w_new, MIN_WEIGHT);

            d_edge_weight[idx_v1_v2_adj_list] = w_new;
            d_edge_weight[idx_v2_v1_adj_list] = w_new;

            if (idx_v2_v1_adj_list >= 0) {
                d_edge_weight[idx_v2_v1_adj_list] = w_new;
            }
        }
        
        idx += gridDim.x*blockDim.x;
    }        
}

__global__ void array_sum_blockwise(
        double *arr,
        int size,
        double *d_partial_sum // Summation for all elements within the block only
){
    __shared__ double cache[BLOCK_SIZE];
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int cacheIndex = threadIdx.x;

    double temp = 0;
	
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
        double *d_edge_weight,
        double total_weight,
        int n_undirected_edges,
        int n_total_edge
) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    double scale = (double)n_undirected_edges / (total_weight / 2.0f);

    while (idx < n_total_edge) {
        d_edge_weight[idx] = d_edge_weight[idx] * scale;
        idx += gridDim.x * blockDim.x;
    }
}

// __global__ void clamp_weights_after_norm(double *d_edge_weight, int n_total_edge) {
//     int idx = threadIdx.x + blockDim.x * blockIdx.x;
//     while (idx < n_total_edge) {
//         if (d_edge_weight[idx] < 1e-10) d_edge_weight[idx] = 1e-10;
//         idx += gridDim.x * blockDim.x;
//     }
// }

__global__ void bfs_loop_using_virtual_warp_for_cc_labeling(
        int *d_v_adj_list, int *d_v_adj_begin, int *d_v_adj_length,         
        int *d_edge_src, int *d_edge_dst,
        double *d_edge_weight,
        int *d_dist_arr, int *d_still_running,
        int iteration_counter, int start_vertex,
        int *component_list, int component_idx,
        double threshold
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
    // double w;
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
    double *d_edge_weight,
    int *h_component_list, int *d_component_list,
    int *d_dist_arr,
    int *d_still_running,
    double threshold){

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
    double *edge_weight, double *edge_curvature,
    int n_undirected_edges
) {
    for (int idx = 0; idx < n_undirected_edges; idx++) {
        int v1 = edge_src[idx];
        int v2 = edge_dst[idx];

        // Find edge weight w_e
        double w_e = 1.0f;
        int idx_v1_v2 = -1;
        for (int j = 0; j < v_adj_length[v1]; j++) {
            if (v_adj_list[v_adj_begin[v1] + j] == v2) {
                idx_v1_v2 = v_adj_begin[v1] + j;
                w_e = edge_weight[idx_v1_v2];
                break;
            }
        }

        double w_v1 = 1.0f;
        double w_v2 = 1.0f;

        // Sum over neighbors of v1 (excluding v2)
        double ev1_sum = 0.0f;
        for (int j = 0; j < v_adj_length[v1]; j++) {
            int v = v_adj_list[v_adj_begin[v1] + j];
            if (v == v2) continue;
            double w_e_prime = edge_weight[v_adj_begin[v1] + j];
            ev1_sum += w_v1 / sqrt(w_e * w_e_prime);
        }

        // Sum over neighbors of v2 (excluding v1)
        double ev2_sum = 0.0f;
        for (int j = 0; j < v_adj_length[v2]; j++) {
            int v = v_adj_list[v_adj_begin[v2] + j];
            if (v == v1) continue;
            double w_e_prime = edge_weight[v_adj_begin[v2] + j];
            ev2_sum += w_v2 / sqrt(w_e * w_e_prime);
        }

        double curvature = w_e * (w_v1 / w_e + w_v2 / w_e - (ev1_sum + ev2_sum));
        edge_curvature[idx_v1_v2] = curvature;
    }
}

// Find connected components using BFS
// Returns number of communities and fills component_id array
int find_connected_components_cpu_reference(
    int *v_adj_list, int *v_adj_begin, int *v_adj_length,
    double *edge_weight, double threshold,
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
                double w = edge_weight[idx];
                
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
double calculate_modularity_cpu_reference(
    int *v_adj_list, int *v_adj_begin, int *v_adj_length,
    int *component_id,
    int n_nodes, int n_edges  // n_edges = number of undirected edges
) {
    int m = n_edges;  // Total undirected edges
    double modularity = 0.0f;
    
    // For each edge, check if both endpoints are in same community
    for (int u = 0; u < n_nodes; u++) {
        int k_u = v_adj_length[u];  // Degree of u
        
        for (int j = 0; j < v_adj_length[u]; j++) {
            int v = v_adj_list[v_adj_begin[u] + j];
            int k_v = v_adj_length[v];  // Degree of v
            
            // A_ij = 1 (edge exists)
            // δ(c_i, c_j) = 1 if same community
            if (component_id[u] == component_id[v]) {
                modularity += 1.0f - (double)(k_u * k_v) / (2.0f * m);
            } else {
                modularity += 0.0f - (double)(k_u * k_v) / (2.0f * m);
            }
        }
    }
    
    modularity /= (2.0f * m);
    return modularity;
}

// Find optimal threshold and communities using Ricci flow weights
void find_communities_ricci_cpu_reference(
    int *v_adj_list, int *v_adj_begin, int *v_adj_length,
    double *edge_weight,
    int *edge_src, int *edge_dst,
    int n_nodes, int n_undirected_edges
) {
    int *component_id = (int*)malloc(n_nodes * sizeof(int));
    
    // Collect all unique edge weights to use as thresholds
    double *thresholds = (double*)malloc(n_undirected_edges * sizeof(double));
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
                double tmp = thresholds[i];
                thresholds[i] = thresholds[j];
                thresholds[j] = tmp;
            }
        }
    }
    
    // Try different thresholds and compute modularity
    double best_modularity = -1.0f;
    double best_threshold = 0.0f;
    int best_n_communities = 1;
    
    printf("\n=== Threshold Search (CPU Reference) ===\n");
    printf("%-15s %-12s %-12s\n", "Threshold", "Communities", "Modularity");
    printf("-------------------------------------------\n");
    
    // Sample thresholds (don't try all - too slow)
    int step = n_thresholds / 50;  // ~50 samples
    if (step < 1) step = 1;
    
    for (int i = 0; i < n_thresholds; i += step) {
        double threshold = thresholds[i];
        
        int n_communities = find_connected_components_cpu_reference(
            v_adj_list, v_adj_begin, v_adj_length,
            edge_weight, threshold,
            component_id, n_nodes
        );
        
        double modularity = calculate_modularity_cpu_reference(
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

int find_connected_components(
    int *v_adj_list, int *v_adj_begin, int *v_adj_length,
    double *edge_weight, double threshold,
    int *component_id, int n_nodes
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
                
                // Traverse edge only if weight < threshold (internal edge)
                // High-weight edges are bridges that we cut
                if (w < threshold && component_id[neighbor] == -1) {
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

double calculate_modularity(
    int *v_adj_list, int *v_adj_begin, int *v_adj_length,
    int *component_id, int n_nodes, int n_edges
) {
    if (n_edges == 0) return 0.0;

    int m = n_edges;
    double modularity = 0.0f;
    
    for (int u = 0; u < n_nodes; u++) {
        int k_u = v_adj_length[u];
        
        for (int j = 0; j < v_adj_length[u]; j++) {
            int v = v_adj_list[v_adj_begin[u] + j];
            int k_v = v_adj_length[v];
            
            if (component_id[u] == component_id[v]) {
                modularity += 1.0f - (double)(k_u * k_v) / (2.0f * m);
            } else {
                modularity += 0.0f - (double)(k_u * k_v) / (2.0f * m);
            }
        }
    }
    
    modularity /= (2.0f * m);
    return modularity;
}

double calculate_nmi(int *pred_labels, int *true_labels, int n_nodes, 
                    int n_pred_clusters, int n_true_clusters) {
    // Compute confusion matrix
    int max_clusters = (n_pred_clusters > n_true_clusters) ? n_pred_clusters : n_true_clusters;
    int *confusion = (int*)calloc(max_clusters * max_clusters, sizeof(int));
    int *pred_counts = (int*)calloc(max_clusters, sizeof(int));
    int *true_counts = (int*)calloc(max_clusters, sizeof(int));
    
    for (int i = 0; i < n_nodes; i++) {
        int p = pred_labels[i];
        int t = true_labels[i];
        if (p >= 0 && p < max_clusters && t >= 0 && t < max_clusters) {
            confusion[p * max_clusters + t]++;
            pred_counts[p]++;
            true_counts[t]++;
        }
    }
    
    // Compute mutual information
    double mi = 0.0f;
    for (int i = 0; i < n_pred_clusters; i++) {
        for (int j = 0; j < n_true_clusters; j++) {
            if (confusion[i * max_clusters + j] > 0 && 
                pred_counts[i] > 0 && true_counts[j] > 0) {
                double pij = (double)confusion[i * max_clusters + j] / n_nodes;
                double pi = (double)pred_counts[i] / n_nodes;
                double pj = (double)true_counts[j] / n_nodes;
                mi += pij * log(pij / (pi * pj));
            }
        }
    }
    
    // Compute entropies
    double h_pred = 0.0f, h_true = 0.0f;
    for (int i = 0; i < n_pred_clusters; i++) {
        if (pred_counts[i] > 0) {
            double p = (double)pred_counts[i] / n_nodes;
            h_pred -= p * log(p);
        }
    }
    for (int j = 0; j < n_true_clusters; j++) {
        if (true_counts[j] > 0) {
            double p = (double)true_counts[j] / n_nodes;
            h_true -= p * log(p);
        }
    }
    
    free(confusion);
    free(pred_counts);
    free(true_counts);
    
    // NMI = 2 * MI / (H_pred + H_true)
    double denom = h_pred + h_true;
    if (denom < 1e-10f) return 0.0f;
    return 2.0f * mi / denom;
}


int main(){
    // srand(time(NULL));
    srand(42);

    printf("\n=== Graph Configuration ===\n");
    printf("N_NODE:             %d\n", N_NODE);
    printf("N_CLUSTERS:         %d\n", N_CLUSTERS);
    printf("NODES_PER_CLUSTER:  %d\n", NODES_PER_CLUSTER);
    printf("P_IN:               %.4f\n", P_IN);
    printf("P_OUT:              %.6f\n", P_OUT);
    printf("N_EDGES_MAX:        %d\n", N_EDGES_MAX);
    printf("N_ITERATION:        %d\n", N_ITERATION);
    printf("Intra/Inter ratio:  %.4f\n", (P_IN * (N_PER_C - 1)) / (P_OUT * (N_NODE - N_PER_C)));
    printf("===============================\n\n");

    int n_block;
    int *edge_src, *edge_dst;
    int idx, iter, i, j, k;
    // int idx_1, idx_2;    
    // double random_number;
    double edge_weight_sum;

    // Create random edges for an undirected graph
    // Then put it into adjacency list form (v_adj_list, v_adj_begin, v_adj_length)
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

    printf("Number of undirected edges: %d\n", n_undirected_edges);

    // Verify adjacency lists are sorted
    int unsorted_count = 0;
    for (int i = 0; i < N_NODE; i++) {
        for (int j = 1; j < v_adj_length[i]; j++) {
            if (v_adj_list[v_adj_begin[i] + j] < v_adj_list[v_adj_begin[i] + j - 1]) {
                unsorted_count++;
                break;
            }
        }
    }
    if (unsorted_count > 0) {
        printf("WARNING: %d nodes have unsorted adjacency lists!\n", unsorted_count);
    }

    // Verify the graph
    verify_graph(edge_src, edge_dst, n_undirected_edges,
             v_adj_list, v_adj_begin, v_adj_length);

    // Ricci curvature
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
    h_partial_sum = (double*)malloc( n_block*sizeof(double) );
    cudaMalloc(&d_partial_sum, n_block * sizeof(double));
    cudaCheckErrors("cudaMalloc for d_partial_sum failed.");

    int *h_component_list, *d_component_list;
    int *d_dist_arr;
    int *d_still_running;
    int *h_best_labels;

    cudaMalloc(&d_still_running, sizeof(int));
    cudaCheckErrors("cudaMalloc for d_still_running failed.");     
    cudaMalloc(&d_component_list, N_NODE * sizeof(int));  
    cudaCheckErrors("cudaMalloc for d_component_list failed.");
    cudaMalloc(&d_dist_arr, N_NODE * sizeof(int));
    cudaCheckErrors("cudaMalloc for d_dist_arr failed.");

    h_component_list = (int*) malloc(N_NODE * sizeof(int));   
    h_best_labels = (int*)malloc(N_NODE * sizeof(int));

    cudaEvent_t start, stop;
    float milliseconds_ricci_flow = 0.0;
    float milliseconds_threshold_search = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
   
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
            d_edge_weight, d_edge_curvature,
            n_undirected_edges);
        cudaDeviceSynchronize();
        cudaCheckErrors("calc_augmented_forman_ricci_curvature failed");

        cudaMemcpy(h_edge_curvature, d_edge_curvature, n_total_edge*sizeof(double), cudaMemcpyDeviceToHost);
        cudaCheckErrors("Copy from d_edge_curvature to h_edge_curvature failed.");
        
        // After computing curvature, find max absolute value
        double max_curv = 0.0f;
        for (int i = 0; i < n_total_edge; i++) {
            double abs_curv = fabs(h_edge_curvature[i]);
            if (abs_curv > max_curv) max_curv = abs_curv;
        }

        // Adaptive step size (as in paper)
        double adaptive_step = 1.0f / (STEP_SCALE * max_curv + 1e-10f);
        adaptive_step = fmin(adaptive_step, 1.0);

        printf("Iter %d: max_curv=%.2f, step=%.6f\n", iter, max_curv, adaptive_step);

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

        cudaMemcpy(h_partial_sum, d_partial_sum, n_block*sizeof(double), cudaMemcpyDeviceToHost );
        edge_weight_sum = 0.0;
        for (idx=0; idx<n_block; idx++) {
            edge_weight_sum += h_partial_sum[idx];
        }

        n_block = (n_total_edge + BLOCK_SIZE - 1) / BLOCK_SIZE;
        normalize_weights<<<n_block, BLOCK_SIZE>>>(
            d_edge_weight, edge_weight_sum, n_undirected_edges, n_total_edge);
        cudaDeviceSynchronize();

        // clamp_weights_after_norm<<<n_block, BLOCK_SIZE>>>(d_edge_weight, n_total_edge);
        // cudaDeviceSynchronize();

        printf("Iteration %d: sum_weights = %.4f\n", iter, edge_weight_sum);       
    }   
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds_ricci_flow, start, stop);
    // printf("Ricci flow iterations: %.6f ms.\n", milliseconds_ricci_flow);

    // Copy final weights back
    cudaMemcpy(h_edge_weight, d_edge_weight, n_total_edge * sizeof(double), cudaMemcpyDeviceToHost);

    // Analyze weight distribution
    double min_w = 1e9f, max_w = -1e9f;
    double intra_sum = 0.0f, inter_sum = 0.0f;
    int intra_w_count = 0, inter_w_count = 0;
    
    std::vector<double> all_weights;
    
    for (int i = 0; i < n_undirected_edges; i++) {
        int v1 = edge_src[i];
        int v2 = edge_dst[i];
        
        double w = 0.0f;
        for (int j = 0; j < v_adj_length[v1]; j++) {
            if (v_adj_list[v_adj_begin[v1] + j] == v2) {
                w = h_edge_weight[v_adj_begin[v1] + j];
                break;
            }
        }
        
        all_weights.push_back(w);
        if (w < min_w) min_w = w;
        if (w > max_w) max_w = w;
        
        if (node_cluster[v1] == node_cluster[v2]) {
            intra_sum += w;
            intra_w_count++;
        } else {
            inter_sum += w;
            inter_w_count++;
        }
    }

    printf("\n=== Weight Analysis ===\n");
    printf("Min weight: %.6f\n", min_w);
    printf("Max weight: %.6f\n", max_w);
    printf("Ratio max/min: %.2f\n", max_w / min_w);
    printf("Intra-cluster avg weight: %.6f\n", intra_sum / intra_w_count);
    printf("Inter-cluster avg weight: %.6f\n", inter_sum / inter_w_count);
    printf("Weight ratio (inter/intra): %.4f\n", 
           (inter_sum / inter_w_count) / (intra_sum / intra_w_count));


    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Sort the weights in ascending order
    unsigned int next_pow = next_power_of_2(n_total_edge);
    double *d_edge_weight_padded;
    // h_edge_weight_padded = (double*)malloc(next_pow * sizeof(double));
    cudaMalloc((void**)&d_edge_weight_padded, next_pow * sizeof(double));

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
    double *h_sorted_weights = (double*)malloc(n_total_edge * sizeof(double));
    cudaMemcpy(h_sorted_weights, d_edge_weight_padded, n_total_edge * sizeof(double), cudaMemcpyDeviceToHost);

    // Print sorted weights
    printf("\n=== Sorted Weights (ascending) ===\n");
    printf("First 20 weights (smallest):\n");
    for (i = 0; i < 20; i++) {
        printf("  [%d] %.8f\n", i, h_sorted_weights[i]);
    }

    min_w = h_sorted_weights[0];

    // Get 99th quantile
    int q_idx = (int)(QUANTILE_Q * n_total_edge);
    if (q_idx >= n_total_edge) q_idx = n_total_edge - 1;
    double w_quantile = h_sorted_weights[q_idx]; 
    printf("Weight quantile (%.3f): %.6f\n", QUANTILE_Q, w_quantile);

    // Get cut-off points
    int n_cutoff=0;
    double w_stop = 1.1 * min_w;
    int cutoff_idx;
    double threshold;

    for (idx=0; idx<n_total_edge; idx +=2){
        // idx+=2 because the weights store both (i, j) and (j, i) for completeness
        if (h_sorted_weights[idx] > w_quantile){
            n_cutoff++;
        }
    }
    for (double t = w_quantile - DELTA_STEP; t >= w_stop; t -= DELTA_STEP) {
        n_cutoff++;
    }

    double *cutoff_list = (double*)malloc(n_cutoff * sizeof(double));

    cutoff_idx = 0;
    for (idx=0; idx<n_total_edge; idx +=2){
        // idx+=2 because the weights store both (i, j) and (j, i) for completeness
        if (h_sorted_weights[idx] > w_quantile) {
            if (cutoff_idx == 0 || h_sorted_weights[idx] != cutoff_list[cutoff_idx - 1]) {
                    cutoff_list[cutoff_idx++] = h_sorted_weights[idx];
            }
        }
    }
    for (double t = w_quantile - DELTA_STEP; t >= w_stop; t -= DELTA_STEP) {
        cutoff_list[cutoff_idx++] = t;
    }
    n_cutoff = cutoff_idx;

    
    double best_modularity = -1.0;
    double best_threshold = 0.0;
    int best_n_communities = 0;
    // double best_nmi = 0.0;
    

    

    struct Result {
        double threshold;
        int n_communities;
        double modularity;
        double nmi;
    };
    std::vector<Result> all_results; // No point being a C purist at this point lol 

    for (cutoff_idx=0; cutoff_idx<n_cutoff; cutoff_idx+=1){
        threshold = cutoff_list[cutoff_idx]; 
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
            continue;
        }
        
        double modularity = calculate_modularity(
            v_adj_list, v_adj_begin, v_adj_length,
            h_component_list, N_NODE, n_undirected_edges);
        
        double nmi = calculate_nmi(h_component_list, node_cluster, N_NODE, 
                                  n_communities, N_CLUSTERS);

        all_results.push_back({threshold, n_communities, modularity, nmi});
        
        if (n_communities <= 30) {
            char marker = (modularity > best_modularity) ? '*' : ' ';
            printf("%.6e  %-12d %-12.6f %-10.4f %c\n", 
                   threshold, n_communities, modularity, nmi, marker);
        }
        
        if (modularity > best_modularity) {
            best_modularity = modularity;
            best_threshold = threshold;
            best_n_communities = n_communities;
            // best_nmi = nmi;
            memcpy(h_best_labels, h_component_list, N_NODE * sizeof(int));
        }
    }

    // Now find connected components using the best threshold
    bfs_loop_using_virtual_warp_for_cc_labeling_wrapper(
        d_v_adj_list, d_v_adj_begin, d_v_adj_length,
        d_edge_src, d_edge_dst,
        d_edge_weight,
        h_best_labels, d_component_list,
        d_dist_arr, 
        d_still_running,
        best_threshold
    );
    double final_nmi = calculate_nmi(h_best_labels, node_cluster, N_NODE, 
                                    best_n_communities, N_CLUSTERS);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds_threshold_search, start, stop);
    float total_ms = milliseconds_ricci_flow + milliseconds_threshold_search;
    printf("Ricci flow:       %.3f ms (%.3f s)\n", milliseconds_ricci_flow, milliseconds_ricci_flow / 1000.0f);
    printf("Threshold search: %.3f ms (%.3f s)\n", milliseconds_threshold_search, milliseconds_threshold_search / 1000.0f);
    printf("TOTAL TIME:       %.3f ms (%.3f s)\n", total_ms, total_ms / 1000.0f);
 
    
    // Final Results
    printf("\n=======================================================================\n");
    printf("=== FINAL RESULTS ===\n");
    printf("=======================================================================\n");
    printf("Threshold: %.6e\n", best_threshold);
    printf("Communities found: %d (ground truth: %d)\n", best_n_communities, N_CLUSTERS);
    printf("Modularity: %.6f\n", best_modularity);
    printf("NMI: %.6f\n", final_nmi);
    
    int *community_sizes = (int*)calloc(best_n_communities, sizeof(int));
    for (int i = 0; i < N_NODE; i++) {
        if (h_best_labels[i] >= 0 && h_best_labels[i] < best_n_communities) {
            community_sizes[h_best_labels[i]]++;
        }
    }
    
    printf("\nCommunity sizes:\n");
    for (int c = 0; c < best_n_communities && c < 15; c++) {
        printf("  Community %d: %d nodes\n", c, community_sizes[c]);
    }
    if (best_n_communities > 15) {
        printf("  ... (%d more communities)\n", best_n_communities - 15);
    }

    // ==========================================================================
    // Save graph to file
    // ==========================================================================
    FILE *fp = fopen("graph_output.txt", "w");
    if (fp) {
        // Header: num_nodes, num_edges
        fprintf(fp, "%d %d\n", N_NODE, n_undirected_edges);
        
        // Edges: src dst
        for (int i = 0; i < n_undirected_edges; i++) {
            fprintf(fp, "%d %d\n", edge_src[i], edge_dst[i]);
        }
        
        // Cluster assignments (predicted)
        for (int i = 0; i < N_NODE; i++) {
            fprintf(fp, "%d\n", h_best_labels[i]);
        }
        
        fclose(fp);
        printf("\nSaved graph to graph_output.txt\n");
    } else {
        printf("\nError: Could not open graph_output.txt for writing\n");
    }

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
    // cudaFree(d_faces);

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
    // free(h_edge_weight_padded);  
    free(h_best_labels);
    free(cutoff_list);
    free(community_sizes);
    free(node_cluster);
}