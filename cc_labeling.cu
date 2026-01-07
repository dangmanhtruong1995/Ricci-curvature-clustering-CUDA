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

#define N_NODE 10000
#define NODES_PER_CLUSTER 100
// #define STEP_SIZE 1.0
#define STEP_SIZE 0.0001
// #define N_EDGES_TARGET 30000 // Target number of unique undirected edges

#define N_CLUSTERS (N_NODE / NODES_PER_CLUSTER)
#define EDGES_PER_CLUSTER ((NODES_PER_CLUSTER * (NODES_PER_CLUSTER - 1)) / 2)  // C(n,2)
#define N_EDGES_TARGET (N_CLUSTERS * EDGES_PER_CLUSTER)  // Max possible edges

#define BLOCK_SIZE 256 // CUDA maximum is 1024
#define VERY_LARGE_NUMBER 99999
#define VWARP_SIZE 16 // Size of virtual warp
// #define N_ITERATION 10 // Number of iterations in calculating Ricci curvature
#define N_ITERATION 100
// #define N_ITERATION 1 // For debugging purpose
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

// =========================================================================
// Graph operations
// =========================================================================
void create_k_cluster_graph(
        int **v_adj_length,
        int **v_adj_begin,
        int **v_adj_list,
        int **edge_src,
        int **edge_dst,
        int **v_adj_begin_2,
        int *n_undirected_edges,
        int *ground_truth_component
    ){
    int idx_1, idx_2, idx;
    float random_number; 
    int n_total_edge = 0;

    *v_adj_length = (int*)malloc(N_NODE * sizeof(int));
    *v_adj_begin = (int*)malloc(N_NODE * sizeof(int));
    *edge_src = (int*)malloc(N_EDGES_TARGET * sizeof(int));
    *edge_dst = (int*)malloc(N_EDGES_TARGET * sizeof(int));

    init_zero(*v_adj_length, N_NODE);
    init_zero(*v_adj_begin, N_NODE);
    init_zero(*edge_src, N_EDGES_TARGET);
    init_zero(*edge_dst, N_EDGES_TARGET);

    *n_undirected_edges = 0;

    // Ground truth
    for (int i = 0; i < N_NODE; i++) {
        ground_truth_component[i] = i / NODES_PER_CLUSTER;
    }

    for (idx_1 = 0; idx_1 < N_NODE; idx_1++) {
        for (idx_2 = idx_1 + 1; idx_2 < N_NODE; idx_2++) {
            
            // Only connect if same cluster
            if (idx_1 / NODES_PER_CLUSTER != idx_2 / NODES_PER_CLUSTER) {
                continue;
            }

            random_number = (float)rand() / (float)(RAND_MAX);
            
            if (random_number <= 0.5) {
                (*edge_src)[*n_undirected_edges] = idx_1;
                (*edge_dst)[*n_undirected_edges] = idx_2;

                (*v_adj_length)[idx_1]++;
                (*v_adj_length)[idx_2]++;

                (*n_undirected_edges)++;
            }
            
            if (*n_undirected_edges >= N_EDGES_TARGET) {
                break;
            }
        }
        if (*n_undirected_edges >= N_EDGES_TARGET) {
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

int main(){
    srand(time(NULL));
    int n_block;
    int *edge_src, *edge_dst;
    int idx, i, j, idx_1;

    // Create random edges for an undirected graph
    // Then put it into adjacency list form (v_adj_list, v_adj_begin, v_adj_length)
    int *v_adj_length, *v_adj_begin, *v_adj_list;  
    int *v_adj_begin_2;
    int n_total_edge, n_undirected_edges;
    int *ground_truth_component = (int*)malloc(N_NODE * sizeof(int));

    create_k_cluster_graph(
        &v_adj_length, &v_adj_begin, &v_adj_list,
        &edge_src, &edge_dst, &v_adj_begin_2,
        &n_undirected_edges, ground_truth_component);  
    n_total_edge = 2*n_undirected_edges;

    // Print results to verify
    printf("Generated %d undirected edges (%d directed entries)\n", n_undirected_edges, n_total_edge);

    printf("\nAdjacency list:\n");
    for (i = 0; i < N_NODE; i++) {
        printf("Node %d, component %d\n", i, ground_truth_component[i]);
        printf("\n");
    }

    for (idx=0; idx<n_undirected_edges; idx++){
        printf("(%d, %d)\n", edge_src[idx], edge_dst[idx]);
    }

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

    // Find connected components using BFS
    int component_idx=0;
    int *h_component_list, *d_component_list;
    int *d_dist_arr;
    int *h_dist_arr;
    int start_vertex = 0;
    int h_still_running = 1;
    int *d_still_running;
    int still_has_cc=1;

    cudaMalloc(&d_still_running, sizeof(int));
    h_dist_arr = (int*) malloc(N_NODE * sizeof(int));
    h_component_list = (int*) malloc(N_NODE * sizeof(int));
    cudaMalloc(&d_component_list, N_NODE * sizeof(int));
    
    for(idx = 0; idx < N_NODE; idx++) h_dist_arr[idx] = VERY_LARGE_NUMBER;
    cudaMalloc(&d_dist_arr, N_NODE * sizeof(int));
    cudaMemcpy(d_dist_arr, h_dist_arr, N_NODE * sizeof(int), cudaMemcpyHostToDevice);    

    n_block = (N_NODE+BLOCK_SIZE-1)/BLOCK_SIZE; 
    init_on_device<<<n_block, BLOCK_SIZE>>>(d_component_list, N_NODE, -1);
    cudaMemcpy(h_component_list, d_component_list, N_NODE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    while (1){
        still_has_cc = 0;
        
        for(idx = 0; idx < N_NODE; idx++){
            if (h_component_list[idx] < 0){
                still_has_cc = 1;

                // Do a BFS for this current component                
                for(idx_1 = 0; idx_1 < N_NODE; idx_1++) h_dist_arr[idx_1] = VERY_LARGE_NUMBER;
                start_vertex = idx;
                
                for(idx_1 = 0; idx_1 < N_NODE; idx_1++) h_dist_arr[idx_1] = VERY_LARGE_NUMBER;
                h_dist_arr[start_vertex] = 0;
                cudaMemcpy(d_dist_arr, h_dist_arr, N_NODE * sizeof(int), cudaMemcpyHostToDevice);    

                h_still_running = 1;
                h_component_list[start_vertex] = component_idx;
                cudaMemcpy(d_component_list, h_component_list, N_NODE * sizeof(int), cudaMemcpyHostToDevice);

                n_block = (N_NODE+BLOCK_SIZE-1)/BLOCK_SIZE;    
                while (h_still_running == 1){
                    h_still_running = 0;
                    cudaMemcpy(d_still_running, &h_still_running, sizeof(int), cudaMemcpyHostToDevice);

                    bfs_loop_for_cc_labeling<<<n_block, BLOCK_SIZE>>>(
                        d_v_adj_list, d_v_adj_begin, d_v_adj_length, 
                        d_dist_arr, d_still_running, start_vertex,
                        d_component_list, component_idx);
                    cudaDeviceSynchronize();

                    cudaMemcpy(&h_still_running, d_still_running, sizeof(int), cudaMemcpyDeviceToHost);
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

    // =========================================================================
    // Verification
    // =========================================================================
    printf("\nFound %d connected components\n", component_idx);

    int expected_components = N_NODE / NODES_PER_CLUSTER;
    printf("Expected %d components\n", expected_components);

    int errors = 0;

    // Check 1: Component count
    if (component_idx != expected_components) {
        printf("ERROR: Expected %d components, got %d\n", expected_components, component_idx);
        errors++;
    }

    // Check 2: Same ground truth cluster -> same computed component
    // Check 3: Different ground truth cluster -> different computed component
    for (i = 0; i < N_NODE && errors < 20; i++) {
        for (j = i + 1; j < N_NODE && errors < 20; j++) {
            int same_gt = (ground_truth_component[i] == ground_truth_component[j]);
            int same_computed = (h_component_list[i] == h_component_list[j]);

            if (same_gt && !same_computed) {
                printf("ERROR: Nodes %d and %d should be same component (gt=%d), got %d and %d\n",
                       i, j, ground_truth_component[i], h_component_list[i], h_component_list[j]);
                errors++;
            }
            if (!same_gt && same_computed) {
                printf("ERROR: Nodes %d and %d should be different components (gt=%d,%d), both got %d\n",
                       i, j, ground_truth_component[i], ground_truth_component[j], h_component_list[i]);
                errors++;
            }
        }
    }

    if (errors == 0) {
        printf("VERIFICATION PASSED\n");
    } else {
        printf("VERIFICATION FAILED: %d errors\n", errors);
    }

    // Print sample results
    printf("\nSample results:\n");
    for (i = 0; i < N_NODE; i += NODES_PER_CLUSTER) {
        printf("  Node %d: ground_truth=%d, computed=%d\n",
               i, ground_truth_component[i], h_component_list[i]);
    }

    // Cleanup
    free(v_adj_length);
    free(v_adj_begin);
    free(v_adj_list);
    free(edge_src);
    free(edge_dst);
    free(v_adj_begin_2);
    free(ground_truth_component);
    free(h_dist_arr);
    free(h_component_list);

    cudaFree(d_v_adj_list);
    cudaFree(d_v_adj_begin);
    cudaFree(d_v_adj_length);
    cudaFree(d_dist_arr);
    cudaFree(d_component_list);
    cudaFree(d_still_running);

    return 0;
}