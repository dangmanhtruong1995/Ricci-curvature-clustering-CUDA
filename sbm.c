#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_NODE 10000
#define NODES_PER_CLUSTER 100
#define N_CLUSTERS (N_NODE / NODES_PER_CLUSTER)
#define N_EDGES_MAX 500000  // Maximum edges to allocate

// SBM probabilities
#define P_IN 0.3    // Probability of edge within same cluster
#define P_OUT 0.01  // Probability of edge between different clusters

void init_zero(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = 0;
    }
}

void prefix_sum_exclusive(int *input, int *output, int n) {
    output[0] = 0;
    for (int i = 1; i < n; i++) {
        output[i] = output[i - 1] + input[i - 1];
    }
}

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

void print_graph_stats(int *v_adj_length, int *node_cluster, int n_undirected_edges) {
    int within_cluster_edges = 0;
    int between_cluster_edges = 0;
    
    printf("=== Stochastic Block Model Graph Statistics ===\n");
    printf("Number of nodes: %d\n", N_NODE);
    printf("Number of clusters: %d\n", N_CLUSTERS);
    printf("Nodes per cluster: %d\n", NODES_PER_CLUSTER);
    printf("P_in (within cluster): %.3f\n", P_IN);
    printf("P_out (between clusters): %.3f\n", P_OUT);
    printf("Total undirected edges: %d\n", n_undirected_edges);
    
    // Calculate average degree
    int total_degree = 0;
    for (int i = 0; i < N_NODE; i++) {
        total_degree += v_adj_length[i];
    }
    printf("Average degree: %.2f\n", (float)total_degree / N_NODE);
}

void free_graph(int *v_adj_length, int *v_adj_begin, int *v_adj_list,
                int *edge_src, int *edge_dst, int *v_adj_begin_2, int *node_cluster) {
    free(v_adj_length);
    free(v_adj_begin);
    free(v_adj_list);
    free(edge_src);
    free(edge_dst);
    free(v_adj_begin_2);
    free(node_cluster);
}

int main() {
    int *v_adj_length, *v_adj_begin, *v_adj_list;
    int *edge_src, *edge_dst, *v_adj_begin_2;
    int *node_cluster;
    int n_undirected_edges;

    srand(time(NULL));

    printf("Generating SBM graph...\n");
    
    create_sbm_graph(
        &v_adj_length,
        &v_adj_begin,
        &v_adj_list,
        &edge_src,
        &edge_dst,
        &v_adj_begin_2,
        &n_undirected_edges,
        &node_cluster
    );

    print_graph_stats(v_adj_length, node_cluster, n_undirected_edges);

    // Example: print adjacency list for first few nodes
    printf("\n=== Sample Adjacency Lists (first 5 nodes) ===\n");
    for (int i = 0; i < 5 && i < N_NODE; i++) {
        printf("Node %d (cluster %d): ", i, node_cluster[i]);
        int start = v_adj_begin[i];
        int len = v_adj_length[i];
        for (int j = 0; j < len && j < 10; j++) {  // Print at most 10 neighbors
            printf("%d ", v_adj_list[start + j]);
        }
        if (len > 10) printf("...");
        printf("(degree: %d)\n", len);
    }

    free_graph(v_adj_length, v_adj_begin, v_adj_list,
               edge_src, edge_dst, v_adj_begin_2, node_cluster);

    return 0;
}
