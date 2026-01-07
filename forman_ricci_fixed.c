#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define N_NODE 500
#define NODES_PER_CLUSTER 50
#define N_CLUSTERS (N_NODE / NODES_PER_CLUSTER)
#define N_EDGES_MAX 30000
#define STEP_SIZE 1.0
#define N_ITERATION 30

// SBM probabilities
#define P_IN 0.4
#define P_OUT 0.001

// =========================================================================
// Utility functions
// =========================================================================
int get_cluster(int node_idx) {
    return node_idx / NODES_PER_CLUSTER;
}

double get_edge_probability(int node_1, int node_2) {
    if (get_cluster(node_1) == get_cluster(node_2)) return P_IN;
    return P_OUT;
}

void init_zero(int *arr, int size) {
    for (int i = 0; i < size; i++) arr[i] = 0;
}

void prefix_sum_exclusive(int *arr, int *result, int n) {
    result[0] = 0;
    for (int i = 1; i < n; i++) {
        result[i] = result[i-1] + arr[i-1];
    }
}

// =========================================================================
// Graph data structure
// =========================================================================
typedef struct {
    int n_nodes;
    int n_edges;
    int *v_adj_length;
    int *v_adj_begin;
    int *v_adj_list;
    int *edge_src;
    int *edge_dst;
    double *edge_weight;
    double *edge_curvature;
    int *node_cluster;  // ground truth
    int *edge_active;   // 1 if edge is active, 0 if removed
} Graph;

Graph* create_sbm_graph() {
    Graph *g = (Graph*)malloc(sizeof(Graph));
    g->n_nodes = N_NODE;
    
    g->v_adj_length = (int*)calloc(N_NODE, sizeof(int));
    g->v_adj_begin = (int*)malloc(N_NODE * sizeof(int));
    g->edge_src = (int*)malloc(N_EDGES_MAX * sizeof(int));
    g->edge_dst = (int*)malloc(N_EDGES_MAX * sizeof(int));
    g->node_cluster = (int*)malloc(N_NODE * sizeof(int));
    
    for (int i = 0; i < N_NODE; i++) {
        g->node_cluster[i] = get_cluster(i);
    }
    
    g->n_edges = 0;
    for (int i = 0; i < N_NODE; i++) {
        for (int j = i + 1; j < N_NODE; j++) {
            double p = get_edge_probability(i, j);
            if ((double)rand() / RAND_MAX <= p) {
                g->edge_src[g->n_edges] = i;
                g->edge_dst[g->n_edges] = j;
                g->v_adj_length[i]++;
                g->v_adj_length[j]++;
                g->n_edges++;
                if (g->n_edges >= N_EDGES_MAX) break;
            }
        }
        if (g->n_edges >= N_EDGES_MAX) break;
    }
    
    int n_total = g->n_edges * 2;
    g->v_adj_list = (int*)malloc(n_total * sizeof(int));
    g->edge_weight = (double*)malloc(n_total * sizeof(double));
    g->edge_curvature = (double*)malloc(n_total * sizeof(double));
    g->edge_active = (int*)malloc(g->n_edges * sizeof(int));
    
    for (int i = 0; i < n_total; i++) g->edge_weight[i] = 1.0;
    for (int i = 0; i < g->n_edges; i++) g->edge_active[i] = 1;
    
    prefix_sum_exclusive(g->v_adj_length, g->v_adj_begin, N_NODE);
    
    int *temp_begin = (int*)malloc(N_NODE * sizeof(int));
    memcpy(temp_begin, g->v_adj_begin, N_NODE * sizeof(int));
    
    for (int e = 0; e < g->n_edges; e++) {
        int s = g->edge_src[e], d = g->edge_dst[e];
        g->v_adj_list[temp_begin[s]++] = d;
        g->v_adj_list[temp_begin[d]++] = s;
    }
    free(temp_begin);
    
    return g;
}

void free_graph(Graph *g) {
    free(g->v_adj_length);
    free(g->v_adj_begin);
    free(g->v_adj_list);
    free(g->edge_src);
    free(g->edge_dst);
    free(g->edge_weight);
    free(g->edge_curvature);
    free(g->node_cluster);
    free(g->edge_active);
    free(g);
}

// =========================================================================
// Augmented Forman-Ricci curvature
// =========================================================================
void calc_augmented_frc(Graph *g) {
    for (int e = 0; e < g->n_edges; e++) {
        int v1 = g->edge_src[e];
        int v2 = g->edge_dst[e];
        
        // Find adjacency list indices
        int idx_v1_v2 = -1, idx_v2_v1 = -1;
        for (int j = 0; j < g->v_adj_length[v1]; j++) {
            if (g->v_adj_list[g->v_adj_begin[v1] + j] == v2) {
                idx_v1_v2 = g->v_adj_begin[v1] + j;
                break;
            }
        }
        for (int j = 0; j < g->v_adj_length[v2]; j++) {
            if (g->v_adj_list[g->v_adj_begin[v2] + j] == v1) {
                idx_v2_v1 = g->v_adj_begin[v2] + j;
                break;
            }
        }
        
        double w_e = g->edge_weight[idx_v1_v2];
        
        // Count triangles and compute contribution
        double triangle_contrib = 0.0;
        for (int j = 0; j < g->v_adj_length[v1]; j++) {
            int n = g->v_adj_list[g->v_adj_begin[v1] + j];
            if (n == v2) continue;
            for (int k = 0; k < g->v_adj_length[v2]; k++) {
                if (g->v_adj_list[g->v_adj_begin[v2] + k] == n) {
                    double w1 = g->edge_weight[g->v_adj_begin[v1] + j];
                    double w2 = g->edge_weight[g->v_adj_begin[v2] + k];
                    double s = (w_e + w1 + w2) / 2.0;
                    double area_sq = s * (s - w_e) * (s - w1) * (s - w2);
                    double w_tri = (area_sq > 0) ? sqrt(area_sq) : 0.0001;
                    triangle_contrib += w_e / w_tri;
                    break;
                }
            }
        }
        
        // Sum over non-triangle edges
        double sum1 = 0.0, sum2 = 0.0;
        for (int j = 0; j < g->v_adj_length[v1]; j++) {
            int n = g->v_adj_list[g->v_adj_begin[v1] + j];
            if (n == v2) continue;
            int is_tri = 0;
            for (int k = 0; k < g->v_adj_length[v2]; k++) {
                if (g->v_adj_list[g->v_adj_begin[v2] + k] == n) { is_tri = 1; break; }
            }
            if (!is_tri) {
                sum1 += 1.0 / sqrt(w_e * g->edge_weight[g->v_adj_begin[v1] + j]);
            }
        }
        for (int j = 0; j < g->v_adj_length[v2]; j++) {
            int n = g->v_adj_list[g->v_adj_begin[v2] + j];
            if (n == v1) continue;
            int is_tri = 0;
            for (int k = 0; k < g->v_adj_length[v1]; k++) {
                if (g->v_adj_list[g->v_adj_begin[v1] + k] == n) { is_tri = 1; break; }
            }
            if (!is_tri) {
                sum2 += 1.0 / sqrt(w_e * g->edge_weight[g->v_adj_begin[v2] + j]);
            }
        }
        
        double curv = w_e * (triangle_contrib + 2.0/w_e - (sum1 + sum2));
        g->edge_curvature[idx_v1_v2] = curv;
        g->edge_curvature[idx_v2_v1] = curv;
    }
}

void update_weights(Graph *g, double step_size) {
    for (int e = 0; e < g->n_edges; e++) {
        int v1 = g->edge_src[e];
        int v2 = g->edge_dst[e];
        
        int idx1 = -1, idx2 = -1;
        for (int j = 0; j < g->v_adj_length[v1]; j++) {
            if (g->v_adj_list[g->v_adj_begin[v1] + j] == v2) { idx1 = g->v_adj_begin[v1] + j; break; }
        }
        for (int j = 0; j < g->v_adj_length[v2]; j++) {
            if (g->v_adj_list[g->v_adj_begin[v2] + j] == v1) { idx2 = g->v_adj_begin[v2] + j; break; }
        }
        
        double w_new = (1.0 - step_size * g->edge_curvature[idx1]) * g->edge_weight[idx1];
        if (w_new < 1e-10) w_new = 1e-10;
        g->edge_weight[idx1] = w_new;
        g->edge_weight[idx2] = w_new;
    }
}

void normalize_weights(Graph *g) {
    double sum = 0.0;
    int n_total = g->n_edges * 2;
    for (int i = 0; i < n_total; i++) sum += g->edge_weight[i];
    double scale = (double)g->n_edges / (sum / 2.0);
    for (int i = 0; i < n_total; i++) g->edge_weight[i] *= scale;
}

double find_max_abs_curvature(Graph *g) {
    double max_abs = 0.0;
    int n_total = g->n_edges * 2;
    for (int i = 0; i < n_total; i++) {
        double v = fabs(g->edge_curvature[i]);
        if (v > max_abs) max_abs = v;
    }
    return max_abs;
}

// =========================================================================
// Connected components using only active edges below threshold
// =========================================================================
int find_components(Graph *g, double threshold, int *comp_id) {
    for (int i = 0; i < g->n_nodes; i++) comp_id[i] = -1;
    
    int n_comp = 0;
    int *queue = (int*)malloc(g->n_nodes * sizeof(int));
    
    for (int start = 0; start < g->n_nodes; start++) {
        if (comp_id[start] != -1) continue;
        
        int q_start = 0, q_end = 0;
        queue[q_end++] = start;
        comp_id[start] = n_comp;
        
        while (q_start < q_end) {
            int node = queue[q_start++];
            
            for (int j = 0; j < g->v_adj_length[node]; j++) {
                int idx = g->v_adj_begin[node] + j;
                int neighbor = g->v_adj_list[idx];
                double w = g->edge_weight[idx];
                
                // Keep edges BELOW threshold (cut high-weight bridges)
                if (w < threshold && comp_id[neighbor] == -1) {
                    comp_id[neighbor] = n_comp;
                    queue[q_end++] = neighbor;
                }
            }
        }
        n_comp++;
    }
    
    free(queue);
    return n_comp;
}

double calculate_modularity(Graph *g, int *comp_id) {
    double mod = 0.0;
    int m = g->n_edges;
    
    for (int u = 0; u < g->n_nodes; u++) {
        int k_u = g->v_adj_length[u];
        for (int j = 0; j < g->v_adj_length[u]; j++) {
            int v = g->v_adj_list[g->v_adj_begin[u] + j];
            int k_v = g->v_adj_length[v];
            if (comp_id[u] == comp_id[v]) {
                mod += 1.0 - (double)(k_u * k_v) / (2.0 * m);
            } else {
                mod -= (double)(k_u * k_v) / (2.0 * m);
            }
        }
    }
    return mod / (2.0 * m);
}

double calculate_nmi(int *detected, int *truth, int n, int n_det, int n_truth) {
    int *cnt_det = (int*)calloc(n_det, sizeof(int));
    int *cnt_tru = (int*)calloc(n_truth, sizeof(int));
    int **joint = (int**)malloc(n_det * sizeof(int*));
    for (int i = 0; i < n_det; i++) joint[i] = (int*)calloc(n_truth, sizeof(int));
    
    for (int i = 0; i < n; i++) {
        cnt_det[detected[i]]++;
        cnt_tru[truth[i]]++;
        joint[detected[i]][truth[i]]++;
    }
    
    double mi = 0.0;
    for (int i = 0; i < n_det; i++) {
        for (int j = 0; j < n_truth; j++) {
            if (joint[i][j] > 0) {
                double p_ij = (double)joint[i][j] / n;
                double p_i = (double)cnt_det[i] / n;
                double p_j = (double)cnt_tru[j] / n;
                mi += p_ij * log(p_ij / (p_i * p_j));
            }
        }
    }
    
    double h_det = 0.0, h_tru = 0.0;
    for (int i = 0; i < n_det; i++) {
        if (cnt_det[i] > 0) { double p = (double)cnt_det[i]/n; h_det -= p*log(p); }
    }
    for (int j = 0; j < n_truth; j++) {
        if (cnt_tru[j] > 0) { double p = (double)cnt_tru[j]/n; h_tru -= p*log(p); }
    }
    
    double nmi = (h_det + h_tru > 0) ? 2.0 * mi / (h_det + h_tru) : 0.0;
    
    free(cnt_det); free(cnt_tru);
    for (int i = 0; i < n_det; i++) free(joint[i]);
    free(joint);
    return nmi;
}

// =========================================================================
// Main
// =========================================================================
int main() {
    srand(42);
    
    Graph *g = create_sbm_graph();
    
    printf("=== Forman-Ricci Clustering ===\n");
    printf("Nodes: %d, Clusters: %d\n", N_NODE, N_CLUSTERS);
    printf("P_IN: %.2f, P_OUT: %.3f\n", P_IN, P_OUT);
    printf("Edges: %d\n", g->n_edges);
    
    int intra = 0, inter = 0;
    for (int e = 0; e < g->n_edges; e++) {
        if (g->node_cluster[g->edge_src[e]] == g->node_cluster[g->edge_dst[e]]) intra++;
        else inter++;
    }
    printf("Intra: %d (%.1f%%), Inter: %d (%.1f%%)\n", 
           intra, 100.0*intra/g->n_edges, inter, 100.0*inter/g->n_edges);
    
    // Ricci flow
    printf("\n=== Ricci Flow ===\n");
    for (int iter = 0; iter < N_ITERATION; iter++) {
        calc_augmented_frc(g);
        double max_abs = find_max_abs_curvature(g);
        double step = 1.0 / (1.1 * max_abs + 1e-10);
        if (step > STEP_SIZE) step = STEP_SIZE;
        
        update_weights(g, step);
        normalize_weights(g);
        
        if (iter % 10 == 0 || iter == N_ITERATION - 1) {
            printf("Iter %d: max_curv=%.2f, step=%.6f\n", iter, max_abs, step);
        }
    }
    
    // Analyze weights
    printf("\n=== Weight Analysis ===\n");
    double intra_sum = 0, inter_sum = 0;
    double intra_min = 1e30, intra_max = 0, inter_min = 1e30, inter_max = 0;
    
    for (int e = 0; e < g->n_edges; e++) {
        int s = g->edge_src[e], d = g->edge_dst[e];
        double w = 0;
        for (int j = 0; j < g->v_adj_length[s]; j++) {
            if (g->v_adj_list[g->v_adj_begin[s] + j] == d) {
                w = g->edge_weight[g->v_adj_begin[s] + j];
                break;
            }
        }
        
        if (g->node_cluster[s] == g->node_cluster[d]) {
            intra_sum += w;
            if (w < intra_min) intra_min = w;
            if (w > intra_max) intra_max = w;
        } else {
            inter_sum += w;
            if (w < inter_min) inter_min = w;
            if (w > inter_max) inter_max = w;
        }
    }
    
    printf("Intra: avg=%.4f, min=%.4f, max=%.4f\n", intra_sum/intra, intra_min, intra_max);
    printf("Inter: avg=%.4f, min=%.4f, max=%.4f\n", inter_sum/inter, inter_min, inter_max);
    printf("Ratio (inter/intra avg): %.2f\n", (inter_sum/inter)/(intra_sum/intra));
    
    // Try different thresholds
    printf("\n=== Threshold Search ===\n");
    printf("%-12s %-10s %-10s %-10s\n", "Threshold", "Clusters", "Modularity", "NMI");
    
    int *comp_id = (int*)malloc(g->n_nodes * sizeof(int));
    double best_nmi = 0;
    int best_clusters = 0;
    double best_threshold = 0;
    
    double min_w = intra_min < inter_min ? intra_min : inter_min;
    double max_w = intra_max > inter_max ? intra_max : inter_max;
    
    for (int i = 0; i <= 100; i++) {
        double thresh = min_w + (max_w - min_w) * i / 100.0;
        int n_comp = find_components(g, thresh, comp_id);
        
        if (n_comp >= 2 && n_comp <= N_NODE/2) {
            double mod = calculate_modularity(g, comp_id);
            double nmi = calculate_nmi(comp_id, g->node_cluster, g->n_nodes, n_comp, N_CLUSTERS);
            
            if (nmi > best_nmi) {
                best_nmi = nmi;
                best_clusters = n_comp;
                best_threshold = thresh;
            }
            
            if (n_comp <= 20 || n_comp == N_CLUSTERS) {
                printf("%.6f    %-10d %-10.4f %-10.4f", thresh, n_comp, mod, nmi);
                if (n_comp == N_CLUSTERS) printf(" <-- GROUND TRUTH COUNT");
                printf("\n");
            }
        }
    }
    
    printf("\n=== Best Result ===\n");
    printf("Best NMI: %.4f with %d clusters at threshold %.6f\n", best_nmi, best_clusters, best_threshold);
    printf("Ground truth: %d clusters\n", N_CLUSTERS);
    
    if (best_nmi > 0.8) {
        printf("\n*** SUCCESS: High NMI indicates good clustering! ***\n");
    }
    
    free(comp_id);
    free_graph(g);
    return 0;
}
