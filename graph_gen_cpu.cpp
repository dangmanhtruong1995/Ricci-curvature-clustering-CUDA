#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <set>
#include <vector>

#define N_NODES 1000
#define N_EDGES_TARGET 5000  // Undirected edges

int main() {
    srand(time(NULL));
    
    // Step 1: Generate random undirected edges (no duplicates, no self-loops)
    std::set<std::pair<int,int>> edge_set;
    
    while (edge_set.size() < N_EDGES_TARGET) {
        int a = rand() % N_NODES;
        int b = rand() % N_NODES;
        if (a == b) continue;  // No self-loops
        if (a > b) std::swap(a, b);  // Canonical form
        edge_set.insert({a, b});
    }
    
    // Step 2: Build adjacency lists per vertex
    std::vector<std::vector<int>> adj(N_NODES);
    
    for (auto &e : edge_set) {
        adj[e.first].push_back(e.second);
        adj[e.second].push_back(e.first);
    }
    
    // Step 3: Build the 3 arrays
    int n_directed_edges = 2 * edge_set.size();
    
    int *v_adj_list   = (int*)malloc(n_directed_edges * sizeof(int));
    int *v_adj_begin  = (int*)malloc(N_NODES * sizeof(int));
    int *v_adj_length = (int*)malloc(N_NODES * sizeof(int));
    
    int offset = 0;
    for (int v = 0; v < N_NODES; v++) {
        v_adj_begin[v] = offset;
        v_adj_length[v] = adj[v].size();
        
        for (int neighbor : adj[v]) {
            v_adj_list[offset++] = neighbor;
        }
    }
    
    // Print verification
    printf("Graph: %d nodes, %d undirected edges, %d directed edges\n",
           N_NODES, (int)edge_set.size(), n_directed_edges);
    
    printf("\nFirst 5 vertices:\n");
    for (int v = 0; v < 20; v++) {
        printf("  v=%d: begin=%d, len=%d, neighbors=[", 
               v, v_adj_begin[v], v_adj_length[v]);
        for (int i = 0; i < v_adj_length[v] && i < 5; i++) {
            printf("%d", v_adj_list[v_adj_begin[v] + i]);
            if (i < v_adj_length[v] - 1 && i < 4) printf(", ");
        }
        if (v_adj_length[v] > 5) printf(", ...");
        printf("]\n");
    }
    
    // Now copy to GPU and do Forman-Ricci there!
    
    free(v_adj_list);
    free(v_adj_begin);
    free(v_adj_length);
    
    return 0;
}
