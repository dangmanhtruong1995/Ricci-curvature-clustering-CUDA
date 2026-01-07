import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def generate_sbm(sizes, prob_matrix):
    n = sum(sizes)
    # Create community labels
    labels = np.repeat(range(len(sizes)), sizes)
    
    # Generate adjacency matrix
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p = prob_matrix[labels[i], labels[j]]
            if np.random.random() < p:
                adj[i, j] = adj[j, i] = 1
    return adj, labels

# Define parameters
sizes = [30, 30, 30]  # 3 communities of 30 nodes each
p_in, p_out = 0.4, 0.05
prob_matrix = np.array([
    [p_in, p_out, p_out],
    [p_out, p_in, p_out],
    [p_out, p_out, p_in]
])

# Generate the graph
adj, labels = generate_sbm(sizes, prob_matrix)

# Create NetworkX graph from adjacency matrix
G = nx.from_numpy_array(adj)

# Set up colors for communities
colors = ['#e41a1c', '#377eb8', '#4daf4a']
node_colors = [colors[label] for label in labels]

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Network visualization
ax1 = axes[0]
pos = nx.spring_layout(G, seed=42, k=0.5)
nx.draw(G, pos, ax=ax1, node_color=node_colors, node_size=80,
        edge_color='gray', alpha=0.9, width=0.5)
ax1.set_title('Stochastic Block Model\n(Spring Layout)', fontsize=12)

# Right plot: Adjacency matrix
ax2 = axes[1]
im = ax2.imshow(adj, cmap='Blues', aspect='equal')
ax2.set_title('Adjacency Matrix', fontsize=12)
ax2.set_xlabel('Node index')
ax2.set_ylabel('Node index')

# Add community boundary lines
cumsum = np.cumsum([0] + sizes)
for boundary in cumsum[1:-1]:
    ax2.axhline(y=boundary - 0.5, color='red', linewidth=1, linestyle='--')
    ax2.axvline(x=boundary - 0.5, color='red', linewidth=1, linestyle='--')

plt.tight_layout()
plt.savefig('./sbm_plot.png', dpi=150, bbox_inches='tight')
plt.close()

print("Plot saved to sbm_plot.png")
print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
