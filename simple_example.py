import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci

print("\n- Import an example NetworkX karate club graph")
G = nx.karate_club_graph()

print("\n===== Compute the Ollivier-Ricci curvature of the given graph G =====")
# compute the Ollivier-Ricci curvature of the given graph G
orc = OllivierRicci(G, alpha=0.5, verbose="INFO")
orc.compute_ricci_curvature()
print("Karate Club Graph: The Ollivier-Ricci curvature of edge (0,1) is %f" % orc.G[0][1]["ricciCurvature"])

print("\n===== Compute the Forman-Ricci curvature of the given graph G =====")
frc = FormanRicci(G)
frc.compute_ricci_curvature()
print("Karate Club Graph: The Forman-Ricci curvature of edge (0,1) is %f" % frc.G[0][1]["formanCurvature"])

# -----------------------------------
print("\n=====  Compute Ricci flow metric - Optimal Transportation Distance =====")
G = nx.karate_club_graph()
orc_OTD = OllivierRicci(G, alpha=0.5, method="OTD", verbose="INFO")
orc_OTD.compute_ricci_flow(iterations=10)
print("\n=====  Compute Ricci community - by Ricci flow =====")
clustering = orc_OTD.ricci_community()
