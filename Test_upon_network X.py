import numpy as np
import networkx as nx
from scipy.io import mmread
import community as community_louvain

W = mmread("football.txt").toarray().astype(float)
W = 0.5 * (W + W.T)
np.fill_diagonal(W, 0)

n = W.shape[0]
G = nx.Graph()
G.add_nodes_from(range(n))
for i in range(n):
    for j in range(i+1, n):
        if W[i, j] > 0:
            G.add_edge(i, j, weight=W[i, j])

part = community_louvain.best_partition(G, weight="weight", random_state=0)
Q = community_louvain.modularity(part, G, weight="weight")
print("n=", n, "edges=", G.number_of_edges(), "Q=", Q, "k=", len(set(part.values())))
