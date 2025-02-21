from TIAs.C_TIA.C_TIA import LIA_based_classifier, LIA_based_classifier_GAP
from TIAs.I_TIA.I_TIA import LIA_based_inf, \
    LIA_based_inf_GAP

from TIAs.util import construct_multiple_subgraphs, bfs_dense_subgraph_random
import networkx as nx
from torch_geometric.utils import to_networkx
import torch
import numpy as np

def LIAs(algorithm, data, model, dense_matrix, features, regen_adj, labels, device, seed):
    LIA_sim=0
    LIA_inf=0
    regen_adj = np.maximum(regen_adj.cpu().numpy(), regen_adj.cpu().numpy().T)
    regen_adj=torch.tensor(regen_adj,device=device)
    if algorithm == 'GAP_PGR':
        rows = dense_matrix.storage.row()
        cols = dense_matrix.storage.col()
        edges = list(zip(rows.tolist(), cols.tolist()))

        G = nx.Graph()
        G.add_edges_from(edges)
        selected_subgraphs = bfs_dense_subgraph_random(G, 100, 100, 5,seed)
    else:
        G = to_networkx(data, to_undirected=True)
        selected_subgraphs = bfs_dense_subgraph_random(G, 100, 100, 5,seed)

    LIA_sim=LIA_based_classifier(algorithm, selected_subgraphs, model, dense_matrix, features, regen_adj, labels, device, seed)
    LIA_inf=LIA_based_inf(algorithm, selected_subgraphs, model, dense_matrix, features, regen_adj, labels, device, seed)

    return LIA_sim, LIA_inf

def LIAs_GAP(model, features, labels, adj_t, eps, hops, device, seed):
    LIA_sim=0
    LIA_inf=0
    rows = adj_t.storage.row()
    cols = adj_t.storage.col()
    edges = list(zip(rows.tolist(), cols.tolist()))

    G = nx.Graph()
    G.add_edges_from(edges)
    selected_subgraphs = bfs_dense_subgraph_random(G, 100, 100, 5, seed)

    LIA_sim=LIA_based_classifier_GAP(selected_subgraphs,model, features, labels, adj_t, eps, hops, device, seed)
    LIA_inf=LIA_based_inf_GAP(selected_subgraphs,model, features, labels, adj_t, eps, hops, device, seed)
    return LIA_sim, LIA_inf
