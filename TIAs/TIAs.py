import numpy as np
import torch

from TIAs.C_TIA.C_TIA import based_classifier, based_classifier_PGR, based_classifier_GAP
from TIAs.I_TIA.I_TIA import based_inf, based_inf_PGR, based_inf_GAP, based_random
from TIAs.M_TIA.M_TIA import based_metric, based_metric_PGR, based_metric_GAP

from TIAs.util import construct_multiple_subgraphs, \
    bfs_dense_subgraph_random, tensor_to_networkx
import networkx as nx
from torch_geometric.utils import to_networkx

def TIA(algorithm, data, model, dense_matrix, features, regen_adj, labels, device,hops, seed):
    TPL_M=0
    TPL_C=0
    TPL_I=0
    regen_adj = np.maximum(regen_adj.cpu().numpy(), regen_adj.cpu().numpy().T)
    regen_adj=torch.tensor(regen_adj,device=device)
    if algorithm == 'GAP_PGR':
        rows = dense_matrix.storage.row()
        cols = dense_matrix.storage.col()
        edges = list(zip(rows.tolist(), cols.tolist()))  # 将行列组合成边的列表

        G = nx.Graph()
        G.add_edges_from(edges)
        selected_subgraphs = bfs_dense_subgraph_random(G, 100, 100, 5,seed)
    else:
        G = to_networkx(data, to_undirected=True)
        selected_subgraphs = bfs_dense_subgraph_random(G, 100, 100, 5,seed)


    TPL_M=based_metric(algorithm, selected_subgraphs, model, dense_matrix, features, regen_adj, labels, device, seed)
    TPL_C=based_classifier(algorithm, selected_subgraphs, model, dense_matrix, features, regen_adj, labels, device, seed)
    TPL_I=based_inf(algorithm, selected_subgraphs, model, dense_matrix, features, regen_adj, labels, device, hops,seed)
    return TPL_M, TPL_C, TPL_I


def R_TIA(algorithm, data, model, dense_matrix, features, regen_adj, labels, device, hops, seed):
    TPL_M = 0
    TPL_C = 0
    TPL_I = 0
    TPL_R=0
    regen_adj = np.maximum(regen_adj.cpu().numpy(), regen_adj.cpu().numpy().T)
    regen_adj = torch.tensor(regen_adj, device=device)
    if algorithm == 'GAP_PGR':
        rows = dense_matrix.storage.row()
        cols = dense_matrix.storage.col()
        edges = list(zip(rows.tolist(), cols.tolist()))  #

        G = nx.Graph()
        G.add_edges_from(edges)
        selected_subgraphs = bfs_dense_subgraph_random(G, 100, 100, 5, seed)
    else:
        G = to_networkx(data, to_undirected=True)
        selected_subgraphs = bfs_dense_subgraph_random(G, 100, 100, 5, seed)

 #   TPL_I=based_inf(algorithm, selected_subgraphs, model, dense_matrix, features, regen_adj, labels, device, hops,seed)

    TPL_R=based_random(algorithm, selected_subgraphs, model, dense_matrix, features, regen_adj, labels, device, hops,
                      seed)
    return  TPL_R,TPL_I



def TIA_for_all_graph(algorithm, data, model, dense_matrix, features, regen_adj, labels, device,hops, seed):
    TPL_M=0
    TPL_C=0
    TPL_I=0
    regen_adj = np.maximum(regen_adj.cpu().numpy(), regen_adj.cpu().numpy().T)
    regen_adj=torch.tensor(regen_adj,device=device)
    if algorithm == 'GAP_PGR':
        rows = dense_matrix.storage.row()
        cols = dense_matrix.storage.col()
        edges = list(zip(rows.tolist(), cols.tolist()))  #

        G = nx.Graph()
        G.add_edges_from(edges)
        selected_subgraphs = bfs_dense_subgraph_random(G, 100, 100, 5,seed)

    else:
        # G = to_networkx(data, to_undirected=True)
        # selected_subgraphs = bfs_dense_subgraph_random(G, 100, 100, 5,seed)
        selected_subgraphs=[]
        num_rows = dense_matrix.shape[0]
        node_index = list(range(num_rows))
        selected_subgraphs.append((node_index, dense_matrix))


    TPL_M=based_metric(algorithm, selected_subgraphs, model, dense_matrix, features, regen_adj, labels, device, seed)
    TPL_C=based_classifier(algorithm, selected_subgraphs, model, dense_matrix, features, regen_adj, labels, device, seed)
    TPL_I=based_inf(algorithm, selected_subgraphs, model, dense_matrix, features, regen_adj, labels, device, hops,seed)
    return TPL_M, TPL_C, TPL_I

def TIA_PGR(algorithm, data, model, dense_matrix, features, regen_adj, labels, device,hops, seed):
    TPL_M=0
    TPL_C=0
    TPL_I=0
    regen_adj = np.maximum(regen_adj.cpu().numpy(), regen_adj.cpu().numpy().T)
    regen_adj=torch.tensor(regen_adj,device=device)

    G = to_networkx(data, to_undirected=True)
    selected_subgraphs = bfs_dense_subgraph_random(G, 100, 100, 5,seed)
    TPL_M = based_metric_PGR(algorithm, selected_subgraphs, model, dense_matrix, features,
                                                  regen_adj, labels, device, seed)
    TPL_C = based_classifier_PGR(algorithm, selected_subgraphs, model, dense_matrix, features,
                                                  regen_adj, labels, device, seed)
    TPL_I=based_inf_PGR(algorithm, selected_subgraphs, model, dense_matrix, features, regen_adj, labels, device,hops, seed)

    return TPL_M, TPL_C, TPL_I




def TIA_GAP(model, features, labels, adj_t, eps, hops, device, seed):
    TPL_M=0
    TPL_C=0
    TPL_I=0
    rows = adj_t.storage.row()
    cols = adj_t.storage.col()
    edges = list(zip(rows.tolist(), cols.tolist()))  #

    G = nx.Graph()
    G.add_edges_from(edges)
    selected_subgraphs = bfs_dense_subgraph_random(G, 100, 100, 5,seed)

    TPL_M=based_metric_GAP(selected_subgraphs,model, features, labels, adj_t, eps, hops, device, seed)
    TPL_C=based_classifier_GAP(selected_subgraphs,model, features, labels, adj_t, eps, hops, device, seed)
    TPL_I=based_inf_GAP(selected_subgraphs,model, features, labels, adj_t, eps, hops, device, seed)
    return TPL_M,TPL_C, TPL_I


def TIA_GAP_Random(model, features, labels, adj_t, eps, hops, device, seed):
    TPL_M=0
    TPL_C=0
    TPL_I=0
    rows = adj_t.storage.row()
    cols = adj_t.storage.col()
    edges = list(zip(rows.tolist(), cols.tolist()))  #

    G = nx.Graph()
    G.add_edges_from(edges)
    selected_subgraphs = bfs_dense_subgraph_random(G, 100, 100, 5,seed)

    TPL_M=based_metric_GAP(selected_subgraphs,model, features, labels, adj_t, eps, hops, device, seed)
    TPL_C=based_classifier_GAP(selected_subgraphs,model, features, labels, adj_t, eps, hops, device, seed)
    TPL_I=based_inf_GAP(selected_subgraphs,model, features, labels, adj_t, eps, hops, device, seed)
    return TPL_M,TPL_C, TPL_I

def TIA_GAP_full_graph(model, features, labels, adj_t, eps, hops, device, seed):
    TPL_M=0
    TPL_C=0
    TPL_I=0
    selected_subgraphs=[]
    dense_matrix = adj_t.to_dense()
    num_rows = dense_matrix.shape[0]
    node_index = list(range(num_rows))
    selected_subgraphs.append((node_index, dense_matrix))
    TPL_M=based_metric_GAP(selected_subgraphs,model, features, labels, adj_t, eps, hops, device, seed)
    TPL_C=based_classifier_GAP(selected_subgraphs,model, features, labels, adj_t, eps, hops, device, seed)
    TPL_I=based_inf_GAP(selected_subgraphs,model, features, labels, adj_t, eps, hops, device, seed)
    return TPL_M,TPL_C, TPL_I