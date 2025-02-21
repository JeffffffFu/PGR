import random

import networkx as nx
import numpy as np
import torch
import torch_geometric
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import to_networkx
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
from collections import Counter
import pandas as pd



def extract_submatrix(adj_matrix, selected_nodes):
    num_selected = len(selected_nodes)
    submatrix = [[0] * num_selected for _ in range(num_selected)]

    for i in range(num_selected):
        for j in range(num_selected):
            node_i = selected_nodes[i]
            node_j = selected_nodes[j]
            submatrix[i][j] = adj_matrix[node_i][node_j]

    return submatrix

def influence_val_to_matrix(influence_val):
    influence_val_tensor = torch.tensor(influence_val)

    condition_matrix = torch.where(influence_val_tensor != 0, torch.tensor(1), torch.tensor(0))
    diagonal_matrix = torch.eye(influence_val_tensor.shape[0], dtype=torch.int)
    result_matrix = condition_matrix * (1 - diagonal_matrix)
    return result_matrix.T

def val_pre(A,B,n_test):
    pre_exit_edges = 0.
    pre_nonexit_edges = 0.
    print("A:",A)
    print("B:",B)
    for i in tqdm(range(n_test)):
        for j in range(n_test):
            if A[i][j]==1 and B[i][j]==1:
                pre_exit_edges+=1
            if A[i][j]==0 and B[i][j]==0:
                pre_nonexit_edges+=1

    nonzero_indices = torch.nonzero(A)
    nonzero_count = nonzero_indices.size(0)

    pre_acc_exit_edges=pre_exit_edges/nonzero_count
    pre_acc_nonexit_edges=pre_nonexit_edges/((n_test*n_test)-nonzero_count)
    print(pre_acc_exit_edges)
    print(pre_acc_nonexit_edges)


def val_average_precision_score(A,B,n_test):

    # Flatten A and B to one-dimensional lists
    A_flattened = A.view(-1).cpu().numpy().tolist()  # Convert A to a flattened list
    B_flattened = B.flatten().tolist()  # Flatten B and convert to a list
    precision, recall, thresholds_2 = metrics.precision_recall_curve(A_flattened, B_flattened)

    plt.plot(recall, precision, label="PR", color='orange')
    # 添加标签和标题
    plt.xlabel('recall', fontsize=16)
    plt.ylabel('precision', fontsize=16)
    # 显示图形
    plt.show()
    print('ap =', metrics.average_precision_score(A_flattened, B_flattened))


def adjust_list(input_list, a):
    non_zero_count = sum(1 for x in input_list if x != 0)

    if non_zero_count < 1:

        indices_to_set = random.sample(range(len(input_list)), a)

        new_list = input_list[:]
        for i in indices_to_set:
            new_list[i] = 0.1

        return new_list

    return input_list

def compute_and_save(norm_exist, norm_nonexist):

    y = [1] * len(norm_exist) + [0] * len(norm_nonexist)  #，
    pred = norm_exist + norm_nonexist



    auc=roc_auc_score(y, pred)
    print("auc:",auc)

    ap=metrics.average_precision_score(y, pred)
    print("ap:",ap)
    exit()
    precision, recall, thresholds_2 = metrics.precision_recall_curve(y, pred)

    precision, recall, thresholds_2 = metrics.precision_recall_curve(y, pred)
    idx = (np.abs(thresholds_2 - 0.5)).argmin()
    print("precision =", precision[idx])
    print("recall =", recall[idx])
    print("thresholds_2 =", thresholds_2[idx])

    print("precision =", len(precision))
    print('norm_exist=',len(norm_exist))
    # print("thresholds_2 =", len(thresholds_2))



    if len(precision)<len(norm_exist):

        if len(precision)==2:
            print("precision =", precision)
            print("precision =", recall)
            precision=0.
            recall=0.
        else:
            precision = precision[1]
            recall = recall[1]

    else:

        precision=precision[-len(norm_exist)-1]
        recall=recall[-len(norm_exist)-1]

    return ap,auc,precision,recall


def get_edge_sets_among_nodes(private_edge, nodes):

    graph = torch_geometric.data.Data(edge_index=private_edge)
    nx_graph = torch_geometric.utils.to_networkx(graph, to_undirected=True)
    adjacency_list = nx.to_dict_of_lists(nx_graph)

    n_nodes = len(nodes)
    edge_set = []
    nonedge_set = []

    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            u, v = nodes[i], nodes[j]
            if v in adjacency_list[u]:
                edge_set.append((u, v))
            else:
                nonedge_set.append((u, v))

    print('#nodes =', len(nodes))
    print('#edges_set =', len(edge_set))
    print('#nonedge_set =', len(nonedge_set))
    return edge_set, nonedge_set

def construct_edge_sets_from_random_subgraph(num,n_samples,seed):
    indice_all = [i for i in range(num)]
    random.seed(seed)
    nodes = random.sample(indice_all, n_samples)
    sorted_list = sorted(nodes)

    return sorted_list

def unblanced_node_pairs(sparse_matrix,labels, n_test):
    seed = 12345  #
    test_nodes = construct_edge_sets_from_random_subgraph(len(labels), n_test, seed)
    # sparse_adj=dense_adj_to_adj_sparse_adj(private_edge)
    exist_edges, nonexist_edges = get_edge_sets_among_nodes(sparse_matrix, test_nodes)

    return exist_edges, nonexist_edges,test_nodes


def balanced_node_pairs(sparse_matrix, labels, n_test):
    edges = sparse_matrix.t().tolist()
    edges = [(edges[i][0], edges[i][1]) for i in range(len(edges))]

    edges_pair = random.sample(edges, n_test)

    all_nodes = [x for x in range(len(labels))]

    edge_set = set(edges)
    no_edge_pairs = []

    while len(no_edge_pairs) < n_test:
        node_pair = tuple(random.sample(all_nodes, 2))
        if node_pair[0] != node_pair[1] and node_pair not in edge_set and node_pair[::-1] not in edge_set:
            no_edge_pairs.append(node_pair)

    return edges_pair, no_edge_pairs