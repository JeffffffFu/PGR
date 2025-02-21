import community
import networkx as nx
import time
import numpy as np
import torch
import torch_geometric

from numpy.random import laplace
from sklearn import metrics
from torch_geometric.datasets import Planetoid
#from torch_geometric.graphgym import optim
import torch.optim as optim

from baseline.PrivGraph_main import comm
from baseline.PrivGraph_main.utils import *
from mask.add_diagonal_matrix import add_diagonal_and_normalize_edge
from model.GCN import GCN

import os

from utils.compare_adj import compare_adj3
from utils.train import train, test


# 我们做了一个priv_graph在这里，以适应我们的需求
def main_func(dense_matrix, epsilon):
    # set the ratio of the privacy budget
    e1_r = 1/3
    e2_r = 1/3

    # set the number of nodes for community initialization
    N = 20

    # set the resolution parameter
    t = 1.0
    t_begin = time.time()
    mat0 = dense_matrix.numpy().astype(np.uint8)  # 按照privGraph的格式，需要转成np格式，注意这里没有自连接的



    # original graph
    mat0_graph = nx.from_numpy_array(mat0, create_using=nx.Graph)
    mat0_node = mat0_graph.number_of_nodes()

    print(f'mat0_graph:{mat0_graph}')
    print('Node number:%d' % (mat0_graph.number_of_nodes()))
    print('Edge number:%d' % (mat0_graph.number_of_edges()))

    e1 = e1_r * epsilon

    e2 = e2_r * epsilon
    e3_r = 1 - e1_r - e2_r

    e3 = e3_r * epsilon

    ed = e3
    ev = e3

    ev_lambda = 1 / ed
    dd_lam = 2 / ev


    # Community Initialization
    mat1_pvarr1 = community_init(mat0, mat0_graph, epsilon=e1, nr=N, t=t)

    part1 = {}
    for i in range(len(mat1_pvarr1)):
        part1[i] = mat1_pvarr1[i]

    # Community Adjustment
    mat1_par1 = comm.best_partition(mat0_graph, part1, epsilon_EM=e2)
    mat1_pvarr = np.array(list(mat1_par1.values()))

    # Information Extraction
    mat1_pvs = []
    for i in range(max(mat1_pvarr) + 1):
        pv1 = np.where(mat1_pvarr == i)[0]
        pvs = list(pv1)
        mat1_pvs.append(pvs)

    comm_n = max(mat1_pvarr) + 1

    ev_mat = np.zeros([comm_n, comm_n], dtype=np.int64)

    # edge vector
    for i in range(comm_n):
        pi = mat1_pvs[i]
        ev_mat[i, i] = np.sum(mat0[np.ix_(pi, pi)])
        for j in range(i + 1, comm_n):
            pj = mat1_pvs[j]
            ev_mat[i, j] = int(np.sum(mat0[np.ix_(pi, pj)]))
            ev_mat[j, i] = ev_mat[i, j]

    ga = get_uptri_arr(ev_mat, ind=1)
    ga_noise = ga + laplace(0, ev_lambda, len(ga))

    ga_noise_pp = FO_pp(ga_noise)
    ev_mat = get_upmat(ga_noise_pp, comm_n, ind=1)

    # degree sequence
    dd_s = []
    for i in range(comm_n):
        dd1 = mat0[np.ix_(mat1_pvs[i], mat1_pvs[i])]
        dd1 = np.sum(dd1, 1)

        dd1 = (dd1 + laplace(0, dd_lam, len(dd1))).astype(int)
        dd1 = FO_pp(dd1)
        dd1[dd1 < 0] = 0
        dd1[dd1 >= len(dd1)] = len(dd1) - 1

        dd1 = list(dd1)
        dd_s.append(dd1)

    # Graph Reconstruction
    mat2 = np.zeros([mat0_node, mat0_node], dtype=np.int8)
    for i in range(comm_n):
        # Intra-community
        dd_ind = mat1_pvs[i]
        dd1 = dd_s[i]
        mat2[np.ix_(dd_ind, dd_ind)] = generate_intra_edge(dd1)

        # Inter-community
        for j in range(i + 1, comm_n):
            ev1 = ev_mat[i, j]
            pj = mat1_pvs[j]
            if ev1 > 0:
                c1 = np.random.choice(pi, ev1)
                c2 = np.random.choice(pj, ev1)
                for ind in range(ev1):
                    mat2[c1[ind], c2[ind]] = 1
                    mat2[c2[ind], c1[ind]] = 1

    mat2 = mat2 + np.transpose(mat2)
    mat2 = np.triu(mat2, 1)
    mat2 = mat2 + np.transpose(mat2)
    mat2[mat2 > 0] = 1
    return torch.from_numpy(mat2)




def train_with_privGraph(eps,features, adj, labels, idx_train, idx_val, idx_test, hidden, dropout, lr,weight_decay, epochs, device):


    dense_matrix=main_func(adj, eps)



    A_hat = add_diagonal_and_normalize_edge(dense_matrix, device)  # 进行自连接后正则化
    edge_num = torch.count_nonzero(dense_matrix)
    print("生成后的边数（不包括自连接）:", edge_num)

    model = GCN(nfeat=features.shape[1],
                nhid=hidden,
                nclass=labels.max().item() + 1,
                dropout=dropout).to(device)

    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)

    # 正常训练
    train( epochs, features, A_hat, labels, idx_train, idx_val, model, optimizer)
    start_time = time.time()  # 记录开始时间

    acc_test = test(features, A_hat, labels, idx_test, model)
    end_time  = time.time()  # 记录开始时间
    execution_time = end_time - start_time  # 计算运行时间
    print(f'execution_time: {execution_time:.4f}')
    return acc_test.cpu(), edge_num.cpu(), edge_num.cpu(), model, dense_matrix
