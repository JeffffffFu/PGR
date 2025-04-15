from __future__ import division
from __future__ import print_function

import copy
import math
import os
import time
import argparse
from math import ceil

import numpy as np
import pandas as pd
import torch_geometric
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj
from mask.add_diagonal_matrix import diagnoal_matrix, matrix_add, \
    add_diagonal_and_normalize_edge, degree_limit_index, self_connecting, delete_diagonal
from mask.compute_scores import keep_edges_add_many_edge_from_zero_priD2
from mask.mata_grads import compute_matrix_grads

import random

from model.GCN import GCN, GCN_one_hop, GCN_three_hop
from utils.matrix_operation import normalize_edge
from utils.train import train, test
from utils.utils import replace_elements, sample_neighbors


def graph_regenerate_different(algorithm,features,dense_matrix,dense_matrix_DP,labels,idx_train,idx_val,idx_test,lr,weight_decay, epochs,epochs_inner,prune,device,mu,model,network):


    edge_num=torch.count_nonzero(dense_matrix)

    if network=='GCN':
        A_hat = add_diagonal_and_normalize_edge(dense_matrix,device)
    else:
        A_hat = self_connecting(dense_matrix, device)
    dense_matrix_DP=self_connecting(dense_matrix_DP,device)
    matrix_vec = torch.cat([score.flatten() for score in dense_matrix_DP])
    origin_matrix_vec_non_zero_indices = torch.nonzero(matrix_vec)


    model1=copy.deepcopy(model)

    optimizer = optim.Adam(model1.parameters(),
                           lr=lr, weight_decay=weight_decay)

    train( epochs, features, A_hat, labels, idx_train, idx_val, model1, optimizer)
    test(features, A_hat, labels, idx_test, model1)

    output = model1(features, A_hat)
    new_label_all = replace_elements(labels, output.max(1)[1], idx_test)

    model2=copy.deepcopy(model)

    optimizer2 = optim.Adam(model2.parameters(),
                           lr=lr, weight_decay=weight_decay)

    num_to_keep = math.ceil(prune*edge_num)

    keep_adj=torch.eye(len(labels)).to(device)
    if network=='GCN':
        A_hat,_ = normalize_edge(keep_adj,device)
    else:
        A_hat = keep_adj

    train( epochs, features, keep_adj, labels, idx_train, idx_val, model2, optimizer2)
    degree_limit_indexs = torch.tensor([], dtype=torch.long).to(device)
    test(features, A_hat, labels, idx_test, model2)
    torch.cuda.empty_cache()

    print("num_to_keep:",num_to_keep)

    for i in range(num_to_keep):
        print(f'Already generated {i} edges, need to generate {num_to_keep-i} edges')


        train(epochs_inner, features, A_hat, labels, idx_train, idx_val, model2, optimizer2)

        matrix_grads = compute_matrix_grads(A_hat, features, new_label_all, model2, idx_train, idx_test,device)

        edge_num_gen = 1
       # print("matrix_grads:",matrix_grads)
        keep_adj= keep_edges_add_many_edge_from_zero_priD2(matrix_grads, keep_adj, device, edge_num_gen,degree_limit_indexs,origin_matrix_vec_non_zero_indices,matrix_vec,mu)

        if network == 'GCN':
            A_hat, _ = normalize_edge(keep_adj, device)
        else:
            A_hat = keep_adj

        test(features, A_hat, labels, idx_test, model2)

    print("PGR Done")

    acc_test = test(features, A_hat, labels, idx_test, model2)

    keep_adj=delete_diagonal(keep_adj,device)

    return acc_test.cpu(),edge_num.cpu(),num_to_keep,model2,keep_adj
