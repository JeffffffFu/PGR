import time

import torch
from torch import optim

from mask.add_diagonal_matrix import add_diagonal_and_normalize_edge, self_connecting
from model.GCN import GCN
from utils.perturb_adj import perturb_adj_laplace, perturb_adj_discrete
from utils.train import train, test
def graph_normal_training_perturb_RR(eps,features, adj, labels, idx_train, idx_val, idx_test, model, network, lr,
                          weight_decay, epochs,device):

    dense_matrix=perturb_adj_discrete(adj, eps)


    if network=='GCN':
        A_hat = add_diagonal_and_normalize_edge(dense_matrix,device)
    else:
        A_hat = self_connecting(dense_matrix, device)

    edge_num = torch.count_nonzero(dense_matrix)
    print("edges:", edge_num)



    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)

    train(epochs, features, A_hat, labels, idx_train, idx_val, model, optimizer)
    start_time = time.time()

    acc_test = test(features, A_hat, labels, idx_test, model)
    end_time  = time.time()
    execution_time = end_time - start_time
    print(f'execution_time: {execution_time:.4f}')
    return acc_test.cpu(), edge_num.cpu(), edge_num.cpu(), model, dense_matrix

