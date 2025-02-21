import time

import torch
from torch import optim

from mask.add_diagonal_matrix import add_diagonal_and_normalize_edge
from model.GCN import GCN
from utils.perturb_adj import perturb_adj_laplace, perturb_adj_discrete
from utils.train import train, test
def graph_normal_training_perturb_RR(eps,features, adj, labels, idx_train, idx_val, idx_test, hidden, dropout, lr,
                          weight_decay, epochs,device):

    dense_matrix=perturb_adj_discrete(adj, eps)


    A_hat = add_diagonal_and_normalize_edge(dense_matrix, device)  # 进行自连接后正则化
    edge_num = torch.count_nonzero(dense_matrix)
    print("原边数（不包括自连接）:", edge_num)

    model = GCN(nfeat=features.shape[1],
                nhid=hidden,
                nclass=labels.max().item() + 1,
                dropout=dropout).to(device)

    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)

    # 正常训练
    train(epochs, features, A_hat, labels, idx_train, idx_val, model, optimizer)
    start_time = time.time()  # 记录开始时间

    acc_test = test(features, A_hat, labels, idx_test, model)
    end_time  = time.time()  # 记录开始时间
    execution_time = end_time - start_time  # 计算运行时间
    print(f'execution_time: {execution_time:.4f}')
    return acc_test.cpu(), edge_num.cpu(), edge_num.cpu(), model, dense_matrix

