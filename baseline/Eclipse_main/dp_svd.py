import time

from scipy.linalg import svd
import torch
import numpy as np

from mask.add_diagonal_matrix import add_diagonal_and_normalize_edge, self_connecting
from model.GCN import GCN
import torch.optim as optim

from utils.compare_adj import compare_adj3
from utils.train import train, test


def dp_svd(matrix,eps):
    rank=50
    edges=torch.count_nonzero(matrix)
    u, s, Vh = svd(matrix)
    eps_e = 0.01 * eps
    eps_l = eps - eps_e
    lap_noise = torch.distributions.Laplace(0, 1.0 / eps_e).sample((1,))

    edges_dp=int(edges+lap_noise)

    sensitivity = np.sqrt(2)
    delta = 1.0 / edges
    sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / eps_l

    # add noise to low rank, use low rank for A'

    gauss_noise = np.random.normal(0, sigma, rank)
    noisy_s = s[:rank] + gauss_noise
    # clamp noisy_s to be non-negative
    noisy_s = np.clip(noisy_s, 0, None)
    arr = u[:, :rank] @ np.diag(noisy_s) @ Vh[:rank, :]

    arr[np.tril_indices_from(arr, k=0)] = -999999999
    raveled_arr = arr.ravel()
    flat_indices = np.argpartition(raveled_arr, len(raveled_arr) - int(edges_dp))[-int(edges_dp):]

    row_indices, col_indices = np.unravel_index(flat_indices, arr.shape)
    arr.fill(0.0)
    arr[row_indices, col_indices] = 1.0
    arr += arr.T
    arr=torch.tensor(arr).to(torch.float32)


    return arr

#《Edge Private Graph Neural Networks with Singular Value Perturbation》
def train_with_Eclipse(eps,features, adj, labels, idx_train, idx_val, idx_test, model, network, lr,weight_decay, epochs, device):


    dense_matrix= dp_svd(adj,eps)


    if network=='GCN':
        A_hat = add_diagonal_and_normalize_edge(dense_matrix,device)
    else:
        A_hat = self_connecting(dense_matrix, device)


    edge_num = torch.count_nonzero(dense_matrix)
    print("edge:", edge_num)



    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)

    train( epochs, features, A_hat, labels, idx_train, idx_val, model, optimizer)
    start_time = time.time()

    acc_test = test(features, A_hat, labels, idx_test, model)
    end_time  = time.time()
    execution_time = end_time - start_time
    print(f'execution_time: {execution_time:.4f}')

    return acc_test.cpu(), edge_num.cpu(), edge_num.cpu(), model, dense_matrix


