import numpy as np
from tqdm import tqdm
import torch
import math
#RR扰动矩阵,无向图，也就是对称扰动
def perturb_adj_discrete(adj,eps):
    p=math.exp(eps)/ (1+math.exp(eps))  #

    N = adj.shape[0]

    perturbed_matrix = torch.zeros(N, N)

    # 对每个元素进行扰动
    for i in tqdm(range(N)):
        for j in range(N):
            if i < j:  #
                if np.random.rand() < p:
                    perturbed_matrix[i, j] = adj[i, j]
                    perturbed_matrix[j, i] = adj[j, i]

                else:
                    perturbed_matrix[i, j] = 1 - adj[i, j]
                    perturbed_matrix[j, i] = 1 - adj[j, i]

    return perturbed_matrix

def perturb_adj_discrete_direction(adj, eps):
    p = math.exp(eps) / (1 + math.exp(eps))

    N = adj.shape[0]

    perturbed_matrix = torch.zeros(N, N)

    # 对每个元素进行扰动
    for i in tqdm(range(N)):
        for j in range(N):
            if i != j:
                if np.random.rand() < p:
                    perturbed_matrix[i, j] = adj[i, j]
                else:
                    perturbed_matrix[i, j] = 1 - adj[i, j]

    return perturbed_matrix

def perturb_adj_laplace(adj,eps):

    eps_e=eps*0.01
    eps_l=eps-eps_e

    edge_num = torch.count_nonzero(adj).item()
    lap_noise = torch.distributions.Laplace(0, 1.0 / eps_e).sample((1,))

    top_k=int(edge_num+lap_noise)

    N = adj.shape[0]
    perturbed_matrix = torch.zeros(N, N)

    laplace_noise = torch.distributions.Laplace(0, 1 / eps_l).sample(adj.size())
    adj_perturbed = adj + laplace_noise
    adj_perturbed_flat = adj_perturbed.flatten()

    mask = ~torch.eye(N, dtype=bool).flatten()
    adj_perturbed_flat_no_diag = adj_perturbed_flat[mask]

    topk_values, topk_indices = torch.topk(adj_perturbed_flat_no_diag, k=top_k, largest=True)

    flat_indices_no_diag = torch.arange(N * N)[mask]
    actual_topk_indices = flat_indices_no_diag[topk_indices]
    row_indices = actual_topk_indices // N
    col_indices = actual_topk_indices % N

    perturbed_matrix[row_indices, col_indices] = 1.0

    return perturbed_matrix


def perturb_adj_laplace_groups(adj, eps):

    N = adj.shape[0]
    perturbed_matrix = torch.zeros(N, N)

    num_all = torch.count_nonzero(adj).item()

    num_groups= num_all

    K=num_all // num_groups

    non_diag_elements = N**2-N

    group_size, remainder = divmod(non_diag_elements, num_groups)

    #torch.manual_seed(1)
    group_sizes = [group_size + (1 if i < remainder else 0) for i in range(num_groups)]


    adj_flat = adj.flatten()
    mask = ~torch.eye(N, dtype=bool).flatten()
    adj_flat_no_diag = adj_flat[mask]

    indices = torch.randperm(adj_flat_no_diag.numel())
    #print("indices:",indices)

    #
    group_indices = []
    start_index = 0
    for size in group_sizes:
        end_index = start_index + size
        group_indices.append(indices[start_index:end_index])
        start_index = end_index


    full_indices = []
    for i in range(N):
        for j in range(N):
            if i != j:
                full_indices.append(i * N + j)

    for indices in group_indices:
        group = adj_flat_no_diag[indices]

        laplace_noise = torch.distributions.Laplace(0, 1 / eps).sample(group.size())
        perturbed_group = group + laplace_noise
       # print(f'indices:{indices},group:{adj_flat_no_diag[indices]},perturbed_group:{perturbed_group}')

        max_value, max_index  = torch.topk(perturbed_group, k=K, largest=True)

        # print(f'max_value:{max_value},max_index:{max_index}')

        max_flat_index = indices[max_index]
       # print(f'max_flat_index:{max_flat_index}')

        for max_index in max_flat_index:
            index=full_indices[max_index]
           # print(f'index:{index}')

            max_row_index = index // N
            max_col_index = index %  N

            # 将perturbed_matrix中对应索引的值置为1
            perturbed_matrix[max_row_index, max_col_index] = 1.0

    #print("diff:",torch.count_nonzero(perturbed_matrix-adj).item())
    return perturbed_matrix