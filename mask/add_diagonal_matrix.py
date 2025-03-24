import numpy as np
import torch


def diagnoal_matrix(martix):
    diag_matrix = np.diag(np.full(martix[0].shape[0], 1000))
    A = np.add(martix[0], diag_matrix)

    return torch.tensor(A)

def matrix_add(matrix,valid_id,test_id):
    idx_test = range(200, 2708)
    a=torch.zeros_like(matrix[0], dtype=torch.float32)
    a[idx_test]+=1000
    b=np.add(matrix[0],a)
    return (torch.tensor(b),)

def add_diagonal_and_normalize_edge(edge,device):
    edge=edge.to(device)
    diagonal_ones = torch.eye(edge.size(0)).to(device)

    result_matrix = edge + diagonal_ones

    D1 = torch.sum(result_matrix, axis=1)
    D2 = torch.sum(result_matrix, axis=0)

    D1 = D1 ** (-1 / 2)
    D2 = D2 ** (-1 / 2)
    D_inv1 = torch.diag(D1)
    D_inv2 = torch.diag(D2)

    A_hat = torch.mm(torch.mm(D_inv1, result_matrix), D_inv2)

    return A_hat




def delete_diagonal(edge,device):
    edge=edge.to(device)
    diagonal_ones = torch.eye(edge.size(0)).to(device)

    result_matrix = edge - diagonal_ones

    return result_matrix

def degree_limit_index(D,degree):
    indices = torch.where(D >= degree)[0]
    concatenated_vector = torch.tensor([], dtype=torch.long, device=D.device)
    for i in indices:
        end_item=i+(len(D)-1)*len(D)+1
        A = torch.arange(i,end_item,len(D))
        A=A.to(D.device)
        concatenated_vector = torch.cat((concatenated_vector, A))

    return concatenated_vector





def self_connecting(edge,device):
    edge=edge.to(device)
    diagonal_ones = torch.eye(edge.size(0)).to(device)

    result_matrix = edge + diagonal_ones

    return result_matrix

