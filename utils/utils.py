import random

import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric
import math


from utils.matrix_operation import sparse_mx_to_torch_sparse_tensor


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data_local(path="data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    #features = normalize(features)

    #exit()

    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(270)
    idx_val = range(271, 271)
    idx_test = range(272, 2708)

    random.seed(3407)  # 设置随机种子为42
    numbers = range(2708)
    idx_train = random.sample(numbers, 270)
    idx_test = [num for num in numbers if num not in idx_train]


    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()

    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)

    mx = r_mat_inv.dot(mx)

    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)



def replace_elements(A, B, indices):
    C = A.clone()
    C[indices] = B[indices]
    return C

def split_dataset(label,ratio_of_train,seed):

    # 设置随机种子
    random.seed(seed)
    torch.manual_seed(seed)

    numbers=len(label)
    num_of_train=math.ceil(numbers*ratio_of_train)
    idx_all = range(numbers)

    idx_train = random.sample(idx_all, num_of_train)
    idx_val = range(numbers, numbers)
    idx_test = [num for num in idx_all if num not in idx_train]

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return idx_train,idx_val,idx_test

def BinaryMask_to_ListMask(mask):
    l=[index for index, value in enumerate(mask) if value]
    return torch.tensor(l)

def generate_list_C(A, B):
    C = []
    index_A = 0
    for flag in B:
        if flag:
            C.append(A[index_A])
            index_A += 1
        else:
            C.append(A[0])
    return torch.stack(C)


import torch


def sample_neighbors(adj: torch.Tensor) -> torch.Tensor:
    adj_detached = adj.detach()

    N = adj_detached.size(0)
    k = 150 if N in (2708, 3327) else 448

    rand_values = torch.rand(N, adj_detached.size(1), device=adj_detached.device)
    rand_values[adj_detached <= 0] = float('inf')

    _, selected = torch.topk(-rand_values, k, dim=1)

    mask = torch.zeros_like(adj_detached)
    mask.scatter_(1, selected, 1)


    sampled_adj = adj * mask.to(adj.dtype)

    return sampled_adj

