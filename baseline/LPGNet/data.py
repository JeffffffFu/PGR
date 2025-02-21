from re import A
import scipy.sparse as sparse
from scipy.linalg import svd
import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid, FacebookPagePage, WikipediaNetwork
from torch_geometric.utils import to_scipy_sparse_matrix
import numpy as np
# from utils import Dataset
# import utils_linkteller
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import pandas as pd
import os
import time
import gc
import pickle as pkl
import csv

def ListMask_to_BinaryMask(list_mask, size):
    binary_mask = [False] * size
    for index in list_mask:
        binary_mask[index] = True
    return torch.tensor(binary_mask)

class LoadData:
    def __init__(
        self,
        dataset,
        idx_train,
        idx_val,
        idx_test,
        dp=False,
        eps=2.0,
        svd=False,
        rank=20,
        n_val=500,
        n_test=1000,
        rng=None,
        rng_seed=None,
        test_dataset=None,
        split_num_for_geomGCN_dataset=0,
    ):
        self.dataset = dataset
        # self.load_dir = os.path.join(MyGlobals.DATADIR, load_dir)
        # if (
        #     not dataset.value in ["cora", "citeseer", "pubmed"]
        #     and load_dir == "planetoid"
        # ):
        #     self.load_dir = os.path.join(MyGlobals.DATADIR, dataset.value)
        self.idx_train=idx_train
        self.idx_val=idx_val
        self.idx_test=idx_test
        self.dp = dp
        self.eps = eps
        self.svd = svd
        self.rank = rank
        self.n_test = n_test
        self.n_val = n_val
        self.rng = rng
        self.rng_seed = rng_seed

        self.test_dataset = test_dataset
        self.features = None  # N \times F matrix
        self.labels = None  # N labels
        self.num_classes = None

        self.train_features = None
        self.train_labels = None
        self.train_adj_csr = None  # the noised and (or) normalized adjacency matrix in scipy.sparse format used for training
        self.train_adj_orig_csr = (
            None  # original adjacency matrix scipy.sparse format for training
        )

        self.val_features = None
        self.val_labels = None
        self.val_adj_csr = None
        self.val_adj_orig_csr = None

        self.test_features = None
        self.test_labels = None
        self.test_adj_csr = None
        self.test_adj_orig_csr = None

        self.full_adj_csr_after_dp = None
        self.split_num_for_geomGCN_dataset = split_num_for_geomGCN_dataset
        self._load_data()  # fills in the values for above fields.

    def is_inductive(self):

        return False

    def val_on_new_graph(self):
        return False

    def has_no_val(self):

        return False



    #train_mask的修改
    def _load_data(self):
        
        data = self.dataset
        # train_mask = data.train_mask
        # val_mask = data.val_mask
        # test_mask = data.test_mask
        train_mask=ListMask_to_BinaryMask(self.idx_train, len(data.y))
        val_mask=train_mask
        test_mask=ListMask_to_BinaryMask(self.idx_test, len(data.y))

        # read & normalize features
        features = data.x.clone()
        features_sum = features.sum(1).unsqueeze(1)
        features_sum[features_sum == 0] = 1.0
        features = torch.div(features, features_sum)
        self.features = features

        # read train, test, valid labels based on public splits of this data
        # = -100, used to ignore not allowed labels in CE loss
        ignore_index = nn.CrossEntropyLoss().ignore_index
        self.num_classes = len(set(data.y.numpy()))
        self.labels = data.y.clone()
        self.train_features = self.features
        self.train_labels = self.set_labels(data.y.clone(), train_mask, ignore_index)

        self.val_features = self.features
        self.val_labels = self.set_labels(data.y.clone(), val_mask, ignore_index)

        self.test_features = self.features
        self.test_labels = self.set_labels(data.y.clone(), test_mask, ignore_index)
        print(
            "{} {} {}".format(
                len(np.where(self.train_labels > -1)[0]),
                len(np.where(self.val_labels > -1)[0]),
                len(np.where(self.test_labels > -1)[0]),
            )
        )
        print("len(data.x) {}".format(len(data.x)))
        edge_index = data.edge_index
        
        # read & normalize adjacency matrix
        (
            self.train_adj_csr,
            self.train_adj_orig_csr,
        ) = self.get_adjacency_matrix(edge_index, self.dp, self.eps, self.svd, self.rank)
        
        self.test_adj_csr = self.train_adj_csr
        self.test_adj_orig_csr = self.train_adj_orig_csr
        self.val_adj_csr = self.train_adj_csr
        # print(f"Data loading done: {time.time()-start_time}")


    def augNormGCN(self, adj):
        adj += sparse.eye(adj.shape[0])  # add self loops
        # # print(adj)
        degree_for_norm = sparse.diags(
            np.power(np.array(adj.sum(1)), -0.5).flatten()
        )  # D^(-0.5)
        adj_hat_csr = degree_for_norm.dot(
            adj.dot(degree_for_norm)
        )  # D^(-0.5) * A * D^(-0.5)
        adj_hat_coo = adj_hat_csr.tocoo().astype(np.float32)
        return adj_hat_csr, adj_hat_coo
    
    def get_adjacency_matrix(self, edge_index, dp, eps, svd=False, rank=0, dataset=None):
        print(f"get adj matrix with dp:{dp}, eps:{eps}, svd:{svd}, rank:{rank}, dataset:{dataset}")
        adj = to_scipy_sparse_matrix(edge_index)
        # if True:
        nondp_adj_hat_csr = adj.copy()
        nondp_adj_hat_csr = nondp_adj_hat_csr.tocsr()
        assert (adj.toarray() == adj.T.toarray()).all()
        if svd:
            if dp: # dp for singular values
                adj = self.gaussvdgraph(adj, eps, rank)
                self.full_adj_csr_after_dp = adj
            else:
                # svd with rank reconstruction
                adj = self.svdgraph(adj, rank)
        else: # no svd on adj matrix
            if dp:
                adj = self.lapgraph(adj, eps)
                self.full_adj_csr_after_dp = adj
            else:
                print("没有对邻接矩阵加噪")
        _, adj_hat_coo = self.augNormGCN(adj)
        # to torch sparse matrix/pdb
        indices = torch.from_numpy(
            np.vstack((adj_hat_coo.row, adj_hat_coo.col)).astype(np.int64)
        )
        values = torch.from_numpy(adj_hat_coo.data)
        adjacency_matrix = torch.sparse_coo_tensor(
            indices, values, torch.Size(adj_hat_coo.shape)
        )

        return (
            adjacency_matrix,
            nondp_adj_hat_csr,
        )

    def set_labels(self, initial_labels, set_mask, ignore_label):
        initial_labels[~set_mask] = ignore_label
        return initial_labels

