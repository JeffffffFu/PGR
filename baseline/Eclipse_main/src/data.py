from re import A
import scipy.sparse as sparse
from scipy.linalg import svd
import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid, FacebookPagePage, WikipediaNetwork
from torch_geometric.utils import to_scipy_sparse_matrix
import numpy as np
from utils import Dataset
import utils_linkteller
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import pandas as pd
import os
import time
import gc
import pickle as pkl
from globals import MyGlobals
import csv


class LoadData:
    def __init__(
        self,
        dataset: Dataset,
        load_dir="planetoid",
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
        self.load_dir = os.path.join(MyGlobals.DATADIR, load_dir)
        if (
            not dataset.value in ["cora", "citeseer", "pubmed"]
            and load_dir == "planetoid"
        ):
            self.load_dir = os.path.join(MyGlobals.DATADIR, dataset.value)

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
        if self.dataset in (
            Dataset.TwitchES,
            Dataset.TwitchDE,
            Dataset.TwitchENGB,
            Dataset.TwitchFR,
            Dataset.TwitchPTBR,
        ):
            return True
        return False

    def val_on_new_graph(self):
        return False

    def has_no_val(self):
        if self.dataset in (
            Dataset.TwitchES,
            Dataset.TwitchDE,
            Dataset.TwitchENGB,
            Dataset.TwitchFR,
            Dataset.TwitchPTBR,
        ):
            return True
        return False

    def _get_masks_fb_page(self, dataset, te_tr_split=0.2, val_tr_split=0.2):
        nnodes = len(dataset[0].x)
        # get train mask, test mask and validation mask
        # get an 80-20 split for train-test
        # get an 80-20 split for train-val in train

        nodes = np.array(range(nnodes))
        train_mask = np.array([False] * nnodes)
        test_mask = np.array([False] * nnodes)
        val_mask = np.array([False] * nnodes)

        test_ind = self.rng.choice(nodes, int(te_tr_split * nnodes), replace=False)
        test_mask[np.array(test_ind)] = True
        rem_ind = []
        for ind in range(nnodes):
            if not ind in test_ind:
                rem_ind.append(ind)
        val_ind = self.rng.choice(
            rem_ind, int(val_tr_split * (len(rem_ind))), replace=False
        )
        val_mask[np.array(val_ind)] = True
        train_mask[~(test_mask | val_mask)] = True

        # Testing
        assert ~(train_mask & test_mask).all()
        assert ~(val_mask & test_mask).all()
        assert ~(train_mask & val_mask).all()

        return train_mask, val_mask, test_mask

    def _get_edge_index_from_csr(self, csr_m):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t = sparse.find(csr_m)
        return torch.Tensor(np.stack([t[0], t[1]])).to(device)

    def _load_data(self):
        print("Load dir {}; dataset name {}".format(self.load_dir, self.dataset.name))
        start_time = time.time()
        # This dataset is used for transfer learning so uses test_dataset for pred
        if (
            self.dataset == Dataset.TwitchES
            or self.dataset == Dataset.TwitchRU
            or self.dataset == Dataset.TwitchFR
            or self.dataset == Dataset.TwitchDE
            or self.dataset == Dataset.TwitchENGB
            or self.dataset == Dataset.TwitchPTBR
        ):
            if self.test_dataset is None:
                print("Expecting inductive training. Specify the test dataset")
            train_dataset = self.dataset.value
            test_dataset = self.test_dataset.value
            (
                self.train_features,
                self.train_labels,
            ) = utils_linkteller.twitch_feature_reader(dataset=train_dataset)
            (
                self.test_features,
                self.test_labels,
            ) = utils_linkteller.twitch_feature_reader(dataset=test_dataset)

            scaler = StandardScaler()
            scaler.fit(self.train_features)
            self.train_features = torch.FloatTensor(
                scaler.transform(self.train_features)
            )
            self.test_features = torch.FloatTensor(scaler.transform(self.test_features))

            self.num_classes = len(set(self.train_labels.numpy()))
            self.features = self.train_features
            self.labels = self.train_labels

            identifier = self.dataset.value[self.dataset.value.find("/") + 1 :]
            data = pd.read_csv(
                os.path.join(
                    MyGlobals.LK_DATA,
                    "{}/musae_{}_edges.csv".format(train_dataset, identifier),
                )
            )
            print(
                os.path.join(
                    MyGlobals.LK_DATA,
                    "{}/musae_{}_edges.csv".format(train_dataset, identifier),
                )
            )

            edge_index = torch.t(torch.from_numpy(data.values))
            (
                self.train_adj_csr,
                self.train_adj_orig_csr,
            ) = self.get_adjacency_matrix(edge_index, self.dp, self.eps, self.svd, self.rank, train_dataset)

            identifier = self.test_dataset.value[
                self.test_dataset.value.find("/") + 1 :
            ]
            data = pd.read_csv(
                os.path.join(
                    MyGlobals.LK_DATA,
                    "{}/musae_{}_edges.csv".format(test_dataset, identifier),
                )
            )
            print(
                os.path.join(
                    MyGlobals.LK_DATA,
                    "{}/musae_{}_edges.csv".format(test_dataset, identifier),
                )
            )

            test_rank = self.rank
            
            edge_index = torch.t(torch.from_numpy(data.values))
            # perturbed test graph
            (
                self.test_adj_csr,
                self.test_adj_orig_csr,
            ) = self.get_adjacency_matrix(edge_index, self.dp, self.eps, self.svd, test_rank, test_dataset)
            
            # for test accuracy with unperturbed test graph, 
            # (
            #     self.test_adj_csr,
            #     self.test_adj_orig_csr,
            # ) = self.get_adjacency_matrix(edge_index, False, self.eps, False, test_rank, test_dataset)
                        
            print(f"Data loading done: {time.time()-start_time}")
            return
        elif self.dataset == Dataset.Chameleon:
            dataset = WikipediaNetwork(self.load_dir, "chameleon")
            train_mask = dataset[0].train_mask[:, self.split_num_for_geomGCN_dataset]
            val_mask = dataset[0].val_mask[:, self.split_num_for_geomGCN_dataset]
            test_mask = dataset[0].test_mask[:, self.split_num_for_geomGCN_dataset]
            edges = []
            with open(os.path.join(self.load_dir, "edges_final.csv"), "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    items = line.rstrip("\n").split(",")
                    edges.append((int(items[0]), int(items[1])))
            edge_index_chameleon = torch.tensor(edges, dtype=torch.int64).T
        elif self.dataset == Dataset.facebook_page:
            dataset = FacebookPagePage(self.load_dir)
            # get number of nodes
            train_mask, val_mask, test_mask = self._get_masks_fb_page(dataset)
        elif self.dataset in [Dataset.Cora, Dataset.CiteSeer, Dataset.PubMed]:
            dataset = Planetoid(root=self.load_dir, name=self.dataset.name)
            train_mask = dataset[0].train_mask
            val_mask = dataset[0].val_mask
            test_mask = dataset[0].test_mask
        else:
            print(f"Dataset Loading undefined for {self.dataset.value}")
            exit()

        # Preprocessing for transductive datasets
        data = dataset[0]  # a single graph
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
        edge_index = (
            data.edge_index
            if not self.dataset == Dataset.Chameleon
            else edge_index_chameleon
        )
        
        # read & normalize adjacency matrix
        (
            self.train_adj_csr,
            self.train_adj_orig_csr,
        ) = self.get_adjacency_matrix(edge_index, self.dp, self.eps, self.svd, self.rank)
        
        self.test_adj_csr = self.train_adj_csr
        self.test_adj_orig_csr = self.train_adj_orig_csr
        self.val_adj_csr = self.train_adj_csr
        print(f"Data loading done: {time.time()-start_time}")

    def firstOrderGCNNorm(self, adj):
        degree = np.array(adj.sum(1))
        degree_for_norm = np.power(degree.sum(1), -0.5).flatten()
        degree_for_norm[np.isinf(degree_for_norm)] = 0.0
        degree_for_norm = sparse.diags(degree_for_norm)
        adj_hat_csr = degree_for_norm.dot(adj.dot(degree_for_norm))
        adj_hat_csr += sparse.eye(adj.shape[0])
        adj_hat_coo = adj_hat_csr.tocoo().astype(np.float32)
        return adj_hat_csr, adj_hat_coo

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
        if (
            self.dataset == Dataset.TwitchES
            or self.dataset == Dataset.TwitchRU
            or self.dataset == Dataset.TwitchFR
            or self.dataset == Dataset.TwitchDE
            or self.dataset == Dataset.TwitchENGB
            or self.dataset == Dataset.TwitchPTBR
        ):

            adj += adj.T
            nondp_adj_hat_csr = adj.copy()
            if svd:
                if dp: # dp for singular values
                    adj = self.gaussvdgraph(adj, eps, rank, dataset)  #here for svd dp
                    self.full_adj_csr_after_dp = adj
                else:
                    # svd with rank reconstruction
                    adj = self.svdgraph(adj, rank, dataset)
            else:
                if dp:
                    adj = self.lapgraph(adj, eps)
                    self.full_adj_csr_after_dp = adj
            _, adj_hat_coo = self.firstOrderGCNNorm(adj)
        else:
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

    def lapgraph(self, adj, eps):
        arr = adj.toarray()
        eps_e = 0.01 * eps
        eps_l = eps - eps_e
        edges = np.sum(arr) / 2.0
        edges_dp = int(edges + self.rng.laplace(0, 1.0 / eps_e))
        lap_noise = self.rng.laplace(0, 1.0 / eps_l, arr.shape)
        arr += lap_noise
        del lap_noise
        gc.collect()

        arr[np.tril_indices_from(arr, k=0)] = -999999999
        raveled_arr = arr.ravel()
        flat_indices = np.argpartition(raveled_arr, len(raveled_arr) - int(edges_dp))[
            -int(edges_dp) :
        ]
        del raveled_arr
        gc.collect()

        row_indices, col_indices = np.unravel_index(flat_indices, arr.shape)
        arr.fill(0.0)
        arr[row_indices, col_indices] = 1.0
        arr += arr.T
        return sparse.csr_matrix(arr)

    def gaussvdgraph(self, adj, eps, rank, dataset = None):
        arr = adj.todense()
        edges = np.sum(arr) / 2.0   # 除以2 也就是表示无向边
        if dataset == None:
            svd_dir = os.path.join(MyGlobals.SVDDIR, self.dataset.value)
        else:
            svd_dir = os.path.join(MyGlobals.SVDDIR, dataset)
        
        if not os.path.exists(svd_dir):
            u, s, Vh = svd(arr)
            
            # save u, s, Vh to path
            os.makedirs(svd_dir, exist_ok=True)
            np.save(os.path.join(svd_dir, 'u'),u)
            np.save(os.path.join(svd_dir, 's'),s)
            np.save(os.path.join(svd_dir, 'Vh'),Vh)
        else:
            # load u, s, Vh from path
            u = np.load(os.path.join(svd_dir, 'u.npy'))
            s = np.load(os.path.join(svd_dir, 's.npy'))
            Vh = np.load(os.path.join(svd_dir, 'Vh.npy'))
        
        eps_e = 0.01 * eps  #一部分隐私预算给边数量
        eps_l = eps - eps_e
        edges_dp = int(edges + self.rng.laplace(0, 1.0 / eps_e))
        
        sensitivity = np.sqrt(2)
        delta = 1.0 / 10 ** (len(str(edges)))
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / eps_l

        # add noise to low rank, use low rank for A'
        gauss_noise = self.rng.normal(loc=0,scale=sigma,size=rank)
        noisy_s = s[:rank] + gauss_noise
        # clamp noisy_s to be non-negative
        noisy_s = np.clip(noisy_s,0,None)
        arr = u[:,:rank] @ np.diag(noisy_s) @ Vh[:rank,:]
        
        del gauss_noise
        gc.collect()
        
        arr[np.tril_indices_from(arr, k=0)] = -999999999
        raveled_arr = arr.ravel()
        flat_indices = np.argpartition(raveled_arr, len(raveled_arr) - int(edges_dp))[-int(edges_dp):]
        del raveled_arr
        gc.collect()
        
        row_indices, col_indices = np.unravel_index(flat_indices, arr.shape)
        arr.fill(0.0)
        arr[row_indices, col_indices] = 1.0
        arr += arr.T
        return sparse.csr_matrix(arr)
    
    def svdgraph(self, adj, rank, dataset=None):
        arr = adj.todense()
        edges = np.sum(arr) / 2.0
        if dataset == None:
            svd_dir = os.path.join(MyGlobals.SVDDIR, self.dataset.value)
        else:
            svd_dir = os.path.join(MyGlobals.SVDDIR, dataset)
        if not os.path.exists(svd_dir):
            u, s, Vh = svd(arr)
            # save u, s, Vh to path
            os.makedirs(svd_dir, exist_ok=True)
            np.save(os.path.join(svd_dir, 'u'),u)
            np.save(os.path.join(svd_dir, 's'),s)
            np.save(os.path.join(svd_dir, 'Vh'),Vh)
        else:
            # load u, s, Vh from path
            u = np.load(os.path.join(svd_dir, 'u.npy'))
            s = np.load(os.path.join(svd_dir, 's.npy'))
            Vh = np.load(os.path.join(svd_dir, 'Vh.npy'))
            
        # if dataset == "twitch/RU":
        #     with open('svd_twitchRU.csv', 'w', encoding='UTF8', newline='') as file:
        #         writer = csv.writer(file)
        #         writer.writerow(s.T) 
        #     exit()
                
        arr = u[:,:rank] @ np.diag(s[:rank]) @ Vh[:rank,:]

        arr[np.tril_indices_from(arr, k=0)] = -999999999
        raveled_arr = arr.ravel()
        flat_indices = np.argpartition(raveled_arr, len(raveled_arr) - int(edges))[-int(edges):]
        row_indices, col_indices = np.unravel_index(flat_indices, arr.shape)
        arr.fill(0.0)
        arr[row_indices, col_indices] = 1.0
        arr += arr.T
        return sparse.csr_matrix(arr)


class convertToClass:
    def __init__(self, x, y, edge_index):
        self.x = x
        self.y = y
        self.edge_index = edge_index
