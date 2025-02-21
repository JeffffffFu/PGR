from collections import defaultdict
import numpy as np
import os
import os.path as osp
from sklearn import metrics
import time
from tqdm import tqdm

import torch
import torch.nn.functional as F
from utils_linkteller import (
    construct_edge_sets,
    construct_edge_sets_from_random_subgraph,
    construct_edge_sets_from_svd_ind_subgraph,
    construct_edge_sets_from_trans_subgraph,
    construct_balanced_edge_sets,
    construct_edge_sets_from_svd_subgraph,
    _get_adj_rank,
)


class Attacker:
    """
    Re-purposed class without passing args from LinkTeller code
    """

    def __init__(
        self,
        dataset,
        model,
        features,
        test_labels,
        adjacency_matrix,
        orig_adj_csr,
        influence,
        attack_mode,
        mode,
        sample_type,
        seed,
        n_test,
        rng,
        test_dataset,
        rank,
    ):
        self.dataset = dataset

        if test_dataset is not None:
            self.dataset = test_dataset.value

        self.model = model
        self.sample_type = sample_type
        self.attack_mode = attack_mode
        self.influence = influence
        self.mode = mode
        self.features = features
        self.n_features = self.features.shape[1]
        self.test_labels = test_labels
        self.adj_matrix = adjacency_matrix

        self.n_nodes = self.features.shape[0]
        if sample_type == "balanced_full":
            self.n_test = len(self.test_labels)
        else:
            self.n_test = n_test

        self.rng = rng
        self.seed = seed
        self.test_csr = orig_adj_csr
        self.rank = rank
        # this initializes self.exist_edges, self.non_exist_edges, self.test_nodes
        self.prepare_test_data()
        self.avg_grad_time = 0
        print(torch.get_num_threads())

    def prepare_test_data(self):
        func = {
            "balanced": construct_edge_sets,
            "balanced_full": construct_balanced_edge_sets,
            "unbalanced": construct_edge_sets_from_random_subgraph,
            "unbalanced_lo": construct_edge_sets_from_random_subgraph,
            "unbalanced_hi": construct_edge_sets_from_random_subgraph,
            "unbalanced_lorank": construct_edge_sets_from_svd_ind_subgraph,
            "unbalanced_hirank": construct_edge_sets_from_svd_ind_subgraph,
            "balanced_lorank": construct_edge_sets_from_svd_subgraph,
            "balanced_hirank": construct_edge_sets_from_svd_subgraph,
            "balanced_lo": construct_edge_sets_from_trans_subgraph,
            "balanced_hi": construct_edge_sets_from_trans_subgraph,
        }.get(self.sample_type)
        if not func:
            raise NotImplementedError(
                f"sample_type = {self.sample_type} not implemented!"
            )

        if self.sample_type.endswith('rank'):
            (self.exist_edges, self.nonexist_edges), self.test_nodes = func(
                self.dataset, self.sample_type, self.test_csr, self.n_test, self.rng, self.rank
            )
        else:
            (self.exist_edges, self.nonexist_edges), self.test_nodes = func(
                self.dataset, self.sample_type, self.test_csr, self.n_test, self.rng
            )
        print(f"generating testing (non-)edge set done!")
        
    def get_gradient_eps_mat(self, v):
        pert_1 = torch.zeros_like(self.features)
        pert_1[v] = self.features[v] * self.influence
        grad = (
            self.model(self.features + pert_1, self.adj_matrix).detach()
            - self.model(self.features, self.adj_matrix).detach()
        ) / self.influence

        return grad

    def calculate_auc(self, v1, v0):
        v1 = sorted(v1)
        v0 = sorted(v0)
        vall = sorted(v1 + v0)

        TP = self.n_test
        FP = self.n_test
        T = F = self.n_test  # fixed

        p0 = p1 = 0

        TPR = TP / T
        FPR = FP / F

        result = [(FPR, TPR)]
        auc = 0
        for elem in vall:
            if p1 < self.n_test and abs(elem - v1[p1]) < 1e-6:
                p1 += 1
                TP -= 1
                TPR = TP / T
            else:
                p0 += 1
                FP -= 1
                FPR = FP / F
                auc += TPR * 1 / F

            result.append((FPR, TPR))

        return result, auc

    def link_prediction_attack_efficient(self):
        norm_exist = []
        norm_nonexist = []

        t = time.time()

        # 2. compute influence value for all pairs of nodes
        influence_val = np.zeros((self.n_test, self.n_test))

        with torch.no_grad():

            for i in tqdm(range(self.n_test)):
                u = self.test_nodes[i]
                grad_mat = self.get_gradient_eps_mat(u)

                for j in range(self.n_test):
                    v = self.test_nodes[j]

                    grad_vec = grad_mat[v]

                    influence_val[i][j] = grad_vec.norm().item()

            print(f"time for predicting edges: {time.time() - t}")

        node2ind = {node: i for i, node in enumerate(self.test_nodes)}

        for u, v in self.exist_edges:
            i = node2ind[u]
            j = node2ind[v]

            norm_exist.append(influence_val[j][i])

        for u, v in self.nonexist_edges:
            i = node2ind[u]
            j = node2ind[v]

            norm_nonexist.append(influence_val[j][i])

        return norm_exist, norm_nonexist

    def link_prediction_attack_efficient_balanced(self):
        norm_exist = []
        norm_nonexist = []
        
        # organize exist_edges and nonexist_edges into dict
        edges_dict = defaultdict(list)
        nonedges_dict = defaultdict(list)
        for u, v in self.exist_edges:
            edges_dict[u].append(v)
        for u, v in self.nonexist_edges:
            nonedges_dict[u].append(v)

        t = time.time()
        with torch.no_grad():
            for u in tqdm(range(self.n_nodes)):
                if u not in edges_dict and u not in nonedges_dict:
                    continue

                grad_mat = self.get_gradient_eps_mat(u)

                if u in edges_dict:
                    v_list = edges_dict[u]
                    for v in v_list:
                        grad_vec = grad_mat[v]
                        norm_exist.append(grad_vec.norm().item())

                if u in nonedges_dict:
                    v_list = nonedges_dict[u]
                    for v in v_list:
                        grad_vec = grad_mat[v]
                        norm_nonexist.append(grad_vec.norm().item())

            print(f"time for predicting edges: {time.time() - t}")
            # print('avg_grad_time {} s'.format(self.avg_grad_time / self.n_nodes))

        return norm_exist, norm_nonexist

    def baseline_attack(self):
        norm_exist = []
        norm_nonexist = []

        t = time.time()

        with torch.no_grad():
            posterior = F.softmax(self.model(self.features, self.adj_matrix), dim=1)
            # 1. compute the mean posterior of sampled nodes
            mean = torch.mean(posterior[self.test_nodes], dim=0)
            # 2. compute correlation value for all pairs of nodes
            dist = np.zeros((self.n_test, self.n_test))
            for i in tqdm(range(self.n_test)):
                u = self.test_nodes[i]
                for j in range(i + 1, self.n_test):
                    v = self.test_nodes[j]

                    dist[i][j] = (
                        torch.dot(posterior[u] - mean, posterior[v] - mean)
                        / torch.norm(posterior[u] - mean)
                        / torch.norm(posterior[v] - mean)
                    )

            print(f"time for computing correlation value: {time.time() - t}")

        node2ind = {node: i for i, node in enumerate(self.test_nodes)}

        for u, v in self.exist_edges:
            i = node2ind[u]
            j = node2ind[v]
            norm_exist.append(dist[i][j] if i < j else dist[j][i])

        for u, v in self.nonexist_edges:
            i = node2ind[u]
            j = node2ind[v]
            norm_nonexist.append(dist[i][j] if i < j else dist[j][i])

        return norm_exist, norm_nonexist

    def baseline_attack_balanced(self):
        norm_exist = []
        norm_nonexist = []

        # organize exist_edges and nonexist_edges into dict
        edges_dict = defaultdict(list)
        nonedges_dict = defaultdict(list)
        for u, v in self.exist_edges:
            edges_dict[u].append(v)
        for u, v in self.nonexist_edges:
            nonedges_dict[u].append(v)

        t = time.time()

        with torch.no_grad():
            # 0. compute posterior
            posterior = F.softmax(self.model(self.features, self.adj_matrix), dim=1)
            # 1. compute the mean posterior of sampled nodes
            mean = torch.mean(posterior, dim=0)
            # 2. compute correlation value for all pairs
            for u, v in tqdm(self.exist_edges):
                norm_exist.append(
                    (
                        torch.dot(posterior[u] - mean, posterior[v] - mean)
                        / torch.norm(posterior[u] - mean)
                        / torch.norm(posterior[v] - mean)
                    ).item()
                )

            for u, v in tqdm(self.nonexist_edges):
                norm_nonexist.append(
                    (
                        torch.dot(posterior[u] - mean, posterior[v] - mean)
                        / torch.norm(posterior[u] - mean)
                        / torch.norm(posterior[v] - mean)
                    ).item()
                )

            print(f"time for computing correlation value: {time.time() - t}")

        return norm_exist, norm_nonexist
    
    def baseline_attack_nfeat(self):
        norm_exist = []
        norm_nonexist = []

        t = time.time()

        with torch.no_grad():
            feature = self.features
            # 1. compute the mean feature of sampled nodes
            mean = torch.mean(feature[self.test_nodes], dim=0)
            # 2. compute correlation value for all pairs of nodes
            dist = np.zeros((self.n_test, self.n_test))
            for i in tqdm(range(self.n_test)):
                u = self.test_nodes[i]
                for j in range(i + 1, self.n_test):
                    v = self.test_nodes[j]

                    dist[i][j] = (
                        torch.dot(feature[u] - mean, feature[v] - mean)
                        / torch.norm(feature[u] - mean)
                        / torch.norm(feature[v] - mean)
                    )

            print(f"time for computing correlation value: {time.time() - t}")

        node2ind = {node: i for i, node in enumerate(self.test_nodes)}

        for u, v in self.exist_edges:
            i = node2ind[u]
            j = node2ind[v]
            norm_exist.append(dist[i][j] if i < j else dist[j][i])

        for u, v in self.nonexist_edges:
            i = node2ind[u]
            j = node2ind[v]
            norm_nonexist.append(dist[i][j] if i < j else dist[j][i])

        return norm_exist, norm_nonexist

    def baseline_attack_balanced_nfeat(self):
        norm_exist = []
        norm_nonexist = []

        # organize exist_edges and nonexist_edges into dict
        edges_dict = defaultdict(list)
        nonedges_dict = defaultdict(list)
        for u, v in self.exist_edges:
            edges_dict[u].append(v)
        for u, v in self.nonexist_edges:
            nonedges_dict[u].append(v)

        t = time.time()

        with torch.no_grad():
            # 0. get node features
            feature = self.features
            # 1. compute the mean feature values of sampled nodes
            mean = torch.mean(feature, dim=0)
            # 2. compute correlation value for all pairs
            for u, v in tqdm(self.exist_edges):
                norm_exist.append(
                    (
                        torch.dot(feature[u] - mean, feature[v] - mean)
                        / torch.norm(feature[u] - mean)
                        / torch.norm(feature[v] - mean)
                    ).item()
                )

            for u, v in tqdm(self.nonexist_edges):
                norm_nonexist.append(
                    (
                        torch.dot(feature[u] - mean, feature[v] - mean)
                        / torch.norm(feature[u] - mean)
                        / torch.norm(feature[v] - mean)
                    ).item()
                )

            print(f"time for computing correlation value: {time.time() - t}")

        return norm_exist, norm_nonexist

    def compute_and_save(self, norm_exist, norm_nonexist, prefix):
        y = [1] * len(norm_exist) + [0] * len(norm_nonexist)
        pred = norm_exist + norm_nonexist

        fpr, tpr, thresholds = metrics.roc_curve(y, pred)
        print("auc =", metrics.auc(fpr, tpr))

        precision, recall, thresholds_2 = metrics.precision_recall_curve(y, pred)
        np.save("test.y", y)
        np.save("test.pred", pred)
        print("ap =", metrics.average_precision_score(y, pred))
        folder_name = f"eval_{self.dataset}"
        if not osp.exists(folder_name):
            os.makedirs(folder_name)

        if self.sample_type == "balanced_full":
            n_test = self.n_nodes
        else:
            n_test = self.n_test
        filename = osp.join(
            folder_name, f"{prefix}-{self.attack_mode}-{self.sample_type}-{n_test}.pt"
        )

        torch.save(
            {
                "auc": {"fpr": fpr, "tpr": tpr, "thresholds": thresholds},
                "pr": {
                    "precision": precision,
                    "recall": recall,
                    "thresholds": thresholds_2,
                },
                "result": {"y": y, "pred": pred},
            },
            filename,
        )
        print(f"attack results saved to: {filename}")
