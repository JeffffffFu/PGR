from collections import defaultdict
import itertools
import json
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from tqdm import tqdm
from globals import MyGlobals
from scipy.linalg import svd
import scipy.sparse as sparse
import matplotlib.pyplot as plt


def twitch_feature_reader(
    dataset="twitch/", scale="large", train_ratio=0.5, feature_size=-1
):
    identifier = dataset[dataset.find("/") + 1 :]
    filename = os.path.join(
        MyGlobals.LK_DATA, "{}/musae_{}_features.json".format(dataset, identifier)
    )
    with open(filename) as f:
        data = json.load(f)
        n_nodes = len(data)

        items = sorted(set(itertools.chain.from_iterable(data.values())))
        n_features = 3170 if dataset.startswith("twitch") else max(items) + 1

        features = np.zeros((n_nodes, n_features))
        for idx, elem in data.items():
            features[int(idx), elem] = 1

    data = pd.read_csv(
        os.path.join(
            MyGlobals.LK_DATA, "{}/musae_{}_target.csv".format(dataset, identifier)
        )
    )
    mature = list(map(int, data["mature"].values))
    new_id = list(map(int, data["new_id"].values))
    idx_map = {elem: i for i, elem in enumerate(new_id)}
    labels = [mature[idx_map[idx]] for idx in range(n_nodes)]

    labels = torch.LongTensor(labels)
    return features, labels


def construct_balanced_edge_sets(dataset, sample_type, adj, n_samples, rng):
    indices = adj.indices
    indptr = adj.indptr
    n_nodes = adj.shape[0]

    dic = defaultdict(list)
    for u in range(n_nodes):
        begg, endd = indptr[u : u + 2]
        dic[u] = indices[begg:endd]

    edge_set = []
    nonedge_set = []

    # construct edge set
    for u in range(n_nodes):
        for v in dic[u]:
            if v > u:
                edge_set.append((u, v))
    n_samples = len(edge_set)

    # random sample equal number of pairs to compose a nonoedge set
    while 1:
        u = rng.choice(n_nodes)
        v = rng.choice(n_nodes)
        if v not in dic[u] and u not in dic[v]:
            nonedge_set.append((u, v))
            if len(nonedge_set) == n_samples:
                break

    print(
        f"sampling done! len(edge_set) = {len(edge_set)}, len(nonedge_set) = {len(nonedge_set)}"
    )

    return (edge_set, nonedge_set), list(range(n_nodes))


def construct_edge_sets(dataset, sample_type, adj, n_samples, rng):
    indices = adj.indices
    indptr = adj.indptr
    n_nodes = adj.shape[0]

    # construct edge set
    edge_set = []
    while 1:
        u = rng.choice(n_nodes)
        begg, endd = indptr[u : u + 2]
        v_range = indices[begg:endd]
        if len(v_range):
            v = rng.choice(v_range)
            if (u,v) not in edge_set and (v,u) not in edge_set:
                edge_set.append((u, v))
            if len(edge_set) == n_samples:
                break

    # construct non-edge set
    nonedge_set = []

    # randomly select non-neighbors
    for _ in tqdm(range(n_samples)):
        u = rng.choice(n_nodes)
        begg, endd = indptr[u : u + 2]
        v_range = indices[begg:endd]
        while 1:
            v = rng.choice(n_nodes)
            if v not in v_range and (u,v) not in edge_set and (v,u) not in edge_set:
                nonedge_set.append((u, v))
                break
    list1, list2 = zip(*(edge_set + nonedge_set))
    return (edge_set, nonedge_set), list(set(list1 + list2))


def _get_edge_sets_among_nodes(indices, indptr, nodes):
    # construct edge list for each node
    dic = defaultdict(list)

    for u in nodes:
        begg, endd = indptr[u : u + 2]
        dic[u] = indices[begg:endd]

    n_nodes = len(nodes)
    edge_set = []
    nonedge_set = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            u, v = nodes[i], nodes[j]
            if v in dic[u]:
                edge_set.append((u, v))
            else:
                nonedge_set.append((u, v))

    print("#nodes =", len(nodes))
    print("#edges_set =", len(edge_set))
    print("#nonedge_set =", len(nonedge_set))
    return edge_set, nonedge_set

def _get_balanced_edge_sets_among_nodes(indices, indptr, nodes, n_samples, rng):
    n_nodes = len(nodes)

    # construct edge set
    edge_set = []
    while 1:
        u = rng.choice(n_nodes)
        begg, endd = indptr[u : u + 2]
        v_range = indices[begg:endd]
        if len(v_range):
            v = rng.choice(v_range)
            if v in nodes and (u,v) not in edge_set and (v,u) not in edge_set:
                edge_set.append((u, v))
            if len(edge_set) == n_samples:
                break

    # construct non-edge set
    nonedge_set = []

    # randomly select non-neighbors
    for _ in tqdm(range(n_samples)):
        u = rng.choice(n_nodes)
        begg, endd = indptr[u : u + 2]
        v_range = indices[begg:endd]
        while 1:
            v = rng.choice(n_nodes)
            if v not in v_range and v in nodes and (u,v) not in nonedge_set and (v,u) not in edge_set:
                nonedge_set.append((u, v))
                break

    print("#nodes =", len(nodes))
    print("#edges_set =", len(edge_set))
    print("#nonedge_set =", len(nonedge_set))
    return edge_set, nonedge_set

def _get_degree(n_nodes, indptr):
    deg = np.zeros(n_nodes, dtype=np.int32)
    for i in range(n_nodes):
        deg[i] = indptr[i + 1] - indptr[i]

    ind = np.argsort(deg)
    return deg, ind

def _get_adj_rank(dataset, adj,rank):
    adj_dense = adj.todense()
    edges = np.sum(adj_dense) / 2.0
    low_rank_adj = None
    high_rank_adj = None
    # from saved svd results and rank to reconstruct low-rank adj that overlap with adj
    svd_dir = os.path.join(MyGlobals.SVDDIR, dataset)
    if not os.path.exists(svd_dir):
        u, s, Vh = svd(adj.todense())
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
    
    arr = u[:,:rank] @ np.diag(s[:rank]) @ Vh[:rank,:]
    # use sort method for quantization
    arr[np.tril_indices_from(arr, k=0)] = -999999999
    raveled_arr = arr.ravel()
    flat_indices = np.argpartition(raveled_arr, len(raveled_arr) - int(edges))[-int(edges):]
    row_indices, col_indices = np.unravel_index(flat_indices, arr.shape)
    arr.fill(0.0)
    arr[row_indices, col_indices] = 1.0
    arr += arr.T
    
    # construct arr_01
    arr_01 = np.zeros_like(arr)
    arr_01 = np.where((arr!=0),1,0)
    
    # keep edges in arr that overlap with orig adj
    low_rank_adj = np.where((arr_01 == adj_dense), adj_dense, 0)
    high_rank_adj = adj_dense - low_rank_adj
    # print(f"low rank adj edges: {np.sum(low_rank_adj)/2.0}")
    # print(f"high rank adj edges: {np.sum(high_rank_adj)/2.0}")
    # print(f"adj edges: {edges}")
    
    low_rank_adj = sparse.csr_matrix(low_rank_adj)
    high_rank_adj = sparse.csr_matrix(high_rank_adj)
    return low_rank_adj, high_rank_adj

def construct_edge_sets_from_random_subgraph(dataset, sample_type, adj, n_samples, rng):
    indices = adj.indices
    indptr = adj.indptr
    n_nodes = adj.shape[0]

    if sample_type == "unbalanced":
        indice_all = range(n_nodes)

    else:
        deg, ind = _get_degree(n_nodes, indptr)
        if dataset.startswith("twitch"):
            lo = 5 if "PTBR" not in dataset else 10
            hi = 10
        elif dataset in ("cora"):
            lo = 3
            hi = 4
        elif dataset in ("citeseer"):
            lo = 3
            hi = 3
        elif dataset in ("pubmed"):
            lo = 10
            hi = 10
        else:
            raise NotImplementedError(f"lo and hi for dataset = {dataset} not set!")

        if sample_type == "unbalanced_lo":
            indice_all = np.where(deg <= lo)[0]
        else:
            indice_all = np.where(deg >= hi)[0]

    print("#indice =", len(indice_all))

    # choose from low degree nodes
    nodes = rng.choice(indice_all, n_samples, replace=False)

    return _get_edge_sets_among_nodes(indices, indptr, nodes), nodes

def construct_edge_sets_from_trans_subgraph(dataset, sample_type, adj, n_samples, rng):
    indices = adj.indices
    indptr = adj.indptr
    n_nodes = adj.shape[0]

    deg, ind = _get_degree(n_nodes, indptr)

    if dataset.startswith("twitch"):
        lo = 5 if "PTBR" not in dataset else 10
        hi = 10
    elif dataset in ("cora"):
        lo = 3
        hi = 4
    elif dataset in ("citeseer"):
        lo = 3
        hi = 3
    elif dataset in ("pubmed"):
        lo = 10
        hi = 10
    elif dataset in ("facebook_page"):
        lo = 10
        hi = 10
    else:
        raise NotImplementedError(f"lo and hi for dataset = {dataset} not set!")

    if sample_type == "balanced_lo":
        indice_all = np.where(deg <= lo)[0]
    else:
        indice_all = np.where(deg >= hi)[0]

    print("#indice =", len(indice_all))

    # use all low (or high) degree nodes and sample n_samples edges and non-edges from it
    return _get_balanced_edge_sets_among_nodes(indices, indptr, indice_all, n_samples, rng), indice_all

def construct_edge_sets_from_svd_subgraph(dataset, sample_type, adj, n_samples, rng, rank):
    # construct adj_lowrank and adj_highrank
    low_rank_adj, high_rank_adj = _get_adj_rank(dataset, adj,rank)

    # get info on node degree in original A
    indices = adj.indices
    indptr = adj.indptr
    n_nodes = adj.shape[0]
    deg, ind = _get_degree(n_nodes, indptr)
    
    if sample_type == "balanced_lorank":
        adj_all = low_rank_adj

        # get info on node degree in low rank A
        indices = adj_all.indices
        indptr = adj_all.indptr
        n_nodes = adj_all.shape[0]
        deg_lo, ind_lo = _get_degree(n_nodes, indptr)
    else:
        adj_all = high_rank_adj

    # return construct_balanced_edge_sets(dataset, sample_type, adj_all, n_samples, rng)
    return construct_edge_sets(dataset, sample_type, adj_all, n_samples, rng)

def construct_edge_sets_from_svd_ind_subgraph(dataset, sample_type, adj, n_samples, rng, rank):
    # construct adj_lowrank and adj_highrank
    low_rank_adj, high_rank_adj = _get_adj_rank(dataset, adj,rank)
    
    if sample_type == "unbalanced_lorank":
        adj_all = low_rank_adj
    else:
        adj_all = high_rank_adj

    return construct_balanced_edge_sets(dataset, sample_type, adj_all, n_samples, rng)

