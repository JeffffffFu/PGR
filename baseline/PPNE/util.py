from collections import Counter
import networkx as nx
from datetime import datetime
import os, torch, baseline.PPNE.lib, time
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import rand_score as RI
from sklearn.metrics import mutual_info_score as MI
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import v_measure_score as VMS
from sklearn.metrics import calinski_harabasz_score as ch_score
from sklearn.metrics import silhouette_score as ss_score

import numpy as np
import scipy.sparse as sp
from pdb import set_trace as st


def load_npz(file_path):
    loadfile = np.load(file_path, allow_pickle=True)
    return loadfile['adj_matrix'], loadfile['labels']


def sample_private_nodes(adj_matrix, private_ratio=0.1):

    # sample a number of private nodes
    num_nodes = adj_matrix.shape[0]

    num_private_nodes = int(num_nodes * private_ratio)

    while True:
        g_copy = nx.Graph(adj_matrix)
        private_nodes = np.random.choice(num_nodes, num_private_nodes, replace=False)
        private_edges = list(g_copy.subgraph(private_nodes).edges())
        # remove private edges
        g_copy.remove_edges_from(private_edges)
        # if isolated nodes exist, remove them
        isolate_nodes = list(nx.isolates(g_copy))

        if len(isolate_nodes) == 0:
            print('no isolated nodes')
            private_edges = np.array(private_edges)
            num_private_edges = private_edges.shape[0]
            private_non_edges = []
            G_private = nx.Graph()
            G_private.add_edges_from(private_edges)
            while len(private_non_edges) < num_private_edges:
                u, v = np.random.choice(private_nodes, 2, replace=False)
                if G_private.has_edge(u, v):
                    continue
                elif [u, v] in private_non_edges:
                    continue
                elif [v, u] in private_non_edges:
                    continue
                else:
                    private_non_edges.append([u, v])
            adj_matrix = nx.adj_matrix(g_copy).todense()
            private_edges = np.array(private_edges)
            private_non_edges = np.array(private_non_edges)
            break
        else:
            print("isolated nodes exist. re-sample.")
            continue

    return adj_matrix, private_edges, private_non_edges


def sample_private_nodes_fast(adj_matrix, private_ratio, labels):
    # create a graph based on adj matrix
    g = nx.from_scipy_sparse_matrix(adj_matrix)
    g_copy = nx.from_scipy_sparse_matrix(adj_matrix)

    # sample a number of private nodes
    num_nodes = adj_matrix.shape[0]
    node_list = [i for i in range(num_nodes)]
    num_private_nodes = int(num_nodes * private_ratio)
    private_nodes = np.random.choice(num_nodes, num_private_nodes, replace=False)
    private_edges = list(g.subgraph(private_nodes).edges())

    # remove private edges
    g_copy.remove_edges_from(private_edges)
    # if isolated nodes exist, remove them
    isolated_nodes = list(nx.isolates(g_copy))
    print("isolated nodes: {}".format(isolated_nodes))

    if len(isolated_nodes) == 0:
        print("no isolated nodes.")
        private_edges = np.array(private_edges)
        num_private_edges = private_edges.shape[0]
        G_private = nx.Graph()
        G_private.add_edges_from(private_edges)

        all_non_edges = np.array(list(nx.non_edges(G_private)))
        idx_non_edges = np.random.choice(len(all_non_edges), num_private_edges, replace=False)
        private_non_edges = all_non_edges[idx_non_edges]

        # adj_matrix = nx.adj_matrix(g_copy).todense()
        adj_matrix = nx.adj_matrix(g_copy)
    else:
        print("isolated nodes exist.")
        print("num of isolated nodes: {}".format(len(isolated_nodes)))
        left_nodes = sorted(list(set(range(num_nodes)) - set(isolated_nodes)))
        labels = np.array(labels)[np.array(left_nodes)]

        # mapping
        ks = []
        vs = []
        for i, j in enumerate(left_nodes):
            ks.append(j)
            vs.append(i)
        mapping = dict(zip(ks, vs))
        g_copy.remove_nodes_from(isolated_nodes)
        g_copy = nx.relabel_nodes(g_copy, mapping=mapping)

        private_nodes = list(set(private_nodes) - set(isolated_nodes))
        G_private = nx.Graph()
        G_private.add_edges_from(private_edges)
        private_edges = list(G_private.subgraph(private_nodes).edges())
        private_nodes = [mapping[node] for node in private_nodes]

        new_private_edges = []
        for pe in private_edges:
            u = mapping[pe[0]]
            v = mapping[pe[1]]
            new_private_edges.append([u, v])

        private_edges = np.array(new_private_edges)

        num_private_edges = private_edges.shape[0]
        print("num private edges {}".format(num_private_edges))
        g_priv = g_copy.subgraph(private_nodes)
        # seek for private non edges
        print("seek for private non_edges")

        all_non_edges = np.array(list(nx.non_edges(g_priv)))
        idx_non_edges = np.random.choice(len(all_non_edges), num_private_edges, replace=False)
        private_non_edges = all_non_edges[idx_non_edges]

        nodelist = [node for node in range(len(g_copy.nodes))]
        # adj_matrix = nx.adjacency_matrix(g_copy, nodelist=nodelist).todense()
        adj_matrix = nx.adjacency_matrix(g_copy, nodelist=nodelist)

    return adj_matrix, private_edges, private_non_edges, labels


def sample_private_nodes_new(adj_matrix, private_ratio, labels):
    # create a graph based on adj matrix
    # g = nx.Graph(adj_matrix)
    # g_copy = nx.Graph(adj_matrix)
    # create from sparse matrix
    g = nx.from_scipy_sparse_matrix(adj_matrix)
    g_copy = nx.from_scipy_sparse_matrix(adj_matrix)

    # sample a number of private nodes
    num_nodes = adj_matrix.shape[0]
    node_list = [i for i in range(num_nodes)]
    num_private_nodes = int(num_nodes * private_ratio)
    private_nodes = np.random.choice(num_nodes, num_private_nodes, replace=False)

    private_edges = list(g.subgraph(private_nodes).edges())

    # remove private edges
    g_copy.remove_edges_from(private_edges)
    # if isolated nodes exist, remove them
    isolated_nodes = list(nx.isolates(g_copy))
    print("isolated nodes: {}".format(isolated_nodes))

    if len(isolated_nodes) == 0:
        print("no isolated nodes.")
        private_edges = np.array(private_edges)
        num_private_edges = private_edges.shape[0]
        # G_private = nx.Graph()
        # G_private.add_edges_from(private_edges)
        private_non_edges = []
        while len(private_non_edges) < num_private_edges:
            u, v = np.random.choice(private_nodes, 2, replace=False)
            # if G_private.has_edge(u, v):
            #     continue
            # elif [u, v] in private_non_edges:
            #     continue
            # elif [v, u] in private_non_edges:
            #     continue
            # else:
            private_non_edges.append([u, v])
        private_non_edges = np.array(private_non_edges)
        # adj_matrix = nx.adj_matrix(g_copy).todense()
        adj_matrix = nx.adj_matrix(g_copy)
    else:
        print("isolated nodes exist.")
        print("num of isolated nodes: {}".format(len(isolated_nodes)))
        left_nodes = sorted(list(set(range(num_nodes)) - set(isolated_nodes)))

        labels = np.array(labels)[np.array(left_nodes)]

        # mapping
        ks = []
        vs = []
        for i, j in enumerate(left_nodes):
            ks.append(j)
            vs.append(i)
        mapping = dict(zip(ks, vs))

        g_copy.remove_nodes_from(isolated_nodes)
        g_copy = nx.relabel_nodes(g_copy, mapping=mapping)
        private_nodes = list(set(private_nodes) - set(isolated_nodes))

        G_private = nx.Graph()
        G_private.add_edges_from(private_edges)
        private_edges = list(G_private.subgraph(private_nodes).edges())
        private_nodes = [mapping[node] for node in private_nodes]

        new_private_edges = []
        for pe in private_edges:
            u = mapping[pe[0]]
            v = mapping[pe[1]]
            new_private_edges.append([u, v])

        private_edges = np.array(new_private_edges)

        num_private_edges = private_edges.shape[0]
        print("num private edges {}".format(num_private_edges))
        # g_priv = g_copy.subgraph(private_nodes)
        # seek for private non edges
        print("seek for private non_edges")
        private_non_edges = []
        while len(private_non_edges) < num_private_edges:
            u, v = np.random.choice(private_nodes, 2, replace=False)
            # if g_priv.has_edge(u, v):
            #     continue
            # elif [u, v] in private_non_edges:
            #     continue
            # elif [v, u] in private_non_edges:
            #     continue
            # else:
            private_non_edges.append([u, v])
        private_edges = np.array(private_edges)
        private_non_edges = np.array(private_non_edges)

        nodelist = [node for node in range(len(g_copy.nodes))]
        # adj_matrix = nx.adjacency_matrix(g_copy, nodelist=nodelist).todense()
        adj_matrix = nx.adjacency_matrix(g_copy, nodelist=nodelist)

    return adj_matrix, private_edges, private_non_edges, labels


def optimization_edges_sample(G, sample_ratio, save_path):
    num_edges = len(G.edges())
    n_sample_edges = int(sample_ratio * num_edges)
    sample_edges = np.array(list(map(list, np.array(G.edges())[np.random.choice(num_edges,
                                                                                n_sample_edges, replace=False)])))
    sample_non_edges = []
    while len(sample_non_edges) < n_sample_edges:
        i = np.random.choice(len(G.nodes()), 1)[0]
        j = np.random.choice(len(G.nodes()), 1)[0]
        while i == j:
            j = np.random.choice(len(G.nodes()), 1)[0]
        if tuple([i, j]) not in G.edges() and tuple([j, i]) not in G.edges()\
                and [i, j] not in sample_non_edges and [j, i] not in sample_non_edges:
            sample_non_edges.append([i, j])
    sample_non_edges = np.array(sample_non_edges)

    np.save(save_path+"/sample_edges.npy", sample_edges)
    np.save(save_path+"/sample_non_edges.npy", sample_non_edges)

    return sample_edges, sample_non_edges


def testing_edges_sample(G, sample_ratio):
    num_edges = len(G.edges())
    n_sample_edges = int(sample_ratio * num_edges)
    sample_edges = np.array(list(map(list, np.array(G.edges())[np.random.choice(num_edges,
                                                                                n_sample_edges, replace=False)])))
    sample_non_edges = []
    while len(sample_non_edges) < n_sample_edges:
        i = np.random.choice(len(G.nodes()), 1)[0]
        j = np.random.choice(len(G.nodes()), 1)[0]
        while i == j:
            j = np.random.choice(len(G.nodes()), 1)[0]
        if tuple([i, j]) not in G.edges() and tuple([j, i]) not in G.edges()\
                and [i, j] not in sample_non_edges and [j, i] not in sample_non_edges:
            sample_non_edges.append([i, j])
    sample_non_edges = np.array(sample_non_edges)

    return sample_edges, sample_non_edges


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def enforcing_no_change(adj_train, edges):
    adj_train = np.copy(adj_train)
    for edge in edges:
        adj_train[edge[0]][edge[1]] = 0
        adj_train[edge[1]][edge[0]] = 0
    return adj_train


def compute_avail_loss_over_edges(scores_gt, scores_cur, victim_edges):
    loss = 0
    for edge in victim_edges:
        loss += (scores_gt[edge[0]][edge[1]]-scores_cur[edge[0]][edge[1]]) ** 2
    loss = np.sqrt(loss / len(victim_edges))
    return loss


def evaluate(X, test_edges, test_edges_false):
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    scores_matrix = X_normalized.dot(X_normalized.T)
    roc_score, ap_score = lib.get_roc_score(test_edges, test_edges_false, scores_matrix)
    return roc_score, ap_score


def evaluate_link_prediction_csr(X, test_edges, test_edges_false):
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    scores_matrix = X_normalized.dot(X_normalized.T)
    roc_score, ap_score = lib.get_roc_score(test_edges, test_edges_false, scores_matrix)
    return roc_score, ap_score


def evaluate_scenario2(X, test_edges, test_edges_false, dim, n_repeats=20, train_ratio=0.7, seed=0):
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    sample_num = test_edges.shape[0] * 2
    sample_features = []
    labels = []
    for edge in test_edges:
        sample_features.append(np.concatenate((X_normalized[edge[0]], X_normalized[edge[1]])))
        labels.append(1)
    for non_edge in test_edges_false:
        sample_features.append(np.concatenate((X_normalized[non_edge[0]], X_normalized[non_edge[1]])))
        labels.append(0)
    sample_features = np.array(sample_features).reshape(sample_num, dim*2)
    labels = np.array(labels)

    # lr_results = []
    xgb_results = []
    for it_seed in range(n_repeats):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_ratio, random_state=seed + it_seed)
        split_train, split_test = next(sss.split(sample_features, labels))

        features_train = sample_features[split_train]
        features_test = sample_features[split_test]
        labels_train = labels[split_train]
        labels_test = labels[split_test]

        # lr = LogisticRegression(solver="lbfgs", penalty='l2', random_state=0)
        # lr.fit(features_train, labels_train)

        xgb = XGBClassifier(random_state=0, n_estimators=50)
        xgb.fit(features_train, labels_train)

        # lr_z_predict = lr.predict(features_test)
        xgb_z_predict = xgb.predict(features_test)

        # lr_f1_micro = f1_score(labels_test, lr_z_predict, average='micro')
        xgb_f1_micro = f1_score(labels_test, xgb_z_predict, average='micro')

        # lr_results.append(lr_f1_micro)
        xgb_results.append(xgb_f1_micro)

    # return np.mean(lr_results), np.mean(xgb_results)
    return np.mean(xgb_results)

def intermediate_similarity_calculation(X, test_edges):
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    scores_matrix = X_normalized.dot(X_normalized.T)
    xs = []
    ys = []
    for x, y in test_edges:
        xs.append(x)
        ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys)
    similarity_score = np.sum(scores_matrix[xs, ys])

    return similarity_score


def private_non_edge_similarity_calculation(X, test_non_edges):
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    scores_matrix = X_normalized.dot(X_normalized.T)
    xs = []
    ys = []
    for x, y in test_non_edges:
        xs.append(x)
        ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys)
    similarity_score = np.sum(scores_matrix[xs, ys])

    return similarity_score


def evaluate_clf(X, train_test_split):
    def get_edge_embeddings(emb_X, edge_list):
        embs = []
        for edge in edge_list:
            node1 = edge[0]
            node2 = edge[1]
            emb1 = emb_X[node1]
            emb2 = emb_X[node2]
            # edge_emb = np.multiply(emb1 , emb2)
            edge_emb = (emb1 + emb2) / 2.
            embs.append(edge_emb)
        embs = np.array(embs)
        return embs

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = train_test_split

    # Train-set edge embeddings
    pos_train_edge_embs = get_edge_embeddings(X, train_edges)
    neg_train_edge_embs = get_edge_embeddings(X, train_edges_false)
    train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

    # Create train-set edge labels: 1 = real edge, 0 = false edge
    train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

    # Test-set edge embeddings, labels
    pos_test_edge_embs = get_edge_embeddings(X, test_edges)
    neg_test_edge_embs = get_edge_embeddings(X, test_edges_false)
    test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

    # Create val-set edge labels: 1 = real edge, 0 = false edge
    test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

    # Train logistic regression classifier on train-set edge embeddings
    edge_classifier = LogisticRegression(random_state=0)
    edge_classifier.fit(train_edge_embs, train_edge_labels)

    test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]
        
    gae_test_roc = roc_auc_score(test_edge_labels, test_preds)
    # gae_test_roc_curve = roc_curve(test_edge_labels, test_preds)
    gae_test_ap = average_precision_score(test_edge_labels, test_preds)

    return gae_test_roc, gae_test_ap


def get_score_over_edge(X, edge_list):
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    scores_matrix = X_normalized.dot(X_normalized.T)
    score = lib.compute_score_over_edges(scores_matrix, edge_list)
    return score


def get_grad_on_X(X_torch, attack_edges, attack_edges_false):
    X_torch.requires_grad = True
    X_norm = torch.sqrt(torch.sum(X_torch * X_torch, dim=1))
    X_normalized = X_torch / X_norm.reshape(X_norm.shape[0], -1)

    loss = 0
    for edge in attack_edges:
        loss -= torch.dot(X_normalized[edge[0]], X_normalized[edge[1]])
    for edge in attack_edges_false:
        loss += torch.dot(X_normalized[edge[0]], X_normalized[edge[1]])
    loss.backward()

    return X_torch.grad.data


def get_grad_on_X_old(X_torch, attack_edges, attack_edges_false, savepath):
    X_torch.requires_grad = True

    with open(os.path.join(savepath, 'pgd_record_file_modify'), 'a') as file_record:
        file_record.write('start X normalization\n')
        file_record.flush()

    X_norm = torch.sqrt(torch.sum(X_torch * X_torch, dim=1))
    X_normalized = X_torch / X_norm.reshape(X_norm.shape[0], -1)

    with open(os.path.join(savepath, 'pgd_record_file_modify'), 'a') as file_record:
        file_record.write('X normalization done\n')
        file_record.flush()

    loss = 0
    for edge in attack_edges:
        loss -= torch.dot(X_normalized[edge[0]], X_normalized[edge[1]])
    for edge in attack_edges_false:
        loss += torch.dot(X_normalized[edge[0]], X_normalized[edge[1]])

    with open(os.path.join(savepath, 'pgd_record_file_modify'), 'a') as file_record:
        file_record.write('objective loss calculation done, start backward()\n')
        file_record.flush()

    loss.backward()

    with open(os.path.join(savepath, 'pgd_record_file_modify'), 'a') as file_record:
        file_record.write('backward done\n')
        file_record.flush()

    return X_torch.grad.data


def get_grad_on_X_new(X_torch, attack_edges, attack_edges_false, savepath):
    X_torch.requires_grad = True
    # X_norm = torch.sqrt(torch.sum(X_torch * X_torch, dim=1))
    # X_normalized = X_torch / X_norm.reshape(X_norm.shape[0], -1)

    loss = 0

    with open(os.path.join(savepath, 'pgd_record_file_modify'), 'a') as file_record:
        file_record.write('start calculate loss, now:{}\n'.format(datetime.now()))
        file_record.flush()

    n = 0
    for edge in attack_edges:
        x1_n = torch.sqrt(torch.sum(X_torch[edge[0]] * X_torch[edge[0]]))
        x2_n = torch.sqrt(torch.sum(X_torch[edge[1]] * X_torch[edge[1]]))
        x1_norm = X_torch[edge[0]] / x1_n
        x2_norm = X_torch[edge[1]] / x2_n

        loss -= torch.dot(x1_norm, x2_norm)
        if n % 1000 == 0:
            with open(os.path.join(savepath, 'pgd_record_file_modify'), 'a') as file_record:
                file_record.write('{} edge done, now:{}\n'.format(n, datetime.now()))
                file_record.flush()
        n+=1
        # loss -= torch.dot(X_normalized[edge[0]], X_normalized[edge[1]])
    m = 0
    for edge in attack_edges_false:
        x1_n = torch.sqrt(torch.sum(X_torch[edge[0]] * X_torch[edge[0]]))
        x2_n = torch.sqrt(torch.sum(X_torch[edge[1]] * X_torch[edge[1]]))
        x1_norm = X_torch[edge[0]] / x1_n
        x2_norm = X_torch[edge[1]] / x2_n

        loss += torch.dot(x1_norm, x2_norm)
        if m % 1000 == 0:
            with open(os.path.join(savepath, 'pgd_record_file_modify'), 'a') as file_record:
                file_record.write('{} non-edge done, now:{}\n'.format(m, datetime.now()))
                file_record.flush()
        m+=1
        # loss += torch.dot(X_normalized[edge[0]], X_normalized[edge[1]])
    loss.backward()

    return X_torch.grad.data


def get_grad_on_X_fp16(args, X_torch, attack_edges, attack_edges_false):
    X_torch.requires_grad = True
    X_norm = torch.sqrt(torch.sum(X_torch * X_torch, dim=1))
    X_normalized = X_torch / X_norm.reshape(X_norm.shape[0],-1)

    loss = 0
    for edge in attack_edges:
        loss -= torch.dot(X_normalized[edge[0]], X_normalized[edge[1]])
    for edge in attack_edges_false:
        loss += torch.dot(X_normalized[edge[0]], X_normalized[edge[1]])
    loss.backward()

    return X_torch.grad.data


def build_grad_on_A_adv(X, Y, X_grad, mask):
    n = X.shape[0]
    d = X.shape[1]
    y_ts_grad = []
    for i in range(n):
        nonzero_y = np.argwhere(mask[i]).T[0]
        if np.argwhere(X_grad[i]).shape[0] == 0:
            for j in nonzero_y:
                y_ts_grad.append(0)
            continue
        # y_yt_sum = np.ones((d,d)) * 1e-4
        y_yt_sum = np.eye(d) * 1e-4
        for j in nonzero_y:
            y = Y[j].reshape(1,-1)
            y_yt_sum += y.T.dot(y)
        y_yt_sum_inv = np.linalg.inv(y_yt_sum)

        for j in nonzero_y:
            value = X_grad[i].dot(np.matmul(y_yt_sum_inv, Y[j]))
            y_ts_grad.append(value)

    return np.array(y_ts_grad)


# def build_grad_on_Z_adv_torch(X, Y, X_grad, mask_on_Z,args):
#     n, d = X.shape
#     # grad_Z = torch.zeros((n,n)).double().cuda()
#     grad_Z = torch.zeros((n,n)).double().to(args.device)
#     # grad_Z = torch.zeros((n, n), dtype=torch.float32).cuda()
#     for i in range(n):
#         if torch.nonzero(X_grad[i]).numel() == 0:
#             continue
#         nonzero_y = np.argwhere(mask_on_Z[i]).T[0]
#         # y_yt_sum = torch.zeros((d,d)).double().cuda()
#         y_yt_sum = torch.zeros((d,d)).double().to(args.device)
#         # y_yt_sum = torch.zeros((d, d), dtype=torch.float32).cuda()
#         for j in nonzero_y:
#             y = Y[j].reshape(1,-1)
#             y_yt_sum += torch.mm(torch.t(y),y)
#         y_yt_sum_inv = torch.inverse(y_yt_sum)
#
#         # for j in nonzero_y:
#         # 	grad_Z[i][j] = torch.dot(X_grad[i], torch.mm(y_yt_sum_inv, Y[j].view(-1,1)).squeeze(1))
#
#         grad_Z[i] = torch.mm(X_grad[i].view(1,-1), torch.mm(y_yt_sum_inv, Y.t()))[0]
#         grad_Z[i, np.argwhere(~mask_on_Z[i]).T[0]] = 0
#
#     return grad_Z


def build_grad_on_Z_adv_torch(X, Y, X_grad, mask_on_Z, args):
    n, d = X.shape  # n 是样本数量，d 是特征维度

    # 确保所有输入张量都在相同的设备上
    Y = Y.to(args.device) #print(Y.size())   torch.Size([2708, 64])

    X_grad = X_grad.to(args.device)#print(X_grad.size()) torch.Size([2708, 64])

    # 将 mask_on_Z 转换为与 X, Y, X_grad 相同的设备，并确保是布尔类型
    mask_on_Z = torch.from_numpy(mask_on_Z).to(args.device)    # print(mask_on_Z.size())  torch.Size([2708, 2708])

    # 初始化 grad_Z 为零矩阵，大小与 mask_on_Z 相同
    grad_Z = torch.zeros((n, n), dtype=torch.double).to(args.device)

    # 遍历每个样本
    for i in range(n):
        # 找出 mask_on_Z 中第 i 行中为 True 的索引，即与样本 i 相邻的样本索引
        neighbors = torch.nonzero(mask_on_Z[i], as_tuple=False).squeeze()

        # 如果没有邻居，grad_Z[i] 保持为零
        if neighbors.numel() == 0:
            continue

        # 选择 Y 中邻居样本的列向量
        selected_Y_columns = Y[neighbors]

        # 计算 y_yt_sum
        y_yt_sum = torch.mm(selected_Y_columns.t(), selected_Y_columns)

        # 计算 y_yt_sum 的逆矩阵
        y_yt_sum_inv = torch.pinverse(y_yt_sum)

        # 计算 grad_Z[i] 对应的行
        grad_Z[i, neighbors] = torch.mm(X_grad[i].unsqueeze(0), torch.mm(y_yt_sum_inv, selected_Y_columns.t()))

    # 对于 mask_on_Z 中为 False 的位置，将 grad_Z 中的相应元素设置为 0
    grad_Z = grad_Z * mask_on_Z

    return grad_Z



def build_grad_on_Z_adv_torch_float2(X, Y, X_grad, mask_on_Z):
    n, d = X.shape
    # grad_Z = torch.zeros((n,n)).double().cuda()
    # grad_Z = torch.zeros((n, n), dtype=torch.float32).cuda()
    grad_Z = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        if torch.nonzero(X_grad[i]).numel() == 0:
            continue
        nonzero_y = np.argwhere(mask_on_Z[i]).T[0]
        # y_yt_sum = torch.zeros((d,d)).double().cuda()
        # y_yt_sum = torch.zeros((d, d), dtype=torch.float32).cuda()
        y_yt_sum = torch.zeros((d, d), dtype=torch.float32)
        for j in nonzero_y:
            y = Y[j].reshape(1,-1)
            y_yt_sum += torch.mm(torch.t(y),y)
        y_yt_sum_inv = torch.inverse(y_yt_sum)

        # for j in nonzero_y:
        # 	grad_Z[i][j] = torch.dot(X_grad[i], torch.mm(y_yt_sum_inv, Y[j].view(-1,1)).squeeze(1))

        grad_Z[i] = torch.mm(X_grad[i].view(1,-1), torch.mm(y_yt_sum_inv, Y.t()))[0]
        grad_Z[i, np.argwhere(~mask_on_Z[i]).T[0]] = 0

    return grad_Z


def build_grad_on_Z_adv_torch_float(X, Y, X_grad, mask_on_Z):
    n, d = X.shape
    # grad_Z = torch.zeros((n,n)).double().cuda()
    grad_Z = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        if torch.nonzero(X_grad[i]).numel() == 0:
            continue
        nonzero_y = np.argwhere(mask_on_Z[i]).T[0]
        # y_yt_sum = torch.zeros((d,d)).double().cuda()
        y_yt_sum = torch.zeros((d, d), dtype=torch.float32)
        for j in nonzero_y:
            y = Y[j].reshape(1,-1)
            y_yt_sum += torch.mm(torch.t(y),y)
        y_yt_sum_inv = torch.inverse(y_yt_sum)

        # for j in nonzero_y:
        # 	grad_Z[i][j] = torch.dot(X_grad[i], torch.mm(y_yt_sum_inv, Y[j].view(-1,1)).squeeze(1))

        grad_Z[i] = torch.mm(X_grad[i].view(1,-1), torch.mm(y_yt_sum_inv, Y.t()))[0]
        grad_Z[i, np.argwhere(~mask_on_Z[i]).T[0]] = 0

    return grad_Z


def load_dataset(file_name):
    """"Load a graph from a Numpy binary file.

    :param file_name: str
        Name of the file to load.

    :return: dict
        Dictionary that contains:
            * 'A' : The adjacency matrix in sparse matrix format
            * 'X' : The attribute matrix in sparse matrix format
            * 'z' : The ground truth class labels
            * Further dictionaries mapping node, class and attribute IDs
    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                    loader['adj_indptr']), shape=loader['adj_shape'])

        labels = loader.get('labels')

        graph = {
            'adj_matrix': adj_matrix,
            'labels': labels
        }

        idx_to_node = loader.get('idx_to_node')
        if idx_to_node:
            idx_to_node = idx_to_node.tolist()
            graph['idx_to_node'] = idx_to_node

        idx_to_attr = loader.get('idx_to_attr')
        if idx_to_attr:
            idx_to_attr = idx_to_attr.tolist()
            graph['idx_to_attr'] = idx_to_attr

        idx_to_class = loader.get('idx_to_class')
        if idx_to_class:
            idx_to_class = idx_to_class.tolist()
            graph['idx_to_class'] = idx_to_class

        return graph


def standardize(adj_matrix, labels):
    """
    Make the graph undirected and select only the nodes
     belonging to the largest connected component.

    :param adj_matrix: sp.spmatrix
        Sparse adjacency matrix
    :param labels: array-like, shape [n]

    :return:
        standardized_adj_matrix: sp.spmatrix
            Standardized sparse adjacency matrix.
        standardized_labels: array-like, shape [?]
            Labels for the selected nodes.
    """
    # copy the input
    standardized_adj_matrix = adj_matrix.copy()

    # make the graph unweighted
    standardized_adj_matrix[standardized_adj_matrix != 0] = 1

    # make the graph undirected
    standardized_adj_matrix = standardized_adj_matrix.maximum(standardized_adj_matrix.T)

    # select the largest connected component
    _, components = sp.csgraph.connected_components(standardized_adj_matrix)
    c_ids, c_counts = np.unique(components, return_counts=True)
    id_max_component = c_ids[c_counts.argmax()]
    select = components == id_max_component
    standardized_adj_matrix = standardized_adj_matrix[select][:, select]
    standardized_labels = labels[select]

    # remove self-loops
    standardized_adj_matrix = standardized_adj_matrix.tolil()
    standardized_adj_matrix.setdiag(0)
    standardized_adj_matrix = standardized_adj_matrix.tocsr()
    standardized_adj_matrix.eliminate_zeros()

    return standardized_adj_matrix, standardized_labels


def generate_candidates_removal(adj_matrix, seed=0):
    """Generates candidate edge flips for removal (edge -> non-edge),
     disallowing one random edge per node to prevent singleton nodes.

    adj_matrix: sp.csr_matrix, shape [n_nodes, n_nodes]
        Adjacency matrix of the graph
    :param seed: int
        Random seed
    :return: np.ndarray, shape [?, 2]
        Candidate set of edge flips
    """
    n_nodes = adj_matrix.shape[0]

    np.random.seed(seed)
    deg = np.where(adj_matrix.sum(1).A1 == 1)[0]

    hiddeen = np.column_stack(
        (np.arange(n_nodes), np.fromiter(map(np.random.choice, adj_matrix.tolil().rows), dtype=int)))

    adj_hidden = edges_to_sparse(hiddeen, adj_matrix.shape[0])
    adj_hidden = adj_hidden.maximum(adj_hidden.T)

    adj_keep = adj_matrix - adj_hidden

    candidates = np.column_stack((sp.triu(adj_keep).nonzero()))

    candidates = candidates[np.logical_not(np.in1d(candidates[:, 0], deg) | np.in1d(candidates[:, 1], deg))]

    return candidates


def edges_to_sparse(edges, num_nodes, weights=None):
    """Create a sparse adjacency matrix from an array of edge indices and (optionally) values.

    :param edges: array-like, shape [num_edges, 2]
        Array with each row storing indices of an edge as (u, v).
    :param num_nodes: int
        Number of nodes in the resulting graph.
    :param weights: array_like, shape [num_edges], optional, default None
        Weights of the edges. If None, all edges weights are set to 1.
    :return: sp.csr_matrix
        Adjacency matrix in CSR format.
    """
    if weights is None:
        weights = np.ones(edges.shape[0])

    return sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes)).tocsr()


def generate_candidates_addition(adj_matrix, n_candidates, seed=0):
    """Generates candidate edge flips for addition (non-edge -> edge).

    adj_matrix: sp.csr_matrix, shape [n_nodes, n_nodes]
        Adjacency matrix of the graph
    :param n_candidates: int
        Number of candidates to generate.
    :param seed: int
        Random seed
    :return: np.ndarray, shape [?, 2]
        Candidate set of edge flips
    """
    np.random.seed(seed)
    num_nodes = adj_matrix.shape[0]

    candidates = np.random.randint(0, num_nodes, [n_candidates * 5, 2])
    candidates = candidates[candidates[:, 0] < candidates[:, 1]]
    candidates = candidates[adj_matrix[candidates[:, 0], candidates[:, 1]].A1 == 0]
    candidates = np.array(list(set(map(tuple, candidates))))
    candidates = candidates[:n_candidates]

    assert len(candidates) == n_candidates

    return candidates


def flip_candidates(adj_matrix, candidates):
    """Flip the edges in the candidate set to non-edges and vise-versa.

    :param adj_matrix: sp.csr_matrix, shape [n_nodes, n_nodes]
        Adjacency matrix of the graph
    :param candidates: np.ndarray, shape [?, 2]
        Candidate set of edge flips
    :return: sp.csr_matrix, shape [n_nodes, n_nodes]
        Adjacency matrix of the graph with the flipped edges/non-edges.
    """
    adj_matrix_flipped = adj_matrix.copy().tolil()
    adj_matrix_flipped[candidates[:, 0], candidates[:, 1]] = 1 - adj_matrix[candidates[:, 0], candidates[:, 1]]
    adj_matrix_flipped[candidates[:, 1], candidates[:, 0]] = 1 - adj_matrix[candidates[:, 1], candidates[:, 0]]
    adj_matrix_flipped = adj_matrix_flipped.tocsr()
    adj_matrix_flipped.eliminate_zeros()

    return adj_matrix_flipped


def evaluate_embedding_node_classification(embedding_matrix, labels, train_ratio=0.7, norm=True, seed=0, n_repeats=20):
    """Evaluate the node embeddings on the node classification task..

    :param embedding_matrix: np.ndarray, shape [n_nodes, embedding_dim]
        Embedding matrix
    :param labels: np.ndarray, shape [n_nodes]
        The ground truth labels
    :param train_ratio: float
        The fraction of labels to use for training
    :param norm: bool
        Whether to normalize the embeddings
    :param seed: int
        Random seed
    :param n_repeats: int
        Number of times to repeat the experiment
    :return: [float, float], [float, float]
        The mean and standard deviation of the f1_scores
    """
    if norm:
        embedding_matrix = normalize(embedding_matrix)

    results = []
    accs = []
    for it_seed in range(n_repeats):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_ratio, random_state=seed + it_seed)
        split_train, split_test = next(sss.split(embedding_matrix, labels))

        features_train = embedding_matrix[split_train]
        features_test = embedding_matrix[split_test]
        labels_train = labels[split_train]
        labels_test = labels[split_test]

        lr = LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='auto')
        lr.fit(features_train, labels_train)

        lr_z_predict = lr.predict(features_test)
        f1_micro = f1_score(labels_test, lr_z_predict, average='micro')
        f1_macro = f1_score(labels_test, lr_z_predict, average='macro')

        acc = accuracy_score(labels_test, lr_z_predict)

        results.append([f1_micro, f1_macro])
        accs.append(acc)

    results = np.array(results)
    accs = np.array(accs)

    return results.mean(0), results.std(0), np.mean(accs)


def evaluate_embedding_node_clustering(ori_emb_matrix, emb_matrix, k, norm=True, random_seed=0):
    """Evaluate the node embedding on the node clustering task

    :return: nmi_score: Normalized Mutual Information (external metric)
             ss: Silhouette Coefficient (internal metric)
             ch: Caliniski-Harabaz score (internal metric)

    """
    if norm:
        ori_emb_matrix = normalize(ori_emb_matrix)
        emb_matrix = normalize(emb_matrix)

    # k = len(set(labels))
    kmeans_ori = KMeans(n_clusters=len(k), random_state=random_seed)
    kmeans_ori.fit(ori_emb_matrix)

    kmeans = KMeans(n_clusters=len(k), random_state=random_seed)
    kmeans.fit(emb_matrix)

    labels_ori = kmeans_ori.labels_
    labels_pred = kmeans.labels_
    nmi_score = NMI(labels_ori, labels_pred, average_method='arithmetic')
    ami_score = AMI(labels_ori, labels_pred)
    mi_score = MI(labels_ori, labels_pred)
    ari_score = ARI(labels_ori, labels_pred)
    ri_score = RI(labels_ori, labels_pred)
    vmi_score = VMS(labels_ori, labels_pred)

    # ss = ss_score(emb_matrix, labels_pred, metric='euclidean')
    # ch = ch_score(emb_matrix, labels_pred)

    # return nmi_score, ss, ch
    return nmi_score, ami_score, mi_score, ari_score, ri_score, vmi_score


def evaluate_embedding_node_clustering_hier(ori_emb_matrix, emb_matrix, k, norm=True, random_seed=0):
    """Evaluate the node embedding on the node clustering task

    :return: nmi_score: Normalized Mutual Information (external metric)
             ss: Silhouette Coefficient (internal metric)
             ch: Caliniski-Harabaz score (internal metric)

    """
    if norm:
        ori_emb_matrix = normalize(ori_emb_matrix)
        emb_matrix = normalize(emb_matrix)

    hier_ori = AgglomerativeClustering(k, affinity='cosine', linkage='average')
    hier_ori.fit(ori_emb_matrix)

    hier = AgglomerativeClustering(k, affinity='cosine', linkage='average')
    hier.fit(emb_matrix)

    labels_ori = hier_ori.labels_
    labels_pred = hier.labels_
    nmi_score = NMI(labels_ori, labels_pred, average_method='arithmetic')
    ami_score = AMI(labels_ori, labels_pred)
    mi_score = MI(labels_ori, labels_pred)
    ari_score = ARI(labels_ori, labels_pred)
    ri_score = RI(labels_ori, labels_pred)
    vmi_score = VMS(labels_ori, labels_pred)
    # ss = ss_score(emb_matrix, labels_pred, metric='euclidean')
    # ch = ch_score(emb_matrix, labels_pred)

    # return nmi_score, ss, ch
    return nmi_score, ami_score, mi_score, ari_score, ri_score, vmi_score


def k_largest_index_argsort(a, k):
    idx = np.argsort(a.ravel())[:-k-1:-1]
    return np.column_stack(np.unravel_index(idx, a.shape))


def differential_privacy_manipulation(adj_train, epsilon=10):
    # obtain the in degree of each node
    dg = nx.DiGraph(adj_train)
    in_degrees = np.array([val for (node, val) in dg.in_degree()])

    # generate noise vector
    sensitivity = 1
    noise_vec = np.random.laplace(0, sensitivity / epsilon, len(in_degrees))

    # result vector
    result_vec = np.round(in_degrees + noise_vec)

    # at least 1
    result_vec[result_vec <= 0] = 1

    add_or_delete_vec = result_vec - in_degrees

    return add_or_delete_vec