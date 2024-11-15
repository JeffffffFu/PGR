import torch.nn.functional as F
import torch
import torch_geometric
from scipy.stats import pearsonr
from torch_geometric.datasets import Planetoid


import torch

import numpy as np
from tqdm import tqdm

from TIA.attack_model import attack_step, attack_model_inference, attack_model_inference_cluster, \
    attack_model_inference_PGR_worst_case
from TIA.util import construct_subgraph, process_matrix, topology_loss_sim, \
    euclidean_distance, chebyshev_distance, cosine_distance, construct_subgraph_for_GAP, construct_subgraph_shadow, \
    k_hat_subgraph, process_matrix_PGR_worst_case
from mask.add_diagonal_matrix import add_diagonal_and_normalize_edge
from utils.matrix_operation import dense_adj_to_adj_sparse_adj
from scipy.spatial.distance import cosine, euclidean, correlation, cityblock


def topology_audit_based_sim( selected_subgraphs, model, priv_adj, features, regen_edge, labels, device, seed):
    print("topology_audit_based_sim -------------------------")
    model = model.to(device)
    regen_edge = regen_edge.to(device)
    features = features.to(device)
    N = len(labels)
    similarity_list = [cosine_distance, chebyshev_distance,euclidean_distance]


    tpl_values1 = []

    for i, (test_nodes, adj_subgraph) in enumerate(selected_subgraphs):

        K = torch.count_nonzero(adj_subgraph).item()
        K = int(K / 2)
        K_hat=k_hat_subgraph(test_nodes, regen_edge)

        TPL_list=[]

        A_hat = add_diagonal_and_normalize_edge(regen_edge, device)
        posterior = model.logist(features, A_hat).detach().cpu().numpy()

        for similarity in similarity_list:
            dist = torch.zeros((N, N))
            for u in test_nodes:
                for v in test_nodes:
                    if v > u:
                        dist[u][v] = torch.tensor(similarity(posterior[u], posterior[v]), dtype=torch.float32)

            binary_matrix = process_matrix(dist, K_hat)


            K_hat = torch.count_nonzero(torch.tensor(binary_matrix)).item() // 2
            TPL = topology_loss_sim(adj_subgraph, binary_matrix, K, K_hat)

            TPL_list.append(TPL)

        tpl_values1.append(max(TPL_list))


    tpl_avg1 = sum(tpl_values1) / len(tpl_values1)


    return tpl_avg1



def topology_audit_based_sim_PGR_worst_case(selected_subgraphs, model, priv_adj, features, regen_edge, labels, device, seed):
    model = model.to(device)
    regen_edge = regen_edge.to(device)
    features = features.to(device)
    N = len(labels)
    similarity_list = [cosine_distance, chebyshev_distance,euclidean_distance]


    tpl_values1 = []

    for i, (test_nodes, adj_subgraph) in enumerate(selected_subgraphs):


        K = torch.count_nonzero(adj_subgraph).item()
        K = int(K / 2)
        K_hat=k_hat_subgraph(test_nodes, regen_edge)


        A_hat = add_diagonal_and_normalize_edge(regen_edge, device)
        posterior = model.logist(features, A_hat).detach().cpu().numpy()
        TPL_list=[]
        for similarity in similarity_list:
            dist = torch.zeros((N, N))
            for u in test_nodes:
                for v in test_nodes:
                    if v > u:
                        dist[u][v] = torch.tensor(similarity(posterior[u], posterior[v]), dtype=torch.float32)


            binary_matrix = process_matrix_PGR_worst_case(dist, K,regen_edge)

            K_hat = torch.count_nonzero(torch.tensor(binary_matrix)).item() // 2
            TPL = topology_loss_sim(adj_subgraph, binary_matrix, K, K_hat)

            TPL_list.append(TPL)

        tpl_values1.append(max(TPL_list))


    tpl_avg1 = sum(tpl_values1) / len(tpl_values1)


    return tpl_avg1

def topology_audit_based_sim_classifier( selected_subgraphs, model, priv_adj, features, regen_edge, labels, device, seed):
    model = model.to(device)
    regen_edge = regen_edge.to(device)
    features = features.to(device)
    N = len(labels)

    if len(selected_subgraphs)==1:
        train_subgraph = selected_subgraphs[0]
        test_subgraphs = selected_subgraphs[0:]
    else:
        train_subgraph = selected_subgraphs[0]
        test_subgraphs = selected_subgraphs[1:]

    test_nodes_shadow, adj_subgraph_shadow = train_subgraph

    similarity_list = [cosine, euclidean, correlation, cityblock]


    A_hat = add_diagonal_and_normalize_edge(regen_edge, device)
    posterior = model.logist(features, A_hat).detach().cpu().numpy()

    mia_train_feature = []
    mia_train_label = []
    for u in tqdm(test_nodes_shadow):
        for v in test_nodes_shadow:
            if v > u:
                tmp_feature = [sim(posterior[u], posterior[v]) for sim in similarity_list]
                mia_train_feature.append(tmp_feature)
                mia_train_label.append([adj_subgraph_shadow[u][v]])

    attack_model=attack_step(mia_train_feature, mia_train_label, device)
    tpl_values1 = []

    for i, (test_nodes, adj_subgraph) in enumerate(test_subgraphs):

        K = torch.count_nonzero(adj_subgraph).item() // 2
        K_hat=k_hat_subgraph(test_nodes, regen_edge)

        mia_test_feature = []
        mia_test_label = []
        for u in tqdm(test_nodes):
            for v in test_nodes:
                if v > u:
                    tmp_feature = [sim(posterior[u], posterior[v]) for sim in similarity_list]
                    mia_test_feature.append(tmp_feature)
                    mia_test_label.append([adj_subgraph[u][v]])


        TPL=attack_model_inference(attack_model, mia_test_feature, mia_test_label, K, K_hat,N,device)
        tpl_values1.append(TPL)

    tpl_avg1 = sum(tpl_values1) / len(tpl_values1)


    return tpl_avg1



def topology_audit_based_sim_classifier_PGR_worst_case( selected_subgraphs, model, priv_adj, features, regen_edge, labels, device, seed):
    model = model.to(device)
    regen_edge = regen_edge.to(device)
    features = features.to(device)
    N = len(labels)

    if len(selected_subgraphs)==1:
        train_subgraph = selected_subgraphs[0]
        test_subgraphs = selected_subgraphs[0:]
    else:
        train_subgraph = selected_subgraphs[0]
        test_subgraphs = selected_subgraphs[1:]

    test_nodes_shadow, adj_subgraph_shadow = train_subgraph

    similarity_list = [cosine, euclidean, correlation, cityblock]

    A_hat = add_diagonal_and_normalize_edge(regen_edge, device)
    posterior = model.logist(features, A_hat).detach().cpu().numpy()

    mia_train_feature = []
    mia_train_label = []
    for u in tqdm(test_nodes_shadow):
        for v in test_nodes_shadow:
            if v > u:
                tmp_feature = [sim(posterior[u], posterior[v]) for sim in similarity_list]
                mia_train_feature.append(tmp_feature)
                mia_train_label.append([adj_subgraph_shadow[u][v]])

    attack_model=attack_step(mia_train_feature, mia_train_label, device)
    tpl_values1 = []

    for i, (test_nodes, adj_subgraph) in enumerate(test_subgraphs):

        K = torch.count_nonzero(adj_subgraph).item() // 2
        K_hat=k_hat_subgraph(test_nodes, regen_edge)

        mia_test_feature = []
        mia_test_label = []
        Knowledge=[]
        for u in tqdm(test_nodes):
            for v in test_nodes:
                if v > u:
                    tmp_feature = [sim(posterior[u], posterior[v]) for sim in similarity_list]
                    mia_test_feature.append(tmp_feature)
                    mia_test_label.append([adj_subgraph[u][v]])
                    Knowledge.append([regen_edge[u][v]])



        TPL=attack_model_inference_PGR_worst_case(attack_model, mia_test_feature, mia_test_label, K, K_hat,N,device,Knowledge)
        tpl_values1.append(TPL)

    tpl_avg1 = sum(tpl_values1) / len(tpl_values1)


    return tpl_avg1

