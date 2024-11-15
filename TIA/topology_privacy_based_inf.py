import copy

import torch
from tqdm import tqdm

from TIA.util import process_matrix, construct_subgraph, overlaping_edges, topology_loss_sim, \
    construct_subgraph_for_GAP, k_hat_subgraph, cluster_matrix, process_matrix_PGR_worst_case
def link_infer_node_implantation_3_hop( target_u, regen_matrix, model, features, labels, device):
    est_features = copy.deepcopy(features)
    est_features = torch.cat((est_features, torch.zeros(1, est_features.shape[1]).to(device)), dim=0)
    est_features = torch.cat((est_features, torch.zeros(1, est_features.shape[1]).to(device)), dim=0)


    N = regen_matrix.shape[0]
    new_size = N + 2
    new_dense_matrix = torch.zeros((new_size, new_size), device=device)

    new_dense_matrix[:N, :N] = regen_matrix

    adv_idx_v = N
    adv_idx_g = N + 1

    # 将新节点与目标节点的连接关系添加到矩阵中
    new_dense_matrix[target_u, adv_idx_v] = 1
    new_dense_matrix[adv_idx_v, target_u] = 1
    new_dense_matrix[adv_idx_v, adv_idx_g] = 1
    new_dense_matrix[adv_idx_g, adv_idx_v] = 1

    model.eval()

    logist = model.logist(est_features, new_dense_matrix).detach()

    return logist

def link_infer_node_implantation_2_hop(target_u,regen_matrix, model,features,labels ,device):
    est_features = copy.deepcopy(features)
    est_features = torch.cat((est_features, torch.ones(1, est_features.shape[1]).to(device)),
                             dim=0)  # cat dim 0: (nodes, feature)

    N = regen_matrix.shape[0]
    new_size = N + 1
    new_dense_matrix = torch.zeros((new_size, new_size), device=device)

    new_dense_matrix[:N, :N] = regen_matrix

    adv_idx_u = N

    new_dense_matrix[target_u, adv_idx_u] = 1
    new_dense_matrix[adv_idx_u, target_u] = 1
    model.eval()
    logist = model.logist(est_features, new_dense_matrix).detach()

    return logist

def link_infer_1_hop(target_u,regen_matrix, model,features,labels ,device):
    influence = 1.1
    pert = torch.zeros_like(features)
    pert[target_u] = features[target_u] * influence
    model.eval()
    logist=model.logist(features + pert, regen_matrix).detach()


    return logist




def topology_audit_based_inf(selected_subgraphs, model, priv_adj, features, regen_edge, labels, device, hops,seed):
    model = model.to(device)
    regen_edge = regen_edge.to(device)
    features = features.to(device)
    N = len(labels)


    tpl_values1 = []

    for i, (test_nodes, adj_subgraph) in enumerate(selected_subgraphs):

        K = torch.count_nonzero(adj_subgraph).item() // 2
        K_hat=k_hat_subgraph(test_nodes, regen_edge)

        origial_logists = model.logist(features, regen_edge).detach()

        if hops==2:
            origial_logists = torch.cat((origial_logists, torch.zeros(1, origial_logists.shape[1]).to(origial_logists.device)), dim=0)
        elif hops==3:
            origial_logists = torch.cat((origial_logists, torch.zeros(1, origial_logists.shape[1]).to(origial_logists.device)), dim=0)
            origial_logists = torch.cat((origial_logists, torch.zeros(1, origial_logists.shape[1]).to(origial_logists.device)), dim=0)

        influence_val = torch.zeros((N, N))

        for u in tqdm(test_nodes):
            if hops==1:
                new_logists = link_infer_1_hop( u, regen_edge, model, features, labels, device)
            elif hops==2:
                new_logists = link_infer_node_implantation_2_hop( u, regen_edge, model, features, labels, device)
            elif hops==3:
                new_logists = link_infer_node_implantation_3_hop( u, regen_edge, model, features, labels, device)
            grad = new_logists - origial_logists
            for v in test_nodes:
                if v > u:
                    influence_tempo = torch.zeros_like(influence_val)
                    influence_tempo[v][u] = grad[v].norm().item()
                    influence_val = influence_val + influence_tempo

        attacked_adj = process_matrix(influence_val, K)

        TPL = topology_loss_sim(adj_subgraph, attacked_adj, K,K_hat)
        tpl_values1.append(TPL)


    tpl_avg1 = sum(tpl_values1) / len(tpl_values1)
    return tpl_avg1

def topology_audit_based_inf_worst_case(selected_subgraphs, model, priv_adj, features, regen_edge, labels, device,hops, seed):
    model = model.to(device)
    regen_edge = regen_edge.to(device)
    features = features.to(device)
    N = len(labels)


    tpl_values1 = []

    for i, (test_nodes, adj_subgraph) in enumerate(selected_subgraphs):

        K = torch.count_nonzero(adj_subgraph).item() // 2
        K_hat=k_hat_subgraph(test_nodes, regen_edge)

        origial_logists = model.logist(features, regen_edge).detach()

        if hops==2:
            origial_logists = torch.cat((origial_logists, torch.zeros(1, origial_logists.shape[1]).to(origial_logists.device)), dim=0)
        elif hops==3:
            origial_logists = torch.cat((origial_logists, torch.zeros(1, origial_logists.shape[1]).to(origial_logists.device)), dim=0)
            origial_logists = torch.cat((origial_logists, torch.zeros(1, origial_logists.shape[1]).to(origial_logists.device)), dim=0)

        influence_val = torch.zeros((N, N))

        for u in tqdm(test_nodes):
            if hops==1:
                new_logists = link_infer_1_hop( u, regen_edge, model, features, labels, device)
            elif hops==2:
                new_logists = link_infer_node_implantation_2_hop( u, regen_edge, model, features, labels, device)
            elif hops==3:
                new_logists = link_infer_node_implantation_3_hop( u, regen_edge, model, features, labels, device)
            grad = new_logists - origial_logists
            for v in test_nodes:
                if v > u:
                    influence_tempo = torch.zeros_like(influence_val)
                    influence_tempo[v][u] = grad[v].norm().item()
                    influence_val = influence_val + influence_tempo

        attacked_adj = process_matrix_PGR_worst_case(influence_val, K,regen_edge)

        TPL = topology_loss_sim(adj_subgraph, attacked_adj, K,K_hat)
        tpl_values1.append(TPL)


    tpl_avg1 = sum(tpl_values1) / len(tpl_values1)


    return tpl_avg1