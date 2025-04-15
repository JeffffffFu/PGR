import copy

import torch
from tqdm import tqdm

from TIAs.I_TIA.Influence import link_infer_node_implantation_2_hop, link_infer_1_hop, \
    link_infer_node_implantation_3_hop, link_infer_node_implantation_2_hop_GAP
from attacks.util import construct_edge_sets_from_random_subgraph, get_edge_sets_among_nodes, \
    compute_and_save
from TIAs.util import construct_subgraph, overlaping_edges, topology_loss_sim, \
    construct_subgraph_for_GAP, k_hat_subgraph, cluster_matrix, get_edges, TOP_K_index_to_matrix, \
    TOP_K_index_to_matrix_PGR, topology_loss_sim_PGR
from baseline.GAP_master.core.modules.node.em import EncoderModule
from baseline.GAP_master.train import compute_aggregations_dp
from mask.add_diagonal_matrix import add_diagonal_and_normalize_edge
from torch_sparse import SparseTensor




def based_inf(algorithm, selected_subgraphs, model, priv_adj, features, regen_edge, labels, device,hops, seed):
    print("I_MIA begin-------------------------")
    model = model.to(device)
    regen_edge = regen_edge.to(device)
    features = features.to(device)
    N = len(labels)

    # if algorithm =='LPGNet':
    #     selected_subgraphs=selected_subgraphs[0]
    tpl_values1 = []
    A_hat = add_diagonal_and_normalize_edge(regen_edge, device)

    for i, (test_nodes, adj_subgraph) in enumerate(selected_subgraphs):

        K = torch.count_nonzero(adj_subgraph).item() // 2
     #   K_hat=k_hat_subgraph(test_nodes, regen_edge)

        if algorithm == 'LPGNet':
            origial_logists = model.logist(features, regen_edge, labels).detach()
        elif algorithm == 'privGraph':
            origial_logists = model.logist(features, A_hat).detach()
        else:
            origial_logists = model.logist(features, regen_edge).detach()


        if hops==2:
            origial_logists = torch.cat((origial_logists, torch.zeros(1, origial_logists.shape[1]).to(origial_logists.device)), dim=0)
        elif hops==3:
            origial_logists = torch.cat((origial_logists, torch.zeros(1, origial_logists.shape[1]).to(origial_logists.device)), dim=0)
            origial_logists = torch.cat((origial_logists, torch.zeros(1, origial_logists.shape[1]).to(origial_logists.device)), dim=0)

        influence_val = torch.zeros((N, N))

        test_nodes.sort(reverse=False)

        for i in tqdm(range(len(test_nodes))):
            u = test_nodes[i]
            if hops == 1:
                new_logists = link_infer_1_hop(u, regen_edge, model, features, labels, device)
            elif hops == 2:
                new_logists = link_infer_node_implantation_2_hop(algorithm,u, regen_edge, model, features, labels, device)
            elif hops == 3:
                new_logists = link_infer_node_implantation_3_hop(algorithm,u, regen_edge, model, features, labels, device)
            grad = new_logists - origial_logists
            for j in range(i + 1, len(test_nodes)):
                v = test_nodes[j]
                influence_tempo = torch.zeros_like(influence_val)
                influence_tempo[v][u] = grad[v].norm().item()
                influence_val = influence_val + influence_tempo


        attacked_adj = TOP_K_index_to_matrix(influence_val, K,test_nodes)

        item1 = topology_loss_sim(adj_subgraph, attacked_adj, K,K)

        tpl_values1.append(item1)

    tpl_avg1 = sum(tpl_values1) / len(tpl_values1)

    print(f"I_MIA done: {tpl_avg1}")

    return tpl_avg1


def based_random(algorithm, selected_subgraphs, model, priv_adj, features, regen_edge, labels, device,hops, seed):
    print("I_MIA begin-------------------------")
    model = model.to(device)
    regen_edge = regen_edge.to(device)
    features = features.to(device)
    N = len(labels)


    tpl_values1 = []

    for i, (test_nodes, adj_subgraph) in enumerate(selected_subgraphs):

        K = torch.count_nonzero(adj_subgraph).item() // 2
        K_hat=k_hat_subgraph(test_nodes, regen_edge)

        influence_val = torch.zeros((N, N))

        noise = torch.randn_like(influence_val)

        influence_val += noise

        attacked_adj = TOP_K_index_to_matrix(influence_val, K_hat,test_nodes)
        item1 = topology_loss_sim(adj_subgraph, attacked_adj, K,K_hat)
        tpl_values1.append(item1)

    tpl_avg1 = sum(tpl_values1) / len(tpl_values1)

    print(f"I_MIA done: {tpl_avg1}")

    return tpl_avg1

def based_inf_PGR(algorithm,selected_subgraphs, model, priv_adj, features, regen_edge, labels, device,hops, seed):
    print("I_MIA_PGR begin-------------------------")

    model = model.to(device)
    regen_edge = regen_edge.to(device)
    features = features.to(device)
    N = len(labels)


    tpl_values1 = []

    for i, (test_nodes, adj_subgraph) in enumerate(selected_subgraphs):

        K = torch.count_nonzero(adj_subgraph).item() // 2

        origial_logists = model.logist(features, regen_edge).detach()

        if hops==2:
            origial_logists = torch.cat((origial_logists, torch.zeros(1, origial_logists.shape[1]).to(origial_logists.device)), dim=0)
        elif hops==3:
            origial_logists = torch.cat((origial_logists, torch.zeros(1, origial_logists.shape[1]).to(origial_logists.device)), dim=0)
            origial_logists = torch.cat((origial_logists, torch.zeros(1, origial_logists.shape[1]).to(origial_logists.device)), dim=0)

        influence_val = torch.zeros((N, N))
        test_nodes.sort(reverse=False)

        for i in tqdm(range(len(test_nodes))):
            u = test_nodes[i]
            if hops==1:
                new_logists = link_infer_1_hop( u, regen_edge, model, features, labels, device)
            elif hops==2:
                new_logists = link_infer_node_implantation_2_hop(algorithm, u, regen_edge, model, features, labels, device)
            elif hops==3:
                new_logists = link_infer_node_implantation_3_hop(algorithm, u, regen_edge, model, features, labels, device)
            grad = new_logists - origial_logists
            for j in range(i + 1, len(test_nodes)):
                v = test_nodes[j]
                influence_tempo = torch.zeros_like(influence_val)
                influence_tempo[v][u] = grad[v].norm().item()
                influence_val = influence_val + influence_tempo

        attacked_adj = TOP_K_index_to_matrix_PGR(influence_val, K,regen_edge,test_nodes)

        TPL = topology_loss_sim_PGR(adj_subgraph, attacked_adj, K,K,test_nodes)
        tpl_values1.append(TPL)


    tpl_avg1 = sum(tpl_values1) / len(tpl_values1)

    print(f"I_MIA_PGR done: {tpl_avg1}")

    return tpl_avg1



def based_inf_GAP(selected_subgraphs,model, features, labels, adj_t, eps, hops, device, seed):
    print("I_MIA_GAP begin-------------------------")
    model = model.to(device)
    features = features.to(device)

    tpl_values1 = []
    tpl_values2 = []

    for i, (test_nodes, adj_subgraph) in tqdm(enumerate(selected_subgraphs)):

        K = torch.count_nonzero(adj_subgraph).item() // 2
        K_hat = K

        Encoder = EncoderModule(
            num_classes=labels.max().item() + 1,
            hidden_dim=16,
            encoder_layers=2,
            head_layers=1,
            normalize=True,
            activation_fn=torch.selu_,
            dropout=0.0,
            batch_norm=True,
        )

        encoded_features = Encoder.predict2(features.cpu())
        encoded_features = compute_aggregations_dp(eps, hops, encoded_features, adj_t)
        encoded_features = encoded_features.to(device)

        influence_val = torch.zeros((len(labels), len(labels)))

        origial_logists = model.logist(encoded_features).detach()
        origial_logists = torch.cat(
            (origial_logists, torch.zeros(1, origial_logists.shape[1]).to(origial_logists.device)), dim=0
        )
        test_nodes.sort(reverse=False)

        for i in tqdm(range(len(test_nodes))):
            u = test_nodes[i]
            new_logists = link_infer_node_implantation_2_hop_GAP(u, adj_t, model, features, labels, eps, hops, device)
            grad = new_logists - origial_logists
            for j in range(len(test_nodes)):
                v = test_nodes[j]
                influence_tempo = torch.zeros_like(influence_val)
                influence_tempo[v][u] = grad[v].norm().item()
                influence_val += influence_tempo

        attacked_adj = TOP_K_index_to_matrix(influence_val, K,test_nodes)
        #  K_hat = torch.count_nonzero(torch.tensor(attacked_adj)).item() // 2

        item1= topology_loss_sim(adj_subgraph, attacked_adj, K, K_hat)
        tpl_values1.append(item1)


    tpl_avg1 = sum(tpl_values1) / len(tpl_values1)

    print(f"I_MIA_GAP done: {tpl_avg1}")

    return tpl_avg1

def LIA_based_inf(algorithm, selected_subgraphs, model, priv_adj, features, regen_edge, labels, device, seed):
    print("开始进行 topology_audit_based_inf -------------------------")
    model = model.to(device)
    regen_edge = regen_edge.to(device)
    features = features.to(device)
    N = len(labels)
    F1_values = []

    A_hat = add_diagonal_and_normalize_edge(regen_edge, device)

    for i, (test_nodes, adj_subgraph) in enumerate(selected_subgraphs):
        print(f"attack {i + 1} subgraph")

        if algorithm == 'LPGNet':
            origial_logists = model.logist(features, regen_edge, labels).detach()
        elif algorithm == 'PPRL':
            origial_logists = model.logist(features, regen_edge).detach()
        elif algorithm == 'privGraph':
            origial_logists = model.logist(features, A_hat).detach()
        else:
            origial_logists = model.logist(features, regen_edge).detach()

        influence_val = torch.zeros((N, N))

        for u in tqdm(test_nodes):
            pert = torch.zeros_like(features)
            pert[u] = features[u] * 0.1

            if algorithm == 'LPGNet':
                new_logists = model.logist(features + pert, regen_edge, labels).detach()
            elif algorithm == 'privGraph':
                new_logists = model.logist(features + pert, A_hat).detach()
            else:
                new_logists = model.logist(features + pert, regen_edge).detach()
            grad = new_logists - origial_logists
            for v in test_nodes:
                if v > u:
                    influence_tempo = torch.zeros_like(influence_val)
                    influence_tempo[v][u] = grad[v].norm().item()
                    influence_val = influence_val + influence_tempo

        attacked_adj = cluster_matrix(influence_val)
        matrix = adj_subgraph + attacked_adj
        C = torch.sum(torch.eq(matrix, 2))
        attacked_adj_tensor = torch.tensor(attacked_adj)
        precision = C / torch.count_nonzero(attacked_adj_tensor)
        recall = C / (torch.count_nonzero(adj_subgraph) / 2)
        print(C,precision,recall)
        F1 = 2 * precision * recall / (precision + recall + 0.0001)
        F1_values.append(F1)

        print(f" F1 value: {F1}")


    F1_avg = sum(F1_values) / len(F1_values)
    print(f"LIA_based_inf F1_avg : {F1_avg}")
#
    return F1_avg




def LIA_based_inf_GAP(selected_subgraphs,model, features, labels, adj_t, eps, hops, device, seed):
    print("topology_audit_based_inf_GAP -------------------------")
    model = model.to(device)
    features = features.to(device)

    F1_values = []

    for i, (test_nodes, adj_subgraph) in enumerate(selected_subgraphs):
        print(f"attack{i + 1} subgraph")


        Encoder = EncoderModule(
            num_classes=labels.max().item() + 1,
            hidden_dim=16,
            encoder_layers=2,
            head_layers=1,
            normalize=True,
            activation_fn=torch.selu_,
            dropout=0.0,
            batch_norm=True,
        )

        encoded_features = Encoder.predict2(features.cpu())
        encoded_features = compute_aggregations_dp(eps, hops, encoded_features, adj_t)
        encoded_features = encoded_features.to(device)

        influence_val = torch.zeros((len(labels), len(labels)))

        origial_logists = model.logist(encoded_features).detach()
        origial_logists = torch.cat(
            (origial_logists, torch.zeros(1, origial_logists.shape[1]).to(origial_logists.device)), dim=0
        )

        for u in tqdm(test_nodes):
            new_logists = link_infer_node_implantation_2_hop_GAP(u, adj_t, model, features, labels, eps, hops, device)
            grad = new_logists - origial_logists
            for v in test_nodes:
                if v > u:
                    influence_tempo = torch.zeros_like(influence_val)
                    influence_tempo[v][u] = grad[v].norm().item()
                    influence_val += influence_tempo

        attacked_adj = cluster_matrix(influence_val)
        matrix = adj_subgraph + attacked_adj
        C = torch.sum(torch.eq(matrix, 2))
        attacked_adj_tensor = torch.tensor(attacked_adj)
        precision = C / torch.count_nonzero(attacked_adj_tensor)
        recall = C / (torch.count_nonzero(adj_subgraph) / 2)
        F1 = 2 * precision * recall / (precision + recall)
        F1_values.append(F1)

        print(f"F1 value: {F1}")
    F1_avg = sum(F1_values) / len(F1_values)
    print(f"topology_audit_based_inf F1_avg : {F1_avg}")

    return F1_avg
