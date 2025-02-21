import numpy as np
import torch

from tqdm import tqdm

from TIAs.C_TIA.attack_model import attack_step, attack_model_inference, attack_model_inference_cluster, \
    topk_confidence_to_TPL, topk_confidence_to_TPL_PGR, confidence_to_LIA

from mask.add_diagonal_matrix import add_diagonal_and_normalize_edge
from scipy.spatial.distance import cosine, euclidean, correlation, cityblock

from tqdm import tqdm

from TIAs.C_TIA.attack_model import attack_step, attack_model_inference, attack_model_inference_cluster
from TIAs.util import topology_loss_sim, \
    euclidean_distance, chebyshev_distance, cosine_distance, k_hat_subgraph
from baseline.GAP_master.core.modules.node.em import EncoderModule
from baseline.GAP_master.train import compute_aggregations_dp
from mask.add_diagonal_matrix import add_diagonal_and_normalize_edge
from scipy.spatial.distance import cosine, euclidean, correlation, cityblock

def based_classifier(algorithm, selected_subgraphs, model, priv_adj, features, regen_edge, labels, device, seed):
    print("C_TIA begin -------------------------")
    model = model.to(device)
    regen_edge = regen_edge.to(device)
    features = features.to(device)

    train_subgraph = selected_subgraphs[-1]
    test_subgraphs = selected_subgraphs[0:]

    test_nodes_shadow, adj_subgraph_shadow = train_subgraph

    similarity_list = [cosine, euclidean, correlation, cityblock]

    if algorithm == 'LPGNet':
        posterior = model.logist(features, regen_edge, labels).detach().cpu().numpy()
    elif algorithm == 'PPRL':
        posterior = model.logist(features, regen_edge).detach().cpu().numpy()
    else:
        A_hat = add_diagonal_and_normalize_edge(regen_edge, device)
        posterior = model.logist(features, A_hat).detach().cpu().numpy()

    mia_train_feature = []
    mia_train_label = []
    test_nodes_shadow.sort(reverse=False)

    for i in range(len(test_nodes_shadow)):
        for j in range(i + 1, len(test_nodes_shadow)):
            u=test_nodes_shadow[i]
            v=test_nodes_shadow[j]
            tmp_feature = [sim(posterior[u], posterior[v]) for sim in similarity_list]
            mia_train_feature.append(tmp_feature)
            mia_train_label.append([adj_subgraph_shadow[u][v]])

    attack_model=attack_step(mia_train_feature, mia_train_label, device)
    tpl_values1 = []

    for i, (test_nodes, adj_subgraph) in enumerate(test_subgraphs):

        K = torch.count_nonzero(adj_subgraph).item() // 2
        K_hat=k_hat_subgraph(test_nodes, regen_edge) ## regen_edge is the G_hat

        mia_test_feature = []
        mia_test_label = []
        test_nodes.sort(reverse=False)

        for i in range(len(test_nodes)):
            for j in range(i + 1, len(test_nodes)):
                u = test_nodes[i]
                v = test_nodes[j]
                tmp_feature = [sim(posterior[u], posterior[v]) for sim in similarity_list]
                mia_test_feature.append(tmp_feature)
                mia_test_label.append([adj_subgraph[u][v]])


        P=attack_model_inference(attack_model, mia_test_feature,device)
        TPL=topk_confidence_to_TPL(P, mia_test_label, K, K_hat, device)
        tpl_values1.append(TPL)

    tpl_avg1 = sum(tpl_values1) / len(tpl_values1)

    print(f"C_TIA done: {tpl_avg1}")

    return tpl_avg1



def based_classifier_PGR(algorithm, selected_subgraphs, model, priv_adj, features, regen_edge, labels, device, seed):
    print("C_TIA_PGR begin -------------------------")
    model = model.to(device)
    regen_edge = regen_edge.to(device)
    features = features.to(device)
    N = len(labels)

    train_subgraph = selected_subgraphs[-1]
    test_subgraphs = selected_subgraphs[0:]

    test_nodes_shadow, adj_subgraph_shadow = train_subgraph


    similarity_list = [cosine, euclidean, correlation, cityblock]

    if algorithm == 'LPGNet':
        posterior = model.logist(features, regen_edge, labels).detach().cpu().numpy()
    elif algorithm == 'PPRL':
        posterior = model.logist(features, regen_edge).detach().cpu().numpy()
    else:
        A_hat = add_diagonal_and_normalize_edge(regen_edge, device)
        posterior = model.logist(features, A_hat).detach().cpu().numpy()

    mia_train_feature = []
    mia_train_label = []
    test_nodes_shadow.sort(reverse=False)
    for i in range(len(test_nodes_shadow)):
        for j in range(i + 1, len(test_nodes_shadow)):
            u=test_nodes_shadow[i]
            v=test_nodes_shadow[j]
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
        test_nodes.sort(reverse=False)

        for i in range(len(test_nodes)):
            for j in range(i + 1, len(test_nodes)):
                u = test_nodes[i]
                v = test_nodes[j]
                tmp_feature = [sim(posterior[u], posterior[v]) for sim in similarity_list]
                mia_test_feature.append(tmp_feature)
                mia_test_label.append([adj_subgraph[u][v]])
                Knowledge.append([regen_edge[u][v]])

        P=attack_model_inference(attack_model, mia_test_feature, device)
        TPL=topk_confidence_to_TPL_PGR(P, mia_test_label, K, K_hat, device,Knowledge,test_nodes)
        tpl_values1.append(TPL)

    tpl_avg1 = sum(tpl_values1) / len(tpl_values1)

    print(f"C_TIA_PGR done: {tpl_avg1}")

    return tpl_avg1


def based_classifier_GAP(selected_subgraphs,model,features,labels,adj_t,eps,hops,device,seed):
    print("M_TIA begin-------------------------")
    model = model.to(device)
    features = features.to(device)

    train_subgraph = selected_subgraphs[0]
    test_subgraphs = selected_subgraphs[0:]

    test_nodes_shadow, adj_subgraph_shadow = train_subgraph

    similarity_list = [cosine, euclidean, correlation, cityblock]

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

    posterior = model.logist(encoded_features).detach().cpu().numpy()

    mia_train_feature = []
    mia_train_label = []
    test_nodes_shadow.sort(reverse=False)
    for i in range(len(test_nodes_shadow)):
        for j in range(i + 1, len(test_nodes_shadow)):
            u=test_nodes_shadow[i]
            v=test_nodes_shadow[j]
            tmp_feature = [sim(posterior[u], posterior[v]) for sim in similarity_list]
            mia_train_feature.append(tmp_feature)
            mia_train_label.append([adj_subgraph_shadow[u][v]])

    attack_model = attack_step(mia_train_feature, mia_train_label, device)
    tpl_values1 = []

    for i, (test_nodes, adj_subgraph) in enumerate(test_subgraphs):

        K = torch.count_nonzero(adj_subgraph).item() // 2
        K_hat = K

        mia_test_feature = []
        mia_test_label = []
        test_nodes.sort(reverse=False)
        for i in range(len(test_nodes)):
            for j in range(i + 1, len(test_nodes)):
                u = test_nodes[i]
                v = test_nodes[j]
                tmp_feature = [sim(posterior[u], posterior[v]) for sim in similarity_list]
                mia_test_feature.append(tmp_feature)
                mia_test_label.append([adj_subgraph[u][v]])


        P=attack_model_inference(attack_model, mia_test_feature,device)
        TPL=topk_confidence_to_TPL(P, mia_test_label, K, K_hat, device)
        tpl_values1.append(TPL)

    tpl_avg1 = sum(tpl_values1) / len(tpl_values1)

    print(f"M_TIA done: {tpl_avg1}")

    return tpl_avg1


def LIA_based_classifier(algorithm, selected_subgraphs, model, priv_adj, features, regen_edge, labels, device, seed):
    print("C_TIA_LIA begin -------------------------")
    model = model.to(device)
    regen_edge = regen_edge.to(device)
    features = features.to(device)

    train_subgraph = selected_subgraphs[-1]
    test_subgraphs = selected_subgraphs[0:]

    test_nodes_shadow, adj_subgraph_shadow = train_subgraph

    similarity_list = [cosine, euclidean, correlation, cityblock]

    if algorithm == 'LPGNet':
        posterior = model.logist(features, regen_edge, labels).detach().cpu().numpy()
    elif algorithm == 'PPRL':
        posterior = model.logist(features, regen_edge).detach().cpu().numpy()
    else:
        A_hat = add_diagonal_and_normalize_edge(regen_edge, device)
        posterior = model.logist(features, A_hat).detach().cpu().numpy()

    mia_train_feature = []
    mia_train_label = []
    test_nodes_shadow.sort(reverse=False)

    for i in range(len(test_nodes_shadow)):
        for j in range(i + 1, len(test_nodes_shadow)):
            u=test_nodes_shadow[i]
            v=test_nodes_shadow[j]
            tmp_feature = [sim(posterior[u], posterior[v]) for sim in similarity_list]
            mia_train_feature.append(tmp_feature)
            mia_train_label.append([adj_subgraph_shadow[u][v]])

    attack_model=attack_step(mia_train_feature, mia_train_label, device)
    f1_values1 = []

    for i, (test_nodes, adj_subgraph) in enumerate(test_subgraphs):

        mia_test_feature = []
        mia_test_label = []
        test_nodes.sort(reverse=False)

        for i in range(len(test_nodes)):
            for j in range(i + 1, len(test_nodes)):
                u = test_nodes[i]
                v = test_nodes[j]
                tmp_feature = [sim(posterior[u], posterior[v]) for sim in similarity_list]
                mia_test_feature.append(tmp_feature)
                mia_test_label.append([adj_subgraph[u][v]])


        P=attack_model_inference(attack_model, mia_test_feature,device)
        F1=confidence_to_LIA(P, mia_test_label)
        f1_values1.append(F1)

    f1_values = sum(f1_values1) / len(f1_values1)

    print(f"C_TIA done: {f1_values}")

    return f1_values


def LIA_based_classifier_GAP(selected_subgraphs,model,features,labels,adj_t,eps,hops,device,seed):
    print("M_TIA begin-------------------------")
    model = model.to(device)
    features = features.to(device)

    train_subgraph = selected_subgraphs[0]
    test_subgraphs = selected_subgraphs[0:]

    test_nodes_shadow, adj_subgraph_shadow = train_subgraph

    similarity_list = [cosine, euclidean, correlation, cityblock]

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

    posterior = model.logist(encoded_features).detach().cpu().numpy()

    mia_train_feature = []
    mia_train_label = []
    test_nodes_shadow.sort(reverse=False)
    for i in range(len(test_nodes_shadow)):
        for j in range(i + 1, len(test_nodes_shadow)):
            u=test_nodes_shadow[i]
            v=test_nodes_shadow[j]
            tmp_feature = [sim(posterior[u], posterior[v]) for sim in similarity_list]
            mia_train_feature.append(tmp_feature)
            mia_train_label.append([adj_subgraph_shadow[u][v]])

    attack_model = attack_step(mia_train_feature, mia_train_label, device)
    f1_values1 = []

    for i, (test_nodes, adj_subgraph) in enumerate(test_subgraphs):


        mia_test_feature = []
        mia_test_label = []
        test_nodes.sort(reverse=False)
        for i in range(len(test_nodes)):
            for j in range(i + 1, len(test_nodes)):
                u = test_nodes[i]
                v = test_nodes[j]
                tmp_feature = [sim(posterior[u], posterior[v]) for sim in similarity_list]
                mia_test_feature.append(tmp_feature)
                mia_test_label.append([adj_subgraph[u][v]])


        P=attack_model_inference(attack_model, mia_test_feature,device)
        F1 = confidence_to_LIA(P, mia_test_label)
        f1_values1.append(F1)

    f1_values = sum(f1_values1) / len(f1_values1)

    print(f"C_TIA done: {f1_values}")
    return f1_values