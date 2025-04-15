import torch
from tqdm import tqdm


from TIAs.util import topology_loss_sim, \
    euclidean_distance, chebyshev_distance, cosine_distance, k_hat_subgraph, TOP_K_index_to_matrix, \
    TOP_K_index_to_matrix_PGR, topology_loss_sim_PGR
from baseline.GAP_master.core.modules.node.em import EncoderModule
from baseline.GAP_master.train import compute_aggregations_dp
from mask.add_diagonal_matrix import add_diagonal_and_normalize_edge
from scipy.spatial.distance import cosine, euclidean, correlation, cityblock



def based_metric(algorithm, selected_subgraphs, model, priv_adj, features, regen_edge, labels, device, seed):
    print("M_TIA begin -------------------------")
    model = model.to(device)
    regen_edge = regen_edge.to(device)
    features = features.to(device)
    N = len(labels)
    similarity_list = [cosine_distance, chebyshev_distance,euclidean_distance]

    tpl_values1 = []

    for i, (test_nodes, adj_subgraph) in enumerate(selected_subgraphs):
        K = torch.count_nonzero(adj_subgraph).item() // 2
     #   K_hat=k_hat_subgraph(test_nodes, regen_edge)  # regen_edge is the G_hat

        if algorithm == 'LPGNet':
            posterior = model.logist(features, regen_edge, labels).detach().cpu().numpy()
        elif algorithm == 'PPRL':
            posterior = model.logist(features, regen_edge).detach().cpu().numpy()
        else:
            A_hat = add_diagonal_and_normalize_edge(regen_edge, device)
            posterior = model.logist(features, A_hat).detach().cpu().numpy()

        TPL_list=[]
        test_nodes.sort(reverse=False)
        for similarity in similarity_list:
            dist = torch.zeros((N, N))
            for i in range(len(test_nodes)):
                for j in range(i + 1, len(test_nodes)):
                    u = test_nodes[i]
                    v = test_nodes[j]
                    dist[u][v] = torch.tensor(similarity(posterior[u], posterior[v]), dtype=torch.float32)

            binary_matrix = TOP_K_index_to_matrix(dist, K,test_nodes)
            item1= topology_loss_sim(adj_subgraph, binary_matrix, K, K)
            TPL_list.append(item1)

        tpl_values1.append(max(TPL_list))


    tpl_avg1 = sum(tpl_values1) / len(tpl_values1)

    print(f"M_TIA done: {tpl_avg1}")

    return tpl_avg1


def based_metric_PGR(algorithm, selected_subgraphs, model, priv_adj, features, regen_edge, labels, device, seed):
    print("M_TIA_PGR begin -------------------------")
    model = model.to(device)
    regen_edge = regen_edge.to(device)
    features = features.to(device)
    N = len(labels)
    similarity_list = [cosine_distance, chebyshev_distance,euclidean_distance]

    tpl_values1 = []

    for i, (test_nodes, adj_subgraph) in enumerate(selected_subgraphs):


        K = torch.count_nonzero(adj_subgraph).item()
        K = int(K / 2)
     #   K_hat=k_hat_subgraph(test_nodes, regen_edge)

        if algorithm == 'LPGNet':
            posterior = model.logist(features, regen_edge, labels).detach().cpu().numpy()
        elif algorithm == 'PPRL':
            posterior = model.logist(features, regen_edge).detach().cpu().numpy()
        else:
            A_hat = add_diagonal_and_normalize_edge(regen_edge, device)  # 进行自连接后正则化
            posterior = model.logist(features, A_hat).detach().cpu().numpy()

        TPL_list=[]
        test_nodes.sort(reverse=False)
        for similarity in similarity_list:
            dist = torch.zeros((N, N))
            for i in range(len(test_nodes)):
                for j in range(i + 1, len(test_nodes)):
                    u = test_nodes[i]
                    v = test_nodes[j]
                    dist[u][v] = torch.tensor(similarity(posterior[u], posterior[v]), dtype=torch.float32)

            binary_matrix = TOP_K_index_to_matrix_PGR(dist, K,regen_edge,test_nodes)


            item1= topology_loss_sim_PGR(adj_subgraph, binary_matrix, K, K,test_nodes)
            TPL_list.append(item1)
        tpl_values1.append(max(TPL_list))


    tpl_avg1 = sum(tpl_values1) / len(tpl_values1)

    print(f"M_TIA_PGR done: {tpl_avg1}")

    return tpl_avg1


def based_metric_GAP(selected_subgraphs,model, features, labels, adj_t, eps, hops, device, seed):
    print("M_TIA begin -------------------------")
    model = model.to(device)
    features = features.to(device)
    N = len(labels)

    tpl_values = []
    similarity_list = [cosine_distance, chebyshev_distance,euclidean_distance]

    for i, (test_nodes, adj_subgraph) in enumerate(selected_subgraphs):

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

        posterior = model.logist(encoded_features).detach().cpu().numpy()

        TPL_list = []
        for similarity in similarity_list:
            dist = torch.zeros((N, N))
            for u in test_nodes:
                for v in test_nodes:
                    if v > u:
                        dist[u][v] = torch.tensor(similarity(posterior[u], posterior[v]), dtype=torch.float32)

            binary_matrix = TOP_K_index_to_matrix(dist, K_hat,test_nodes)

            TPL = topology_loss_sim(adj_subgraph, binary_matrix, K,K_hat)
            TPL_list.append(TPL)
        tpl_values.append(max(TPL_list))

    tpl_avg = sum(tpl_values) / len(tpl_values)
    print(f"M_TIA done: {tpl_avg}")

    return tpl_avg
