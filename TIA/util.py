from collections import Counter

import numpy as np
import torch
from torch_geometric.utils import to_networkx
import community as community_louvain  # The Louvain method is implemented in this module
import networkx as nx
from sklearn.cluster import KMeans



def cluster_matrix(influence_val):
    # Step 1: Save the original shape and identify non-zero values
    original_shape = influence_val.shape
    non_zero_indices = influence_val != 0
    non_zero_values = influence_val[non_zero_indices].reshape(-1, 1)

    # Step 2: Apply KMeans clustering to only the non-zero values
    kmeans = KMeans(n_clusters=2, random_state=0)
    clustered_matrix = np.zeros_like(influence_val, dtype=int)

    print(len(non_zero_values))
    if len(non_zero_values)<2:
        return clustered_matrix
    else:
        kmeans.fit(non_zero_values)
    labels = kmeans.labels_


    # Step 4: Determine which cluster has the larger centroid and set its values to 1
    if kmeans.cluster_centers_[0] > kmeans.cluster_centers_[1]:
        clustered_matrix[non_zero_indices] = (labels == 0).astype(int)  # Set cluster 0 as 1
    else:
        clustered_matrix[non_zero_indices] = (labels == 1).astype(int)  # Set cluster 1 as 1

    return clustered_matrix



def process_matrix_PGR_worst_case(matrix, K, adj_subgraph):
    matrix_upper = np.triu(matrix)
    matrix_lower = np.tril(matrix, -1)
    matrix_upper += matrix_lower.T
    matrix = np.triu(matrix_upper)


    upper_triangle_indices = np.triu_indices(matrix.shape[0], k=1)
    upper_triangle_values = matrix[upper_triangle_indices]

    adj_subgraph_np = adj_subgraph.cpu().numpy() if isinstance(adj_subgraph, torch.Tensor) else adj_subgraph
    adj_subgraph_upper_values = adj_subgraph_np[upper_triangle_indices]

    valid_indices = adj_subgraph_upper_values == 0
    filtered_values = upper_triangle_values[valid_indices]
    filtered_indices = (upper_triangle_indices[0][valid_indices], upper_triangle_indices[1][valid_indices])

    top_k_indices = np.argpartition(filtered_values, -K)[-K:]

    top_k_values_indices = (filtered_indices[0][top_k_indices],
                            filtered_indices[1][top_k_indices])

    binary_matrix = np.zeros_like(matrix)
    binary_matrix[top_k_values_indices] = 1

    binary_matrix += binary_matrix.T

    return binary_matrix


def process_matrix(matrix, K):
    matrix_upper = np.triu(matrix)
    matrix_lower = np.tril(matrix, -1)
    matrix_upper += matrix_lower.T
    matrix = np.triu(matrix_upper)

    upper_triangle_indices = np.triu_indices(matrix.shape[0], k=1)
    upper_triangle_values = matrix[upper_triangle_indices]
    top_k_indices = np.argpartition(upper_triangle_values, -K)[-K:]  #max

    top_k_values_indices = (upper_triangle_indices[0][top_k_indices],
                            upper_triangle_indices[1][top_k_indices])


    binary_matrix = np.zeros_like(matrix)
    binary_matrix[top_k_values_indices] = 1

    binary_matrix += binary_matrix.T

    return binary_matrix

def k_hat_subgraph(test_nodes,ren_adj):
    k_hat=0
    for u in test_nodes:
        for v in test_nodes:
            if v>u:
                k_hat+=ren_adj[u][v]
    return int(k_hat)

def construct_multiple_subgraphs(G):
    num_communities =5
    partition = community_louvain.best_partition(G)

    value_counts = Counter(partition.values())
    value_counts_dict = dict(value_counts)

    selected_communities = []
    selected_subgraphs = []
    print(value_counts)
    for _ in range(num_communities):
        community_id = -1
        for key, value in value_counts_dict.items():
            if key not in selected_communities and 100 <= value <= 200:
                community_id = key
                break

        if community_id == -1:  # If no community found in the 100-200 range, look in the 200-600 range
            for key, value in value_counts_dict.items():
                if key not in selected_communities and 20 <= value <= 200:
                    community_id = key
                    break

        if community_id != -1:
            selected_communities.append(community_id)
            index_subgraph = [node for node, comm_id in partition.items() if comm_id == community_id]
            subgraph = G.subgraph(index_subgraph)
            edges_list = list(subgraph.edges())

            # Step 5: Create a full adjacency matrix for the subgraph
            original_size = len(G.nodes())
            full_adj = torch.zeros((original_size, original_size), dtype=torch.float32)
            for i, j in edges_list:
                full_adj[i, j] = 1
                full_adj[j, i] = 1

            # Append the results
            selected_subgraphs.append((index_subgraph, full_adj))

            print(f"Selected community ID: {community_id} with node count: {len(index_subgraph)}")

        if len(selected_subgraphs) == num_communities:
            break

    return selected_subgraphs

def construct_subgraph(data):
    # Step 2: Convert the PyTorch Geometric data to a NetworkX graph
    G = to_networkx(data, to_undirected=True)

    partition = community_louvain.best_partition(G)

    value_counts = Counter(partition.values())

    value_counts_dict = dict(value_counts)
    community_id = -1
    print(value_counts_dict)



    if community_id == -1:
        for key, value in value_counts_dict.items():
            if 100 <= value <= 200:
                community_id = key
    if community_id == -1:
        for key, value in value_counts_dict.items():
            if 200 <= value <= 600:
                community_id = key

    print(f'community_id: {community_id}')

    index_subgraph = [key for key, value in partition.items() if value == community_id]

    subgraph = G.subgraph(index_subgraph)

    edges_list = list(subgraph.edges())

    original_size = len(G.nodes())

    full_adj = torch.zeros((original_size, original_size), dtype=torch.float32)

    for i,j in edges_list:
            full_adj[i, j] = 1
            full_adj[j, i] = 1

    return index_subgraph,full_adj,community_id



def construct_subgraph_shadow(data,community_id_attack):
    # Step 2: Convert the PyTorch Geometric data to a NetworkX graph
    G = to_networkx(data, to_undirected=True)

    # Step 3: Apply the Louvain community detection algorithm
    # This returns a dictionary where the key is the node and the value is the community it belongs to
    partition = community_louvain.best_partition(G)

    # Count occurrences of each value  每个社区的节点数
    value_counts = Counter(partition.values())

    # Convert to a dictionary for better readability (optional)
    value_counts_dict = dict(value_counts)
    # Step 4: Extract subgraphs based on the detected communities
    community_id = -1
    print(value_counts_dict)

    if community_id == -1:
        for key, value in value_counts_dict.items():
            if key != community_id_attack and 100 <= value <= 200:
                community_id = key
    if community_id == -1:
        for key, value in value_counts_dict.items():
            if key != community_id_attack and 10 <= value <= 100:
                community_id = key


    print(f'community_id: {community_id}')

    index_subgraph = [key for key, value in partition.items() if value == community_id]

    subgraph = G.subgraph(index_subgraph)

    edges_list = list(subgraph.edges())

    original_size = len(G.nodes())

    full_adj = torch.zeros((original_size, original_size), dtype=torch.float32)

    for i,j in edges_list:
            full_adj[i, j] = 1
            full_adj[j, i] = 1

    return index_subgraph,full_adj


def construct_subgraph_for_GAP_shadow(sparse_tensor,community_id_attack):
    #我们拿一个子图
    # Step 1: Convert SparseTensor to networkx graph
    rows = sparse_tensor.storage.row()
    cols = sparse_tensor.storage.col()
    edges = list(zip(rows.tolist(), cols.tolist()))  # 将行列组合成边的列表

    G = nx.Graph()  # 初始化无向图
    G.add_edges_from(edges)  # 添加边

    # Step 2: Apply the Louvain community detection algorithm
    partition = community_louvain.best_partition(G)

    # Count occurrences of each value  每个社区的节点数
    value_counts = Counter(partition.values())

    # Convert to a dictionary for better readability (optional)
    value_counts_dict = dict(value_counts)

    # Step 4: Extract subgraphs based on the detected communities
    # 查找第一个 节点数 在 50-200 之间的 社区
    community_id = -1
    print(value_counts_dict)
    for key, value in value_counts_dict.items():
        if 50 <= value <= 100 and key!=community_id_attack:
            community_id = key
    if community_id == -1:
        for key, value in value_counts_dict.items():
            if 100 <= value <= 200 and key!=community_id_attack:
                community_id = key
    if community_id == -1:
        for key, value in value_counts_dict.items():
            if 200 <= value <= 600 and key!=community_id_attack:
                community_id = key
    #largest_community = value_counts.most_common(1)[0][0]  # Get the community label with the most nodes
    index_subgraph = [key for key, value in partition.items() if value == community_id]
    # print(f'Nodes in largest community: {index_subgraph}')

    # Step 5: 获取这些节点组成的子图
    subgraph = G.subgraph(index_subgraph)

    # Step 6: 获取子图中的边 (节点对)
    edges_list = list(subgraph.edges())

    # Step 5: 生成和原图 G 一样大小的邻接矩阵
    original_size = len(G.nodes())  # 原图大小 (节点数)

    # 初始化一个全零矩阵，形状为 (original_size, original_size)
    full_adj = torch.zeros((original_size, original_size), dtype=torch.float32)

    # 在对应的 subgraph index 上将相应的位置置为 1
    for i,j in edges_list:
            full_adj[i, j] = 1
            full_adj[j, i] = 1

 #   print(f'subgraph_adj: {subgraph_adj}')

    return index_subgraph,full_adj,community_id

def construct_subgraph_for_GAP(sparse_tensor):
    rows = sparse_tensor.storage.row()
    cols = sparse_tensor.storage.col()
    edges = list(zip(rows.tolist(), cols.tolist()))

    G = nx.Graph()
    G.add_edges_from(edges)

    partition = community_louvain.best_partition(G)

    value_counts = Counter(partition.values())

    value_counts_dict = dict(value_counts)

    community_id = -1
    print(value_counts_dict)
    for key, value in value_counts_dict.items():
        if 100 <= value <= 100:
            community_id = key
    if community_id == -1:
        for key, value in value_counts_dict.items():
            if 200 <= value <= 600:
                community_id = key
    index_subgraph = [key for key, value in partition.items() if value == community_id]

    subgraph = G.subgraph(index_subgraph)

    edges_list = list(subgraph.edges())

    original_size = len(G.nodes())

    full_adj = torch.zeros((original_size, original_size), dtype=torch.float32)

    for i,j in edges_list:
            full_adj[i, j] = 1
            full_adj[j, i] = 1

    return index_subgraph,full_adj,community_id


def topology_loss_sim(subgraph_adj, binary_matrix,K,K_hat):
    N=subgraph_adj.shape[0]
    matrix =subgraph_adj+binary_matrix
    C=torch.sum(torch.eq(matrix, 2))
    C=C //2
    item_1=C/(K+K_hat-C)
    return item_1

def topology_loss_sim_worst_case(subgraph_adj, binary_matrix,K,K_hat):
    N=subgraph_adj.shape[0]
    matrix =subgraph_adj+binary_matrix
    C=torch.sum(torch.eq(matrix, 2))
    C=C //2
    item_1=C/(K+K_hat-C)
    item_2=(K-C)/((N**2-N)/2-K_hat+C)

    return max(item_1,item_2)

def topology_sim_classifier(C,K,K_hat):
    TPL=C/(K+K_hat-C)
    return TPL


def overlaping_edges(private_adj,synthetic_adj):
    device = private_adj.device
    synthetic_adj = synthetic_adj.to(device)
    C=private_adj+synthetic_adj
    count = torch.sum(torch.eq(C, 2))

    return count

def cosine_distance(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    cosine_sim = dot_product / (norm_vec1 * norm_vec2)
    return cosine_sim


def euclidean_distance(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))


def chebyshev_distance(vec1, vec2):
    return np.max(np.abs(np.array(vec1) - np.array(vec2)))