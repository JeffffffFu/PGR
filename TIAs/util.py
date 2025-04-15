from collections import Counter
import random

import numpy as np
import torch
from scipy.spatial.distance import chebyshev
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
        # kmeans.fit(influence_val)
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


def TOP_K_index_to_matrix_PGR(matrix, K, regen_edge, test_nodes):
    """
    """
    # Step 1: Extract subgraph
    test_nodes = np.array(test_nodes)
    sub_matrix = matrix[np.ix_(test_nodes, test_nodes)]
    regen_edge_np = regen_edge.cpu().numpy() if isinstance(regen_edge, torch.Tensor) else regen_edge
    sub_regen_edge = regen_edge_np[np.ix_(test_nodes, test_nodes)]

    # Step 2: Merge upper triangle
    sub_matrix = np.triu(sub_matrix) + np.triu(sub_matrix, k=1).T

    # Step 3: Get upper triangle values
    upper_triangle_indices = np.triu_indices(sub_matrix.shape[0], k=1)
    upper_triangle_values = sub_matrix[upper_triangle_indices]

    # Step 4: Filter by regen_edge
    sub_regen_edge_upper_values = sub_regen_edge[upper_triangle_indices]
    valid_mask = sub_regen_edge_upper_values == 0  # Keep only edges not already connected
    filtered_values = upper_triangle_values[valid_mask]
    filtered_indices = (upper_triangle_indices[0][valid_mask], upper_triangle_indices[1][valid_mask])

    # Step 5: Safeguard against empty values
    if len(filtered_values) == 0 or K <= 0:
        return np.zeros_like(matrix)  # Return an empty binary matrix if no valid edges

    # Step 6: Find Top-K indices
    K = min(K, len(filtered_values))  # Adjust K to the available number of edges
    top_k_indices = np.argpartition(filtered_values, -K)[-K:]
    if len(top_k_indices) == 0:
        return np.zeros_like(matrix)  # Additional safeguard
    top_k_sorted_indices = top_k_indices[np.argsort(filtered_values[top_k_indices])[::-1]]

    # Step 7: Construct binary submatrix
    binary_sub_matrix = np.zeros_like(sub_matrix)
    binary_sub_matrix[filtered_indices[0][top_k_sorted_indices], filtered_indices[1][top_k_sorted_indices]] = 1

    # Step 8: Symmetrize binary matrix
    binary_sub_matrix = binary_sub_matrix + binary_sub_matrix.T

    # Step 9: Map back to original matrix size
    binary_matrix = np.zeros_like(matrix)
    binary_matrix[np.ix_(test_nodes, test_nodes)] = binary_sub_matrix

    return binary_matrix

def TOP_K_index_to_matrix(matrix, K, test_nodes):
    """"""
    test_nodes = np.array(test_nodes)
    sub_matrix = matrix[np.ix_(test_nodes, test_nodes)]

    sub_matrix_upper = np.triu(sub_matrix)
    sub_matrix_lower = np.tril(sub_matrix, -1)
    sub_matrix_upper += sub_matrix_lower.T
    sub_matrix = np.triu(sub_matrix_upper)  #
    #print("sub_matrix:",np.count_nonzero(sub_matrix))
    upper_triangle_indices = np.triu_indices(sub_matrix.shape[0], k=1)
    upper_triangle_values = sub_matrix[upper_triangle_indices]

    top_k_indices = np.argpartition(upper_triangle_values, -K)[-K:]
    top_k_values_indices = (upper_triangle_indices[0][top_k_indices],
                            upper_triangle_indices[1][top_k_indices])

    binary_sub_matrix = np.zeros_like(sub_matrix)
    binary_sub_matrix[top_k_values_indices] = 1

    binary_sub_matrix += binary_sub_matrix.T

    binary_matrix = np.zeros_like(matrix)
    binary_matrix[np.ix_(test_nodes, test_nodes)] = binary_sub_matrix

    return binary_matrix


def k_hat_subgraph(test_nodes, ren_adj):
    k_hat = 0
    test_nodes.sort(reverse=False)
    for i in range(len(test_nodes)):
        for j in range(i + 1, len(test_nodes)):
            u = test_nodes[i]
            v = test_nodes[j]
            if ren_adj[u][v]==1 :
                k_hat += 1
    if k_hat==0:
        k_hat=1
    if k_hat<0:
        raise ValueError("k_hat need to larger than 0")

    return k_hat


import networkx as nx


def bfs_dense_subgraph_random(graph, min_nodes, max_nodes, max_subgraphs,seed):
    subgraphs = []
    nodes = list(graph.nodes())
    visited_global = set()  #
    random.seed(seed)
    while len(subgraphs) < max_subgraphs:
        start_node = random.choice(nodes)
        if start_node in visited_global:
            continue

        visited_local = set()
        queue = [start_node]
        community = []

        while queue and len(community) < max_nodes:
            current = queue.pop(0)
            if current not in visited_local:
                visited_local.add(current)
                community.append(current)
                queue.extend([n for n in graph.neighbors(current) if n not in visited_local])

        if min_nodes <= len(community) <= max_nodes:
            subgraph = graph.subgraph(community)
            subgraphs.append(subgraph)
            visited_global.update(community)  #
        else:
            continue  # ，

    if len(subgraphs) < max_subgraphs:
        raise ValueError("wrong")

    result = []
    for idx, subgraph in enumerate(subgraphs):
        node_index = list(subgraph.nodes())
        original_size = len(graph.nodes())
        full_adj = torch.zeros((original_size, original_size), dtype=torch.float32)

        for i, j in subgraph.edges():
            full_adj[i, j] = 1
            full_adj[j, i] = 1

        print(f"Subgraph {idx + 1}: Node count = {len(node_index)}, Edges count = {len(subgraph.edges())}")

        result.append((node_index, full_adj))

    return result


def construct_multiple_subgraphs(G):
    print("random choose subgraphs -----")
    # Step 1: Convert the PyTorch Geometric data to a NetworkX graph
    num_communities = 5
    # Step 2: Apply the Louvain community detection algorithm
    partition = community_louvain.best_partition(G)

    # Step 3: Count the number of nodes in each community
    value_counts = Counter(partition.values())
    value_counts_dict = dict(value_counts)

    selected_communities = []
    selected_subgraphs = []
    print(value_counts)
    # Step 4: Find multiple communities that meet the node count requirements
    for _ in range(num_communities):
        community_id = -1
        for key, value in value_counts_dict.items():
            # Select a community with the required number of nodes, excluding previously selected communities
            if key not in selected_communities and 100 <= value <= 200:
                community_id = key
                break


        # If we find a valid community, extract its nodes and adjacency matrix
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

        # Stop if we have reached the desired number of communities
        if len(selected_subgraphs) == num_communities:
            break
    return selected_subgraphs

def construct_subgraph(data):
    # Step 2: Convert the PyTorch Geometric data to a NetworkX graph
    G = to_networkx(data, to_undirected=True)

    # Step 3: Apply the Louvain community detection algorithm
    # This returns a dictionary where the key is the node and the value is the community it belongs to
    partition = community_louvain.best_partition(G)

    # Count occurrences of each value
    value_counts = Counter(partition.values())

    # Convert to a dictionary for better readability (optional)
    value_counts_dict = dict(value_counts)
    # Step 4: Extract subgraphs based on the detected communities
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
    # print(f'Nodes in largest community: {index_subgraph}')

    subgraph = G.subgraph(index_subgraph)

    edges_list = list(subgraph.edges())

    original_size = len(G.nodes())

    full_adj = torch.zeros((original_size, original_size), dtype=torch.float32)

    # 在对应的 subgraph index 上将相应的位置置为 1
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

    # Count occurrences of each value
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
    # print(f'Nodes in largest community: {index_subgraph}')

    subgraph = G.subgraph(index_subgraph)

    edges_list = list(subgraph.edges())

    original_size = len(G.nodes())  #

    full_adj = torch.zeros((original_size, original_size), dtype=torch.float32)

    for i,j in edges_list:
            full_adj[i, j] = 1
            full_adj[j, i] = 1

    return index_subgraph,full_adj


def construct_subgraph_for_GAP_shadow(sparse_tensor,community_id_attack):
    # Step 1: Convert SparseTensor to networkx graph
    rows = sparse_tensor.storage.row()
    cols = sparse_tensor.storage.col()
    edges = list(zip(rows.tolist(), cols.tolist()))

    G = nx.Graph()  #
    G.add_edges_from(edges)  #

    # Step 2: Apply the Louvain community detection algorithm
    partition = community_louvain.best_partition(G)

    # Count occurrences of each value
    value_counts = Counter(partition.values())

    # Convert to a dictionary for better readability (optional)
    value_counts_dict = dict(value_counts)

    # Step 4: Extract subgraphs based on the detected communities
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

    subgraph = G.subgraph(index_subgraph)

    edges_list = list(subgraph.edges())

    original_size = len(G.nodes())

    full_adj = torch.zeros((original_size, original_size), dtype=torch.float32)

    for i,j in edges_list:
            full_adj[i, j] = 1
            full_adj[j, i] = 1


    return index_subgraph,full_adj,community_id

def construct_subgraph_for_GAP(sparse_tensor):
    # Step 1: Convert SparseTensor to networkx graph
    rows = sparse_tensor.storage.row()
    cols = sparse_tensor.storage.col()
    edges = list(zip(rows.tolist(), cols.tolist()))  #

    G = nx.Graph()  #
    G.add_edges_from(edges)  #

    # Step 2: Apply the Louvain community detection algorithm
    partition = community_louvain.best_partition(G)

    # Count occurrences of each value
    value_counts = Counter(partition.values())

    # Convert to a dictionary for better readability (optional)
    value_counts_dict = dict(value_counts)

    # Step 4: Extract subgraphs based on the detected communities
    community_id = -1
    print(value_counts_dict)
    for key, value in value_counts_dict.items():
        if 100 <= value <= 100:
            community_id = key
    if community_id == -1:
        for key, value in value_counts_dict.items():
            if 200 <= value <= 600:
                community_id = key
    #largest_community = value_counts.most_common(1)[0][0]  # Get the community label with the most nodes
    index_subgraph = [key for key, value in partition.items() if value == community_id]
    # print(f'Nodes in largest community: {index_subgraph}')

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
    if C>K or C>K_hat:
        print(C,K,K_hat)
        raise ValueError("C need to smaller than K or K_hat")
    TPL=C/(K+K_hat-C)
    print(C, K, K_hat)
    return TPL

def topology_loss_sim_PGR(subgraph_adj, binary_matrix,K,K_hat,test_nodes):
    N=len(test_nodes)
    matrix =subgraph_adj+binary_matrix
    C=torch.sum(torch.eq(matrix, 2))
    C=C //2
    if C>K or C>K_hat:
        print(C,K,K_hat)
        raise ValueError("C need to smaller than K or K_hat")
    TPL_1=C/(K+K_hat-C)
    TPL_2=(K_hat-C)/((N**2/2)-K_hat+C)
    print(TPL_1,TPL_2)
    return max(TPL_1,TPL_2)

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
    consine_sim_norm=(cosine_sim + 1) / 2
    return cosine_sim


def euclidean_distance(vec1, vec2):
    euclidean=np.linalg.norm(np.array(vec1) - np.array(vec2))
    euclidean_norm=1 - euclidean / 10 if euclidean <= 10 else 0
    return euclidean


def chebyshev_distance(vec1, vec2):  # best
    chebyshev=np.max(np.abs(np.array(vec1) - np.array(vec2)))
    chebyshev_norm=1 - chebyshev / 10 if chebyshev <= 10 else 0
    return chebyshev

def get_edges(test_node, adj_subgraph):
    exist_edges = []
    nonexist_edges = []

    num_nodes = len(test_node)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_subgraph[i, j] == 1:
                exist_edges.append((test_node[i], test_node[j]))
            else:
                nonexist_edges.append((test_node[i], test_node[j]))

    return exist_edges, nonexist_edges

def tensor_to_networkx(regen_adj):
    if regen_adj.ndim != 2 or regen_adj.size(0) != regen_adj.size(1):
        raise ValueError("The adjacency matrix must be a square tensor.")

    if regen_adj.is_cuda:
        regen_adj = regen_adj.cpu()

    edge_indices = torch.nonzero(regen_adj, as_tuple=False)

    G = nx.Graph()
    G.add_edges_from(edge_indices.tolist())

    return G