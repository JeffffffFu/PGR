import copy

import torch
from torch_sparse import SparseTensor

from baseline.GAP_master.core.modules.node.em import EncoderModule
from baseline.GAP_master.train import compute_aggregations_dp
from mask.add_diagonal_matrix import add_diagonal_and_normalize_edge

def link_infer_1_hop(target_u,regen_matrix, model,features,labels ,device):
    influence = 1.1
    pert = torch.zeros_like(features)
    pert[target_u] = features[target_u] * influence
    model.eval()
    logist=model.logist(features + pert, regen_matrix).detach()


    return logist
def link_infer_node_implantation_2_hop(algorithm,target_u,regen_matrix, model,features,labels ,device):
    est_features = copy.deepcopy(features)
    est_features = torch.cat((est_features, torch.ones(1, est_features.shape[1]).to(device)),
                             dim=0)  # cat dim 0: (nodes, feature)
    labels = copy.deepcopy(labels)

    if algorithm == 'train_with_PPRL':
        regen_matrix = regen_matrix.coalesce().to(device)
        num_nodes = len(labels)
        adv_idx_u = num_nodes
        # Prepare additional edges in COO format
        additional_indices = torch.tensor([
            [target_u, adv_idx_u],
            [adv_idx_u, target_u]
        ], device=device)

        additional_values = torch.ones(additional_indices.shape[1], device=device)

        # Concatenate the new edges with the existing ones
        new_indices = torch.cat([regen_matrix.indices(), additional_indices], dim=1)
        new_values = torch.cat([regen_matrix.values(), additional_values])

        # Create a new sparse tensor with the new edges
        regen_matrix = torch.sparse_coo_tensor(
            new_indices,
            new_values,
            (num_nodes + 1, num_nodes + 1),
            device=device
        ).coalesce()  #

        model.eval()
        model.to(device)  #
        logist = model.logist(est_features, regen_matrix).detach()

    else:

        #
        N = regen_matrix.shape[0]
        new_size = N + 1
        new_dense_matrix = torch.zeros((new_size, new_size), device=device)

        #
        new_dense_matrix[:N, :N] = regen_matrix

        #
        adv_idx_u = N

        #
        new_dense_matrix[target_u, adv_idx_u] = 1
        new_dense_matrix[adv_idx_u, target_u] = 1
        model.eval()
        if algorithm == 'LPGNet':
            logist = model.logist(est_features, new_dense_matrix, labels).detach()
        else:
            logist = model.logist(est_features, new_dense_matrix).detach()

    return logist


def link_infer_node_implantation_3_hop(algorithm, target_u, regen_matrix, model, features, labels, device):
    est_features = copy.deepcopy(features)
    #
    est_features = torch.cat((est_features, torch.zeros(1, est_features.shape[1]).to(device)), dim=0)
    est_features = torch.cat((est_features, torch.zeros(1, est_features.shape[1]).to(device)), dim=0)

    labels = copy.deepcopy(labels)

    if algorithm == 'train_with_PPRL':
        regen_matrix = regen_matrix.coalesce().to(device)
        num_nodes = len(labels)
        adv_idx_v = num_nodes
        adv_idx_g = num_nodes + 1

        # Prepare additional edges in COO format
        additional_indices = torch.tensor([
            [target_u, adv_idx_v],
            [adv_idx_v, target_u],
            [adv_idx_v, adv_idx_g],
            [adv_idx_g, adv_idx_v]
        ], device=device)

        additional_values = torch.ones(additional_indices.shape[1], device=device)

        new_indices = torch.cat([regen_matrix.indices(), additional_indices], dim=1)
        new_values = torch.cat([regen_matrix.values(), additional_values])

        regen_matrix = torch.sparse_coo_tensor(
            new_indices,
            new_values,
            (num_nodes + 2, num_nodes + 2),
            device=device
        ).coalesce()

        model.eval()
        model.to(device)
        logist = model.logist(est_features, regen_matrix).detach()

    else:
        N = regen_matrix.shape[0]
        new_size = N + 2
        new_dense_matrix = torch.zeros((new_size, new_size), device=device)

        new_dense_matrix[:N, :N] = regen_matrix

        adv_idx_v = N
        adv_idx_g = N + 1

        new_dense_matrix[target_u, adv_idx_v] = 1
        new_dense_matrix[adv_idx_v, target_u] = 1
        new_dense_matrix[adv_idx_v, adv_idx_g] = 1
        new_dense_matrix[adv_idx_g, adv_idx_v] = 1

        model.eval()
        if algorithm == 'train_with_LPGNet':
            logist = model.logist(est_features, new_dense_matrix, labels).detach()
        else:
            logist = model.logist(est_features, new_dense_matrix).detach()

    return logist


def link_infer_node_implantation_2_hop_GAP(target_u,adj_t, model,features,labels ,eps,hops,device):
    # 提取行和列索引
    rows = adj_t.storage.row()
    cols = adj_t.storage.col()

    sparse_adj = torch.stack([rows, cols])

    est_features = copy.deepcopy(features).to(device)
    sparse_adj = sparse_adj  #
    num_nodes = len(labels)

    #
    adv_idx_u = num_nodes  #

    #
    est_features = torch.cat(
        (est_features, torch.full((1, est_features.shape[1]), 100).to(device)), dim=0
    )
    #
    additional_indices = torch.tensor([
        [target_u, adv_idx_u],
        [adv_idx_u, target_u]
    ], device=device)

    additional_values = torch.ones(additional_indices.shape[1], device=device)

    sparse_adj = sparse_adj.to(device)

    new_indices = torch.cat([sparse_adj, additional_indices], dim=1)
    new_values = torch.cat([torch.ones(sparse_adj.shape[1], device=device), additional_values])

    sparse_adj = torch.sparse_coo_tensor(
        new_indices,
        new_values,
        (num_nodes + 1, num_nodes + 1),  #
        device=device
    ).coalesce()  #

    #
    rows, cols = sparse_adj.indices()[0], sparse_adj.indices()[1]

    #
    adj_t = SparseTensor(row=rows, col=cols, sparse_sizes=(num_nodes + 1, num_nodes + 1)).to(device)

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

    #
    encoded_features = Encoder.predict2(est_features.cpu())


    encoded_features = encoded_features.to(device)

    encoded_features = compute_aggregations_dp(eps, hops, encoded_features, adj_t)
    model = model.to(device)

    model.eval()
    logits = model.logist(encoded_features)

    return logits