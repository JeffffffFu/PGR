import numpy as np
import scipy.sparse as sp
import torch

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def edge_matrix():
    path = "data/cora/"
    dataset = "cora"
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj = adj.to_dense()
    return adj

def normalize_edge(edge,device):
    edge=edge.to(device)

    D1 = torch.sum(edge, axis=1)
    D2 = torch.sum(edge, axis=0)

    D11 = D1 ** (-1 / 2)
    D22 = D2 ** (-1 / 2)
    D_inv1 = torch.diag(D11)
    D_inv2 = torch.diag(D22)

    A_hat = torch.mm(torch.mm(D_inv1, edge), D_inv2)
    return A_hat,D2

def dense_adj_to_adj_sparse_adj(dense_adj):
    nonzero_indices = torch.nonzero(dense_adj == 1, as_tuple=False).t()


    row_indices = nonzero_indices[0]
    col_indices = nonzero_indices[1]

    mask = row_indices < col_indices
    row_indices = row_indices[mask]
    col_indices = col_indices[mask]

    sparse_adj = torch.vstack((row_indices, col_indices))

    return sparse_adj
if __name__ == '__main__':
    adj=edge_matrix()
    A_hat=normalize_edge(adj)
    print(A_hat)
