import torch.nn as nn
import torch.nn.functional as F
import torch

from utils import sample_neighbors


class GraphSAGE(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GraphSAGE, self).__init__()

        self.sage1 = SAGEConvolution(nfeat, nhid)
        self.sage2 = SAGEConvolution(nhid, nclass)

        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.sage2(x, adj)
        return F.log_softmax(x, dim=1)

    def logist(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.sage2(x, adj)
        return F.log_softmax(x, dim=1)


class SAGEConvolution(nn.Module):

    def __init__(self, in_features, out_features):
        super(SAGEConvolution, self).__init__()
        self.linear = nn.Linear(in_features , out_features)

    def forward(self, x, adj):
        adj=sample_neighbors(adj)

        neighbor_mean = torch.spmm(adj, x) / (torch.spmm(adj, torch.ones_like(x)) + 1e-6)


        return self.linear(neighbor_mean)