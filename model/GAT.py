import copy
import math

import torch
from torch import nn

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

# class GAT(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, alpha, dropout, nheads=8):
#         super(GAT, self).__init__()
#         self.dropout = dropout
#
#         self.attentions = nn.ModuleList([GATLayer(nfeat, nhid, alpha, concat=True) for _ in range(nheads)])
#
#         self.out_att = GATLayer(nheads * nhid, nclass, alpha, concat=False)
#
#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         x = F.elu(x)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.out_att(x, adj)
#         return F.log_softmax(x, dim=1)
#
#     def logist(self, x, adj):
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         x = F.elu(x)
#         x = self.out_att(x, adj)
#         return F.log_softmax(x, dim=1)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, alpha, dropout):
        super(GAT, self).__init__()
        self.gat1 = GATLayer(nfeat, nhid, alpha, concat=True)
        self.gat2 = GATLayer(nhid, nclass, alpha, concat=False)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gat1(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gat2(x, adj)
        return F.log_softmax(x, dim=1)

    def logist(self, x, adj):
        x = self.gat1(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gat2(x, adj)
        return F.log_softmax(x, dim=1)


class GATLayer(Module):

    def __init__(self, in_features, out_features, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()


    def forward(self, input_h, adj):
        h = torch.mm(input_h, self.W)
        N = h.size(0)

        input_concat = torch.cat([h.repeat(1, N).view(N * N, -1),
                                  h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
       # e = self.leakyrelu(torch.matmul(input_concat, self.a).squeeze(2))
        e = self.sigmoid(torch.matmul(input_concat, self.a).squeeze(2))
        attention = e * adj

        if adj.size(0)==2708 or adj.size(0)==2709 :  # for cora
            attention = l1_norm(attention, dim=1)
        else:
            attention = l2_norm(attention, dim=1)

        output_h = torch.mm(attention, h)
        return output_h

def l1_norm(x, dim=1, eps=1e-10):
    abs_sum = x.abs().sum(dim, keepdim=True) + eps

    return x / abs_sum

def l2_norm(x, dim=1, eps=1e-10):

    norm = x.norm(p=2, dim=dim, keepdim=True) + eps
    return x / norm