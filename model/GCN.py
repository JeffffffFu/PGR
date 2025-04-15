import torch.nn as nn
import torch.nn.functional as F

from model.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)

        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


    def logist(self,x,adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)



class GCN_three_hop(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_three_hop, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid*2)
        self.gc2 = GraphConvolution(nhid*2, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)

        self.dropout = dropout


    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        return F.log_softmax(x, dim=1)

    def logist(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        return F.log_softmax(x, dim=1)

class GCN_one_hop(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_one_hop, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x=self.gc1(x,adj)
        return F.log_softmax(x, dim=1)

    def logist(self,x,adj):
        x=self.gc1(x,adj)
        return x
