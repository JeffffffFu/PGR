from model.GAT import GATLayer, GAT
from model.GCN import GCN_one_hop, GCN, GCN_three_hop
from model.GNN import GNN


def get_network(network,hops,features,labels,hidden,dropout,device):
    if network=='GCN':
        if hops==1:
            model = GCN_one_hop(nfeat=features.shape[1],
                        nhid=hidden,
                        nclass=labels.max().item() + 1,
                        dropout=dropout).to(device)
        elif hops==2:
            model = GCN(nfeat=features.shape[1],
                        nhid=hidden,
                        nclass=labels.max().item() + 1,
                        dropout=dropout).to(device)
        elif hops==3:
            model = GCN_three_hop(nfeat=features.shape[1],
                        nhid=hidden,
                        nclass=labels.max().item() + 1,
                        dropout=dropout).to(device)
    elif network=='GNN':
        model = GNN(nfeat=features.shape[1],
                    nhid=hidden,
                    nclass=labels.max().item() + 1,
                    dropout=dropout).to(device)
    elif network=='GAT':
        model = GAT(nfeat=features.shape[1],
                    nhid=hidden,
                    nclass=labels.max().item() + 1,
                    alpha=0.2,
                    dropout=dropout).to(device)
    else:
        raise ValueError('Invalid network choice.')

    return model