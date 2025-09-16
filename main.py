from __future__ import division
from __future__ import print_function

import copy
import os

from TIAs.TIAs import TIA_GAP, TIA, TIA_PGR
from baseline.Eclipse_main.dp_svd import train_with_Eclipse
from baseline.GAP_master.train import train_with_GAP
from baseline.LPGNet.LPGNet import train_with_LPGNet
from baseline.Lap_and_RR.LapEdge import graph_normal_training_perturb_LAP
from baseline.Lap_and_RR.RandEdge import graph_normal_training_perturb_RR
from baseline.PPRL.GNNPrivacy import train_with_PPRL
from baseline.PrivGraph_main.priv_graph import train_with_privGraph
from utils.get_network import get_network
from utils.train import graph_normal_training

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import argparse
import numpy as np
import pandas as pd

import torch
import torch_geometric


from data.dataload import load_data
from graph_reconstruction.graph_regenerate import graph_regenerate_different

from utils.utils import split_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='Original',choices=['PGR','Original','LapEdge','EdgeRand','LPGNet','PPRL','privGraph','GAP','Eclipse'])
    parser.add_argument('--dataset', type=str, default='cora'
                            ,choices=['cora', 'citeseer','duke','lastfm','emory'])
    parser.add_argument('--device', type=str, default='cuda:3',choices=['cpu','cuda:3','cuda:0','cuda:1','cuda:2'])
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--epochs_inner', type=int, default=1,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=32,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='units dropout')
    parser.add_argument('--prune', type=float, default=0.5,
                        help='how many need to keep')
    parser.add_argument('--ratio_of_train_set', type=float, default='0.1')
    parser.add_argument('--mu', type=float, default='0.0')
    parser.add_argument('--attacks', type=str, default='None',choices=['None','TIA','TIA-PGR'])
    parser.add_argument('--network', type=str, default='GCN',choices=['GCN','GAT','GraphSAGE'])
    parser.add_argument('--hops', type=int, default=2)
    parser.add_argument('--eps', type=float, default=7)


    args = parser.parse_args()
    algorithm=args.algorithm
    device=args.device
    dataset_name=args.dataset
    seed=args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    ratio_of_train_set=args.ratio_of_train_set
    hidden=args.hidden
    dropout=args.dropout
    lr=args.lr
    epochs=args.epochs

    epochs_inner=args.epochs_inner
    weight_decay=args.weight_decay
    prune=args.prune
    mu=args.mu
    attack=args.attacks
    hops=args.hops
    eps=args.eps
    network=args.network
    dataset=load_data(dataset_name)
    data = dataset[0]
    dense_matrix = torch_geometric.utils.to_dense_adj(data.edge_index)[0]

    features=data.x.to(device)
    labels=data.y.to(device)
    model=get_network(network, hops, features, labels, hidden, dropout, device)
    preds=None

    idx_train,idx_val,idx_test=split_dataset(labels,ratio_of_train_set,seed)

    if algorithm=='PGR':
        acc_test, num_priv_edges, num_rengen_edges, model, regen_adj = graph_regenerate_different(algorithm,features,
                                                                                                  dense_matrix, dense_matrix,labels,
                                                                                                  idx_train, idx_val,
                                                                                                  idx_test, lr,
                                                                                                  weight_decay, epochs,
                                                                                                  epochs_inner, prune,
                                                                                                  device,
                                                                                                  mu, model,network)


    elif algorithm == 'Original':
        acc_test,num_priv_edges,num_rengen_edges,model,regen_adj=graph_normal_training(features, dense_matrix, labels, idx_train, idx_val, idx_test, hidden, dropout, lr,weight_decay, epochs,network,model,device)

    elif algorithm == 'privGraph':
        acc_test, num_priv_edges, num_rengen_edges, model, regen_adj = train_with_privGraph(eps,features, dense_matrix, labels, idx_train, idx_val, idx_test, model, network, lr,weight_decay, epochs, device)
    elif algorithm == 'GAP':
        acc_test,preds,data_init,model=train_with_GAP(dataset_name,eps,hops,device)
        regen_adj=dense_matrix
    elif algorithm == 'LPGNet':
        acc_test,model,features,regen_adj=train_with_LPGNet(copy.deepcopy(data), eps, idx_train, idx_val, idx_test)
    elif algorithm == 'PPRL':
        acc_test,model,features,regen_adj=train_with_PPRL(copy.deepcopy(data), idx_train, idx_val, idx_test)
    elif algorithm == 'Eclipse':
        acc_test, num_priv_edges, num_rengen_edges, model, regen_adj = train_with_Eclipse(eps,features, dense_matrix, labels, idx_train, idx_val, idx_test, model, network, lr,weight_decay, epochs, device)
    elif algorithm == 'LapEdge':
        acc_test, num_priv_edges, num_rengen_edges, model, regen_adj = graph_normal_training_perturb_LAP(eps, features,
                                                                                                         dense_matrix,
                                                                                                         labels,
                                                                                                         idx_train,
                                                                                                         idx_val,
                                                                                                         idx_test,
                                                                                                         model, network, lr,
                                                                                                         weight_decay,
                                                                                                         epochs, device)
    elif algorithm=='EdgeRand':
        acc_test, num_priv_edges, num_rengen_edges, model, regen_adj = graph_normal_training_perturb_RR(eps, features,
                                                                                                        dense_matrix,
                                                                                                        labels,
                                                                                                        idx_train,
                                                                                                        idx_val,
                                                                                                        idx_test,
                                                                                                        model, network,
                                                                                                        lr,
                                                                                                        weight_decay,
                                                                                                        epochs, device)




    else:
        raise ValueError("this algorithm is not exist")

    print(f'{attack}|{network}|{algorithm}|{dataset_name}|test_acc:{acc_test}')
    # pd.DataFrame([acc_test]).to_csv(
    #     f"TPL_result_baseline/{attack}_{network}_{algorithm}_{dataset_name}_{eps}_acc.csv",
    #     index=False, header=False)
    # if algorithm == 'PGR':
    #     File_Path_Csv = os.getcwd() + f"/result_PGR/{network}/{algorithm}/{dataset_name}/{prune}/{mu}/{epochs_inner}/{hops}//"
    #     if not os.path.exists(File_Path_Csv):
    #         os.makedirs(File_Path_Csv)
    #     pd.DataFrame([acc_test,num_priv_edges,num_rengen_edges]).to_csv(f"{File_Path_Csv}/acc.csv",index=False, header=False)
    #     torch.save(regen_adj, f"{File_Path_Csv}/regen_edge.pth")
    #     torch.save(model.state_dict(), f'{File_Path_Csv}/model.pt')
    #     torch.save(labels, f'{File_Path_Csv}/labels.pth')
    #     torch.save(features, f'{File_Path_Csv}/features.pth')
    #     torch.save(device, f'{File_Path_Csv}/device.pth')

    if attack=='TIA':
        if algorithm == 'GAP':
            TPL_M,TPL_C, TPL_I = TIA_GAP(model,  data_init.x, data_init.y, data_init.adj_t, eps, hops,device, seed)
        else:

            TPL_M,TPL_C, TPL_I = TIA(algorithm, data, model, dense_matrix, features,regen_adj,labels, device,hops, seed)
        # pd.DataFrame([TPL_M, TPL_C, TPL_I]).to_csv(
        #     f"TPL_result_baseline/{attack}_{network}_{algorithm}_{dataset_name}_{eps}.csv",
        #     index=False, header=False)
        print(f'{attack}|{network}|{algorithm}|{dataset_name}|{TPL_M}|{TPL_C}|{TPL_I}')


    if attack=='TIA-PGR':

        TPL_M, TPL_C,TPL_I = TIA_PGR( data, model, dense_matrix, features,regen_adj,labels, device,hops, seed)

        print(f'{attack}|{network}|{algorithm}|{dataset_name}|{TPL_M}|{TPL_C}|{TPL_I}')

    if algorithm == 'PGR':
        print(f"model acc loss (utility):{(acc_test-0.8338)/0.8338} | M-TIA:{TPL_M*100.0} | C-TIA:{TPL_C*100.0} | I-TIA :{TPL_I*100.0}")
    else:
        print(f"model accuracy (utility):{acc_test*100.0} | M-TIA:{TPL_M*100.0} | C-TIA:{TPL_C*100.0} | I-TIA :{TPL_I*100.0}")

if __name__ == '__main__':
     main()