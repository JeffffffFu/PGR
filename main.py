from __future__ import division
from __future__ import print_function

import copy
import os

from TIA.TPL_audit import TIA, TIA_PGR

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import argparse
import numpy as np
import pandas as pd

import torch
import torch_geometric


from data.dataload import load_data
from graph_normal_training.normal_training import graph_normal_training
from graph_reconstruction.graph_regenerate import graph_regenerate_different

from utils.utils import split_dataset, BinaryMask_to_ListMask, generate_list_C, replace_elements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='Original',choices=['PGR','Original'])
    parser.add_argument('--dataset', type=str, default='cora'
                            ,choices=['cora', 'citeseer','duke','lastfm','emory'])
    parser.add_argument('--device', type=str, default='cuda:0',choices=['cpu','cuda:3','cuda:0','cuda:1','cuda:2'])
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
    parser.add_argument('--prune', type=float, default=0.1,
                        help='how many need to keep')
    parser.add_argument('--ratio_of_train_set', type=float, default='0.1')
    parser.add_argument('--mu', type=float, default='0.0')
    parser.add_argument('--attacks', type=str, default='None',choices=['None','TIA','TIA-PGR'])
    parser.add_argument('--hops', type=int, default=2)


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

    dataset=load_data(dataset_name)
    data = dataset[0]
    dense_matrix = torch_geometric.utils.to_dense_adj(data.edge_index)[0]

    features=data.x.to(device)
    labels=data.y.to(device)

    preds=None

    idx_train,idx_val,idx_test=split_dataset(labels,ratio_of_train_set,seed)

    if algorithm=='PGR':
        acc_test, num_priv_edges, num_rengen_edges, model, regen_adj = graph_regenerate_different(algorithm,features,
                                                                                                  dense_matrix, dense_matrix,labels,
                                                                                                  idx_train, idx_val,
                                                                                                  idx_test, hidden,
                                                                                                  dropout, lr,
                                                                                                  weight_decay, epochs,
                                                                                                  epochs_inner, prune,
                                                                                                  device,
                                                                                                  mu, preds)


    elif algorithm == 'Original':
        acc_test,num_priv_edges,num_rengen_edges,model,regen_adj=graph_normal_training(features, dense_matrix, labels, idx_train, idx_val, idx_test, hidden, dropout, lr,weight_decay, epochs,hops,device)



    else:
        raise ValueError("this algorithm is not exist")



    print("acc_test:",acc_test)

    if attack=='TIA':

        TPL_M, TPL_C,TPL_I = TIA(data, model, dense_matrix, features,regen_adj,labels, device, hops,seed)

        print(f'{attack}|{algorithm}|{dataset_name}|{TPL_M}|{TPL_C}|{TPL_I}')


    if attack=='TIA-PGR':

        TPL_M, TPL_C,TPL_I = TIA_PGR( data, model, dense_matrix, features,regen_adj,labels, device,hops, seed)

        print(f'{attack}|{algorithm}|{dataset_name}|{TPL_M}|{TPL_C}|{TPL_I}')



if __name__ == '__main__':
     main()