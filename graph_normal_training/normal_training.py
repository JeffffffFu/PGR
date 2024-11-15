from __future__ import division
from __future__ import print_function

import math
import os
import time
import argparse
from math import ceil

import numpy as np
import pandas as pd
import torch_geometric
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj

from mask.add_diagonal_matrix import add_diagonal_and_normalize_edge
import random

from model.GCN import GCN, GCN_one_hop, GCN_three_hop
from utils.train import test, train


def graph_normal_training(features,dense_matrix,labels,idx_train,idx_val,idx_test,hidden,dropout,lr,weight_decay,epochs,hops,device):

    A_hat = add_diagonal_and_normalize_edge(dense_matrix,device)
    edge_num=torch.count_nonzero(dense_matrix)

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
    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)


    train(epochs, features, A_hat, labels, idx_train, idx_val, model, optimizer)
    acc_test=test(features, A_hat, labels, idx_test, model)

    return acc_test.cpu(),edge_num.cpu(),edge_num.cpu(),model,dense_matrix

