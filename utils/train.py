from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from mask.mata_grads import compute_matrix_grads
from utils.utils import accuracy



import time

import torch
import torch.nn.functional as F
import torch.optim as optim

from mask.add_diagonal_matrix import add_diagonal_and_normalize_edge, self_connecting



def graph_normal_training(features,dense_matrix,labels,idx_train,idx_val,idx_test,hidden,dropout,lr,weight_decay,epochs,network,model,device):

    if network=='GCN':
        A_hat = add_diagonal_and_normalize_edge(dense_matrix,device)

    else:
        A_hat = self_connecting(dense_matrix, device)

    edge_num=torch.count_nonzero(dense_matrix)


    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)


    train(epochs, features, A_hat, labels, idx_train, idx_val, model, optimizer)
    acc_test=test(features, A_hat, labels, idx_test, model)

    return acc_test.cpu(),edge_num.cpu(),edge_num.cpu(),model,dense_matrix



# Training settings


def train(epoch,features,adj,labels,idx_train,idx_val,model,optimizer):
    t = time.time()
    for i in range(epoch):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        print('Epoch: {:04d}'.format(i+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'time: {:.4f}s'.format(time.time() - t))



def test(features,adj,labels,idx_test,model):
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test

