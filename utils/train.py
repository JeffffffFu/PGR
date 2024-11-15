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


#这个是最纯粹的GCN，没有调用各个库的

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

def train2(epoch,features,adj,labels,idx_train,idx_val,model,optimizer):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output, labels)
    acc_train = accuracy(output, labels)
    loss_train.backward()
    optimizer.step()


    print('Epoch: {:04d}'.format(epoch+1),
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


def test_label(output, labels, idx_test):
    preds = output[idx_test]
    labels = labels[idx_test]
    correct = preds.eq(labels).sum().item()
    acc_test = correct / len(labels)

    print("Test set results:", "accuracy= {:.4f}".format(acc_test))