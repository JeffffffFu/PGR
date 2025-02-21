from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time

import numpy as np
import torch

from baseline.PPRL import optimizers
from baseline.PPRL.config import parser
from baseline.PPRL.models.base_models import ADVNCModel, ADVLPModel
from baseline.PPRL.utils.data_utils import load_data
from baseline.PPRL.utils.train_utils import get_dir_name, format_metrics
from utils.utils import accuracy


def PPRL(dataset, idx_train, idx_val, idx_test):
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.patience = args.epochs if not args.patience else  int(args.patience)
    logging.getLogger().setLevel(logging.INFO)

    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))

    reserve_mark = 0

    if args.task == 'nc':
        reserve_mark = 0
    else:
        args.task = 'nc'
        reserve_mark = 1
    # Load data
    data = load_data(args, dataset, idx_train, idx_val, idx_test)
    args.n_nodes, args.feat_dim = data['features'].shape
    if args.task == 'nc':
        Model = ADVNCModel
        args.n_classes = int(data['labels'].max() + 1)
        logging.info(f'Num classes: {args.n_classes}')
    else:
        args.nb_false_edges = len(data['train_edges_false'])
        args.nb_edges = len(data['train_edges'])
        if args.task == 'lp':
            Model = ADVLPModel


    #transfer loading
    if reserve_mark == 1:
        args.task = 'lp'
        # reset reserve mark
        reserve_mark = 0

    if args.task == 'lp':
        reserve_mark = 0
    else:
        args.task = 'lp'
        reserve_mark = 1

    data1 = load_data(args,dataset, idx_train, idx_val, idx_test )
    args.n_nodes, args.feat_dim = data1['features'].shape
    if args.task == 'nc':
        Model = ADVNCModel
        args.n_classes = int(data1['labels'].max() + 1)
        logging.info(f'Num classes: {args.n_classes}')
    else:
        print('*****')
        args.nb_false_edges = len(data1['train_edges_false'])
        args.nb_edges = len(data1['train_edges'])
        if args.task == 'lp':
            Model = ADVLPModel


    if reserve_mark == 1:
        args.task = 'nc'

    if args.task == 'nc':
        Model = ADVNCModel
    else:
        Model = ADVLPModel



    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    # Model and optimizer
    model = Model(args)
    logging.info(str(model))
    # print('model name')
    # print("-"*50)
    # for name, params in model.named_parameters():
    #     print("name=",name)
    #     print('-'*25)

    # print("-"*50)
    # for name, params in model.encoder.named_parameters():
    #     print("name=",name)
    #     print('-'*25)

    # print("-"*50)
    # for name, params in model.net.named_parameters():
    #     print("name=",name)
    #     print('-'*25)
    
    # print("end")

    optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                    weight_decay=args.weight_decay)
    optimizer_en = getattr(optimizers, args.optimizer)(params=model.encoder.parameters(), lr=args.lr,
                                                       weight_decay=args.weight_decay)
    optimizer_de = getattr(optimizers, args.optimizer)(params=model.net.parameters(), lr=args.lr,
                                                    weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    lr_scheduler_en = torch.optim.lr_scheduler.StepLR(
        optimizer_en,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    lr_scheduler_de = torch.optim.lr_scheduler.StepLR(
        optimizer_de,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    logging.info(f"Total number of parameters: {tot_params}")
    if args.cuda is not None and int(args.cuda) >= 0 :
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)
        for x, val in data1.items():
            if torch.is_tensor(data1[x]):
                data1[x] = data1[x].to(args.device)
    # Train model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        # if epoch%3==0:
        
        # 节点分类
        optimizer.zero_grad()
        optimizer_de.zero_grad()
        embeddings = model.encode(data['features'], data['adj_train_norm'])
        train_metrics,_ = model.compute_metrics(embeddings, data, 'train')
        train_metrics['loss'].backward()
        if args.grad_clip is not None:
            max_norm = float(args.grad_clip)
            all_params = list(model.parameters())
            for param in all_params:
                torch.nn.utils.clip_grad_norm_(param, max_norm)
        optimizer.step()

        optimizer_en.zero_grad()
        # model.load_emb()
        embeddings1 = model.encode(data1['features'], data1['adj_train_norm'])
        embeddings1=embeddings1.to(args.device)
        train_metrics1 = model.compute_metrics1(embeddings1, data1, 'train')
        loss1 = -(train_metrics1['loss'])  # - train_metrics1['loss_shuffle'])
        loss1.backward()
        optimizer_en.step()
        # model.load_net()
        #     # lr_scheduler.step()
        #
        # # if epoch%3==2:

        optimizer_de.zero_grad()
        embeddings2 = model.encode(data1['features'], data1['adj_train_norm']).detach_()
        embeddings2=embeddings2.to(args.device)
        train_metrics2 = model.compute_metrics1(embeddings2, data1, 'train')
        loss2 = (train_metrics2['loss'])  # - train_metrics2['loss_shuffle'])
        loss2.backward()
        optimizer_de.step()
        lr_scheduler.step()
        lr_scheduler_de.step()
        lr_scheduler_en.step()

    feature=data['features']
    adj=data['adj_train_norm']
    label=data['labels']

    return model,feature,adj,label

def PPRL_inference(model,features,adj):
    model.eval()
    emb = model.encode(features, adj)
    output=model.decode2(emb, adj)
    return output


def train_with_PPRL(data,idx_train, idx_val, idx_test):
    model,features,adj,label=PPRL(data, idx_train, idx_val, idx_test)

    output=PPRL_inference(model,features,adj)
    acc_test = accuracy(output[idx_test], data.y[idx_test])
    return acc_test,model,features,adj

# if __name__ == '__main__':
#     from torch_geometric.datasets import Planetoid
#     import torch_geometric.transforms as T
#     dataset = Planetoid(root='data/Planetoid', name='Cora', transform=T.LargestConnectedComponents())
#     data = dataset[0]
#     model,data_ = train(args,data)
#     output = test(model,data_)
#     print(output)

