import os
from baseline.LPGNet import utils
from baseline.LPGNet import trainer
from baseline.LPGNet.data import LoadData
import argparse
import numpy as np
import torch
import time
# import attacker
import baseline.LPGNet.models
from baseline.LPGNet.globals import MyGlobals
from torch_geometric.utils import to_dense_adj ,dense_to_sparse
import psutil
import torch.multiprocessing as mp

from utils.utils import accuracy


def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss
# decorator function
def profile(func):
    def wrapper(*args, **kwargs):
 
        mem_before = process_memory()
        result = func(*args, **kwargs)
        mem_after = process_memory()
        print("{}:consumed memory: {:,}".format(
            func.__name__,
            mem_before, mem_after, mem_after - mem_before))
 
        return result
    return wrapper

def train(args,data,device,idx_train,idx_val,idx_test,):
    # Some sanity checks on the arguments
    if args.w_dp == True and args.eps == 0.0:
        print("You selected with DP but eps=0.0")
        exit()

    if args.w_dp == False and args.eps > 0:
        print("No DP selected")
        exit()

    # Some sanity checks on the SVD arguments
    if args.svd == True and args.rank == 0:
        print("You selected with SVD but rank=0")
        exit()
    
    if args.svd == False and args.rank > 0:
        print("No SVD selected")
        exit()



    # if args.no_cuda:
    #     device = torch.device("cpu")
    # else:
    #     device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
    #     # Run the train/attack/eval on the selected GPU id
    #     if torch.cuda.is_available():
    #         # torch.cuda.set_device(args.cuda_id)
    #         # print("Current CUDA device: {}".format(torch.cuda.current_device()))
    #         print(f"Current CUDA device: {torch.cuda.current_device()}")

    run_config = trainer.RunConfig(
        learning_rate=args.lr,
        num_epochs=args.num_epochs,
        save_each_epoch=False,
        save_epoch=args.save_epoch,
        weight_decay=MyGlobals.weight_decay,
        output_dir=os.path.join(args.outdir, "models"),
        eps=args.eps,
        rank=args.rank,
        nl=args.nl,
        hidden_size=args.hidden_size,
        num_hidden=args.num_hidden,
        dropout=args.dropout,
    )

    seeds = utils.get_seeds(args.num_seeds, args.sample_seed)

    print("Running training")
    return run_training(
        run_config,
        data,
        device,
        idx_train,
        idx_val,
        idx_test,
        dp=args.w_dp,
        svd=args.svd,
        seeds=seeds,
        test_dataset=args.test_dataset,
    ), device



def run_training(
    run_config, dataset, device,idx_train,idx_val,
    idx_test, dp=False, svd=False, seeds=[1], test_dataset=None
):
    if test_dataset:
        print("Transfer learning with separate test graph")

    return trainer.train_mmlp_on_dataset(
        run_config,
        dataset,
        device,
        idx_train,
        idx_val,
        idx_test,
        dp=dp,
        seeds=seeds,
        test_dataset=test_dataset,
    )


#《LPGNet: Link Private Graph Networks for Node Classification》
@profile
def LPGNet(data,eps,idx_train,idx_val,idx_test):
    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--dataset",
    #     type=utils.Dataset,
    #     choices=utils.Dataset,
    #     default=utils.Dataset.Cora,
    #     help="cora|citeseer|pubmed...",
    # )
    # parser.add_argument(
    #     "--arch",
    #     type=utils.Architecture,
    #     choices=utils.Architecture,
    #     default=utils.Architecture.MMLP,
    #     required=True,
    #     help="Type of architecture to train: mmlp|gcn|mlp",
    # )
    parser.add_argument(
        "--algorithm",
        type=str,
        default='LPGNet',
    )
    parser.add_argument(
        "--attacks",
        type=str,
        default='TIAs',
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='cora',
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda:0',
    )
    parser.add_argument(
        "--nl",
        type=int,
        default=MyGlobals.nl,
        help="Only use for MMLP, Number of stacked models, default=-1",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=MyGlobals.num_seeds,
        help="Run over num_seeds seeds",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=MyGlobals.sample_seed,
        help="Run for this seed",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=eps,
        help="The privacy budget. If 0, then do not DP train the arch",
    )
    parser.add_argument(
        "--w_dp",
        default=MyGlobals.with_dp,
        action="store_true",
        help="Run with DP guarantees - if eps=0.0 it throws a warning",
    )
    parser.add_argument(
        "--svd",
        default=MyGlobals.svd,
        action="store_true",
        help="Run with SVD - if rank=0 it throws a warning",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=MyGlobals.rank,
        help="The rank. If 0, then do not perform SVD before train the arch",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=MyGlobals.hidden_size,
        help="Size of the first hidden layer",
    )
    parser.add_argument(
        "--num_hidden",
        type=int,
        default=MyGlobals.num_hidden,
        help="Number of hidden layers",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=MyGlobals.RESULTDIR,
        help="Directory to save the models and results",
    )
    parser.add_argument(
        "--test_dataset",
        type=utils.Dataset,
        choices=utils.Dataset,
        default=None,
        help="Test on this dataset, used for Twitch",
    )

    
    parser.add_argument(
        "--lr", type=float, default=MyGlobals.lr, help="Learning rate"
    )
    parser.add_argument("--num_epochs", type=int, default=MyGlobals.num_epochs)
    parser.add_argument(
        "--save_epoch",
        type=int,
        default=MyGlobals.save_epoch,
        help="Save at every save_epoch",
    )
    parser.add_argument("--dropout", type=float, default=MyGlobals.dropout)
    parser.add_argument("--prune", type=float, default=0.1)
    parser.add_argument("--priD", type=float, default=0.0)

    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Store the results in a pkl in args.outdir",
    )

    parser.add_argument(
        "--epochs_inner",
        action="store_true",
    )


    parser.set_defaults(func=train)
    args = parser.parse_args()
    device=args.device

    # args.subparser_name = 'train'
    output=args.func(args, data, device,idx_train,idx_val,idx_test)
    model=output[0][0]
    features=output[0][1]
    adj= output[0][2]
    adj=torch.tensor(adj.toarray()).to(torch.float32)
    return model,features,data.y,adj,device

def LPGNet_inference(model, features,labels,adj,device):
    model.eval()
    comms_file = None
    model=model.to(device)
    adj=adj.to(device)
    features = features.to(device)
    labels = labels.to(device)
    model.prepare_for_fwd(
        features, adj, comms_file
    )

    outputs = model(features, labels)
    return outputs

def train_with_LPGNet(data,eps,idx_train,idx_val,idx_test):
    mp.set_start_method('spawn', force=True)

    model,features,labels,adj,device=LPGNet(data,eps,idx_train,idx_val,idx_test)
    outputs=LPGNet_inference(model, features,labels,adj,device)
    output=outputs[0]
    start_time = time.time()

    acc_test = accuracy(output[idx_test], data.y[idx_test])
    end_time  = time.time()
    execution_time = end_time - start_time
    print(f'execution_time: {execution_time:.4f}')


    return acc_test,model,features,adj

def train_with_LPGNet_output(data,eps,idx_train,idx_val,idx_test):
    mp.set_start_method('spawn', force=True)

    model,features,labels,adj,device=LPGNet(data,eps,idx_train,idx_val,idx_test)
    outputs=LPGNet_inference(model, features,labels,adj,device)
    output=outputs[0]

    acc_test = accuracy(output[idx_test], data.y[idx_test])


    return acc_test,model,features,adj,output

# if __name__ == "__main__":
#     from torch_geometric.datasets import Planetoid
#     dataset = Planetoid(root="../data/planetoid", name="Cora")
#     data = dataset[0]
#     # mmlp_model, test_features, test_adj_orig_csr, device = train_model(data)
#     eps = 4.0

#     ret = train_model(data,eps)
#     print("ret[0]:",ret[0])
#     exit()
#     logits = test(ret[0][0],ret[0][1],ret[0][2],ret[1])

