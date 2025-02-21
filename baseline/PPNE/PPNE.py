from datetime import datetime
import networkx as nx
import numpy as np
import scipy.sparse as sp
import argparse, pickle, time, os, collections, torch
import sys

from baseline.PPNE import lib, deepwalk, line, util, util_utility

sys.path.append('..')
import baseline.PPNE.lib
import baseline.PPNE.line
import baseline.PPNE.deepwalk
import baseline.PPNE.util
import baseline.PPNE.util_utility
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from scipy.sparse import coo_matrix

#《Privacy-Preserving Network Embedding Against Private Link Inference Attacks 》
def PPNE(data,device):

    parser = argparse.ArgumentParser("deepwalk", formatter_class=argparse.ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--matfile-variable-name', default='network', help='variable name of adjacency matrix inside a .mat file.')
    parser.add_argument('--max-memory-data-size', default=1000000000, type=int, help='Size to start dumping walks to disk, instead of keeping them in memory.')
    parser.add_argument('--num_walks', default=5, type=int, help='Number of random walks to start at each node')
    parser.add_argument('--dim', default=64, type=int, help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--seed', default=3407, type=int, help='Seed for random walk generator.')
    parser.add_argument('--vertex-freq-degree', default=False, action='store_true')
    parser.add_argument('--walk_length', default=10, type=int, help='Length of the random walk started at each node')
    parser.add_argument('--window-size', default=3, type=int, help='Window size of skipgram model.')
    parser.add_argument('--workers', default=1, type=int, help='Number of parallel processes.')
    parser.add_argument('--iter', default=1, type=int, help='number of iterations in word2vec')
    parser.add_argument('--dataset',default='cora',type=str)
    parser.add_argument('--edge-range', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mask', type=str,default='adj')
    parser.add_argument('--p', type=float, default=1,help='Return hyperparameter. Default is 1.')
    parser.add_argument('--q', type=float, default=1,help='Inout hyperparameter. Default is 1.')
    parser.add_argument('--directed', dest='directed', action='store_true')
    parser.add_argument('--method',default='deepwalk',type=str)
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--opt-iter',type=int,default=10)
    parser.add_argument('--ratio',type=float,default=0.05)
    parser.add_argument('--init', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--add-delete-prob', type=float, default=0.5)
    parser.add_argument('--candidate-add-num', type=int, default=100)
    parser.add_argument('--save',default='./baseline/PPNE/tradeoff_cora_ppne', type=str)
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('--algorithm', type=str, default='regen',choices=['regen','spars','train','train_with_perturb','train_with_Lap','train_with_PPNE','train_with_privGraph','train_with_GAP','train_with_Eclipse','regen_base_privGraph','regen_base_Eclipse','regen_base_GAP'])
    parser.add_argument('--device', type=str, default='cuda:0',choices=['cpu','cuda:3','cuda:0','cuda:1','cuda:2'])

    args = parser.parse_args()

    args.device = device
    #
    # if not os.path.exists(args.save):
    #     os.makedirs(args.save)
    ###########################################################################################################

    # Set hyperparameter
    RANDOM_SEED = args.seed

    # if not os.path.exists(os.path.join(args.save, 'tmp')):
    #     os.makedirs(os.path.join(args.save, 'tmp'))
    #
    # IN_PATH = os.path.join(args.save, 'tmp', 'input.txt')
    # SAVE_PATH = os.path.join(args.save, 'tmp', 'output.txt')
    # SAVE_PATH_CT = os.path.join(args.save, 'tmp', 'context.txt')

    with open(os.path.join(args.save, 'cora_ppne_modify'), 'a') as file_record:
        file_record.write('ppne_modify_iter, now time: {}\n'.format(datetime.now()))
        file_record.write('-------------------------------------------\n')
        file_record.flush()

    #------------------------------------------------------------------------------------------------
    # Make the split.
    # modify

    edge_index = data.edge_index

    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    num_nodes = data.num_nodes  # 获取节点数量
    edge_index_coo = coo_matrix((torch.ones(edge_index.shape[1]), (row, col)), shape=(num_nodes, num_nodes))
    adj_train,_,__,___,____,test_edges,test_edges_false = lib.split.mask_test_edges(edge_index_coo)
    labels = data.y.numpy()

    # adj_train = np.load('data/cora_data/adj_train.npy')
    # test_edges = np.load('data/cora_data/test_edges.npy')
    # test_edges_false = np.load('data/cora_data/test_edges_false.npy')
    # labels = np.load('data/cora_data/labels.npy')

    # count_1 = 0
    # for i in range(adj_train.shape[0]):
    #     for j in range(adj_train.shape[0]):
    #         if adj_train[i][j] == 1:
    #             count_1+=1
    # print(count_1)

    g = nx.from_numpy_array(adj_train, parallel_edges=False, create_using=nx.Graph())
    g.remove_edges_from(test_edges)
    adj_train = nx.adjacency_matrix(g).todense()

    adj_train = adj_train.astype('float64')

    #--------------------------------------------------------------------------------------------------
    graph_first = nx.from_numpy_array(adj_train, parallel_edges=False, create_using=nx.Graph())

    if args.method == 'deepwalk':
        print('start deepwalk!')
        X, Y = deepwalk.deepwalk(args, graph_first, verbose=args.verbose, random_seed=RANDOM_SEED)
    # elif args.method == 'line':
    #     X, Y = line.line(args, graph_first, IN_PATH, SAVE_PATH, SAVE_PATH_CT)

    X_ori = X


    #----------------------------------------------------------------------------------------------------
    if args.method == 'deepwalk':
        MF = lib.DeepWalkMF(args,T=args.window_size).to(args.device)
    elif args.method == 'line':
        MF = lib.LINEMF(args).to(args.device)
    #-----------------------------------------------------------
    victim_edges = np.concatenate((test_edges, test_edges_false), axis=0)

    adj_train_first = np.copy(adj_train)

    # with open(os.path.join(args.save, 'cora_ppne_modify'), 'a') as file_record:
    #     file_record.write('start optimization, now time: {}\n'.format(datetime.now()))
    #     file_record.write('-------------------------------------------\n')
    #     file_record.flush()
    adj_train_opt = adj_train_first
    for _ in range(1):
        #------------------------------------------------------------------------
        ITER = 1000
        # ITER = 1
        adds_or_deletes = np.random.choice([0, 1], ITER, p=[1 - args.add_delete_prob, args.add_delete_prob])
        
        deleted_edges = []
        added_edges = []
        for i in range(ITER):
            print("now ITER: {}".format(i))
            # with open(os.path.join(args.save, 'cora_ppne_modify'), 'a') as file_record:
            #     file_record.write('ITER {}, now time: {}\n'.format(i+1, datetime.now()))
            #     file_record.flush()

            print("deleted edge num: {}".format(len(deleted_edges)))
            print("added edge num: {}".format(len(added_edges)))

            # add or delete an edge in this iter
            # 0 for delete, 1 for add
            if adds_or_deletes[i] == 1:
                print("add an edge")

                print("utility loss estimation")
                adj_train_csr = sp.csr_matrix(adj_train_opt)
                candidates = util.generate_candidates_addition(adj_train_csr, n_candidates=args.candidate_add_num)
                mask = sp.csr_matrix(np.zeros(adj_train.shape))
                mask = util.flip_candidates(mask, candidates).toarray()
                mask = (mask == 1)

                mask_low = np.tril(mask, k=-1)
                nonzero = mask_low.nonzero()
                i_ts, j_ts = nonzero

                candidates = np.array([[i_ts[n], j_ts[n]] for n in range(len(i_ts))])

                loss_for_candidates = util_utility.perturbation_utility_loss(adj_train_csr, candidates,
                                                                            args.dim, args.window_size)

                adj_train = np.copy(adj_train_opt)
                adj_train[i_ts, j_ts] = args.init
                adj_train[j_ts, i_ts] = args.init

                # -------------------------------------------------------------------------
                print("Begin optimization")
                train_graph = nx.from_numpy_array(adj_train, parallel_edges=False, create_using=nx.Graph())
                if args.method == 'deepwalk':
                    X, Y = deepwalk.deepwalk(args, train_graph, verbose=args.verbose, random_seed=RANDOM_SEED)
                # elif args.method == 'line':
                #     X, Y = line.line(args, train_graph, IN_PATH, SAVE_PATH, SAVE_PATH_CT)

                # roc_score, ap_score = util.evaluate(X, test_edges, test_edges_false)
                # scenario2_f1_lr, scenario2_f1_xgb = util.evaluate_scenario2(X, test_edges, test_edges_false, args.dim)
                # similarity_score = util.intermediate_similarity_calculation(X, test_edges)
                # f1_score_mean, _, acc = util.evaluate_embedding_node_classification(X, labels)
                # nmi_scores = []
                # for n in range(20):
                #     nmi_score = util.evaluate_embedding_node_clustering(X_ori, X, labels, random_seed=n)
                #     nmi_scores.append(nmi_score)
                # nmi = np.mean(nmi_scores)
                # -----------------------------------------------------------------------------------------------------
                ###### Phase 1 ######
                print("Start Phase 1: calculate X_grad")
                X_torch, Y_torch = torch.tensor(X).double().to(args.device), torch.tensor(Y).double().to(args.device)
                X_grad = util.get_grad_on_X(X_torch, test_edges, test_edges_false).to(args.device)
                # -----------------------------------------------------------------------------------------------------
                print("Start Phase 2: calculate grad_Z")
                d_rt = np.sum(adj_train, axis=1, dtype=np.float64)

                MF.set_d_rt(d_rt)

                adj_torch = torch.tensor(adj_train, dtype=torch.float64).to(args.device)

                Z_forward = MF(adj_torch)
                mask_on_Z = (Z_forward.cpu().data.numpy() != -float('inf'))

                grad_Z = util.build_grad_on_Z_adv_torch(X_torch, Y_torch, X_grad, mask_on_Z,args)
                # ------------------------------------------------------------------------------------
                # st()
                print("Start Phase 3: calculate adj_grad")
                i1_ts, j1_ts = mask_on_Z.nonzero()

                loss = (grad_Z.double()[i1_ts, j1_ts] * Z_forward[i1_ts, j1_ts]).sum()

                loss.backward()
                adj_grad = adj_torch.grad.cpu().data.numpy()
                # ------------------------------------------------------------------------------------
                adj_grad = (adj_grad + adj_grad.T) / 2.
                y_ts_grad = adj_grad[i_ts, j_ts]

                # only consider positive gradient
                pos_grad_indice = np.where(y_ts_grad >= 0)[0]

                # privacy / utility_loss
                pu_vals = abs(y_ts_grad[pos_grad_indice] / loss_for_candidates[pos_grad_indice])
                max_idx = pos_grad_indice[np.argmax(pu_vals)]

                i_max = i_ts[max_idx]
                j_max = j_ts[max_idx]

                # the optimal edge
                edge = [i_max, j_max]
                print("Following edge is added: {}".format(edge))

                # add the optimal edge
                adj_train_opt[i_max][j_max] = 1
                adj_train_opt.T[i_max][j_max] = 1

                added_edges.append(edge)



            else:
                print("delete an edge")

                print("utility loss estimation")
                adj_train_csr = sp.csr_matrix(adj_train_opt)
                candidates = util.generate_candidates_removal(adj_train_csr)
                mask = sp.csr_matrix(np.zeros(adj_train.shape))
                mask = util.flip_candidates(mask, candidates).toarray()
                mask = (mask == 1)

                mask_low = np.tril(mask, k=-1)
                nonzero = mask_low.nonzero()
                i_ts, j_ts = nonzero

                candidates = np.array([[i_ts[n], j_ts[n]] for n in range(len(i_ts))])

                loss_for_candidates = util_utility.perturbation_utility_loss(adj_train_csr, candidates,
                                                                            args.dim, args.window_size)

                adj_train = np.copy(adj_train_opt)

                y_ts = adj_train[i_ts, j_ts]
                # -------------------------------------------------------------------------
                print("Begin optimization")
                train_graph = nx.from_numpy_array(adj_train, parallel_edges=False, create_using=nx.Graph())
                if args.method == 'deepwalk':
                    X, Y = deepwalk.deepwalk(args, train_graph, verbose=args.verbose, random_seed=RANDOM_SEED)
                # elif args.method == 'line':
                #     X, Y = line.line(args, train_graph, IN_PATH, SAVE_PATH, SAVE_PATH_CT)

                # roc_score, ap_score = util.evaluate(X, test_edges, test_edges_false)
                # scenario2_f1_lr, scenario2_f1_xgb = util.evaluate_scenario2(X, test_edges, test_edges_false, args.dim)
                # similarity_score = util.intermediate_similarity_calculation(X, test_edges)
                # f1_score_mean, _, acc = util.evaluate_embedding_node_classification(X, labels)
                # nmi_scores = []
                # for n in range(20):
                #     nmi_score = util.evaluate_embedding_node_clustering(X_ori, X, labels, random_seed=n)
                #     nmi_scores.append(nmi_score)
                # nmi = np.mean(nmi_scores)
                # -----------------------------------------------------------------------------------------------------
                ###### Phase 1 ######
                print("Start Phase 1: calculate X_grad")
                X_torch, Y_torch = torch.tensor(X).double().to(args.device), torch.tensor(Y).double().to(args.device)
                X_grad = util.get_grad_on_X(X_torch, test_edges, test_edges_false).to(args.device)
                # -----------------------------------------------------------------------------------------------------
                print("Start Phase 2: calculate grad_Z")
                d_rt = np.sum(adj_train, axis=1, dtype=np.float64)

                MF.set_d_rt(d_rt)

                adj_torch = torch.tensor(adj_train, dtype=torch.float64).to(args.device)

                Z_forward = MF(adj_torch)
                mask_on_Z = (Z_forward.cpu().data.numpy() != -float('inf'))
                print("here2")
                grad_Z = util.build_grad_on_Z_adv_torch(X_torch, Y_torch, X_grad, mask_on_Z,args)
                # ------------------------------------------------------------------------------------
                # st()
                print("Start Phase 3: calculate adj_grad")
                i1_ts, j1_ts = mask_on_Z.nonzero()

                loss = (grad_Z.double()[i1_ts, j1_ts] * Z_forward[i1_ts, j1_ts]).sum()

                loss.backward()
                adj_grad = adj_torch.grad.cpu().data.numpy()
                # ------------------------------------------------------------------------------------
                adj_grad = (adj_grad + adj_grad.T) / 2.
                y_ts_grad = adj_grad[i_ts, j_ts]

                # only consider negative gradient
                neg_grad_indice = np.where(y_ts_grad <= 0)[0]

                # privacy / utility_loss
                pu_vals = abs(y_ts_grad[neg_grad_indice] / loss_for_candidates[neg_grad_indice])

                max_idx = neg_grad_indice[np.argmax(pu_vals)]

                i_max = i_ts[max_idx]
                j_max = j_ts[max_idx]

                # the optimal edge
                edge = [i_max, j_max]
                print("Following edge is removed: {}".format(edge))

                # remove the optimal edge
                adj_train_opt[i_max][j_max] = 0
                adj_train_opt.T[i_max][j_max] = 0

                deleted_edges.append(edge)



    print("\n\n\nOptimization Done")
    return torch.tensor(adj_train_opt).to(torch.float32)

