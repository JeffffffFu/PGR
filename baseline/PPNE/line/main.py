import networkx as nx
import os, time, torch
import numpy as np
from pdb import set_trace as st

def graph_to_edgelist(graph, save_path):
    with open(save_path,'w') as f:
        for edge in graph.edges.data('weight', default=1):
            f.write("%d %d %.8f\n"%(edge[0], edge[1], edge[2]))
            f.write("%d %d %.8f\n"%(edge[1], edge[0], edge[2]))

def graph_from_edgelist(save_path):
    g = nx.Graph()
    edge_list = []
    with open(save_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            edge = line[:-1].split(" ")
            edge_list.append((int(edge[0]), int(edge[1]), float(edge[2])))
    g.add_weighted_edges_from(edge_list)
    return g

def get_emb_from_file(n, dim, out_path, out_path_ct):
    X = np.zeros((n, dim))
    with open(out_path, 'r') as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            if index == 0:
                continue
            emb = line[:-1].split(" ")
            node = int(emb[0])
            node_emb = [float(t) for t in emb[1:-1]]
            assert len(node_emb) == dim
            X[node] = node_emb

    Y = np.zeros((n, dim))
    with open(out_path_ct, 'r') as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            if index == 0:
                continue
            emb = line[:-1].split(" ")
            node = int(emb[0])
            node_emb = [float(t) for t in emb[1:-1]]
            assert len(node_emb) == dim
            Y[node] = node_emb
    return X, Y
    # return X

# def line_inner(args, in_path, out_path, out_path_ct):
def line_inner(args, in_path, out_path, out_path_ct):
    # os.system('LD_LIBRARY_PATH=/home/xiaocw/sunmj/gsl/lib \    >/dev/null 2>&1
    os.system('G:\graph-attack\line\LINE\windows\line -train %s \
    -output %s -output2 %s \
    -size %d -order 2 -negative 5 \
    -samples 20 -rho 0.025 -threads 20'%(in_path, out_path, out_path_ct, args.dim))
    # default samples 5
    # pubmed 20

def line(args, graph, in_path, out_path, out_path_ct): 
    n = graph.number_of_nodes()
    graph_to_edgelist(graph, in_path)
    line_inner(args, in_path, out_path, out_path_ct)
    X, Y = get_emb_from_file(n, args.dim, out_path, out_path_ct)
    # X = get_emb_from_file(n, args.dim, out_path, out_path_ct)
    return X, Y
    # return X

if __name__ == '__main__':
    g = nx.erdos_renyi_graph(100, 0.3)
    graph_to_edgelist(g, "tmp.txt")
    # g = graph_from_edgelist('tmp.txt')
    line_inner('tmp.txt','emd.txt','context.txt')
    X = get_emb_from_file(100, 128, 'emd.txt', 'context.txt')