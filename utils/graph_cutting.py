import os
import numpy as np
import torch
import community as community_louvain

from torch_geometric.datasets import Planetoid, Flickr, Coauthor, CitationFull, Yelp, Reddit2, AmazonProducts
from torch_geometric.utils import to_networkx


def store_subgraph(dataset_name, partition_groups, upper=1e6, lower=1):

    dataset_name = 'Cora'
    data_path = '../data/'

    # good_dataset: CiteSeer, PubMed, Coauthor
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root=f'{data_path}/{dataset_name}', name=dataset_name)
    elif dataset_name == 'Flickr':
        dataset = Flickr(root=f'{data_path}/{dataset_name}')
    elif dataset_name in ['CS', 'Physics']:
        dataset = Coauthor(root=f'{data_path}/{dataset_name}', name=dataset_name)
    elif dataset_name in ['DBLP']:
        dataset = CitationFull(root=f'{data_path}/{dataset_name}', name=dataset_name)
    elif dataset_name == 'Yelp':
        dataset = Yelp(root=f'{data_path}/{dataset_name}')
    elif dataset_name == 'Reddit2':
        dataset = Reddit2(root=f'{data_path}/{dataset_name}')
    elif dataset_name == 'AmazonProducts':
        dataset = AmazonProducts(root=f'{data_path}/{dataset_name}')

    data = dataset[0]
    G = to_networkx(data, to_undirected=True)

    number_of_modules = 3
    print('Executing Louvain algorithm...')
    partition = community_louvain.best_partition(G, random_state=1)
    print('Louvain algorithm done...')
    groups = []
    print(partition)
    for key in partition.keys():
        if partition[key] not in groups:
            groups.append(partition[key])
    print(groups)   #
    partition_groups = {group_i: [] for group_i in groups}

    for key in partition.keys():
        partition_groups[partition[key]].append(key)

    print(partition_groups)   #

    partition_size = []
    for key in partition_groups:
        partition_size.append(len(partition_groups[key]))

    partition_size.sort(reverse=True)
    print(partition_size)   #

if __name__ == '__main__':
     store_subgraph("cora",'sss')
