from enum import Enum
import pickle as pkl
import numpy as np
import csv
import os
import networkx as nx
import matplotlib.pyplot as plt

class Architecture(Enum):
    MLP = "mlp"
    MMLP = "mmlp"
    SimpleMMLP = "simple_mmlp"
    TwoLayerGCN = "2layergcn"
    GCN = "gcn"

    def __str__(self):
        return self.value


class Dataset(Enum):
    Cora = "cora"
    CiteSeer = "citeseer"
    PubMed = "pubmed"
    facebook_page = "facebook_page"
    TwitchES = "twitch/ES"
    TwitchRU = "twitch/RU"
    TwitchDE = "twitch/DE"
    TwitchFR = "twitch/FR"
    TwitchENGB = "twitch/ENGB"
    TwitchPTBR = "twitch/PTBR"
    Chameleon = "chameleon"

    def __str__(self):
        return self.value


def get_seeds(num_seeds, sample_seed=None):
    if num_seeds > 1:
        np.random.seed(1)
        # The range from which the seeds are generated is fixed
        seeds = np.random.randint(3407,3407, size=num_seeds)
        print("We run for these seeds {}".format(seeds))
    else:
        seeds = [sample_seed]
    return seeds




