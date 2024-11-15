from functools import partial

from torch_geometric.datasets import Twitch, LastFMAsia, Planetoid

from torch_geometric.transforms import Compose

from data.Facebook import Facebook, FilterClassByCount
from data.LastFM import KarateClub


def load_data(dataset_name):
    if dataset_name == "cora":
        dataset = Planetoid(root='./data', name=dataset_name)
    elif dataset_name == "citeseer":
        dataset = Planetoid(root='./data', name=dataset_name)
    elif dataset_name == "pubmed":
        dataset = Planetoid(root='./data', name=dataset_name)
    elif dataset_name == 'twitch_es':
        dataset = Twitch(root='./data/Twitch',name='ES')
    elif dataset_name == 'twitch_pt':
        dataset = Twitch(root='./data/Twitch',name='PT')
    elif dataset_name == 'twitch_de':
        dataset = Twitch(root='./data/Twitch',name='DE')
    elif dataset_name == 'twitch_fr':
        dataset = Twitch(root='./data/Twitch',name='FR')
    elif dataset_name == 'twitch_en':
        dataset = Twitch(root='./data/Twitch',name='EN')
    elif dataset_name == "lastfm":
        dataset = LastFMAsia(root='./data/LastFMAsia')
    elif dataset_name == "raw_lastfm":
        dataset = partial(KarateClub, name='lastfm')(root='./data/')
    elif dataset_name == 'Duke':
        dataset = partial(Facebook, name='Duke14', target='year'
                ,transform=Compose([FilterClassByCount(min_count=1000, remove_unlabeled=True)]))
        dataset = dataset(root=r'./data/facebook/')
    elif dataset_name == 'Yale':
        dataset = partial(Facebook, name='Yale4', target='year'
                ,transform=Compose([FilterClassByCount(min_count=1000, remove_unlabeled=True)]))
        dataset = dataset(root=r'./data/facebook/')
    elif dataset_name == 'Emory':
        dataset = partial(Facebook, name='Emory27', target='year'
                ,transform=Compose([FilterClassByCount(min_count=1000, remove_unlabeled=True)]))
        dataset = dataset(root=r'./data/facebook/')
    elif dataset_name == 'UChicago':
        dataset = partial(Facebook, name='UChicago30', target='year'
                ,transform=Compose([FilterClassByCount(min_count=1000, remove_unlabeled=True)]))
        dataset = dataset(root=r'./data/facebook/')
    else:
        raise ValueError("Dataset No Claim")
    return dataset
