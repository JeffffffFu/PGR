import os
from functools import partial
from typing import Annotated
import torch
from torch_geometric.datasets import Planetoid, Twitch, LastFMAsia, Reddit

from baseline.GAP_master.core import console
from torch_geometric.data import Data
from torch_geometric.transforms import Compose, ToSparseTensor, RandomNodeSplit
from baseline.GAP_master.core.args.utils import ArgInfo
from baseline.GAP_master.core.data.transforms import FilterClassByCount
from baseline.GAP_master.core.data.transforms import RemoveSelfLoops
from baseline.GAP_master.core.data.transforms import RemoveIsolatedNodes
from baseline.GAP_master.core.datasets import Facebook
from baseline.GAP_master.core.datasets import Amazon
from baseline.GAP_master.core.utils import dict2table
import numpy as np
import math
import random

def generate_boolean_lists(label,ratio_of_train):
    # 创建包含 n 个元素的列表 A, 需要划分验证集，把训练集划分为两部分，一部分是最后真正的训练集，另一部分是验证集
    n=len(label)
    m_train=math.ceil(n*ratio_of_train*0.7)
    m_val=math.ceil(n*ratio_of_train*0.3)

    train_mask = [False] * n
    val_mask = [False] * n
    np.random.seed(3407)

    # 随机选择 m 个位置将其设置为 true
    indices = random.sample(range(n), m_train)
    for idx in indices:
        train_mask[idx] = True

    # 随机选择 m_val 个位置将其设置为 True，但要确保这些位置不在 train_mask 中
    indices_val = random.sample([i for i in range(n) if not train_mask[i]], m_val)
    for idx in indices_val:
        val_mask[idx] = True

    # 生成与 train_mask 和 val_mask 完全相反的列表 test_mask
    test_mask = [not train_mask[i] and not val_mask[i] for i in range(n)]

    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    return train_mask, val_mask,test_mask

def generate_boolean_lists2(label,ratio_of_train):
    # 创建包含 n 个元素的列表 A, 需要划分验证集，把训练集划分为两部分，一部分是最后真正的训练集，另一部分是验证集
    n=len(label)
    m_train=math.ceil(n*ratio_of_train)

    train_mask = [False] * n
    np.random.seed(3407)
    # 随机选择 m 个位置将其设置为 true
    indices = random.sample(range(n), m_train)
    for idx in indices:
        train_mask[idx] = True

    # 生成与 train_mask 和 val_mask 完全相反的列表 test_mask
    test_mask = [not train_mask[i] for i in range(n)]

    train_mask = torch.BoolTensor(train_mask)
    test_mask = torch.BoolTensor(test_mask)

    return train_mask, train_mask,test_mask

class DatasetLoader:
    supported_datasets = {
        'reddit': partial(Reddit,
            transform=Compose([
                RandomNodeSplit(num_val=0.1, num_test=0.15),
                FilterClassByCount(min_count=10000, remove_unlabeled=True)
            ])
        ),
        'amazon': partial(Amazon, 
            transform=Compose([
                RandomNodeSplit(num_val=0.1, num_test=0.15), 
                FilterClassByCount(min_count=100000, remove_unlabeled=True)
            ])
        ),
        'duke': partial(Facebook, name='Duke14', target='year',
            transform=Compose([
                RandomNodeSplit(num_val=0.1, num_test=0.9),
                FilterClassByCount(min_count=1000, remove_unlabeled=True)
            ])
        )(root='./data/facebook'),
        'emory': partial(Facebook, name='Emory27', target='year',
                            transform=Compose([
                                RandomNodeSplit(num_val=0.1, num_test=0.9),
                                FilterClassByCount(min_count=1000, remove_unlabeled=True)
                            ])
                            )(root='./data/Emory'),
        'cora': Planetoid(root='./data', name='cora'),
        'citeseer': Planetoid(root='./data', name='citeseer'),
        'pubmed': Planetoid(root='./data', name='pubmed'),
      #  'twitch_es': Twitch(root='./data/Twitch',name='ES') ,
      #  'lastfm': LastFMAsia(root='./data/LastFMAsia'),
    }

    def __init__(self,
                 dataset:    Annotated[str, ArgInfo(help='name of the dataset', choices=supported_datasets)] = 'facebook',
                 data_dir:   Annotated[str, ArgInfo(help='directory to store the dataset')] = './datasets',
                 ):

        self.name = dataset
        self.data_dir = data_dir

    def load(self, verbose=False) -> Data:

        data=self.supported_datasets[self.name][0]
        if isinstance(data, Data):
            data = data
        else:
            data = Data(**data)

        #组建或者重装train_mask,val_mask和text_mask
        # mask的组装有区别，他们的是每个位置以True或False进行的，我们的是把对应的index直接填进去的
        train_mask, val_mask,test_mask = generate_boolean_lists2(data.y, 0.1)

        data.train_mask=train_mask
        data.val_mask=val_mask
        data.test_mask=test_mask
        data.num_nodes=len(data.y)



        #data = self.supported_datasets[self.name](root=os.path.join(self.data_dir, self.name))[0]
        data = Compose([RemoveSelfLoops(), RemoveIsolatedNodes(), ToSparseTensor()])(data)

        if verbose:
            self.print_stats(data)

        return data

    def print_stats(self, data: Data):
        nodes_degree: torch.Tensor = data.adj_t.sum(dim=1)
        baseline: float = (data.y[data.test_mask].unique(return_counts=True)[1].max().item() * 100 / data.test_mask.sum().item())
        train_ratio: float = data.train_mask.sum().item() / data.num_nodes * 100
        val_ratio: float = data.val_mask.sum().item() / data.num_nodes * 100
        test_ratio: float = data.test_mask.sum().item() / data.num_nodes * 100

        stat = {
            'nodes': f'{data.num_nodes:,}',
            'edges': f'{data.num_edges:,}',
            'features': f'{data.num_features:,}',
            'classes': f'{int(data.y.max() + 1)}',
            'mean degree': f'{nodes_degree.mean():.2f}',
            'median degree': f'{nodes_degree.median()}',
            'train/val/test (%)': f'{train_ratio:.1f}/{val_ratio:.1f}/{test_ratio:.1f}',
            'baseline acc (%)': f'{baseline:.2f}'
        }

        table = dict2table(stat, num_cols=2, title=f'dataset: [yellow]{self.name}[/yellow]')
        console.info(table)
        console.print()
