from baseline.GAP_master.core.data.loader import NodeDataLoader
from baseline.GAP_master.core.datasets.loader import DatasetLoader
from baseline.GAP_master.core.loggers.factory import Logger
from baseline.GAP_master.core.methods.node.base import NodeClassification
from baseline.GAP_master.core import console
import torch
import numpy as np
from rich import box
from rich.table import Table
from time import time
from typing import Annotated
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from baseline.GAP_master.core import globals
from baseline.GAP_master.core.args.utils import print_args, create_arguments, strip_kwargs, ArgInfo
from baseline.GAP_master.core.methods.node import supported_methods
from baseline.GAP_master.core.modules.node.em import EncoderModule
from baseline.GAP_master.core.privacy.algorithms import PMA
from baseline.GAP_master.core.utils import seed_everything, confidence_interval
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul

from privacy_analyze.RDP.get_MaxSigma_or_MaxSteps import get_noise_multiplier


def run(seed:    Annotated[int,   ArgInfo(help='initial random seed')] = 3407,
        repeats: Annotated[int,   ArgInfo(help='number of times the experiment is repeated')] = 1,
        debug:   Annotated[bool,  ArgInfo(help='enable global debug mode')] = False,
        **kwargs
    ):

    seed_everything(seed)

    if debug:
        console.info('debug mode enabled')
        globals['debug'] = True
        console.log_level = console.DEBUG

    with console.status('loading dataset'):
        loader_args = strip_kwargs(DatasetLoader, kwargs)
        data_initial = DatasetLoader(**loader_args).load(verbose=True)


    num_classes = data_initial.y.max().item() + 1
    config = dict(**kwargs, seed=seed, repeats=repeats)
    logger_args = strip_kwargs(Logger.setup, kwargs)
    logger = Logger.setup(enabled=False, config=config, **logger_args)

    ### initiallize method ###
    Method = supported_methods[kwargs['method']]
    method_args = strip_kwargs(Method, kwargs)
    method: NodeClassification = Method(num_classes=num_classes, **method_args)

    run_metrics = {}

    ### run experiment ###
    for iteration in range(repeats):
        start_time = time()
        data = Data(**data_initial.to_dict())
        metrics,preds,model = method.fit(data)  # finding,6
        end_time = time()
        metrics['duration'] = end_time - start_time
        ### process results ###
        for metric, value in metrics.items():
            run_metrics[metric] = run_metrics.get(metric, []) + [value]

         ### print results ###
        table = Table(title=f'run {iteration + 1}', box=box.HORIZONTALS)
        table.add_column('metric')
        table.add_column('last', style="cyan")
        table.add_column('mean', style="cyan")
        table.add_row('test/acc', f'{run_metrics["test/acc"][-1]:.2f}', f'{np.mean(run_metrics["test/acc"]):.2f}')
        console.info(table)
        console.print()

        ### reset method's parameters for the next run ###
        method.reset_parameters()

    logger.enable()
    summary = {}
    
    for metric, values in run_metrics.items():
        summary[metric + '_mean'] = np.mean(values)
        summary[metric + '_std'] = np.std(values)
        summary[metric + '_ci'] = confidence_interval(values, size=1000, ci=95, seed=seed)
        logger.log_summary(summary)

    logger.finish()
    return run_metrics["test/acc"][-1],preds,data_initial,model


def main():

    init_parser = ArgumentParser(add_help=False, conflict_handler='resolve')
    method_subparser = init_parser.add_subparsers(dest='method', required=True, title='algorithm to use')

    for method_name, method_class in supported_methods.items():
        method_parser = method_subparser.add_parser(
            name=method_name,
            help=method_class.__doc__,
            formatter_class=ArgumentDefaultsHelpFormatter
        )

        # dataset args
        group_dataset = method_parser.add_argument_group('dataset arguments')
        create_arguments(DatasetLoader, group_dataset)

        # method args
        group_method = method_parser.add_argument_group('method arguments')
        create_arguments(method_class, group_method)

        # experiment args
        group_expr = method_parser.add_argument_group('experiment arguments')
        create_arguments(run, group_expr)
        create_arguments(Logger.setup, group_expr)

    parser = ArgumentParser(parents=[init_parser], formatter_class=ArgumentDefaultsHelpFormatter)

    kwargs = vars(parser.parse_args())

    print_args(kwargs, num_cols=2)

    try:
        start = time()
        preds = run(**kwargs)
        end = time()
        console.info(f'Total running time: {(end - start):.2f} seconds.')
    except KeyboardInterrupt:
        print('\n')
        console.warning('Graceful Shutdown')
    except RuntimeError:
        raise
    finally:
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
            console.info(f'Max GPU memory used = {gpu_mem:.2f} GB\n')
    return preds

def train_with_GAP(dataset,epsilon,hops,device):


    init_parser = ArgumentParser(add_help=False, conflict_handler='resolve')
    method_subparser = init_parser.add_subparsers(dest='method', required=True, title='algorithm to use')

    for method_name, method_class in supported_methods.items():
        method_parser = method_subparser.add_parser(
            name=method_name, 
            help=method_class.__doc__, 
            formatter_class=ArgumentDefaultsHelpFormatter
        )

        # dataset args
        group_dataset = method_parser.add_argument_group('dataset arguments')
        create_arguments(DatasetLoader, group_dataset)

        # method args
        group_method = method_parser.add_argument_group('method arguments')
        create_arguments(method_class, group_method)
        
        # experiment args
        group_expr = method_parser.add_argument_group('experiment arguments')

        create_arguments(run, group_expr)
        create_arguments(Logger.setup, group_expr)

    parser = ArgumentParser(parents=[init_parser], formatter_class=ArgumentDefaultsHelpFormatter)

    #change parameters
    kwargs = vars(parser.parse_args(['gap-inf']))
    method = 'gap-edp'
    kwargs.update({'method':method, 'dataset':dataset, 'epsilon': epsilon, 'hops': hops, 'device':device})

    #print(f'kwargs: {kwargs}')    # all parameters are in here

    print_args(kwargs, num_cols=2)

    try:
        start = time()
        acc,preds,data_initial,model=run(**kwargs)
        end = time()
        console.info(f'Total running time: {(end - start):.2f} seconds.')
    except KeyboardInterrupt:
        print('\n')
        console.warning('Graceful Shutdown')
    except RuntimeError:
        raise
    finally:
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
            console.info(f'Max GPU memory used = {gpu_mem:.2f} GB\n')


    return acc,preds,data_initial,model



def compute_aggregations_dp(eps, hops, features,adj):

    alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]
    delta = 1e-5
    noise_multiplier = get_noise_multiplier(eps, delta, 1, hops, alphas)
    x = F.normalize(features, p=2, dim=-1)

    x_list = [x]
    pma_mechanism = PMA(noise_scale=noise_multiplier, hops=hops)
    for _ in range(hops):
        x = matmul(adj, x)
        x = pma_mechanism(x, 1)
        x = F.normalize(x, p=2, dim=-1)
        x_list.append(x)

    features = torch.stack(x_list, dim=-1)
    return features

def GAP_forward_inference(eps,hops,model,data,device):
    Encoder = EncoderModule(
        num_classes=data.y.max().item() + 1,
        hidden_dim=16,
        encoder_layers=2,
        head_layers=1,
        normalize=True,
        activation_fn= torch.selu_,
        dropout=0.0,
        batch_norm=True,
    )


    features=Encoder.predict2(data.x)
    features=compute_aggregations_dp(eps,hops,features,data.adj_t)
    model=model.to(device)
    features=features.to(device)
    # loss, metrics, preds = model.step(data, stage='test')

    logists = model.logist(features)
    return logists
# if __name__ == '__main__':
#     dataset='cora2'
#     epsilon= '5'
#     hops=2
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     train_with_GAP(dataset,epsilon,hops,device)
    # python train.py gap-edp --epsilon 5  --database cora --hops 2   acc: 59.5%

