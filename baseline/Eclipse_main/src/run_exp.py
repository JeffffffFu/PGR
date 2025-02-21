import multiprocessing
import argparse
from sys import stdout
import numpy as np
import shutil
import torch
from pathlib import Path
import os
from trainer import RunConfig
import utils
import main
from itertools import product
import pickle as pkl
from trainer import TrainStats

import io


def fix(map_loc):
    # Closure rather than a lambda to preserve map_loc
    return lambda b: torch.load(io.BytesIO(b), map_location=map_loc)


class MappedUnpickler(pkl.Unpickler):
    # https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219

    def __init__(self, *args, map_location="cpu", **kwargs):
        self._map_location = map_location
        super().__init__(*args, **kwargs)

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return fix(self._map_location)
        else:
            return super().find_class(module, name)


def mapped_loads(s, map_location="cpu"):
    bs = io.BytesIO(s)
    unpickler = MappedUnpickler(bs, map_location=map_location)
    return unpickler.load()


def read_best_config(out_path):
    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            best_config = pkl.load(f)

        return best_config


def parse_best_config_from_dir(outdir, config_name):
    filenames = os.listdir(outdir)
    best_config = {}

    for filename in filenames:
        if os.path.isdir(filename):
            continue
        if not filename.endswith(".pkl"):
            continue

        with open(os.path.join(outdir, filename), "rb") as f:
            content = f.read()
            results_dict = mapped_loads(content)

        for dataset in results_dict:
            for eps in results_dict[dataset]:
                for arch in results_dict[dataset][eps]:
                    for config in results_dict[dataset][eps][arch]:
                        train_stats = results_dict[dataset][eps][arch][config]
                        best_val_loss, _ = train_stats.get_best_avg_val()
                        if not dataset in best_config:
                            best_config[dataset] = {
                                eps: {arch: [config, best_val_loss, train_stats]}
                            }
                        elif not eps in best_config[dataset]:
                            best_config[dataset][eps] = {
                                arch: [config, best_val_loss, train_stats]
                            }
                        elif not arch in best_config[dataset][eps]:
                            best_config[dataset][eps][arch] = [
                                config,
                                best_val_loss,
                                train_stats,
                            ]
                        else:
                            if best_val_loss < best_config[dataset][eps][arch][1]:
                                best_config[dataset][eps][arch] = [
                                    config,
                                    best_val_loss,
                                    train_stats,
                                ]

    with open(config_name, "wb") as f:
        pkl.dump(best_config, f)

    return best_config


def create_todos(datasets, eps_list, architecture_names, configs, TODOS_DIR):
    todos = []
    tasks_to_run = list(product(datasets, eps_list, architecture_names, configs))
    if not os.path.exists(TODOS_DIR):
        for task in tasks_to_run:
            dataset = task[0]
            eps = task[1]
            arch = task[2]
            lr = task[3][0]
            hidden_size = task[3][1]
            num_hidden = task[3][2]
            dropout = task[3][3]

            if arch.startswith("GCN"):
                if len(arch.split("_")) > 1:
                    rank = int(arch.split("_")[1][4:])
                else:
                    rank = 0
            else:
                rank = 0
                
            if arch.startswith("MLP") or arch.startswith("Simple"):
                if eps > 0:
                    continue
            
            if arch.startswith("MLP") or arch.startswith("Simple") or arch.startswith("MMLP"):
                if rank > 0:
                    continue
            
            if dataset == utils.Dataset.TwitchES:
                for test_dataset in (
                    utils.Dataset.TwitchDE,
                    utils.Dataset.TwitchENGB,
                    utils.Dataset.TwitchFR,
                    utils.Dataset.TwitchPTBR,
                    utils.Dataset.TwitchRU,
                ):
                    todos.append(
                        f"{arch}-{dataset.name}_{test_dataset.name}-{eps}-{lr}"
                        f"-{hidden_size}-{num_hidden}-{dropout}"
                    )
            else:
                todos.append(
                    f"{arch}-{dataset.name}-{eps}-{lr}-{hidden_size}-{num_hidden}-{dropout}"
                )

        Path(TODOS_DIR).mkdir(exist_ok=True, parents=True)
        for todo in todos:
            if not os.path.exists(os.path.join(TODOS_DIR, todo)):
                f = open(os.path.join(TODOS_DIR, todo), "w")
                f.close()
        print("{} created successfully.".format(TODOS_DIR))
    else:
        print("{} already exists. You should delete it if you want to create new todos.".format(TODOS_DIR))
    # try:
    #     os.mkdir(TODOS_DIR)
    # except Exception:
    #     print("Cannot create {}".format(TODOS_DIR))
    #     pass


def create_best_todos(datasets, eps_list, architecture_names, best_configs, TODOS_DIR):
    todos = []
    tasks_to_run = list(product(datasets, eps_list, architecture_names))
    if not os.path.exists(TODOS_DIR):
        for task in tasks_to_run:
            dataset = task[0]
            eps = task[1]
            arch = task[2]

            if arch.startswith("MLP") or arch.startswith("Simple"):
                if eps > 0:
                    continue

            # Inconsistent but :(
            if arch in ("GCN", "MLP"):
                arch = utils.Architecture[arch]
                arch_name = arch.name
            elif arch.startswith("GCN"):
                arch_name = arch    # keep arch_name in sweep rank experiment
                arch = "GCN_rank" + arch.split("_")[1][4:] # change arch to use the same best config file of rank 20 for different ranks
            else:
                arch_name = arch
            if dataset == utils.Dataset.TwitchES:
                for test_dataset in (
                    utils.Dataset.TwitchDE,
                    utils.Dataset.TwitchENGB,
                    utils.Dataset.TwitchFR,
                    utils.Dataset.TwitchPTBR,
                    utils.Dataset.TwitchRU,
                ):

                    if (dataset, test_dataset) not in best_configs:
                        print("{}_{} missing".format(dataset, test_dataset))
                        pass
                    bc = best_configs[(utils.Dataset.TwitchES, utils.Dataset.TwitchFR)]
                    lr = bc[0][arch][2].run_config.learning_rate
                    hidden_size = bc[0][arch][2].run_config.hidden_size
                    num_hidden = bc[0][arch][2].run_config.num_hidden
                    dropout = bc[0][arch][2].run_config.dropout
                    todos.append(
                        f"{arch_name}-{dataset.name}_{test_dataset.name}-{eps}-{lr}"
                        f"-{hidden_size}-{num_hidden}-{dropout}"
                    )
            elif dataset not in best_configs:
                print("{} missing".format(dataset))
                continue
            else:
                lr = best_configs[dataset][0][arch][2].run_config.learning_rate
                hidden_size = best_configs[dataset][0][arch][2].run_config.hidden_size
                num_hidden = best_configs[dataset][0][arch][2].run_config.num_hidden
                dropout = best_configs[dataset][0][arch][2].run_config.dropout

                todos.append(
                    f"{arch_name}-{dataset.name}-{eps}-{lr}-{hidden_size}-{num_hidden}-{dropout}"
                )

        Path(TODOS_DIR).mkdir(exist_ok=True, parents=True)
        for todo in todos:
            if not os.path.exists(os.path.join(TODOS_DIR, todo)):
                f = open(os.path.join(TODOS_DIR, todo), "w")
                f.close()
        print("{} created successfully.".format(TODOS_DIR))
    else:
        print("{} already exists. You should delete it if you want to create new todos.".format(TODOS_DIR))
    # try:
    #     os.mkdir(TODOS_DIR)
    # except Exception:
    #     print("Cannot create {}".format(TODOS_DIR))
    #     pass


def attack_for_config(
    device,
    outdir,
    num_epochs,
    seeds,
    attack_modes,
    sample_types,
    influence,
    TODOS_DIR,
    DONE_DIR,
):
    Path(outdir).mkdir(exist_ok=True, parents=True)
    Path(DONE_DIR).mkdir(exist_ok=True, parents=True)
    todos_files = os.listdir(TODOS_DIR)

    while todos_files:
        todo = np.random.choice(todos_files)
        todos_files.remove(todo)
        todo_path = os.path.join(TODOS_DIR, todo)
        working_path = os.path.join(DONE_DIR, todo)
        print("Working on {}".format(working_path))
        try:
            shutil.move(todo_path, working_path)
        except IOError:
            continue

        params = todo.split("-")
        arch = utils.Architecture[params[0].split("_")[0]]

        if params[1].startswith("Twitch"):
            datasets = params[1].split("_")
            dataset = utils.Dataset[datasets[0]]
            test_dataset = utils.Dataset[datasets[1]]
        else:
            dataset = utils.Dataset[params[1]]
            test_dataset = None
        
        eps = float(todo.split("-")[2])
        if int(eps) == 0 and len(todo.split("-")[2])==1:
            w_dp = False
        else:
            w_dp = True
        lr = float(params[3])
        hidden_size = int(params[4])
        num_hidden = int(params[5])
        dropout = float(params[6])

        if arch == utils.Architecture.GCN:
            if len(params[0].split("_")) > 1:
                rank = int(params[0].split("_")[1][4:])
            else:
                rank = 0
        else:
            rank = 0
                
        if rank == 0:
            svd = False
        else:
            svd = True

        if arch == utils.Architecture.MLP and num_hidden == 1:
            continue

        if arch == utils.Architecture.MMLP or arch == utils.Architecture.SimpleMMLP:
            nl = int(params[0].split("_")[1][2:])
            print(
                "Running dataset={} arch={} nl={} eps={}".format(dataset, arch, nl, eps)
            )
        else:
            nl = 1
            if rank == 0:
                print("Running dataset={} arch={} eps={}".format(dataset, arch, eps))
            else:
                print("Running dataset={} arch={} rank={} eps={}".format(dataset, arch, rank, eps))

        run_config = RunConfig(
            learning_rate=lr,
            num_epochs=num_epochs,
            save_each_epoch=False,
            save_epoch=100,
            weight_decay=5e-4,
            output_dir=os.path.join(outdir, "models"),
            eps=eps,
            rank=rank,
            nl=nl,
            hidden_size=hidden_size,
            num_hidden=num_hidden,
            dropout=dropout,
        )
        print(run_config)
        for attack_mode in attack_modes:
            for sample_type in sample_types:
                for seed in seeds:
                    model_paths = []
                    if (
                        arch == utils.Architecture.MMLP
                        or arch == utils.Architecture.SimpleMMLP
                    ):
                        if not test_dataset:
                            it = 0
                            model_path = (
                                f"arch_{arch}_{nl}_{it}-dataset_{dataset.name}-seed_{seed}-lr_{lr}-"
                                f"hidden_size_{hidden_size}-num_hidden_{num_hidden}-dropout_{dropout}-eps_{eps}"
                            )

                            model_paths.append(
                                os.path.join(model_path, model_path + ".pth")
                            )
                            for it in range(1, nl + 1):
                                model_path = (
                                    f"arch_{arch}_{nl}_{it}-dataset_{dataset.name}-seed_{seed}-lr_{lr}-"
                                    f"hidden_size_{hidden_size}-num_hidden_{num_hidden}-dropout_{dropout}-eps_{eps}"
                                )

                                model_paths.append(
                                    os.path.join(model_path, model_path + ".pth")
                                )
                        else:
                            it = 0
                            model_path = (
                                f"arch_{arch}_{nl}_{it}-dataset_{dataset.name}_{test_dataset.name}-seed_{seed}-lr_{lr}-"
                                f"hidden_size_{hidden_size}-num_hidden_{num_hidden}-dropout_{dropout}-eps_{eps}"
                            )

                            model_paths.append(
                                os.path.join(model_path, model_path + ".pth")
                            )
                            for it in range(1, nl + 1):
                                model_path = (
                                    f"arch_{arch}_{nl}_{it}-dataset_{dataset.name}_{test_dataset.name}-seed_{seed}-lr_{lr}-"
                                    f"hidden_size_{hidden_size}-num_hidden_{num_hidden}-dropout_{dropout}-eps_{eps}"
                                )

                                model_paths.append(
                                    os.path.join(model_path, model_path + ".pth")
                                )
                    else:
                        nl = -1
                        if num_hidden == 2:
                            if arch == utils.Architecture.GCN:
                                if rank == 0:
                                    arch_name = "2layergcn"
                                else:
                                    arch_name = f"2layergcn-rank_{rank}"
                            else:
                                arch_name = "mlp"
                        elif num_hidden == 3:
                            if arch == utils.Architecture.GCN:
                                if rank == 0:
                                    arch_name = "3layergcn"
                                else:
                                    arch_name = f"3layergcn-rank_{rank}"
                            else:
                                arch_name = "mlp"
                        else:
                            print("num hidden not recognized")
                            exit()
                        
                        model_path = (
                            f"arch_{arch_name}-dataset_{dataset.name}-seed_{seed}-lr_{lr}-"
                            f"hidden_size_{hidden_size}-num_hidden_{num_hidden}-dropout_{dropout}-eps_{eps}"
                        )
                        if test_dataset:
                            model_path = (
                                f"arch_{arch_name}-dataset_{dataset.name}_{test_dataset.name}-seed_{seed}-lr_{lr}-"
                                f"hidden_size_{hidden_size}-num_hidden_{num_hidden}-dropout_{dropout}-eps_{eps}"
                            )

                        model_paths = [os.path.join(model_path, model_path + ".pth")]
                    if rank > 0:
                        print(
                            "Running dataset={} arch={} model_paths={} eps={} rank={} "
                            "attack_mode={} sample_type={}".format(
                                dataset, arch, model_paths, eps, rank, attack_mode, sample_type
                            )
                        )
                    else:
                        print(
                            "Running dataset={} arch={} model_paths={} eps={} "
                            "attack_mode={} sample_type={}".format(
                                dataset, arch, model_paths, eps, attack_mode, sample_type
                            )
                        )
                    rng = np.random.default_rng(seed)
                    main.run_attack(
                        dataset,
                        arch,
                        run_config,
                        model_paths,
                        device,
                        influence,
                        attack_mode,
                        sample_type,
                        "vanilla-clean",
                        seed,
                        rng,
                        outdir,
                        500,
                        test_dataset,
                    )
    # utils.write_to_csv(results_dict, "training.csv")


def train_for_config(device, outdir, num_epochs, seeds, TODOS_DIR, DONE_DIR):

    Path(outdir).mkdir(exist_ok=True, parents=True)
    Path(DONE_DIR).mkdir(exist_ok=True, parents=True)
    todos_files = os.listdir(TODOS_DIR)

    results_dict = {}
    while todos_files:
        results_dict = {}
        todo = np.random.choice(todos_files)
        todos_files.remove(todo)
        todo_path = os.path.join(TODOS_DIR, todo)
        working_path = os.path.join(DONE_DIR, todo)
        print("Working on {}".format(working_path))
        try:
            shutil.move(todo_path, working_path)
        except IOError:
            continue

        params = todo.split("-")
        arch = utils.Architecture[params[0].split("_")[0]]

        if params[1].startswith("Twitch"):
            datasets = params[1].split("_")
            dataset = utils.Dataset[datasets[0]]
            test_dataset = utils.Dataset[datasets[1]]
        else:
            dataset = utils.Dataset[params[1]]
            test_dataset = None

        eps = float(todo.split("-")[2])
        if int(eps) == 0 and len(todo.split("-")[2])==1:
            w_dp = False
        else:
            w_dp = True
        
        rank = 0
        if arch == utils.Architecture.GCN:
            if len(params[0].split("_")) > 1:
                rank = int(params[0].split("_")[1][4:])
            else:
                rank = 0
                
        if rank == 0:
            svd = False
        else:
            svd = True

        lr = float(params[3])
        hidden_size = int(params[4])
        num_hidden = int(params[5])
        dropout = float(params[6])

        if arch == utils.Architecture.MLP and num_hidden == 1:
            continue

        if arch == utils.Architecture.MMLP or arch == utils.Architecture.SimpleMMLP:
            nl = int(params[0].split("_")[1][2:])
            print(
                "Running dataset={} arch={} nl={} eps={}".format(dataset, arch, nl, eps)
            )
        else:
            nl = 1
            if rank == 0:
                print("Running dataset={} arch={} eps={}".format(dataset, arch, eps))
            else:
                print("Running dataset={} arch={} rank={} eps={}".format(dataset, arch, rank, eps))

        run_config = RunConfig(
            learning_rate=lr,
            num_epochs=num_epochs,
            save_each_epoch=False,
            save_epoch=100,
            weight_decay=5e-4,
            output_dir=os.path.join(outdir, "models"),
            eps=eps,
            rank=rank,
            nl=nl,
            hidden_size=hidden_size,
            num_hidden=num_hidden,
            dropout=dropout,
        )

        train_stats = main.run_training(
            run_config,
            arch,
            dataset,
            device,
            dp=w_dp,
            svd=svd,
            seeds=seeds,
            test_dataset=test_dataset,
        )
        if arch == utils.Architecture.MMLP or arch == utils.Architecture.SimpleMMLP:
            arch = arch.name + "_nl" + str(nl)

        if arch == utils.Architecture.GCN:
            if rank > 0:
                arch = arch.name + "_rank" + str(rank)
            
        if dataset == utils.Dataset.TwitchES:
            es_identifier = dataset.value[dataset.value.find("/") + 1 :]
            identifier = test_dataset.value[test_dataset.value.find("/") + 1 :]
            dataset_name = es_identifier + "_" + identifier
            dataset = (dataset, test_dataset)
        else:
            dataset_name = str(dataset)

        config = f"{lr}-{hidden_size}-{num_hidden}-{dropout}"
        curr_dict = {dataset: {eps: {arch: {config: train_stats}}}}

        if not dataset in results_dict:
            results_dict = curr_dict
        elif not eps in results_dict[dataset]:
            results_dict[dataset][eps] = curr_dict[dataset][eps]
        elif not arch in results_dict[dataset][eps]:
            results_dict[dataset][eps][arch] = curr_dict[dataset][eps][arch]
        else:
            results_dict[dataset][eps][arch][config] = curr_dict[dataset][eps][arch][
                config
            ]

        utils.save_results_pkl(curr_dict, outdir, arch, dataset_name, run_config)
    return results_dict


def run_all_commands(device, args):
    seeds = utils.get_seeds(args.num_seeds)
    if args.hyperparameters == True and args.best_config_file is not None:
        print("You either do hyperparameter search or run for best config")
        exit()
    if args.parse_config_dir:
        print("Parsing configs and writing them to {}".format(args.best_config_file))
        parse_best_config_from_dir(args.parse_config_dir, args.best_config_file)
        return ()

    print("We run for these seeds {}".format(seeds))
    TODOS_DIR = os.path.join(args.todos_dir, "working")
    DONE_DIR = os.path.join(args.todos_dir, "done")
    
    eps_list = [x for x in range(0, args.max_eps + 1)] + [x/10.0 for x in range(1, 10)]
    # eps_list = [0.1,0.5,1,5,10]
    architecture_names = []
    
    if args.sweep_rank:
        rank_list = [10,20,50,100,200,300,400]
    else:
        rank_list = [args.rank]

    for rank in rank_list:   
        for arch in utils.Architecture:
            # Skip SimpleMMLP for experiments
            if arch == utils.Architecture.SimpleMMLP:
                continue
            if arch == utils.Architecture.TwoLayerGCN:
                continue
            if "mmlp" in arch.value:
                for i in range(args.max_stacked, args.max_stacked + 1): # only check lpgnet-2
                    arch_name = arch.name + "_nl" + str(i)
                    architecture_names.append(arch_name)
            else:
                if arch == utils.Architecture.GCN:                
                    # vanilla GCN (w/o SVD)
                    architecture_names.append(arch.name)
                    
                    # GCN with SVD
                    if rank > 0:
                        arch_name = arch.name + "_rank" + str(rank)
                        architecture_names.append(arch_name)
                else: # MLP
                    arch_name = arch.name
                    architecture_names.append(arch_name)
    
    # run LPA and LINKTELLER attacks
    attack_modes = ["baseline", "efficient"]

    if args.datasets:
        datasets = [utils.Dataset[x] for x in args.datasets.split(",")]
        sample_types = ["balanced"]
        if utils.Dataset.TwitchES in datasets:
            sample_types = ["unbalanced_hi", "unbalanced_lo", "unbalanced"]
    else:
        if not args.inductive:
            if args.attack_bd:  # attack number breakdown experiments
                sample_types = ["balanced","balanced_lorank", "balanced_hirank","balanced_lo","balanced_hi"]
            else:
                sample_types = ["balanced"]
        
            # Transductive setting datasets
            datasets = [
                utils.Dataset.Chameleon, 
                utils.Dataset.Cora,      
                utils.Dataset.CiteSeer,
                utils.Dataset.PubMed,
                utils.Dataset.facebook_page,
            ]
        else:
            if args.attack_bd:  # attack number breakdown experiments
                sample_types = ["unbalanced_hi", "unbalanced_lo", "unbalanced", "unbalanced_lorank", "unbalanced_hirank"]
            else:
                sample_types = ["unbalanced_hi", "unbalanced_lo", "unbalanced"]
    
            # Inductive setting datasets
            datasets = [
                utils.Dataset.TwitchES, 
            ]

    print("TODOS DIR {} DONE DIR {}".format(TODOS_DIR, DONE_DIR))
    # training parameters, there is no batch size as we use the whole set in each iteration
    if args.hyperparameters:
        learning_rates = [0.01, 0.05, 0.005, 0.001]
        hidden_sizes_1 = [16, 64, 256]
        num_hidden = [2, 3]
        dropout_rates = [0.5, 0.3, 0.1]
        eps_list = [0]
        configs = list(
            product(learning_rates, hidden_sizes_1, num_hidden, dropout_rates)
        )
        create_todos(datasets, eps_list, architecture_names, configs, TODOS_DIR)
        if args.only_create_todos:
            return
    elif args.best_config_file:
        configs = read_best_config(args.best_config_file)
        create_best_todos(datasets, eps_list, architecture_names, configs, TODOS_DIR)
        if args.only_create_todos:
            return
    else:
        print(args.datasets)
        # The hyperparameters in the LinkTeller paper give better performance
        learning_rates = [0.01]
        hidden_sizes_1 = [16]
        num_hidden = [2]
        dropout_rates = [0.5]
        configs = list(
            product(learning_rates, hidden_sizes_1, num_hidden, dropout_rates)
        )
        create_todos(datasets, eps_list, architecture_names, configs, TODOS_DIR)
        if args.only_create_todos:
            return

    if args.command == "train":
        train_for_config(
            device, args.outdir, args.num_epochs, seeds, TODOS_DIR, DONE_DIR
        )
    elif args.command == "attack":

        attack_for_config(
            device,
            args.outdir,
            args.num_epochs,
            seeds,
            attack_modes,
            sample_types,
            args.influence,
            TODOS_DIR,
            DONE_DIR,
        )
    else:
        print(args.command)
        exit()


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_seeds", type=int, default=30)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--max_stacked", type=int, default=2)
    parser.add_argument("--max_eps", type=int, default=10)
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument("--sweep_rank", default=False, action="store_true")
    parser.add_argument("--attack_bd", default=False, action="store_true")
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument(
        "--outdir",
        type=str,
        default="../data-test",
        help="Directory to save the models and results",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--influence", type=float, default=0.001)
    parser.add_argument("--inductive", default=False, action="store_true")
    parser.add_argument(
        "--command", type=str, choices=["train", "attack"], default="attack"
    )
    parser.add_argument("--hyperparameters", action="store_true", default=False)
    parser.add_argument("--parse_config_dir", type=str, default=None)
    parser.add_argument("--best_config_file", type=str, default=None)
    parser.add_argument("--distribute", action="store_true", default=False)
    parser.add_argument("--todos_dir", type=str, default=None)
    parser.add_argument("--only_create_todos", action="store_true", default=False)
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Specific datasets separated by comma",
    )

    args = parser.parse_args()

    device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')

    if torch.cuda.is_available():
        print("Current CUDA device: {}".format(torch.cuda.current_device()))

    run_all_commands(device, args)


if __name__ == "__main__":
    run()
