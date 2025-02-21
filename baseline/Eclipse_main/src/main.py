import os
import trainer
import utils
from data import LoadData
import argparse
import numpy as np
import torch
import time
import attacker
import models
from globals import MyGlobals

import psutil

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

def train(args):
    # Some sanity checks on the arguments
    if args.w_dp == True and args.eps == 0.0:
        print("You selected with DP but eps=0.0")
        exit()

    if args.w_dp == False and args.eps > 0:
        print("No DP selected")
        exit()

    if (args.w_dp == True or args.eps > 0) and args.arch == utils.Architecture.MLP:
        print("MLP is by default private, no need to select DP")
        exit()

    # Some sanity checks on the SVD arguments
    if args.svd == True and args.rank == 0:
        print("You selected with SVD but rank=0")
        exit()
    
    if args.svd == False and args.rank > 0:
        print("No SVD selected")
        exit()
            
    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
        # Run the train/attack/eval on the selected GPU id
        if torch.cuda.is_available():
            # torch.cuda.set_device(args.cuda_id)
            # print("Current CUDA device: {}".format(torch.cuda.current_device()))
            print(f"Current CUDA device: {torch.cuda.current_device()}")

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
    train_stats = run_training(
        run_config,
        args.arch,
        args.dataset,
        device,
        dp=args.w_dp,
        svd=args.svd,
        seeds=seeds,
        test_dataset=args.test_dataset,
    )

    results_dict = {args.dataset: {args.eps: {}}}
    results_dict[args.dataset][args.eps][args.arch] = train_stats

    # Saving the results
    arch_name = args.arch.name
    if args.arch == utils.Architecture.MMLP:
        arch_name = args.arch.name + "_nl" + str(run_config.nl)
    if args.arch == utils.Architecture.GCN:
        if run_config.rank > 0:
            arch_name = args.arch.name + "_rank" + str(run_config.rank)
    if args.dataset == utils.Dataset.TwitchES:
        es_identifier = args.dataset.value[args.dataset.value.find("/") + 1 :]
        identifier = args.test_dataset.value[args.test_dataset.value.find("/") + 1 :]
        args.dataset = (args.dataset, args.test_dataset)
        dataset_name = es_identifier + "_" + identifier
    else:
        dataset_name = str(args.dataset)
    utils.save_results_pkl(
        results_dict, args.outdir, arch_name, dataset_name, run_config
    )


def run_training(
    run_config, arch, dataset, device, dp=False, svd=False, seeds=[1], test_dataset=None
):
    print("Train and evaluate arch {}; seeds {}".format(arch.name, seeds))
    start_time = time.time()
    if test_dataset:
        print("Transfer learning with separate test graph")

    if arch == utils.Architecture.GCN:
        train_stats = trainer.train_gcn_on_dataset(
            run_config, dataset, device, dp=dp, svd=svd, seeds=seeds, test_dataset=test_dataset
        )
    elif arch == utils.Architecture.MLP:
        if run_config.eps > 0 or dp == True:
            print("MLP does not require DP")
            exit()
        train_stats = trainer.train_mlp_on_dataset(
            run_config,
            dataset,
            device,
            seeds=seeds,
            test_dataset=test_dataset,
        )
    elif arch == utils.Architecture.MMLP:
        train_stats = trainer.train_mmlp_on_dataset(
            run_config,
            dataset,
            device,
            dp=dp,
            seeds=seeds,
            test_dataset=test_dataset,
        )
    else:
        print("Arch {} not supported".format(arch))
        return None
    print(f"Training time: {time.time()-start_time}")
    return train_stats


def attack(args):
    print(args)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        # torch.cuda.set_device(args.cuda_id)
        print("Current CUDA device: {}".format(torch.cuda.current_device()))

    seeds = utils.get_seeds(args.num_seeds, args.sample_seed)

    run_config = trainer.RunConfig(
        learning_rate=args.lr,
        num_epochs=500,
        save_each_epoch=False,
        save_epoch=None,
        weight_decay=MyGlobals.weight_decay,
        output_dir=os.path.join(args.outdir, "models"),
        eps=args.eps,
        rank=args.rank,
        nl=args.nl,
        hidden_size=args.hidden_size,
        num_hidden=args.num_hidden,
        dropout=args.dropout,
    )

    print(run_config)
    for i, seed in enumerate(seeds):
        trainer.set_torch_seed(seed)
        rng = np.random.default_rng(seed=seed)
        model_paths = []
        if args.model_path:
            model_paths = args.model_path.split(",")
        else:
            model_paths = utils.construct_model_paths(
                args.arch, args.dataset, run_config, seed, args.test_dataset
            )

        run_attack(
            args.dataset,
            args.arch,
            run_config,
            model_paths,
            device,
            args.influence,
            args.attack_mode,
            args.sample_type,
            "vanilla_clean",
            seed,
            rng,
            args.outdir,
            args.n_test,
            args.test_dataset,
            spnum=i % 10,
        )
    return


def run_attack(
    dataset,
    arch,
    run_config,
    model_paths,
    device,
    influence,
    attack_mode,
    sample_type,
    mode,
    seed,
    rng,
    outdir,
    n_test,
    test_dataset,
    spnum=0,
):
    if run_config.eps > 0 and arch == utils.Architecture.GCN:
        dp_for_loading_data = True
    else:
        dp_for_loading_data = False
    if run_config.rank > 0 and arch == utils.Architecture.GCN:
        svd_for_loading_data = True
    else:
        svd_for_loading_data = False
    
    data_loader = LoadData(
        dataset,
        eps=run_config.eps,
        dp=dp_for_loading_data,
        rank=run_config.rank,
        svd=svd_for_loading_data,
        rng=rng,
        rng_seed=seed,
        test_dataset=test_dataset,
        split_num_for_geomGCN_dataset=spnum,
    )
    print("Loaded data")
    comms_file = None
    if arch == utils.Architecture.MMLP:
        comms_file = utils.get_comms_pkl_file_name(
            f"mmlp_nl{run_config.nl}", dataset, seed, test_dataset, run_config
        )
    model = load_model(
        arch,
        run_config,
        model_paths,
        device,
        data_loader,
        outdir,
        rng,
        attack=True,
        comms_file=comms_file,
    )

    features = data_loader.test_features.to(device)
    test_labels = data_loader.test_labels.to(device)
    if arch == utils.Architecture.GCN:
        adjacency_matrix = data_loader.test_adj_csr.to(
            device
        )  # This is for the attack after loading the model
    else:
        adjacency_matrix = None  # No adjacency matrix required for MLP and LPGNet
    # if arch == utils.Architecture.MMLP:
    #    model.prepare_for_fwd(features, test_labels)
    attack = attacker.Attacker(
        dataset.value,
        model,
        features,
        test_labels,
        adjacency_matrix,
        data_loader.test_adj_orig_csr,
        influence,
        attack_mode,
        mode,
        sample_type,
        seed,
        n_test,
        rng,
        test_dataset,
        rank=run_config.rank,
    )
    print(model.model_list)
    if attack_mode == "efficient":
        if sample_type in [
        "balanced_full", 
        "balanced", 
        "balanced_lorank", 
        "balanced_hirank",
        "balanced_lo", 
        "balanced_hi"
        ]:
            (
                norm_exist,
                norm_nonexist,
            ) = attack.link_prediction_attack_efficient_balanced()
        else:
            norm_exist, norm_nonexist = attack.link_prediction_attack_efficient()
    elif attack_mode == "baseline" and sample_type in [
        "balanced_full", 
        "balanced", 
        "balanced_lorank",
        "balanced_hirank",
        "balanced_lo",
        "balanced_hi"
    ]:
        # LPA with posterior correlation
        norm_exist, norm_nonexist = attack.baseline_attack_balanced()
    elif attack_mode == "baseline" and sample_type in [
        "unbalanced",
        "unbalanced_hi",
        "unbalanced_lo",
    ]:
        # LPA with posterior correlation
        norm_exist, norm_nonexist = attack.baseline_attack()
    elif attack_mode == "baseline_nfeat" and sample_type in [
        "balanced_full", 
        "balanced", 
        "balanced_lorank",
        "balanced_hirank",
        "balanced_lo",
        "balanced_hi"
    ]:
        # LPA with node feature correlation
        norm_exist, norm_nonexist = attack.baseline_attack_balanced_nfeat()
    elif attack_mode == "baseline_nfeat" and sample_type in [
        "unbalanced",
        "unbalanced_hi",
        "unbalanced_lo",
    ]:
        # LPA with node feature correlation
        norm_exist, norm_nonexist = attack.baseline_attack_nfeat()
    else:
        print("{} not supported".format(attack_mode))

    # Saving attack results
    prefix = utils.get_attack_results_save_file_prefix(
        arch, dataset, run_config, seed, test_dataset
    )
    attack.compute_and_save(norm_exist, norm_nonexist, prefix)


def load_model(
    arch,
    run_config,
    model_paths,
    device,
    data_loader,
    outdir,
    rng,
    attack=False,
    comms_file=None,
):
    numpy_adjacency_matrix = None
    if arch == utils.Architecture.MMLP:
        if attack or data_loader.is_inductive():
            numpy_adjacency_matrix = data_loader.test_adj_orig_csr.toarray()
        else:
            numpy_adjacency_matrix = data_loader.train_adj_orig_csr.toarray()

    if arch == utils.Architecture.SimpleMMLP:
        print("{} Not implemented. Skip.".format(arch))
        exit()

    eps = run_config.eps
    if arch == utils.Architecture.MMLP:
        eps = eps * 1.0 / run_config.nl

    model = models.create_model(
        run_config,
        arch,
        data_loader.train_features.size(1),
        data_loader.num_classes,
        device=device,
        adjacency_matrix=numpy_adjacency_matrix,
        eps=eps,
        rng=rng,
    )

    path = [os.path.join(outdir, "models", m_path) for m_path in model_paths]
    if arch == utils.Architecture.MMLP:
        model.load_model_from(path, device, comms_file)
    else:
        model.load_model_from(path, device)
    return model


def load_and_test(args):
    print(args)
    print("Load and evaluate {} model".format(args.model_path))
    device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
    # Run the train/attack/eval on the selected GPU id
    if torch.cuda.is_available():
        # torch.cuda.set_device(args.cuda_id)
        print("Current CUDA device: {}".format(torch.cuda.current_device()))

    run_config = trainer.RunConfig(
        learning_rate=args.lr,
        num_epochs=args.num_epochs,
        save_each_epoch=False,
        save_epoch=None,
        weight_decay=MyGlobals.weight_decay,
        output_dir=os.path.join(args.outdir, "models"),
        eps=args.eps,
        nl=args.nl,
        hidden_size=args.hidden_size,
        num_hidden=args.num_hidden,
        dropout=args.dropout,
    )
    model_paths = []
    if args.model_path:
        model_paths = args.model_path.split(",")
    else:
        model_paths = utils.construct_model_paths(
            args.arch, args.dataset, run_config, seed, args.test_dataset
        )

    accuracies = []
    seeds = utils.get_seeds(args.num_seeds, args.sample_seed)
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(seed=seed)
        if args.arch == utils.Architecture.GCN:
            data_loader = LoadData(
                args.dataset,
                load_dir=str(args.dataset),
                dp=args.w_dp,
                eps=args.eps,
                rng=rng,
                rng_seed=seed,
                test_dataset=args.test_dataset,
                split_num_for_geomGCN_dataset=i % 10,
            )
        else:
            data_loader = LoadData(
                args.dataset,
                dp=False,
                rng=rng,
                rng_seed=seed,
                test_dataset=args.test_dataset,
                split_num_for_geomGCN_dataset=i % 10,
            )

        model = load_model(
            args.arch, run_config, model_paths, device, data_loader, args.outdir, rng
        )
        is_rare = "twitch" in args.dataset.value
        if args.arch == utils.Architecture.MMLP:
            test_loss, test_acc, f1_score, rare_f1_score = trainer.evaluate_mmlp(
                model,
                data_loader.test_features,
                data_loader.test_labels,
                data_loader.test_adj_orig_csr.toarray(),
                device,
                is_rare,
            )
        elif args.arch == utils.Architecture.GCN:
            tr = trainer.Trainer(model, rng, seed=seed)
            test_loss, test_acc, f1_score, rare_f1_score = tr.evaluate(
                data_loader.test_features,
                data_loader.test_labels,
                device,
                data_loader.test_adj_csr,
                test_dataset=args.test_dataset,
                is_rare=is_rare,
            )
        elif args.arch == utils.Architecture.MLP:
            tr = trainer.Trainer(model, rng, seed=seed)
            test_loss, test_acc, f1_score, rare_f1_score = tr.evaluate(
                data_loader.test_features,
                data_loader.test_labels,
                device,
                test_dataset=args.test_dataset,
                is_rare=is_rare,
            )
        print(
            f"\nPerformance on {args.dataset.name}:\n- "
            f"test accuracy = {test_acc:.3f} +- 0.000\n"
        )
        accuracies.append(test_acc)

    return accuracies

@profile
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=utils.Dataset,
        choices=utils.Dataset,
        default=utils.Dataset.Cora,
        help="cora|citeseer|pubmed...",
    )
    parser.add_argument(
        "--arch",
        type=utils.Architecture,
        choices=utils.Architecture,
        default=utils.Architecture.MMLP,
        required=True,
        help="Type of architecture to train: mmlp|gcn|mlp",
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
    parser.add_argument("--cuda_id", type=int, default=MyGlobals.cuda_id, help="Cuda ID to use")
    parser.add_argument("--no_cuda", action="store_true", default=False, help="Disables CUDA training.")
    parser.add_argument(
        "--eps",
        type=float,
        default=MyGlobals.eps,
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

    # Train model commands
    # Should add more -- perhaps set a config file to make it easier to set all of these parameters
    subparsers = parser.add_subparsers(help="sub-command help")

    train_parser = subparsers.add_parser("train", help="train sub-menu help")
    train_parser.add_argument(
        "--lr", type=float, default=MyGlobals.lr, help="Learning rate"
    )
    train_parser.add_argument("--num_epochs", type=int, default=MyGlobals.num_epochs)
    train_parser.add_argument(
        "--save_epoch",
        type=int,
        default=MyGlobals.save_epoch,
        help="Save at every save_epoch",
    )
    train_parser.add_argument("--dropout", type=float, default=MyGlobals.dropout)
    train_parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Store the results in a pkl in args.outdir",
    )
    train_parser.set_defaults(func=train)

    # Load and evaluate trained model commands
    eval_parser = subparsers.add_parser("evaluate", help="test sub-menu help")
    eval_parser.add_argument("--model_path", type=str, default="", help="Names of the saved models (not full paths) seperated by commas.")
    eval_parser.set_defaults(func=load_and_test)

    # model commands
    attack_parser = subparsers.add_parser("attack", help="attack sub-menu help")
    attack_parser.add_argument(
        "--lr", type=float, default=MyGlobals.lr, help="Need this for loading the correct saved model"
    )
    attack_parser.add_argument("--dropout", type=float, default=MyGlobals.dropout, help="Need this for loading the correct saved model")
    attack_parser.add_argument("--model_path", type=str, default=None, help="Name of the saved models (not full paths) seperated by commas.")
    attack_parser.add_argument("--influence", type=float, default=MyGlobals.influence, help="I_TIA factor for LinkTeller attack")
    attack_parser.add_argument(
        "--sample_type",
        type=str,
        default="balanced",
        choices=[
            "balanced",
            "balanced_lo",
            "balanced_hi",
            "unbalanced",
            "unbalanced_lo",
            "unbalanced_hi",
            "balanced_lorank",
            "balanced_hirank",
            "balanced_full"
        ],
        help="Type of samples to use for attack",
    )
    attack_parser.add_argument(
        "--attack_mode",
        type=str,
        default="efficient",
        choices=["efficient", "baseline", "baseline_nfeat"],
        help="Type of attack to run. Baseline: LPA with posterior, Efficient: LinkTeller, Baseline_nfeat: LPA with raw node feature",
    )
    attack_parser.add_argument(
        "--n-test", type=int, default=MyGlobals.n_test, help="The number of nodes"
    )
    attack_parser.set_defaults(func=attack)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
