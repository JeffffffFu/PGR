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
        seeds = np.random.randint(3407, 3407, size=num_seeds)
        print("We run for these seeds {}".format(seeds))
    else:
        seeds = [sample_seed]
    return seeds


def plot_graph(adj, labels, filename="./test.png"):
    cmap = {
        0: "b",
        1: "r",
        2: "g",
        3: "y",
        4: "c",
        5: "m",
        6: "k",
        7: "slateblue",
        8: "orange",
        9: "grey",
    }
    C = [cmap[labels[i]] for i in range(len(adj))]
    G = nx.from_numpy_matrix(adj)
    nx.draw_networkx(G, node_color=C, pos=nx.spring_layout(G, seed=1))
    plt.savefig(filename)


def get_folder_name(run_config, dataset, model_name, seed, epoch=-1, test_dataset=None):
    if model_name == "2layergcn" and run_config.rank > 0:
        model_name = model_name + "-rank_" + str(run_config.rank)
    
    if model_name == "3layergcn" and run_config.rank > 0:
        model_name = model_name + "-rank_" + str(run_config.rank)
        
    if epoch > 0:
        if not test_dataset:
            dir_name = (
                f"arch_{model_name}-dataset_{dataset.name}-"
                f"seed_{seed}-lr_{run_config.learning_rate}-"
                f"hidden_size_{run_config.hidden_size}-"
                f"num_hidden_{run_config.num_hidden}-dropout_{run_config.dropout}-"
                f"epoch_{epoch+1}-eps_{run_config.eps}"
            )
        else:
            dir_name = (
                f"arch_{model_name}-dataset_{dataset.name}_{test_dataset.name}-"
                f"seed_{seed}-lr_{run_config.learning_rate}-"
                f"hidden_size_{run_config.hidden_size}-"
                f"num_hidden_{run_config.num_hidden}-dropout_{run_config.dropout}-"
                f"epoch_{epoch+1}-eps_{run_config.eps}"
            )
    else:
        if not test_dataset:
            dir_name = (
                f"arch_{model_name}-dataset_{dataset.name}-"
                f"seed_{seed}-lr_{run_config.learning_rate}-"
                f"hidden_size_{run_config.hidden_size}-"
                f"num_hidden_{run_config.num_hidden}-dropout_{run_config.dropout}-"
                f"eps_{run_config.eps}"
            )
        else:
            dir_name = (
                f"arch_{model_name}-dataset_{dataset.name}_{test_dataset.name}-"
                f"seed_{seed}-lr_{run_config.learning_rate}-"
                f"hidden_size_{run_config.hidden_size}-"
                f"num_hidden_{run_config.num_hidden}-dropout_{run_config.dropout}-"
                f"eps_{run_config.eps}"
            )
    return dir_name


def save_results_pkl(results_dict, outdir, arch, dataset, run_config):
    filename = os.path.join(
        outdir,
        "{}-{}-{}-lr_{}-hidden_size_{}-num_hidden_{}"
        "-dropout_{}-training.pkl".format(
            arch,
            dataset,
            run_config.eps,
            run_config.learning_rate,
            run_config.hidden_size,
            run_config.num_hidden,
            run_config.dropout,
        ),
    )

    print("Saved results at {}".format(filename))
    with open(filename, "wb") as f:
        pkl.dump(results_dict, f)


def get_comms_pkl_file_name(model_name, dataset, seed, test_dataset, run_config):
    dir_name = get_folder_name(
        run_config,
        dataset,
        model_name,
        seed,
        test_dataset=test_dataset,
    )
    comms_file = os.path.join(run_config.output_dir, dir_name) + "_comms.txt"
    return comms_file


def save_comms_pkl(comms_file, comms):
    print("Saved comms at {}".format(comms_file))
    with open(comms_file, "wb") as f:
        pkl.dump(comms, f)


def load_comms_pkl(comms_file):
    print("Loading comms from {}".format(comms_file))
    with open(comms_file, "rb") as f:
        comms = pkl.load(f)
    return comms


def write_to_csv(results, path):
    """Expecting to have the following keys: lr, dataset, eps, arch.
    The columns will be the Dataset, Epsilon, Lr, {all of the arch keys}
    """
    column_names = []
    arch_names = set()
    architectures = set()

    for lr in results:
        for dataset in results[lr]:
            for eps in results[lr][dataset]:
                for arch in results[lr][dataset][eps]:
                    arch_names.add(arch + " F1 Mean Score")
                    arch_names.add(arch + " F1 Std Score")
                    architectures.add(arch)

    column_names = ["Dataset", "Epsilon", "Lr"] + list(arch_names)

    with open(path, "w") as f:
        csv_writer = csv.writer(f, delimiter=",")
        csv_writer.writerow(column_names)
        for lr in results:
            for dataset in results[lr]:
                for eps in results[lr][dataset]:
                    stats = []
                    for arch in architectures:
                        if arch in results[lr][dataset][eps]:
                            # This returns a TrainStats object
                            f1_scores = np.stack(
                                results[lr][dataset][eps][arch].f1_scores, axis=1
                            )
                            mean = np.mean(f1_scores[0])
                            std = np.std(f1_scores[0])
                            stats.append(mean)
                            stats.append(std)
                        else:
                            stats.append(-1)
                            stats.append(-1)

                    row = [dataset, eps, lr] + stats
                    csv_writer.writerow(row)


def construct_model_paths(arch, dataset, run_config, seed, test_dataset):
    hidden_size = run_config.hidden_size
    num_hidden = run_config.num_hidden
    dropout = run_config.dropout
    lr = run_config.learning_rate
    eps = run_config.eps
    rank = run_config.rank
    nl = run_config.nl
    model_paths = []
    if arch == Architecture.MMLP or arch == Architecture.SimpleMMLP:
        if not test_dataset:
            it = 0
            model_path = (
                f"arch_{arch}_{nl}_{it}-dataset_{dataset.name}-seed_{seed}-lr_{lr}-"
                f"hidden_size_{hidden_size}-num_hidden_{num_hidden}-dropout_{dropout}-eps_{eps}"
            )

            model_paths.append(os.path.join(model_path, model_path + ".pth"))
            for it in range(1, nl + 1):
                model_path = (
                    f"arch_{arch}_{nl}_{it}-dataset_{dataset.name}-seed_{seed}-lr_{lr}-"
                    f"hidden_size_{hidden_size}-num_hidden_{num_hidden}-dropout_{dropout}-eps_{eps}"
                )

                model_paths.append(os.path.join(model_path, model_path + ".pth"))
        else:
            it = 0
            model_path = (
                f"arch_{arch}_{nl}_{it}-dataset_{dataset.name}_{test_dataset.name}-seed_{seed}-lr_{lr}-"
                f"hidden_size_{hidden_size}-num_hidden_{num_hidden}-dropout_{dropout}-eps_{eps}"
            )

            model_paths.append(os.path.join(model_path, model_path + ".pth"))
            for it in range(1, nl + 1):
                model_path = (
                    f"arch_{arch}_{nl}_{it}-dataset_{dataset.name}_{test_dataset.name}-seed_{seed}-lr_{lr}-"
                    f"hidden_size_{hidden_size}-num_hidden_{num_hidden}-dropout_{dropout}-eps_{eps}"
                )

                model_paths.append(os.path.join(model_path, model_path + ".pth"))
    else:
        nl = -1
        if num_hidden == 2:
            if arch == Architecture.GCN:
                if rank == 0:
                    arch_name = "2layergcn"
                else:
                    arch_name = f"2layergcn-rank_{rank}"
            else:
                arch_name = "mlp"
        elif num_hidden == 3:
            if arch == Architecture.GCN:
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
    return model_paths


def get_attack_results_save_file_prefix(arch, dataset, run_config, seed, test_dataset):
    prefix = ""
    if arch == Architecture.MMLP or arch == Architecture.SimpleMMLP:
        if test_dataset:
            prefix = (
                f"{arch.value}_nl{run_config.nl}-{dataset.name},{test_dataset.name}-"
                f"seed_{seed}-lr_{run_config.learning_rate}-hidden_size_{run_config.hidden_size}-"
                f"num_hidedn_{run_config.num_hidden}-dropout_{run_config.dropout}-"
                f"{run_config.eps}"
            )
        else:
            prefix = (
                f"{arch.value}_nl{run_config.nl}-{dataset.name}-"
                f"seed_{seed}-lr_{run_config.learning_rate}-hidden_size_{run_config.hidden_size}-"
                f"num_hidedn_{run_config.num_hidden}-dropout_{run_config.dropout}-"
                f"{run_config.eps}"
            )
    else:
        if arch == Architecture.GCN:
            if run_config.rank > 0:
                arch_name = arch.value + "_rank" + str(run_config.rank)
            else:
                arch_name = arch.value
        else:
            arch_name = arch.value
            
        if test_dataset:
            prefix = (
                f"{arch_name}-{dataset.name},{test_dataset.name}-"
                f"seed_{seed}-lr_{run_config.learning_rate}-hidden_size_{run_config.hidden_size}-"
                f"num_hidedn_{run_config.num_hidden}-dropout_{run_config.dropout}-"
                f"{run_config.eps}"
            )
        else:
            prefix = (
                f"{arch_name}-{dataset.name}-"
                f"seed_{seed}-lr_{run_config.learning_rate}-hidden_size_{run_config.hidden_size}-"
                f"num_hidedn_{run_config.num_hidden}-dropout_{run_config.dropout}-"
                f"{run_config.eps}"
            )
    return prefix
