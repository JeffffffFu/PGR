import numpy as np
import torch
from sklearn import metrics
import os
from globals import MyGlobals


def getF1Scores(tobj):
    f1_scores = np.stack(tobj.f1_scores, axis=1)
    mean = np.mean(f1_scores[0])
    std = np.std(f1_scores[0])
    return mean, std


def write_to_csv(results, datasets, att_types, sample_types, epss, archs,results_dir):
    import csv

    archs = list(archs)
    archs_n = []
    archs.sort()
    for arch in archs:
        archs_n.append(arch + "_mean")
        archs_n.append(arch + "_std")
    epss = list(epss)
    epss.sort()
    datasets = list(datasets)
    att_types = list(att_types)
    sample_types = list(sample_types)
    column_names = ["Epsilon"] + list(archs_n)

    for dataset in datasets:
        for att_type in att_types:
            for sample_type in sample_types:
                # t = f"result_attack_{dataset}_{att_type}_{sample_type}_nfeat.csv" #for node feature attack
                t = f"result_attack_{dataset}_{att_type}_{sample_type}.csv"
                # fname = os.path.join(MyGlobals.RESULTDIR, t)
                fname = os.path.join(results_dir, t)
                print(f"Writing to {fname}")
                with open(fname, "w") as f:
                    csv_writer = csv.writer(f, delimiter=",")
                    csv_writer.writerow(column_names)
                    for eps in epss:
                        line = [eps]
                        for arch in archs:
                            mean, std = -1, -1
                            try:
                                mean, std = results[dataset][(att_type, sample_type)][
                                    eps
                                ][arch]
                            except:
                                pass
                            line.append(mean)
                            line.append(std)
                        csv_writer.writerow(line)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Folder to parse with all the models (pth files)",
        required=True,
    )
    # parser.add_argument("--out_path", type=str, help="Path to the parsed CSV", required=True)

    args = parser.parse_args()
    results_files = os.listdir(args.results_dir)

    datasets = set()
    epss = set()
    archs = set()
    att_types = set()
    sample_types = set()
    results = {}
    for result_file in results_files:
        if not result_file.endswith(".pt"):
            continue
        
        splits = result_file.split("-")
        arch = splits[0]
        if not "mmlp" in arch:
            if not "rank" in arch:
                arch = arch.split("_")[0]
        
        archs.add(arch)
        dataset = splits[1]
        dataset = dataset.lower()
        
        datasets.add(dataset)
        eps = float(splits[7])
        epss.add(eps)
        att_type = splits[8]
        att_types.add(att_type)
        sample_type = splits[9]
        sample_types.add(sample_type)
        if not dataset in results:
            results[dataset] = {}
        if not (att_type, sample_type) in results[dataset]:
            results[dataset][(att_type, sample_type)] = {}
        if not eps in results[dataset][(att_type, sample_type)]:
            results[dataset][(att_type, sample_type)][eps] = {}

        f = os.path.join(args.results_dir, result_file)
        try:
            res = torch.load(f)
        except:
            print(f"Something wrong with this file {f}")
            continue
        auc = metrics.auc(res["auc"]["fpr"], res["auc"]["tpr"])
        if not arch in results[dataset][(att_type, sample_type)][eps]:
            results[dataset][(att_type, sample_type)][eps][arch] = []
        results[dataset][(att_type, sample_type)][eps][arch].append(auc)

    # replace the auc score list with mean and std
    for dataset in results:
        print(f"in first loop")
        for t in results[dataset]:
            print(f"in second loop")
            for eps in results[dataset][t]:
                for arch in results[dataset][t][eps]:
                    a = results[dataset][t][eps][arch]
                    results[dataset][t][eps][arch] = (
                        round(np.mean(a), 4),
                        round(np.std(a), 4),
                    )
    write_to_csv(results, datasets, att_types, sample_types, epss, archs,args.results_dir)
