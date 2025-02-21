import numpy as np
import pickle as pkl
import os
import torch
import io
from globals import MyGlobals


class CPU_Unpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def getF1Scores(tobj, rare=True):
    f1_scores = np.stack(tobj.f1_scores, axis=1)
    if rare:
        f1_scores = np.stack(tobj.rare_f1_scores, axis=1)
    mean = np.mean(f1_scores[0])
    std = np.std(f1_scores[0])
    return mean, std


def write_to_csv(results, datasets, epss, archs, results_dir):
    import csv

    archs = list(archs)
    archs.sort()
    archs_n = []
    for arch in archs:
        archs_n.append(arch + "_mean")
        archs_n.append(arch + "_std")
    epss = list(epss)
    epss.sort()
    datasets = list(datasets)
    column_names = ["Epsilon"] + list(archs_n)

    for dataset in datasets:
        t = "result" + dataset + ".csv"
        # fname = os.path.join(MyGlobals.RESULTDIR, t)
        fname = os.path.join(results_dir, t)
        with open(fname, "w") as f:
            csv_writer = csv.writer(f, delimiter=",")
            csv_writer.writerow(column_names)
            for eps in epss:
                line = [eps]
                for arch in archs:
                    mean, std = -1, -1
                    try:
                        mean, std = results[dataset][eps][arch]
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
        help="Folder to parse with all the pkl results",
        required=True,
    )
    # parser.add_argument("--out_path", type=str, help="Path to the parsed CSV", required=True)

    args = parser.parse_args()

    results_files = os.listdir(args.results_dir)
    datasets = set()
    epss = set()
    archs = set()
    results = {}
    for result_file in results_files:
        if not result_file.endswith(".pkl"):
            continue

        splits = result_file.split("-")
        arch = splits[0]
        arch = arch.lower()
        archs.add(arch)
        dataset = splits[1]
        datasets.add(dataset)
        eps = float(splits[2])
        epss.add(eps)

        if not dataset in results:
            results[dataset] = {}
        if not eps in results[dataset]:
            results[dataset][eps] = {}

        with open(os.path.join(args.results_dir, result_file), "rb") as f:
            res = CPU_Unpickler(f).load()
            tobj = list(
                list(list(list(res.items())[0][1].items())[0][1].items())[0][1].items()
            )[0][1]
        
        mean, std = getF1Scores(tobj, rare=True)

        results[dataset][eps][arch] = (round(mean, 4), round(std, 4))
        write_to_csv(results, datasets, epss, archs,args.results_dir)
