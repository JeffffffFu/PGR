# Edge Private Graph Neural Networks with Singular Value Perturbation
This repository contains code for the PETS 2024 paper "Edge Private Graph Neural Networks with Singular Value Perturbation".

## Dependencies
We suggest using Ananconda for managing the environment. The complete list of required software pacakges are in environment.yml file.

To set up conda and create a conda environment based on environment.yml,
```
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh
conda env create -n [env_name] --file environment.yml
```

Some key dependency are listed below:

```
- Python >= 3.8
- PyTorch >= 2.0
- PyG >= 2.3.1
```

---

## Repository content

### Hardware Requirement
The codebase has been tested on the following environments.

#### local setup

```
- CPU: AMD EPYC 7502, 64 cores
- RAM: 504 GB
- Disk space: 3.5 T
- GPU: Nvidia Quadro RTX 5000, 16 GB per GPU
- GPU memory required: 8 GB
```

#### VM provided from PETS artifact site

```
- CPU: Xeon E5-2643, 4 cores
- RAM: 8GB
- Disk space: 40 GB
- GPU: none
```

Note: this setup was only tested on Cora, Citeseer, Chameleon dataset 
due to resource constraints in the provided VMs.

In general, the code can be executed on CPU or GPU. The CPU memory should be at least 32GB to avoid Out-of-Memory error when executing on Pubmed and facebook page dataset. For all other datasets, 8GB CPU memory is sufficient. 

---

### Estimated Time and Storage Consumption
Depending on the dataset size, the evalutation for one specific choice of (dataset, random seed, model, epsilon value) will take at most 5 minutes and consume at most 4GB space on the disk. The total space to reserve on disk to accomondate all datasets and intermediate results should be 8GB.

---

### Usage ###
```bash
usage: python main.py [-h]
               --arch {mlp,mmlp,gcn}
               [--dataset {cora,citeseer,pubmed,facebook_page,twitch/ES,flickr,bipartite,chameleon}]
               [--test_dataset {twitch/RU,twitch/DE,twitch/FR,twitch/ENGB,twitch/PTBR}]
               [--hidden_size HIDDEN_SIZE]
               [--num_hidden NUM_HIDDEN]
               [--nl NL]
               [--w_dp]
               [--eps EPS]
               [--outdir OUTDIR]
               [--todos_dir TODODIR]
               [--num_seeds NUM_SEEDS] 
               [--sample_seed SAMPLE_SEED]
               [--svd]
               [--rank]
               {train,attack} ...
```
### Arguments ###
`--dataset` (Default: <Dataset.Cora: 'cora'>)
cora|citeseer|pubmed...

`--arch` (Default: <Architecture.MMLP: 'mmlp'>)
Type of architecture to train: mmlp|gcn|mlp

`--nl` (Default: -1)
Only use for MMLP, Number of stacked models, default=-1

`--eps` (Default: 0.0)
The privacy budget. If 0, then do not DP train the arch

`--w_dp`
Run with DP guarantees - if eps=0.0 it throws a warning

`--svd` (Default: False)
Enable Singular Value Decomposition used in Eclipse.

`--rank` (Default: 20)
Choose the rank for Singular Value Decomposition used in Eclipse.

`--hidden_size` (Default: 16)
Size of the hidden layers

`--num_hidden` (Default: 2)
Number of hidden layers

`--outdir` (Default: ../results)
Directory to save the models and results

`--todos_dir` (Default: ../results/)
Directory to save the todo tasks for all experiments

`--test_dataset` (Default: None)
Test on this dataset, used for Twitch

`--sample_type` (Default: balanced)
Determines how we sample edges for attack.

`--attack_mode` (Default: efficient)
Choose baseline for running LPA and efficient for LinkTeller.


## Quick Start: Training and Attacking single models

### Output directory
We arrange the output in the following way. During running, all intermediate results will
be saved in **results/**. Depending on whether the task is model training or attack, 
results will be saved in **results/train** or **results/attack**.

```
results/
|----train/: model checkpoints (.pth) and metrics (e.g., F1 score, .pkl) after training
|         |
|         |----experiment1/: results generated after training
|         |         |
|         |         |---- model/: model checkpoints (.pth)
|         |         |
|         |         |---- *.pkl: metrics (F1 score)
|         |
|         |-----todos_experiment1/: all tasks (that specify models, datasets, epsilon, …)
|                   |
|                   |----done/: completed training tasks
|                   |
|                   |----working/: pending training tasks
|
| 
|----attack/:
           |
           |----todos_experiment1/: all tasks (that specify models, datasets, epsilon, …)
                    |
                    |----done/: completed attack tasks
                    |
                    |----working/: pending attack tasks

```

---

### Run training for a single model and dataset with DP
```
python main.py --dataset [Dataset] --arch [mmlp|gcn|mlp] --nl [# stack layers for mmlp] --w_dp --eps [Eps] --svd --rank 20 --sample_seed [Seed] --hidden_size [HID_s] --num_hidden [HID_n] train --lr [Lr] --dropout [Dropout]
```

- Here is an example to train GCN with Eclipse of rank 20
```
python main.py --dataset cora --arch gcn --w_dp --eps 4.0 --svd --rank 20 --sample_seed 42 --hidden_size 256 --num_hidden 2 train --lr 0.01 --dropout 0.2
```
You can also run for multiple seeds using the `--num_seeds` option. The results are stored in the folder defined in globals.py or the directory specified using the `--outdir` option. 
The trained models are stored in the `outdir/models` directory.

### Run the attacks on a single trained model
To run attack on a trained model, we need all the options used for training that model and a few options in addition such as the attack_mode and sample_type (samples for evaluation).
```
python main.py --dataset [Dataset] --arch [mmlp|gcn|mlp] --nl [# stack layers for mmlp] --w_dp --eps [Eps] --sample_seed [Seed] --hidden_size [HID_s] --num_hidden [HID_n] --outdir [Outdir] **attack** --lr [Lr] --dropout [Dropout] --attack_mode [bbaseline (lpa) | efficient (linkteller)] --sample_type [balanced | unbalanced]
```
- Here is an example to attack a GCN model trained with Eclipse of rank 20 and stored in ../results/models/
```
python main.py --dataset cora --arch gcn --w_dp --eps 4.0 --svd --rank 20 --sample_seed 42 --hidden_size 256 --num_hidden 2 --outdir ../results attack --lr 0.01 --dropout 0.2  --attack_mode baseline --sample_type balanced
```
The attack results are stored in the directory with name eval_[dataset] which is placed in the current directory.


## Reproducing the results
The procedure to reproduce the results follows the steps described in [Kolluri et al. (2022)](https://github.com/aashishkolluri/lpgnet-prototype). We show the commands to include Eclipse in the script.

### Training Models

#### Training Procedure

The training consists of three steps as follows:

- **Step 1**: Generate to-do tasks for training
- **Step 2**: Train the models
- **Step 3**: Parse training results

Commands for each step are listed as follows:

#### Step 1: Generate to-do tasks for training

- For transductive datasets (e.g., Cora)
```
python run_exp.py --rank 20  --num_seeds 5 --command train --outdir [results-dir] --todos_dir [todos] --best_config_file best_config.pkl --only_create_todos

# Example command
python run_exp.py --rank 20  --num_seeds 5 --command train --outdir ../results --todos_dir ../todos_train --best_config_file best_config.pkl --only_create_todos
```  
- For inductive datasets (e.g., Twitch)
```
python run_exp.py --rank 20  --num_seeds 5 --num_epochs 200 --command train --inductive --datasets TwitchES --outdir [results-dir] --todos_dir [todos] --best_config_file best_config.pkl --only_create_todos

# Example command
python run_exp.py --rank 20  --num_seeds 5 --num_epochs 200 --command train --inductive --datasets TwitchES --outdir ../results_ind --todos_dir ../todos_train_ind --best_config_file best_config.pkl --only_create_todos
```
#### Step 2: Training the models

- For transductive datasets (e.g., Cora)

```
python run_exp.py --rank 20  --num_seeds 5 --command train --outdir [results-dir] --todos_dir [todos] --best_config_file best_config.pkl

# Example command
python run_exp.py --rank 20  --num_seeds 5 --command train --outdir ../results --todos_dir ../todos_train --best_config_file best_config.pkl
```

- For inductive datasets (e.g., Twitch)

```
python run_exp.py --rank 20  --num_seeds 5 --num_epochs 200 --command train --inductive --datasets TwitchES --outdir [results-dir] --todos_dir [todos] --best_config_file best_config.pkl

# Example command
python run_exp.py --rank 20  --num_seeds 5 --num_epochs 200 --command train --inductive --datasets TwitchES --outdir ../results_ind --todos_dir ../todos_train_ind --best_config_file best_config.pkl
```

#### Step 3: Parse training results
To parse results to get utility scores, provide path to the results directory used during the training.

- For transductive datasets
```
python parser_ash.py --results_dir [results-dir]

# Example command
python parser_ash.py --results_dir ../results
```
- For inductive datasets
```
python parser_ash_ind_utility.py --results_dir [results-dir]

# Example command
python parser_ash_ind_utility.py --results_dir ../results_ind
```
The parsed results will be output in the `[results-dir]` folder.

The expected result is stored in the `expected_results` folder.

**Note**: The training experiment for transductive in our paper runs for *5 datasets*(Cora, Citeseer, Pubmed, Facebook page, Chameleon),
*5 random seeds*, *4 model architectures*(GCN,MLP,LPGNet,Eclipse) and *20 epsilon values* ([0.1,0.2,...,0.9], [0,1,2,...,10]).
In total, there are *1525 tasks* (MLP only uses one epsilon value since no adjacency matrix is involved in MLP training). 
The estimated running time for running all these experiments is around *128 hours on VMs provided by PETS*. 
You can run the command simultaneously in different terminals to expediate the execution of the job.

#### Quick Verification on the Cora Dataset
To quickly verify that the training experiment can be executed correctly, we also provide `run_exp_demo.py` to run the training experiment on Cora dataset and 3 epsilon values([0,1,10]) only. 

```
- generate to-do tasks
python run_exp_demo.py --rank 20  --num_seeds 5 --command train --outdir ../results_cora --todos_dir ../todos_train_cora --best_config_file best_config.pkl --only_create_todos

- run the tasks
python run_exp_demo.py --rank 20  --num_seeds 5 --command train --outdir ../results_cora --todos_dir ../todos_train_cora --best_config_file best_config.pkl
```
**Note**: In the above commands, --outdir can be any other directories specified by [results-dir], `../results_cora` is just an example for illustration purpose. Similarly, --todos_dir can be any other directories specified by [todos], `../todos_train_cora` is an example for illustration purpose.

**Note**: There will be 10 to-do tasks generated in the run_exp_demo.py experiment, with each one consisting of 5 random seeds. 
The estimated running time is 4 hours on one VM provided by PETS. 
After parsing the results following the instruction above, 
the results should be similar to the results provided in `expected_results/result_cora.csv` (lines corresponding to epsilon values 0, 1 and 10). 

**Note**: We reported the mean values in the results for plotting the figures in our paper, and the models and column names correspond as follows.
Since MLP training does not involve adjacency matrix, and the epsilon values are applied to add DP noise to adjacency matrix only,
our code automatically assigns value of -1 to MLP experiments with non-zero epsilon values.
```
gcn_mean -> GCN
gcn_rank20_mean -> Eclipse
mlp_mean -> MLP
mmlp_nl2_mean -> LPGNet
```

---

### Attacking the Trained Models

**Note**: DO NOT delete the `results-dir` obtained during training the models, as the trained model information is saved 
in `results-dir` and is needed for loading the model and running the attack on the trained models.

#### Attacking Procedure

- **Step 1**: Generate to-do tasks
- **Step 2**: Attack the models
- **Step 3**: Parse attacking results


#### Step 1: Generating to-do tasks for attack

- For transductive datasets
```
python run_exp.py --rank 20  --num_seeds 5 --command attack --outdir [results-dir] --todos_dir [todos] --best_config_file best_config.pkl --only_create_todos

# Example command
python run_exp.py --rank 20  --num_seeds 5 --command attack --outdir ../results --todos_dir ../todos_attack --best_config_file best_config.pkl --only_create_todos
```

- For inductive datasets
```
python run_exp.py --rank 20  --num_seeds 5 --num_epochs 200 --command attack --datasets TwitchES --outdir [results-dir] --todos_dir [todos] --best_config_file best_config.pkl --only_create_todos

# Example command
python run_exp.py --rank 20  --num_seeds 5 --num_epochs 200 --command attack --datasets TwitchES --outdir ../results_ind --todos_dir ../todos_attack_ind --best_config_file best_config.pkl --only_create_todos
```
#### Step 2: Attack the trained models

- For transductive datasets

```
python run_exp.py --rank 20  --num_seeds 5 --command attack --outdir [results-dir] --todos_dir [todos] --best_config_file best_config.pkl 

# Example command
python run_exp.py --rank 20  --num_seeds 5 --command attack --outdir ../results --todos_dir ../todos_attack --best_config_file best_config.pkl
```

- For inductive datasets
```
python run_exp.py --rank 20  --num_seeds 5 --num_epochs 200 --command attack --datasets TwitchES --outdir [results-dir] --todos_dir [todos] --best_config_file best_config.pkl 

# Example command
python run_exp.py --rank 20  --num_seeds 5 --num_epochs 200 --command attack --datasets TwitchES --outdir ../results_ind --todos_dir ../todos_attack_ind --best_config_file best_config.pkl
```

**Note**: The attack commands save the results in the current directory with "eval_" as a prefix.

#### Step 3: Parse attack results
To parse results to get attack AUC (Area Under Curve) scores, provide the path to the directory with saved models ( "eval_" as prefix in `./src` directory).

- For both transductive and inductive datasets

```
python parser_ash_trans_attack.py --results_dir [results-dir]

# Example command for parsing attack results for Cora dataset
python parser_ash_trans_attack.py --results_dir ./eval_cora
```
**Note**: The parsed results will be output in the `results-dir` folder (That is, `./src/eval_[dataset_name]/` folder). The expected result is stored in the `expected_results` folder.

#### Quick Verification on the Cora Dataset

Using the trained models from the quick verification experiment above, we can quickly verify if the attack experiments can be executed correctly on Cora dataset and 3 epsilon values.

```
- generate to-do tasks
python run_exp_demo.py --rank 20  --num_seeds 5 --command attack --outdir ../results_cora --todos_dir ../todos_attack_cora --best_config_file best_config.pkl --only_create_todos

- attack the trained model
python run_exp_demo.py --rank 20  --num_seeds 5 --command attack --outdir ../results_cora --todos_dir ../todos_attack_cora --best_config_file best_config.pkl 
```
**Note**: In the above commands, --outdir should be the same as the [results-dir] used during training the model. Here, `../results` was used in training the model above, so we continue using it in attack steps. However, --todos_dir should be different from the [todos] used during training the model. `todos_train` was used in training the model above, so we use a different directory `../todos_attack` to store the to-do tasks for attacking the models.

**Note**: There are 10 attack tasks generated in the run_exp_demo.py experiment, with each task consisting of 2 attack methods (LPA and LINKTELLER) and 5 random seeds.
The estimated running time is ~8 hours on the VM provided by PETS. 
After parsing the results following the instruction above, the results should be similar to the results provided in 
`expected_results/result_attack_cora_baseline_balanced.csv` (lines corresponding to epsilon values 0, 1 and 10) for the LPA attack, 
and `expected_results/result_attack_cora_efficient_balanced.csv` (lines corresponding to epsilon values 0, 1 and 10) for the LINKTELLER attack.
For MLP model with non-zero epsilon values, -1 will be automatically assigned.

### Script for Quick Verification
We have also provided two scripts, `run_exp_demo.sh` and `run_exp_demo_ind.sh` which include all the commands for running the training and attack experiments on specified dataset(s) and 3 epsilon values([0,1,10]) only. Make sure the mode of the scripts are changed to executable before running the scripts. The parsed training result will be in `results` folder, and the parsed attack result will be in `/src/eval_[dataset]/` folder (e.g. Cora dataset will have the parsed atttack result in `/src/eval_cora/`, CiteSeer dataset will have the parsed attack result in `/src/eval_citeseer`). 
```
# Change script mode to executable
chmod 777 run_exp_demo.sh
chmod 777 run_exp_demo_ind.sh

# Run the script for transductive setting with Cora dataset
./run_exp_demo.sh Cora

# Run the script for transductive setting with CiteSeer dataset
./run_exp_demo.sh CiteSeer

# Run the script for transductive setting with Cora and CiteSeer datasets, dataset names separated by comma and NO space between the names
./run_exp_demo.sh Cora,CiteSeer

# Run the script for inductive setting with Twitch datasets
./run_exp_demo_ind.sh TwitchES
```
**Note**: When running the script with multiple datasets, dataset names should be separated by comma. There should be NO space following the comma.

**Note**: One should run Twitch datasets using `run_exp_demo_ind.sh`. The transductive datasets, namely Cora, CiteSeer, Pubmed, Chameleon and facebook_page, should be run using `run_exp_demo.sh`.

**Note**: The dataset names are case-sensitive and should strictly follow the dataset names listed below.
```
Cora
CiteSeer
PubMed
facebook_page
Chameleon
TwitchES
```