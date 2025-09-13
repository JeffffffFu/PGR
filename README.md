# PGR


This repository is the official implementation of the paper: 
#### Safeguarding Graph Neural Networks against Topology Inference Attacks


## Installation

You can install all requirements with:
```bash
conda create --name PGR python=3.9
conda activate PGR
pip install -r requirements.txt

# Then install them manually
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch-scatter==2.1.1 -f https://pytorch-geometric.com/whl/torch-1.13.1+cu116.html
pip install torch-sparse==0.6.16 -f https://pytorch-geometric.com/whl/torch-1.13.1+cu116.html
```

## Code Structure of PGR
* **baseline**: provide the implementation of baselines. 
* **data**: provide data downloading scripts and raw data loader to process original data .
* **graph_reconstruction**: provide the implementation of edge regenerate in PGR.
* **mask**: provide the components of edge regenerate.
* **model**: provide the GNN models.
* **privacy_analyze**: provide the calculation of privacy budget in DP.
* **TIAs**: provide the implementation of C-TIA, M-TIA and I-TIA.
* **utils**: provide graph and adjacency matrix processing functions


## Demo Experiments
You can run the baseline directly with the following default parameters (For example in Table 2):

'Original' means 'No Edge-DP'
```bash
python main.py --attacks TIA --algorithm Original --dataset cora --eps 7
python main.py --attacks TIA --algorithm GAP --dataset cora --eps 7
python main.py --attacks TIA --algorithm LPGNet--dataset cora --eps 7 
python main.py --attacks TIA --algorithm Eclipse --dataset cora --eps 7 
python main.py --attacks TIA --algorithm privGraph --dataset cora --eps 7 
python main.py --attacks TIA --algorithm LapEdge --dataset cora --eps 7  
python main.py --attacks TIA --algorithm EdgeRand --dataset cora --eps 7  

python main.py --attacks TIA --algorithm Original --dataset citeseer --eps 7
python main.py --attacks TIA --algorithm GAP --dataset citeseer --eps 7
python main.py --attacks TIA --algorithm LPGNet--dataset citeseer --eps 7 
python main.py --attacks TIA --algorithm Eclipse --dataset citeseer --eps 7 
python main.py --attacks TIA --algorithm privGraph --dataset citeseer --eps 7 
python main.py --attacks TIA --algorithm LapEdge --dataset citeseer --eps 7  
python main.py --attacks TIA --algorithm EdgeRand --dataset citeseer --eps 7  

python main.py --attacks TIA --algorithm Original --dataset lastfm --eps 7
python main.py --attacks TIA --algorithm GAP --dataset lastfm --eps 7
python main.py --attacks TIA --algorithm LPGNet--dataset lastfm --eps 7 
python main.py --attacks TIA --algorithm Eclipse --dataset lastfm --eps 7 
python main.py --attacks TIA --algorithm privGraph --dataset lastfm --eps 7 
python main.py --attacks TIA --algorithm LapEdge --dataset lastfm --eps 7  
python main.py --attacks TIA --algorithm EdgeRand --dataset lastfm --eps 7  

# you can choose different eps value (Figure 6)
```

You can run the PGR directly with the following default parameters on GCN model (For example in Table 3):

```bash
python main.py --attacks TIA --algorithm PGR --dataset cora --prune 0.5 --mu 0.0 --epochs_inner 1 
python main.py --attacks TIA --algorithm PGR --dataset citeseer --prune 0.5 --mu 0.0 --epochs_inner 1 
python main.py --attacks TIA --algorithm PGR --dataset lastfm --prune 0.5 --mu 0.0 --epochs_inner 1

python main.py --attacks TIA --algorithm PGR --dataset cora --prune 0.2 --mu 0.0 --epochs_inner 1 --network GAT
python main.py --attacks TIA --algorithm PGR --dataset citeseer --prune 0.2 --mu 0.0 --epochs_inner 1 --network GAT
python main.py --attacks TIA --algorithm PGR --dataset emory --prune 0.05 --mu 0.0 --epochs_inner 1 -- epochs 200 --network GAT

python main.py --attacks TIA --algorithm PGR --dataset cora --prune 0.2 --mu 0.0 --epochs_inner 1 --network GraphSAGE
python main.py --attacks TIA --algorithm PGR --dataset citeseer --prune 0.2 --mu 0.0 --epochs_inner 1 --network GraphSAGE
python main.py --attacks TIA --algorithm PGR --dataset emory --prune 0.05 --mu 0.0 --epochs_inner 1 --network GraphSAGE
```

