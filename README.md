# PGR


This repository is the official implementation of the paper:



## Installation

You can install all requirements with:
```bash
pip install -r requirements.txt
```


You can run the PGR directly with the following default parameters:

```bash
python main.py --algorithm PGR --dataset cora --prune 0.5 --mu 0.0 --epochs_inner 1 
python main.py --algorithm PGR --dataset citeseer --prune 0.5 --mu 0.0 --epochs_inner 1 
python main.py --algorithm PGR --dataset lastfm --prune 0.5 --mu 0.0 --epochs_inner 1
python main.py --algorithm PGR --dataset duke --prune 0.05 --mu 0.0 --epochs_inner 1 
python main.py --algorithm PGR --dataset emory --prune 0.05 --mu 0.0 --epochs_inner 1 
```
You can run the Original model (no dp) directly with the following default parameters:

```bash
python main.py --algorithm Original --dataset cora 
python main.py --algorithm Original--dataset citeseer 
python main.py --algorithm Original --dataset lastfm 
python main.py --algorithm Original --dataset duke 
python main.py --algorithm Original --dataset emory  
```
You can run the baseline directly with the following default parameters:
```bash
python main.py --algorithm GAP --dataset cora --eps 7
python main.py --algorithm LPGNet--dataset cora --eps 7 
python main.py --algorithm Eclipse --dataset cora --eps 7 
python main.py --algorithm privGraph --dataset cora --eps 7 
python main.py --algorithm LapEdge --dataset cora --eps 7  
python main.py --algorithm EdgeRand --dataset cora --eps 7  

```

You can run the TIA or TIA-PGR by adding the following
```bash
-- attacks TIA
-- attacks TIA-PGR