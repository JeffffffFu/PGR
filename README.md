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
python main.py --algorithm PGR --dataset citeseer --prune 0.4 --mu 0.0 --epochs_inner 1 
python main.py --algorithm PGR --dataset lastfm --prune 0.1 --mu 0.0 --epochs_inner 1
python main.py --algorithm PGR --dataset duke --prune 0.02 --mu 0.0 --epochs_inner 1 
python main.py --algorithm PGR --dataset emory --prune 0.05 --mu 0.0 --epochs_inner 1 
```
You can run the PGR directly with the following default parameters:

```bash
python main.py --algorithm Original --dataset cora 
python main.py --algorithm Original--dataset citeseer 
python main.py --algorithm Original --dataset lastfm 
python main.py --algorithm Original --dataset duke 
python main.py --algorithm Original --dataset emory  
```

You can run the TIA by adding the following
```bash
-- attacks TIA
-- attacks TIA-PGR