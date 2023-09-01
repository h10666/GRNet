# GRNet: Graph-based remodeling network for multi-view semi-supervised classification

This repo contains the code and data of our PRL'2021 paper GRNet: Graph-based remodeling network for multi-view semi-supervised classification

## Requirements

pytorch>1.2.0 

numpy>=1.19.1

scikit-learn>=0.23.2

## Datasets

The CUB, Handwritten, MSRCV1, and NUS-WIDE datasets are placed in "data" folder.

## Usage

The code includes:

- an example implementation of the model,

```bash
python run.py --lr=0.001 --view_num=2 --total_num=1324 --dataset='cub.mat' --num_class=10 --epochs=50 --method=Exclusivity --display_epoch=1 --lr_decay_epochs=20,30 --rate=0.05
```


