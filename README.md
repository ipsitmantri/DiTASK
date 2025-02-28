# DiTASK: Multi-Task Fine-Tuning with Diffeomorphic Transformations

This repository is the official implementation of [DiTASK: Multi-Task Fine-Tuning with Diffeomorphic Transformations](https://arxiv.org/abs/2407.02013). 

<image src="DiTASK.png" width="100%">


## Setup
Create a virtual environment in python (recommended to use Python 3.10):

```bash
conda create -n ditask python=3.10
```

Activate the virtual environment:
```bash
conda activate ditask
```

Clone the repository:
```bash
git clone git@github.com:ipsitmantri/dynamic-graph-activation.git
```

Do a local install of the repository :
```bash
cd dynamic-graph-activation
pip install -e .
```

Now install other requirements:

```bash
pip install -r requirements.txt
```

## Running DiGRAF

The hyperparameter sweep config files are defined in `conf/wandb_sweep` directory. Follow the below instructions to reproduce our main results.

### Node Classification
To obtain a `<sweep-id>` for node classification datasets, run
```bash
wandb sweep conf/wandb_sweep/<variant>_nodecls_<dataset>.yaml
```
where `<dataset>` can be choosen from `[cora, citeseer, pubmed, flickr, blog]` and `<variant>` can be chosen from `[digraf, digraf_wo]`

Then run `wandb agent <sweep-id>` to launch the sweep.

### Open Graph Benchmark
To obtain a `<sweep-id>` for OGB datasets, run
```bash
wandb sweep conf/wandb_sweep/<variant>_ogb_<dataset>.yaml
```
where `<dataset>` can be choosen from `[molhiv, molbace, molesol, moltox]` and `<variant>` can be chosen from `[digraf, digraf_wo]`

Then run `wandb agent <sweep-id>` to launch the sweep.

### ZINC
To obtain a `<sweep-id>` for ZINC dataset, run
```bash
wandb sweep conf/wandb_sweep/<variant>_zinc.yaml
```
where `<variant>` can be chosen from `[digraf, digraf_wo]`

Then run `wandb agent <sweep-id>` to launch the sweep.

### TU Datasets
To obtain a `<sweep-id>` for TU datasets, run
```bash
wandb sweep conf/wandb_sweep/<variant>_graphcls_tud_<dataset>.yaml
```
where `<dataset>` can be choosen from `[mutag, proteins, nci1, nci109, ptcmr]` and `<variant>` can be chosen from `[digraf, digraf_wo]`

Then run `wandb agent <sweep-id>` to launch the sweep.


## Running Baselines
The hyperparameter sweep config files can be found at `conf/wandb_sweep/baseline_*`. To run different baseline activations i.e `[relu, identity, sigmoid, tanh, gelu, elu, leakyrelu, tanh, prelu, maxout, swish, max, median, grelu]`, you have to change the `parameters.activation` value in the respective config file. 

Follow the steps below to get `<sweep-id>`:

### Node Classification
```bash
wandb sweep conf/wandb_sweep/baseline_nodecls_<dataset>.yaml
```
and choose `<dataset>` from `[cora, citeseer, pubmed, flickr, blog]`

### Open Graph Benchmark
```bash
wandb sweep conf/wandb_sweep/baseline_ogb_<dataset>.yaml
```
and choose `<dataset>` from `[molhiv, molbace, molesol, moltox]`

### ZINC
```bash
wandb sweep conf/wandb_sweep/baseline_zinc.yaml
```

### TU Datasets
```bash
wandb sweep conf/wandb_sweep/baseline_graphcls_tud_<dataset>.yaml
```
and choose `<dataset>` from `[mutag, proteins, nci1, nci109, ptcmr]`

After obtaining a `<sweep-id>`, launch the agent using `wandb agent <sweep-id>`. By default, all baselines use `relu` activation.


## Cite Us
```bibtex
@inproceedings{
mantri2024digraf,
title={Di{GRAF}: Diffeomorphic Graph-Adaptive Activation Function},
author={Krishna Sri Ipsit Mantri and Xinzhi Wang and Carola-Bibiane Sch{\"o}nlieb and Bruno Ribeiro and Beatrice Bevilacqua and Moshe Eliasof},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=ZZoW4Z3le4}
}
```
