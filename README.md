# DiTASK: Multi-Task Fine-Tuning with Diffeomorphic Transformations

This repository is the official implementation of [DiTASK: Multi-Task Fine-Tuning with Diffeomorphic Transformations](https://arxiv.org/abs/2407.02013). 

<image src="method.png" width="100%">


## Setup
The repository is built on top of [MTLoRA](https://github.com/scale-lab/MTLoRA) and uses components from [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) and [Multi-Task-Learning-PyTorch](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch).

Clone the repository:
```bash
git clone git@github.com:ipsitmantri/dynamic-graph-activation.git
```

Create a virtual environment in python (recommended to use Python 3.10):

```bash
conda create -n ditask --file environment.yml
```

Activate the virtual environment:
```bash
conda activate ditask
```
## Dataset
Download the PASCAL-Context MTL dataset from [here](https://cs.stanford.edu/~roozbeh/pascal-context/) and extract it.

## Pre-trained Model Backbone
Download the Swin-Transformer weights pre-trained on ImageNet-22K from the [official Swin repository](https://github.com/microsoft/Swin-Transformer).

## Running DiTASK
```bash
torchrun --nproc_per_node=8 --nnodes=1 main.py --cfg configs/ditask/ditask_tiny_448_r64.yaml --pascal <path-to-PASCAL-Context root> --tasks semsge,human_parts,normals,sal --batch-size 64 --ckpt-freq 20 --epoch 300 --resume-backbone swin_tiny_patch4_window7_224_22k.pth
```



## Cite Us
```bibtex
 @inproceedings{mantri2025ditaskmultitaskfinetuningdiffeomorphic,
  title={DiTASK: Multi-Task Fine-Tuning with Diffeomorphic Transformations}, 
  author={Krishna Sri Ipsit Mantri and Carola-Bibiane Schönlieb and Bruno Ribeiro and Chaim Baskin and Moshe Eliasof},
  year={2025},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  url={https://arxiv.org/abs/2502.06029}    
  }
```
