# Deep Residual Learning for Molecular Force Fields

Implementation of ResFF, by Xinyu Jiang.

This repository contains codes, instructions and model weights of ResFF.

![alt text](image.png)

# Installation

We recommend setting up the environment using [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).

Clone the current repo

    git clone https://github.com/Ameki0/ResFF.git

Set up environment

    mamba create -n resff python=3.9 pytorch==1.10*cuda
    mamba install cudatoolkit=11.3 openmm==7.7 -c pytorch -c conda-forge
    mamba install dgl-cuda11.3==0.9.0 -c dglteam
    mamba install openmmforcefields==0.11.2 openff-toolkit==0.10.0 openff-units==0.1.8 -c conda-forge
    pip install click
    wget https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_cluster-1.5.9-cp39-cp39-linux_x86_64.whl
    pip install torch_cluster-1.5.9-cp39-cp39-linux_x86_64.whl
    wget https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
    pip install torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
    pip install torch_geometric==2.5.3

# Model weight and code

1. The model weight is available in ~/ResFF/weight. Load it using:

    net.load_state_dict(torch.load(f'{checkpoint_path}', map_location=torch.device(cuda_device)))

Adjust checkpoint_path and GPU device settings as needed.

2. The ./resff directory provides the implementation of ResFF.

# Train ResFF

    bash  ~/ResFF/train.sh

# Evaluate ResFF

    bash  ~/ResFF/val.sh

A dataset demo for evaluation is provided in ~/ResFF/data.

# License

MIT