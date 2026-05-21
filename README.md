# Deep Residual Learning for Molecular Force Fields

This repo contains a PyTorch implementation for neural network force field ResFF.

If you have any question, feel free to open an issue or contact us at 📧 jiangxinyu@simm.ac.cn

by [Xinyu Jiang](https://github.com/Ameki0)

![image.png](https://github.com/Ameki0/ResFF/blob/main/images/Model.png)

📢 Update News
[2026-02] 
1. We have updated the ResFF repository and README.md file to improve usability and integration.

2. We have added links and preprocessing instructions for major datasets.

3. We have added support for parameterizing small molecules using ResFF within the OpenMM.

4. We have provided an interface for structure optimization using the ASE.


## Installation

We recommend setting up the environment using [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).

First Clone the current repo

``` bash
git clone https://github.com/Ameki0/ResFF.git
cd ResFF
```

And then set up environment

``` bash
mamba create -n resff python=3.9 pytorch==1.10*cuda
mamba activate resff
mamba install cudatoolkit=11.3 openmm==7.7 -c pytorch -c conda-forge
mamba install dgl-cuda11.3==0.9.0 -c dglteam
mamba install openmmforcefields==0.11.2 openff-toolkit==0.10.0 openff-units==0.1.8 -c conda-forge
pip install click
```
Install dependencies

``` bash
wget https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_cluster-1.5.9-cp39-cp39-linux_x86_64.whl
pip install torch_cluster-1.5.9-cp39-cp39-linux_x86_64.whl

wget https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
pip install torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl

pip install torch_geometric==2.5.3
```

## Model weight and code

1. Pretrained ResFF weight is available in [~/ResFF/weight](https://github.com/Ameki0/ResFF/blob/main/weight). 

Load the model using:

```
    net.load_state_dict(torch.load(f'{checkpoint_path}', map_location=torch.device(cuda_device)))
```

   Adjust `checkpoint_path` and `cuda_device` as needed.

2. The [~/ResFF/resff](https://github.com/Ameki0/ResFF/blob/main/resff) directory provides the implementation of ResFF.

## Data preprocessing

1. Data links

The following datasets are supported:

-   SPICE-1.1.4: https://zenodo.org/records/8222043
-   Generation2-Optimzization: https://github.com/openforcefield/qca-dataset-submission/tree/master/submissions/2020-03-20-OpenFF-Gen-2-Optimization-Set-1-Roche
                               
                               https://github.com/openforcefield/qca-dataset-submission/tree/master/submissions/2020-03-20-OpenFF-Gen-2-Optimization-Set-2-Coverage
                               
                               https://github.com/openforcefield/qca-dataset-submission/tree/master/submissions/2020-03-20-OpenFF-Gen-2-Optimization-Set-3-Pfizer-Discrepancy
                               
                               https://github.com/openforcefield/qca-dataset-submission/tree/master/submissions/2020-03-20-OpenFF-Gen-2-Optimization-Set-4-eMolecules-Discrepancy
                               
                               https://github.com/openforcefield/qca-dataset-submission/tree/master/submissions/2020-03-20-OpenFF-Gen-2-Optimization-Set-5-Bayer
                               
                               You can also download the Generation2-Optimzization dataset from https://zenodo.org/records/20305751.
-   DES370K: https://zenodo.org/records/8222043
             
             You can also download the DES370K dataset from https://zenodo.org/records/20305751.
-   TorsionNet-500: https://pubs.acs.org/doi/10.1021/acs.jcim.1c01346
-   TorsionScan: https://pubs.acs.org/doi/10.1021/acs.jcim.6b00614
-   s66x8: https://pubs.acs.org/doi/10.1021/ct2002946
-   OpenFF Industry Benchmark: https://pubs.acs.org/doi/10.1021/acs.jcim.2c01185

2. Dataset construction

You can generate DGL graph datasets by running [~/ResFF/scripts/prepare_data-example.ipynb](https://github.com/Ameki0/ResFF/blob/main/scripts/prepare_data-example.ipynb). HDF5 format is recommended. Other formats such as SDF and XYZ are supported after conversion. An example of building a dataset from an SDF file is provided in [~/ResFF/scripts/prepare_dataset-benchmark.ipynb](https://github.com/Ameki0/ResFF/blob/main/scripts/prepare_dataset-benchmark.ipynb).
Dataset preprocessing may take minutes to hours depending on size.

## Training ResFF

Run the training script:

``` bash
bash ~/ResFF/train.sh
```
Training configuration

| Parameter        | Default | Description |
|:-----------------|:--------|:------------|
| epochs           | 100     | Number of training epochs |
| batch_size       | 1       | Number of molecules per batch |
| n_max_confs      | 64      | Number of conformations per molecule |
| layer_1          | SAGEConv | GNN architecture for MM module |
| layer_2          | TorchMD_ET | GNN architecture for residual module |
| units            | 512     | Hidden dimension size |
| activation       | relu    | Activation function |
| config_1         | `${units} relu 0.1 ${units} relu 0.1 ${units} relu 0.1` | MM module MLP architecture |
| config_2         | torch   | Residual module architecture |
| janossy_config   | `${units} relu 0.1 ${units} relu 0.1 ${units} relu 0.1 ${units} relu 0.1` | Janossy pooling architecture |
| learning_rate    | 1e-4    | Learning rate |
| input_prefix     | data    | Training dataset path |
| datasets         | None    | Training dataset |
| output_prefix    | output  | Output directory |
| force_weight     | 1.0     | Force loss weight |
| residual_weight  | 1.0     | Residual loss weight |
| stage            | stage_1 | Training stage. `stage_1`: MM only (`residual_weight=0`), `stage_2`: Residual only (`residual_weight=1`) |

  
## Evaluation

Run evaluation with:

``` bash
bash ~/ResFF/val.sh
```

You can run a quick start exmaple in [~/ResFF/scripts/exmample/](https://github.com/Ameki0/ResFF/blob/main/scripts/example/) using preprocessed torsion and s66x8 datasets.
Results will be saved in [~/ResFF/results/](https://github.com/Ameki0/ResFF/blob/main/results/), and visualization scripts are available
in [~/ResFF/results/results.ipynb](https://github.com/Ameki0/ResFF/blob/main/results/results.ipynb).

## Using ResFF in OpenMM

An example for parameterizing small molecules in a protein-ligand complex using ResFF is provided in [~/ResFF/scripts/openmm-interface](https://github.com/Ameki0/ResFF/blob/main/scripts/openmm-interface).


## Using ResFF for Structure Optimization in ASE

An interface for ResFF using the ASE is available in [~/ResFF/scripts/ase-interface](https://github.com/Ameki0/ResFF/blob/main/scripts/ase-interface).

## Training a model yourself and using those weights

To train your own model or fine-tune existing weights, please refer to the example in [~/ResFF/scripts/](https://github.com/Ameki0/ResFF/blob/main/scripts/):

``` bash
bash finetune.sh
```

The training will resume from:

``` python
restart_checkpoints = f"{output_prefix}/best_net.th"
net.load_state_dict(restart_checkpoints)
```

## License

MIT
