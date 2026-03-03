#!/usr/bin/env python
import os
import sys
import math
import numpy as np
import random
import click
import glob
import torch
import resff
import dgl
import logging
from tqdm import tqdm
import pandas as pd
import time

# -------------------------
# GLOBAL PARAMETER
# -------------------------
HARTEE_TO_KCALPERMOL = 627.5
BOHR_TO_A = 0.529
RANDOM_SEED = 2666
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
cuda_device='cuda:0'

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

# -------------------------
# LOAD DATASETS
# -------------------------
def _load_datasets(datasets, input_prefix):
    """
    Load unique molecules (nonisomeric smile).
    """
    logging.debug("# LOAD UNIQUE MOLECULES")

    for i, dataset in enumerate(datasets):
        path = os.path.join(input_prefix, dataset)
        _ds = resff.data.dataset.GraphDataset.load(path).shuffle(RANDOM_SEED)
        logging.debug(f"# {dataset}: {len(_ds)} entries")
        
        # Merge datasets
        if i == 0:
            ds = _ds
        else:
            ds += _ds
            
    logging.debug(f"(Total dataset size: {len(ds)})")
        
    return ds

def _augment_conformations(ds_tr, n_max_confs):
    """
    Augment conformations to handle heterographs.

    This is a work around to handle different graph size (shape). DGL requires at least one dimension with same size. 
    Here, we will modify the graphs so that each graph has the same number of conformations instead fo concatenating 
    graphs into heterogenous graphs with the same number of conformations. This will allow batching and shuffling 
    during the training. 
    """
    _ds_tr = []
    for i, g in enumerate(ds_tr):
    
        g.nodes['g'].data['u_ref'] *= HARTEE_TO_KCALPERMOL
        g.nodes['n1'].data['u_ref_prime'] *= HARTEE_TO_KCALPERMOL / BOHR_TO_A
        g.nodes['n1'].data['xyz'] *= BOHR_TO_A
        n = g.nodes['n1'].data['xyz'].shape[1]

        if n == n_max_confs:
            g.nodes['g'].data['u_ref_relative'] = g.nodes['g'].data['u_ref'] - g.nodes['g'].data['u_ref'].mean(dim=-1, keepdims=True)
            g.nodes['g'].data['u_ref_relative'] = g.nodes['g'].data['u_ref_relative'].float()
            g.nodes['g'].data.pop('u_ref')
            _ds_tr.append(g.heterograph)
        elif n < n_max_confs:
            random.seed(RANDOM_SEED)
            index = random.choices(range(n), k=n_max_confs - n)
            import copy
            _g = copy.deepcopy(g)
            _g.nodes["g"].data["u_ref"] = torch.cat((_g.nodes['g'].data['u_ref'], _g.nodes['g'].data['u_ref'][:, index]), dim=-1)
            _g.nodes["n1"].data["xyz"] = torch.cat((_g.nodes['n1'].data['xyz'], _g.nodes['n1'].data['xyz'][:, index, :]), dim=1)
            _g.nodes['n1'].data['u_ref_prime'] = torch.cat((_g.nodes['n1'].data['u_ref_prime'], _g.nodes['n1'].data['u_ref_prime'][:, index, :]), dim=1)
            _g.nodes['g'].data['u_ref_relative'] = _g.nodes['g'].data['u_ref'] - _g.nodes['g'].data['u_ref'].mean(dim=-1, keepdims=True)
            _g.nodes['g'].data['u_ref_relative'] = _g.nodes['g'].data['u_ref_relative'].float()
            _g.nodes['g'].data.pop('u_ref')
            _ds_tr.append(_g.heterograph)
        else:
            random.seed(RANDOM_SEED)
            idx_range = random.sample(range(n), k=n)
            for j in range(n // n_max_confs + 1):
                import copy
                _g = copy.deepcopy(g)
                if (j+1)*n_max_confs > n:
                    _index = range(j*n_max_confs, n)
                    index = random.choices(range(n), k=(j+1)*n_max_confs-n)
                    a = torch.cat((_g.nodes['g'].data['u_ref'][:, index], _g.nodes['g'].data['u_ref'][:, _index]), dim=-1)
                    b = torch.cat((_g.nodes['n1'].data['xyz'][:, index, :], _g.nodes['n1'].data['xyz'][:, _index, :]), dim=1)
                    c = torch.cat((_g.nodes['n1'].data['u_ref_prime'][:, index, :], _g.nodes['n1'].data['u_ref_prime'][:, _index, :]), dim=1)
                else:
                    idx1 = j*n_max_confs
                    idx2 = (j+1)*n_max_confs
                    _index = idx_range[idx1:idx2]
                    a = _g.nodes['g'].data['u_ref'][:, _index]
                    b = _g.nodes['n1'].data['xyz'][:, _index, :]
                    c = _g.nodes['n1'].data['u_ref_prime'][:, _index, :]
                _g.nodes["g"].data["u_ref"] = a
                _g.nodes["n1"].data["xyz"] = b
                _g.nodes["n1"].data["u_ref_prime"] = c
                _g.nodes['g'].data['u_ref_relative'] = _g.nodes['g'].data['u_ref'] - _g.nodes['g'].data['u_ref'].mean(dim=-1, keepdims=True)
                _g.nodes['g'].data['u_ref_relative'] = _g.nodes['g'].data['u_ref_relative'].float()
                _g.nodes['g'].data.pop('u_ref')
                _ds_tr.append(_g.heterograph)
    return _ds_tr

# -------------------------
# MAIN
# -------------------------
def run(kwargs):
    epochs = kwargs['epochs']
    batch_size = kwargs['batch_size']
    layer_1 = kwargs['layer_1']
    layer_2 = kwargs['layer_2']
    units = kwargs['units']
    config_1 = [int(x) if x.isdigit() else x for x in kwargs['config_1'].split()]
    config_2 = [int(x) if x.isdigit() else x for x in kwargs['config_2'].split()]
    janossy_config = [int(x) if x.isdigit() else x for x in kwargs['janossy_config'].split()]
    learning_rate = kwargs['learning_rate']
    output_prefix = kwargs['output_prefix']
    input_prefix = kwargs['input_prefix']
    datasets = [str(x) for x in kwargs['datasets'].split()]
    n_max_confs = kwargs['n_max_confs']
    force_weight = kwargs['force_weight']
    residual_weight = kwargs['residual_weight']
    stage = kwargs['stage']


    logging.debug("# SET HYPERPARAMETERS")
    logging.debug(f"# Training with stage: {stage}, batch_size: {batch_size*n_max_confs}, learning_rate: {learning_rate}, residual_weight: {residual_weight}, force_weight: {force_weight}")
    logging.debug(f"# Loading validation datasets from {input_prefix}{','.join(datasets)}")
    logging.debug(f"# Output will be saved to {output_prefix}")
    
    logging.debug("# LOAD DUPLICATED MOLECULES")
    #ds_tr = _load_datasets(datasets, input_prefix)
    ds_vl = _load_datasets(datasets, input_prefix)
    logging.debug(f"# Validation size: {len(ds_vl)}")
    
    # Define model
    # Representation
    layer_1 = resff.nn.layers.dgl_legacy.gn(layer_1, {"aggregator_type": "mean", "feat_drop": 0.1})
    representation = resff.nn.Sequential(layer_1, config=config_1)
    layer_2 = resff.nn.layers.pyg_layer.gn(layer_2)
    torch_representation = resff.nn.Sequential(layer_2, config=config_2)

    # 2: bond linear combination, enforce positive
    # 3: angle linear combination, enforce positive
    # 4: torsion barrier heights (can be positive or negative)
    readout = resff.nn.readout.janossy.JanossyPooling(
        in_features=units, config=janossy_config,
        out_features={2: {'log_coefficients': 2}, 3: {'log_coefficients': 2}, 4: {'k': 6}},
    )
    readout_improper = resff.nn.readout.janossy.JanossyPoolingWithSmirnoffImproper(
        in_features=units, config=janossy_config, out_features={"k": 6}
    )

    class ExpCoeff(torch.nn.Module):
        def forward(self, g):
            g.nodes['n2'].data['coefficients'] = g.nodes['n2'].data['log_coefficients'].exp()
            g.nodes['n3'].data['coefficients'] = g.nodes['n3'].data['log_coefficients'].exp()
            return g

    net = torch.nn.Sequential(
        torch_representation,
        representation,
        readout,
        readout_improper,
        ExpCoeff(),
        resff.mm.geometry.GeometryInGraph(),
        resff.mm.energy.EnergyInGraph(terms=["n2", "n3", "n4", "n4_improper"]),
    ).to(cuda_device)

    
   
    # Check if checkpoint file exists
    restart_checkpoint = "./weight/best_net.th"
    net.load_state_dict(torch.load(restart_checkpoint, map_location=torch.device(cuda_device)))

    # Validation
    logging.debug('# Validation')
    net.eval()

    import time
    start_time = time.time()
    
    for g in tqdm(ds_vl):

        u_list, u_ref_list=[], []
        mol_ids_list, conf_ids_list, scan_list = [], [], []
        g = g.heterograph.to(cuda_device)
       
        g.nodes["n1"].data["xyz"].requires_grad = True
        net(g)
     

        u_ref_list.append(g.nodes['g'].data['u_ref'] - g.nodes['g'].data['u_ref'].min(dim=-1, keepdims=True)[0])
        g.nodes['g'].data['u_total'] = (g.nodes['g'].data['u'] + g.nodes['g'].data['u_residual'])
        u_list.append(g.nodes['g'].data['u_total'] - g.nodes['g'].data['u_total'].min(dim=-1, keepdims=True)[0]) 
        u_list = torch.cat(u_list, dim=1).view(-1)
        u_ref_list = torch.cat(u_ref_list, dim=1).view(-1)
        
        mol_ids_list.append(g.nodes['g'].data['FirstLineDescription_1'])
        conf_ids_list.append(g.nodes['g'].data['FirstLineDescription_2'])
        scan_list.append(g.nodes['g'].data['ScanVar_1'])
        mol_ids_list = torch.cat(mol_ids_list, dim=1).view(-1)
        conf_ids_list = torch.cat(conf_ids_list, dim=1).view(-1)
        scan_list = torch.cat(scan_list, dim=1).view(-1)
        
        data = pd.DataFrame({
            'mol_id': mol_ids_list.cpu().detach().numpy(),
            'conf_id': conf_ids_list.cpu().detach().numpy(),
            'scan': scan_list.cpu().detach().numpy(),
            'ResFF Energy': u_list.cpu().detach().numpy(),
            'Reference Energy': u_ref_list.cpu().detach().numpy(),
            })
    
        data.to_csv(f'{output_prefix}/{"-".join(datasets)}-results.csv', mode='a', index=False, header=not os.path.exists(f'{output_prefix}/{"-".join(datasets)}-results.csv') ) #

    end_time = time.time() - start_time
    logging.debug(f'Duration: {end_time}')


@click.command()
@click.option("-e", "--epochs", default=1, help="number of epochs", type=int)
@click.option("-b", "--batch_size", default=128, help="batch size", type=int)
@click.option("-l1", "--layer_1", default="SAGEConv", type=click.Choice(["SAGEConv", "GATConv", "TAGConv", "GINConv", "GraphConv"]), help="GNN architecture")
@click.option("-l2", "--layer_2", default="TorchMD_ET", type=click.Choice(["TorchMD_ET"]), help="GNN architecture")
@click.option("-u", "--units", default=128, help="GNN layer", type=int)
@click.option("-act", "--activation", default="relu", type=click.Choice(["relu", "leaky_relu"]), help="activation method")
@click.option("-c1", "--config_1", default="128 relu 128 relu 128 relu", help="sequence of numbers (for units) and strings (for activation functions) for layer_1", type=str)
@click.option("-c2", "--config_2", default="torch", help="configuration for layer_2", type=str)
@click.option("-jc", "--janossy_config", default="128 relu 128 relu 128 relu 128 relu", help="sequence of numbers (for units) and strings (for activation functions)", type=str)
@click.option("-lr", "--learning_rate", default=1e-4, help="learning rate", type=float)
@click.option("-i", "--input_prefix", default="data", help="input prefix to graph data", type=str)
@click.option("-d", "--datasets", help="name of the dataset", type=str)
@click.option("-o", "--output_prefix", default="output", help="output prefix to save checkpoint network models", type=str)
@click.option("-n", "--n_max_confs", default=50, help="number of conformations to reshape the graph", type=int)
@click.option("-w", "--force_weight", default=1.0, type=float)
@click.option("-r", "--residual_weight", default=1.0, type=float, help="weight for residual energy")
@click.option("-s", "--stage", default="stage_1", type=click.Choice(["stage_1", "stage_2"]), help="stage of training")

def cli(**kwargs):
    logging.debug(kwargs)
    run(kwargs)

if __name__ == "__main__":
    cli()
