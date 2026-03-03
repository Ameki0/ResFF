#!/usr/bin/env python
import os, sys, math
import numpy as np
import random
import click
import glob
import torch
import resff 
import dgl
import logging
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

# GLOBAL PARAMETER
HARTEE_TO_KCALPERMOL = 627.5
RANDOM_SEED = 2666
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
cuda_device = 'cuda:0'

# -------------------------
# LOAD DATASETS
# -------------------------
def _load_datasets(datasets, input_prefix):
    for i, dataset in enumerate(datasets):
        path = os.path.join(input_prefix, dataset)
        ds = resff.data.dataset.GraphDataset.load(path).shuffle(RANDOM_SEED)
    return ds

#-------------------------
# MAIN
#-------------------------
def run(kwargs):
    epochs = kwargs['epochs']
    batch_size = kwargs['batch_size']
    layer_1_type = kwargs['layer_1']
    layer_2_type = kwargs['layer_2']
    units = kwargs['units']
    config_1_str = kwargs['config_1']
    config_2_str = kwargs['config_2']
    janossy_config_str = kwargs['janossy_config']
    learning_rate = kwargs['learning_rate']
    output_prefix = kwargs['output_prefix']
    input_prefix = kwargs['input_prefix']
    datasets = kwargs['datasets'].split()
    eval_datasets_name = kwargs['eval_dataset_name']
    n_max_confs = kwargs['n_max_confs']

    def parse_config(cfg_str):
        res = []
        for x in cfg_str.split():
            try: res.append(int(x))
            except: res.append(str(x))
        return res

    config_1 = parse_config(config_1_str)
    config_2 = parse_config(config_2_str)
    janossy_config = parse_config(janossy_config_str)

    logging.debug(f"# LOAD DATASETS")
    ds_vl = _load_datasets(datasets, input_prefix)
    logging.debug(f"# Valid size: {len(ds_vl)}")

    layer_1 = resff.nn.layers.dgl_legacy.gn(layer_1_type, {"aggregator_type": "mean", "feat_drop": 0.1})
    layer_2 = resff.nn.layers.pyg_layer.gn(layer_2_type)
    
    representation = resff.nn.Sequential(layer_1, config=config_1)
    torch_representation = resff.nn.Sequential(layer_2, config=config_2)

    # Readout 
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

    # Model
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
    with torch.no_grad():
        print('# Validation Phase')
        net.eval()
        
        for g in ds_vl:
            data_name = [eval_datasets_name] * g.nodes['g'].data['u_ref'].shape[1]
            g = g.heterograph.to(cuda_device)
            
            g.nodes['g'].data['u_ref_relative'] = g.nodes['g'].data['u_ref'].detach().clone()
            g.nodes['g'].data['u_ref_relative'] -= g.nodes['g'].data['u_ref_relative'].min(dim=-1, keepdims=True)[0]
            g.nodes['g'].data['u_ref_relative'] = g.nodes['g'].data['u_ref_relative'].float()

            net(g)
        
            u_ref = g.nodes['g'].data['u_ref'].cpu().reshape(-1)
            u_total = (g.nodes['g'].data['u'] + g.nodes['g'].data['u_residual']).cpu().reshape(-1)
            u_residual = g.nodes['g'].data['u_residual'].cpu().reshape(-1)
            u2 = g.nodes['n2'].data['u'].sum(dim=0).cpu().reshape(-1)
            u3 = g.nodes['n3'].data['u'].sum(dim=0).cpu().reshape(-1)
            
            def get_sum_energy(node_type):
                try: return g.nodes[node_type].data['u'].sum(dim=0).cpu().reshape(-1)
                except: return torch.zeros(u_ref.shape)

            u4 = get_sum_energy('n4')
            u4_improper = get_sum_energy('n4_improper')

            import pandas as pd
            df = pd.DataFrame({
                'molname': data_name,
                'ResFF Energy': u_ref.numpy(),
                'Reference Energy': u_total.numpy(),
                #'u_residual': u_residual.numpy(),
                #'u2': u2.numpy(),
                #'u3': u3.numpy(),
                #'u4': u4.numpy(),
                #'u4_imp': u4_improper.numpy(),
            })
            
            save_path = './s66x8-results.csv'
            df.to_csv(save_path, mode='a', index=False, header=not os.path.exists(save_path))
        

#-------------------------
# CLI
#-------------------------
@click.command()
@click.option("-e", "--epochs", default=1, type=int)
@click.option("-b", "--batch_size", default=128, type=int)
@click.option("-l1", "--layer_1", default="SAGEConv", type=str)
@click.option("-l2", "--layer_2", default="SAGEConv", type=str)
@click.option("-u", "--units", default=128, type=int)
@click.option("-act", "--activation", default="relu", type=str)
@click.option("-c1", "--config_1", default="128 relu 128 relu 128 relu", type=str)
@click.option("-c2", "--config_2", default="128 relu 128 relu 128 relu", type=str)
@click.option("-jc", "--janossy_config", default="128 relu 128 relu 128 relu 128 relu", type=str)
@click.option("-lr", "--learning_rate", default=1e-5, type=float)
@click.option("-i", "--input_prefix", default="data", type=str)
@click.option("-d", "--datasets", help="dataset name", type=str)
@click.option("-o", "--output_prefix", default="output", type=str)
@click.option("-n", "--n_max_confs", default=50, type=int)
@click.option("-w", "--force_weight", default=1.0, type=float)
@click.option("-eval_d_name", "--eval_dataset_name", type=str)
def cli(**kwargs):
    run(kwargs)

if __name__ == "__main__":
    cli()