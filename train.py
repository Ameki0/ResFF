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
    ds = None
    for dataset in datasets:
        path = os.path.join(input_prefix, dataset)
        ds = resff.data.dataset.GraphDataset.load(path).shuffle(RANDOM_SEED)
        logging.debug(f"# {dataset}: {len(ds)} entries")
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
    vl_input_prefix = kwargs['vl_input_prefix']
    datasets = [str(x) for x in kwargs['datasets'].split()]
    vl_datasets = [str(x) for x in kwargs['vl_datasets'].split()]
    n_max_confs = kwargs['n_max_confs']
    force_weight = kwargs['force_weight']
    residual_weight = kwargs['residual_weight']
    stage = kwargs['stage']


    logging.debug("# SET HYPERPARAMETERS")
    logging.debug(f"# Training with stage: {stage}, batch_size: {batch_size*n_max_confs}, learning_rate: {learning_rate}, residual_weight: {residual_weight}, force_weight: {force_weight}")
    logging.debug(f"# Loading training datasets from {input_prefix}{datasets[0]}")
    logging.debug(f"# Loading validation datasets from {vl_input_prefix}{vl_datasets[0]}")
    logging.debug(f"# Output will be saved to {output_prefix}")
    
    logging.debug("# LOAD DUPLICATED MOLECULES")
    ds_tr = _load_datasets(datasets, input_prefix)
    ds_vl = _load_datasets(vl_datasets, vl_input_prefix)
    logging.debug(f"# Training size: {len(ds_tr)}, Validation size: {len(ds_vl)}")

    logging.debug("# AUGMENT CONFORMATIONS TO HANDLE HETEROGRAPHS")
    ds_tr_augment = _augment_conformations(ds_tr, n_max_confs)
    logging.debug(f"# Training size after augment: {len(ds_tr_augment)}")

    ds_tr_loader = dgl.dataloading.GraphDataLoader(ds_tr_augment, batch_size=batch_size, shuffle=True)

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

    class GetLoss(torch.nn.Module):
        def energy_loss(self, g):
         
            g.nodes['g'].data['u_total'] = g.nodes['g'].data['u'] + residual_weight * g.nodes['g'].data['u_residual']

            return torch.nn.MSELoss()(
                g.nodes['g'].data['u_total'] - g.nodes['g'].data['u_total'].mean(dim=-1, keepdims=True),
                g.nodes['g'].data['u_ref_relative'],
            )
        
        def force_loss(self, g):
            du_dx_hat = torch.autograd.grad(
                g.nodes['g'].data['u_total'].sum(),
                g.nodes['n1'].data['xyz'],
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            du_dx = g.nodes["n1"].data["u_ref_prime"]
            return torch.nn.MSELoss()(du_dx, du_dx_hat)
        
        def forward(self, g):
            energy_loss = self.energy_loss(g)
            force_loss = self.force_loss(g) * force_weight
            loss = energy_loss + force_loss
            if stage == "stage_1":
                if g.number_of_nodes('n4_improper') > 0:
                    loss += g.nodes['n4_improper'].data['k'].pow(2).mean()
                if g.number_of_nodes('n4') > 0:
                    loss += g.nodes['n4'].data['k'].pow(2).mean()
            return loss, energy_loss, force_loss

    net = torch.nn.Sequential(
        torch_representation,
        representation,
        readout,
        readout_improper,
        ExpCoeff(),
        resff.mm.geometry.GeometryInGraph(),
        resff.mm.energy.EnergyInGraph(terms=["n2", "n3", "n4", "n4_improper"]),
        GetLoss(),
    ).to(cuda_device)

    
    # Number of parameters
    logging.debug(f"# Total number of parameters: {sum(p.numel() for p in net.parameters())}")
    logging.debug(f"# Bonded module parameters: {sum(p.numel() for p in representation.parameters())}")
    logging.debug(f"# Residual module parameters: {sum(p.numel() for p in torch_representation.parameters())}")
    logging.debug(f"# Try loading to load model from {output_prefix}")

    # Check if checkpoint file exists
    checkpoints = glob.glob(f"{output_prefix}/*.th")
    if checkpoints:
    
        n = [int(c.split('net')[1].split('.')[0]) for c in checkpoints]
        n.sort()
        last_step = n[-1]
        last_checkpoint = os.path.join(output_prefix, f"net{last_step}.th")
        checkpoint = torch.load(last_checkpoint, map_location="cpu")
        net.load_state_dict(checkpoint)
        step = last_step + 1
        logging.debug(f"# Found checkpoint file ({last_checkpoint}). Restarting from step {step}")
    else:
        step = 1

    # Freeze parameters
    if stage == "stage_1":
        for param in torch_representation.parameters():
            param.requires_grad = False
        for param in representation.parameters():
            param.requires_grad = True
        for param in readout.parameters():
            param.requires_grad = True
        for param in readout_improper.parameters():
            param.requires_grad = True
    elif stage == "stage_2":
        parameters = [
                '0.f_in.0.weight',
                '0.f_in.0.bias',    ]
        for param in net.named_parameters():  
            for parma_sub in parameters:
                if parma_sub in param[0]:
                    param[1].requires_grad = False
                    
        for param in torch_representation.parameters():
            param.requires_grad = True
        for param in representation.parameters():
            param.requires_grad = False
        for param in readout.parameters():
            param.requires_grad = False
        for param in readout_improper.parameters():
            param.requires_grad = False

    # Train
    best_e = float('inf')
    patience = 0
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    with torch.autograd.set_detect_anomaly(False):
        for idx in range(step, step + epochs):
            tr_loss, tr_energy_loss, tr_force_loss = [], [], []
            net.train()
        
            for g in tqdm(ds_tr_loader):
                
                optimizer.zero_grad()
                g = g.to(cuda_device)
                g.nodes["n1"].data["xyz"].requires_grad = True
                loss, energy_loss, force_loss = net(g)
                loss.backward()
                optimizer.step()
                tr_loss.append(loss.pow(0.5).item())
                tr_energy_loss.append(energy_loss.pow(0.5).item())
                tr_force_loss.append(force_loss.pow(0.5).item())
           
            tr_loss = np.mean(tr_loss)
            tr_energy_loss = np.mean(tr_energy_loss)
            tr_force_loss = np.mean(tr_force_loss)
            
            print(f"[Epoch {idx}] Train Loss: {tr_loss:.4f} Energy Loss: {tr_energy_loss:.4f}, Force Loss: {tr_force_loss:.4f}]")

            csv_file = f"{output_prefix}/loss.csv"
            with open(csv_file, 'a', newline='') as file:
                import csv
                writer = csv.writer(file)
                writer.writerow([idx, tr_loss, tr_energy_loss, tr_force_loss])

            # Save model
            torch.save(net.state_dict(), f"{output_prefix}/net{idx}.th")

            # Validation
            print('# Validation')
            net.eval()
            vl_energy_loss, vl_force_loss = [], []
            for g in tqdm(ds_vl):

                g = g.heterograph.to(cuda_device)
                g.nodes['g'].data['u_ref'] *= HARTEE_TO_KCALPERMOL
                g.nodes['n1'].data['u_ref_prime'] *= HARTEE_TO_KCALPERMOL / BOHR_TO_A
                g.nodes['g'].data['u_ref_relative'] = g.nodes['g'].data['u_ref'] - g.nodes['g'].data['u_ref'].mean(dim=-1, keepdims=True)
                g.nodes['g'].data['u_ref_relative'] = g.nodes['g'].data['u_ref_relative'].float()
                g.nodes['n1'].data['xyz'] *= BOHR_TO_A

                g.nodes["n1"].data["xyz"].requires_grad = True
                _, e_loss, f_loss = net(g)
                vl_energy_loss.append(e_loss.pow(0.5).item())
                vl_force_loss.append(f_loss.pow(0.5).item())

            for filename in os.listdir(output_prefix):
                if filename.startswith("net") and filename.endswith(".th"):
                    file_idx = int(filename.split("net")[1].split(".")[0])
                    if file_idx < idx:
                        os.remove(os.path.join(output_prefix, filename))
            if not os.path.exists(output_prefix):
                os.mkdir(output_prefix)
            torch.save(net.state_dict(), f"{output_prefix}/net{idx}.th")

            vl_energy_loss = np.mean(vl_energy_loss)
            vl_force_loss = np.mean(vl_force_loss)
            print(f"[Epoch {idx}] Val Energy Loss: {vl_energy_loss:.4f}, Val Force Loss: {vl_force_loss:.4f}]")

            # Save model
            if vl_energy_loss + vl_force_loss > best_e:
                patience += 1
                if patience >= 20:
                    patience = 0
                    learning_rate *= 0.8
                    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
                    print("Reducing learning rate to", learning_rate)
            else:
                patience = 0
                for filename in os.listdir(output_prefix):
                    if filename.startswith("best_net") and filename.endswith(".th"):
                        os.remove(os.path.join(output_prefix, filename))
                best_e = vl_energy_loss + vl_force_loss
                torch.save(net.state_dict(), f"{output_prefix}/best_net{idx}.th")

            csv_file =f"{output_prefix}/vl_loss.csv"
            with open(csv_file, 'a', newline='') as file:
                import csv
                writer = csv.writer(file)
                writer.writerow([idx, vl_energy_loss, vl_force_loss])
  

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
@click.option("-vl_i", "--vl_input_prefix", default="data", help="input prefix to validation graph data", type=str)
@click.option("-vl_d", "--vl_datasets", help="name of the validation dataset", type=str)
@click.option("-s", "--stage", default="stage_1", type=click.Choice(["stage_1", "stage_2"]), help="stage of training")

def cli(**kwargs):
    logging.debug(kwargs)
    run(kwargs)

if __name__ == "__main__":
    cli()
