import os
import torch
import sys
import numpy as np
import ase_interface
import espaloma as esp
import logging
import subprocess
  
# Model Configuration
layer_1="SAGEConv"
layer_2="TorchMD_ET"
units = 512
config_1=f"{units} relu 0.1 {units} relu 0.1 {units} relu 0.1"
config_2="torch"
janossy_config=f"{units} relu 0.1 {units} relu 0.1 {units} relu 0.1 {units} relu 0.1"
n_max_confs = 1
force_weight = 1.0
cuda_device = 'cuda:0'
_config = []

for _ in config_1.split():
    try:
        _config.append(int(_))
    except:
        _config.append(str(_))
config_1 = _config

_config = []
for _ in config_2.split():
    try:
        _config.append(int(_))
    except:
        _config.append(str(_))
config_2 = _config

_janossy_config = []
for _ in janossy_config.split():
    try:
        _janossy_config.append(int(_))
    except:
        _janossy_config.append(str(_))
janossy_config = _janossy_config


# Representation
layer_1 = esp.nn.layers.dgl_legacy.gn(layer_1,{"aggregator_type": "mean", "feat_drop": 0.1})
layer_2 = esp.nn.layers.pyg_layer.gn(layer_2)
representation = esp.nn.Sequential(layer_1, config=config_1)
torch_representation = esp.nn.Sequential(layer_2, config=config_2)
readout = esp.nn.readout.janossy.JanossyPooling(
    in_features=units, config=janossy_config,
    out_features={
            2: {'log_coefficients': 2},
            3: {'log_coefficients': 2},
            4: {'k': 6},
    },
)
readout_improper = esp.nn.readout.janossy.JanossyPoolingWithSmirnoffImproper(in_features=units, config=janossy_config, out_features={"k": 6})
    
class ExpCoeff(torch.nn.Module):
    def forward(self, g):
        g.nodes['n2'].data['coefficients'] = g.nodes['n2'].data['log_coefficients'].exp()
        g.nodes['n3'].data['coefficients'] = g.nodes['n3'].data['log_coefficients'].exp()
        
        return g

class GetLoss(torch.nn.Module):
    def convert(self, g):
     
        g.nodes['g'].data['u_total'] =  (g.nodes['g'].data['u'] + g.nodes['g'].data['u_torch']) #
        
        du_dx_hat = torch.autograd.grad(
            g.nodes['g'].data['u_total'].sum(),
            g.nodes['n1'].data['xyz'],
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )[0]
        g.nodes["n1"].data["u_prime"] = du_dx_hat
       
    def forward(self, g):
        self.convert(g)
        return g

print('#Loading model weight...')    
net = torch.nn.Sequential(
        torch_representation,
        representation,
        readout,
        readout_improper,
        ExpCoeff(),
        esp.mm.geometry.GeometryInGraph(),
        esp.mm.energy.EnergyInGraph(terms=["n2", "n3", "n4", "n4_improper"]),
        GetLoss(),
).to(cuda_device)

net.load_state_dict(torch.load('ResFF/weights/best_net.th',map_location=cuda_device))
net.eval()

# Input molecule name
molname = sys.argv[1]
molecule_path=f"./opt-results/{molname}/{molname}.sdf"
xyz_path=f"./opt-results/{molname}/{molname}.xyz"
logging.debug(print('#Converting xyz file into sdf file...'))

# Convert xyz to sdf
subprocess.run(['obabel', '-isdf', molecule_path, '-oxyz', '-O', xyz_path])

# Run optimization
print('#Loading ase simulation system...')
ase_run = ase_interface.AseInterface(molecule_path=f"{molname}.xyz", 
                                     ml_model=net,working_dir=f"./opt-results/{molname}/",device=cuda_device)
print('#Start optimization...')
ase_run.optimize()

