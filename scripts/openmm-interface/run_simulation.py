import os
import torch
import numpy as np
import openmm
import openmmtorch
import resff
from openmm.app import *
from openmm import *
from openmm.unit import *
from pdbfixer import PDBFixer
from openff.toolkit.topology import Molecule
from openmmforcefields.generators import EspalomaTemplateGenerator
from residual.models.model import load_model
from typing import Optional

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_NAME = "tyk2"
WORK_DIR = "./tyk2_simulation"
RECEPTOR_PATH = "./data/tyk2_ejm_31.pdb"
LIGAND_PATH = "./data/ejm_31.sdf"
RESIDUAL_WEIGHTS = "./residual.ckpt"
MM_WEIGHTS = "./MM.pt"

USE_RECEPTOR = True
ADD_SOLVENT = True
IMPLICIT_SOLVENT = False

# Ensure GPU selection
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda")

# =============================================================================
# LIGAND AND RECEPTOR PREPARATION
# =============================================================================
print("Loading ligand and receptor...")
ligand_mol = Molecule.from_file(LIGAND_PATH, allow_undefined_stereo=True)

if USE_RECEPTOR:
    fixer = PDBFixer(RECEPTOR_PATH)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.findNonstandardResidues()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)
    fixer.removeHeterogens(True)
    print("Receptor preparation complete.")

# =============================================================================
# FORCE FIELD AND TOPOLOGY SETUP
# =============================================================================
# Initialize ForceField with Protein and Water models
ff_files = ['amber/protein.ff14SB.xml', 'amber14/tip3pfb.xml']
if ADD_SOLVENT and IMPLICIT_SOLVENT:
    ff_files.append('implicit/obc2.xml')

forcefield = ForceField(*ff_files)

# Utilize EspalomaTemplateGenerator to align ligand parameters for MM module
# EspalomaTemplateGenerator is a user-friendly interface for allocating the parameters of GNN predictions. 
# It requires the use of a locally modified espaloma, where nonbonded terms have been removed 
# as we only need the bonded parameters of MM module.
# TODO: write a independent inteface
ligand_generator = EspalomaTemplateGenerator(molecules=ligand_mol, forcefield=MM_WEIGHTS)
forcefield.registerTemplateGenerator(ligand_generator.generator)

# Build the initial topology (Ligand first)
ligand_topology = ligand_mol.to_topology().to_openmm()
ligand_positions = ligand_mol.conformers[0].to_openmm()
modeller = Modeller(ligand_topology, ligand_positions)

# Add Receptor
if USE_RECEPTOR:
    modeller.add(fixer.topology, fixer.positions)

# Add Solvent
if ADD_SOLVENT and not IMPLICIT_SOLVENT:
    modeller.addSolvent(forcefield, padding=1.0*nanometer)

# Create OpenMM System
nonbonded_method = PME if (ADD_SOLVENT and not IMPLICIT_SOLVENT) else NoCutoff
system = forcefield.createSystem(
    modeller.topology, 
    nonbondedMethod=nonbonded_method,
    nonbondedCutoff=0.9*nanometer,
    constraints=None, 
    rigidWater=True
)

# =============================================================================
# RESIDUAL MODULE INTEGRATION
# =============================================================================
class Wrapper(torch.nn.Module):

    def __init__(self, topology, embeddings, mol_atom_indices, model, isPeriodic):
        super(Wrapper, self).__init__()
        self.embeddings = embeddings
        self.mol_atom_indices = mol_atom_indices

        if isPeriodic:
            self.box_vectors = topology.getPeriodicBoxVectors()
            self.box_vectors_np = 10.0 * np.array([[self.box_vectors[0][0].value_in_unit(unit.nanometer), self.box_vectors[0][1].value_in_unit(unit.nanometer), self.box_vectors[0][2].value_in_unit(unit.nanometer)],
                                    [self.box_vectors[1][0].value_in_unit(unit.nanometer), self.box_vectors[1][1].value_in_unit(unit.nanometer), self.box_vectors[1][2].value_in_unit(unit.nanometer)],
                                    [self.box_vectors[2][0].value_in_unit(unit.nanometer), self.box_vectors[2][1].value_in_unit(unit.nanometer), self.box_vectors[2][2].value_in_unit(unit.nanometer)]])
        else:
            self.box_vectors_np = None

        # OpenMM will compute the forces by backpropagating the energy,
        # so we can load the model with derivative=False
        self.model = load_model(model, derivative=False, max_num_neighbors=128,cutoff_upper=10.0, box_vecs=self.box_vectors_np).to(torch.device("cuda"))

    def forward(self, positions, boxvectors: Optional[torch.Tensor] = None):
        # OpenMM works with nanometer positions and kilojoule per mole energies
        # Depending on the model, you might need to convert the units
        positions = positions[:self.mol_atom_indices, :]
        positions = positions.to(torch.float32) * 10.0 # nm -> A
        positions = positions.to(torch.device("cuda"))
        if boxvectors is not None:
            boxvectors = boxvectors.to(torch.float32) * 10.0
            boxvectors = boxvectors.to(torch.device("cuda"))
        energy = self.model(z=self.embeddings, pos=positions, box=boxvectors)[0]
        return energy * 4.184 # kcal/mol -> kJ/mol


# Setup ML atoms and embeddings
molecule_graph = resff.Graph(ligand_mol)
z_embeddings = molecule_graph.nodes["n1"].data["h0"].to(device)
num_lig_atoms = ligand_topology.getNumAtoms()

is_periodic = system.usesPeriodicBoundaryConditions()
wrapper = Wrapper(modeller.topology, z_embeddings, num_lig_atoms, RESIDUAL_WEIGHTS, is_periodic)
torch_script_model = torch.jit.script(wrapper)

# Add the ML Force to the OpenMM System
torch_force = openmmtorch.TorchForce(torch_script_model)
torch_force.setUsesPeriodicBoundaryConditions(is_periodic)
system.addForce(torch_force)

# =============================================================================
# SIMULATION SETUP
# =============================================================================
integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.001*picoseconds)
platform = Platform.getPlatformByName('CUDA')
simulation = Simulation(modeller.topology, system, integrator, platform)
simulation.context.setPositions(modeller.positions)

# ENERGY MINIMIZATION
def print_energy(sim, label):
    state = sim.context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(kilojoules_per_mole/nanometer)
    max_f = np.sqrt((forces**2).sum(axis=1)).max()
    print(f"{label:15} | Potential Energy: {energy:12.2f} kJ/mol | Max Force: {max_f:12.2f} kJ/(mol*nm)")

print_energy(simulation, "Initial State")
print("Minimizing energy...")
simulation.minimizeEnergy(tolerance=1.0*kilojoule/mole, maxIterations=1000)
print_energy(simulation, "Minimized State")

simulation.reporters.append(PDBReporter(f'{WORK_DIR}/{PROJECT_NAME}.pdb', 1000))
simulation.reporters.append(StateDataReporter(
    f'{WORK_DIR}/{PROJECT_NAME}.csv', 1000, step=True, potentialEnergy=True, temperature=True))

# RUN 
print("Starting production MD...")
simulation.step(500000) 
print("Simulation complete.")