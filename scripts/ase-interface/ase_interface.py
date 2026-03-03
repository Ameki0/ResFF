import numpy as np
import sys
import ase
from ase import Atoms, units
from ase.units import mol
from ase.calculators.calculator import Calculator, all_changes
import torch
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md import VelocityVerlet, Langevin, MDLogger
from ase.md.npt import NPT
from ase.md import MDLogger
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.optimize import QuasiNewton
from ase.optimize.bfgs import BFGS,oldBFGS
from ase.optimize.lbfgs import LBFGS
from ase.optimize.precon import PreconLBFGS
from ase.vibrations import Vibrations
import os
import MDAnalysis as mda
#from ase.calculators.combine_mm import CombineMM
# https://github.com/materialsvirtuallab/m3gnet/blob/main/m3gnet/models/_dynamics.py
# https://github.com/atomistic-machine-learning/schnetpack/blob/f2cf162b2b2810a850a5856e46a18b61822515ee/src/schnetpack/interfaces/ase_interface.py#L208
# https://gitlab.com/hyunp2/ai4molcryst_argonne/-/blob/main/train/ase_pub.py

from rdkit import Chem
import os, sys
import numpy as np
import torch
import espaloma as esp
from espaloma.units import *
import pandas as pd
from openff.toolkit.topology import Molecule
from simtk import unit
from simtk.unit import Quantity

class ResFFCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        model,
        device='cuda',
        energy=True,
        forces=True,
        atoms=None,
        g=None,
        restart=None,
        add_atom_energies=False,
        sdf_path=None,
        label="custom_calc",  # ase settings
        **kwargs
    ):
        super().__init__(**kwargs)

        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model_energy = energy
        self.model_forces = forces
        self.sdf_path = sdf_path
        self.g = g

    def calculate(self, atoms=None,  properties=["energy", "force"], system_changes=all_changes):
        """
        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): do not use this, no functionality
            system_changes (list of str): List of changes for ASE.
        """
        # First call original calculator to set atoms attribute
        # (see https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/calculator.html#Calculator)

        super().calculate(atoms, properties, system_changes)
        if self.sdf_path is None:
            raise ValueError("SDF path is not set.")
        
        import logging
        import time 
        start_time = time.time()
        pos=torch.from_numpy(atoms.positions).view(-1, 3).float()
      
    
        self.g.nodes["n1"].data["xyz"] = torch.tensor(
            Quantity(
                pos.unsqueeze(dim=1),
                unit.bohr,
            ).value_in_unit(esp.units.DISTANCE_UNIT),
            requires_grad=True,
            dtype=torch.get_default_dtype(),
        )
     
        start_time = time.time()
      
        # Call model
        with torch.set_grad_enabled(True):
            _g = self.model(self.g) 

        # Extract energy and forces from the graph
        energy = _g.nodes['g'].data['u_total'].detach().cpu().item()
        forces = -_g.nodes["n1"].data["u_prime"].view(len(atoms), 3).detach().cpu().numpy()
        
        # Store results in the calculator's results dictionary
        self.results = {
            "energy": energy,
            "forces": forces
        }

class AseInterface:
    """
    Interface for ASE calculations (optimization and molecular dynamics)
    Args:
        molecule_path (str): Path to initial geometry
        ml_model (object): Trained model
        working_dir (str): Path to directory where files should be stored
        device (str): cpu or cuda
    """

    def __init__(
        self,
        #smi:str,
        molecule_path: str,
        ml_model: torch.nn.Module,
        working_dir: str,
        device='cuda',
        energy=True,
        forces=True
    ):
        # Setup directory
        self.working_dir = working_dir
        self.device = device
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
        self.molecule_path = molecule_path
        from ase import Atoms
        import logging
        import subprocess
      
        file_name_with_extension = os.path.basename(molecule_path)
        file_name, extension = os.path.splitext(file_name_with_extension)
    
        if extension.lower() == '.xyz':
            pass
        else:
            raise ValueError("Input is not a xyz file.")
        
        sdf_file = file_name_with_extension.replace('.xyz', '.sdf')
        self.sdf_path = os.path.join(working_dir, sdf_file)

        import logging
     
        # Load the molecule
        self.molecule = None
        self._load_molecule(os.path.join(self.working_dir, molecule_path))

        supplier = Chem.SDMolSupplier(self.sdf_path, removeHs=False)
        for mol in supplier:
    
            if mol is None:
                print('No molecule in the sdf file!')
                
            mol_h = Chem.AddHs(mol)
            num_atoms = mol_h.GetNumAtoms()
            logging.debug(print('#num_atoms:',num_atoms))

            for atom in mol_h.GetAtoms():
               atom.SetAtomMapNum(atom.GetIdx() + 1)
            mapped_smiles = Chem.MolToSmiles(mol_h)
            offmol = Molecule.from_mapped_smiles(mapped_smiles, allow_undefined_stereo=True)

            logging.debug(print('#Converting sdf file into dgl graph...'))
            self.g = esp.Graph(offmol)
            self.g =self.g.heterograph

        # Set up calculator
        self.molecule = Atoms(positions=self.molecule.get_positions(), numbers=self.molecule.numbers, charges=self.molecule.get_initial_charges())
        logging.debug(print('#Preparing calculator...'))
        calculator = ResFFCalculator(
            ml_model,
            device=self.device,
            energy=energy,
            forces=forces,
            atoms=self.molecule,
            g=self.g,
            sdf_path = self.sdf_path
         
            )
      self.molecule.calc = calculator
        # Unless initialized, set dynamics to False
        self.dynamics = False

    
    def _load_molecule(self, molecule_path):
        """
        Load molecule from file (can handle all ase formats).
        Args:
            molecule_path (str): Path to molecular geometry
        """
        file_format = os.path.splitext(molecule_path)[-1]
        self.molecule = read(molecule_path)

    def save_molecule(self, name, file_format="xyz", append=False):
        """
        Save the current molecular geometry.
        Args:
            name (str): Name of save-file.
            file_format (str): Format to store geometry (default xyz).
            append (bool): If set to true, geometry is added to end of file
                (default False).
        """
        molecule_path = os.path.join(self.working_dir, "%s.%s" % (name, file_format))
        write(molecule_path, self.molecule, format=file_format, append=append)


    def get_gibbs_free_energy(self):
        """Return the Gibb's free energy, which is supposed to be conserved.

        Requires that the energies of the atoms are up to date.

        This is mainly intended as a diagnostic tool.  If called before the
        first timestep, Initialize will be called.
        """
        linalg = np.linalg
        self.eta = np.zeros((3, 3), float)
        self.zeta = 0.0
        self.zeta_integrated = 0.0
        self.ttime = None
        self.pfactor_given = None
        self.externalstress = tuple(self.molecule.get_stress(
                include_ideal_gas=True) / units.GPa)
        n = self.molecule.get_global_number_of_atoms()
        # tretaTeta = sum(diagonal(matrixmultiply(transpose(self.eta),
        #                                        self.eta)))
        contractedeta = np.sum((self.eta * self.eta).ravel())
        gibbs = (self.molecule.get_potential_energy() +
                 self.molecule.get_kinetic_energy()
                 - np.sum(self.externalstress[0:3]) * linalg.det(self.h) / 3.0)
        if self.ttime is not None:
            gibbs += (1.5 * n * self.temperature *
                      (self.ttime * self.zeta)**2 +
                      3 * self.temperature * (n - 1) * self.zeta_integrated)
        else:
            assert self.zeta == 0.0
        if self.pfactor_given is not None:
            gibbs += 0.5 / self.pfact * contractedeta
        else:
            assert contractedeta == 0.0
        return gibbs
    
    
    def calculate_single_point(self):
        """
        Perform a single point computation of the energies and forces and
        store them to the working directory. The format used is the extended
        xyz format. This functionality is mainly intended to be used for
        interfaces.
        """
        energy = self.molecule.get_potential_energy()
        forces = self.molecule.get_forces()
        self.molecule.energy = energy
        self.molecule.forces = forces

        self.save_molecule("single_point", file_format="extxyz")

    def print_energy(self):
        """Print potential, kinetic, and total energy per atom with density."""
        epot = self.molecule.get_potential_energy() / len(self.molecule)
        ekin = self.molecule.get_kinetic_energy() / len(self.molecule)
        
        # Calculate density
        mass_amu = self.molecule.get_masses().sum()
        mass_g = mass_amu / mol
        vol_A3 = self.molecule.get_volume()
        vol_cm3 = vol_A3 * 1e-24  # Convert A³ to cm³
        density = mass_g / vol_cm3  # g/cm³
        
        temperature = ekin / (1.5 * units.kB)
        etot = epot + ekin
        
        print(f'Energy per atom: Epot = {epot:.4f} eV  Ekin = {ekin:.4f} eV  '
            f'T = {temperature:.0f} K  Etot = {etot:.4f} eV  Density = {density:.4f} g/cm³')
        
        # Save density to file
        with open('density.txt', 'a') as f:
            f.write(f"{density}\n")

    def init_md(
        self,
        name,
        time_step=0.1,
        temp_init=300,
        temp_bath=None,
        reset=False,
        interval=1,
    ):
        """
        Initialize an ase molecular dynamics trajectory. The logfile needs to
        be specifies, so that old trajectories are not overwritten. This
        functionality can be used to subsequently carry out equilibration and
        production.
        Args:
            name (str): Basic name of logfile and trajectory
            time_step (float): Time step in fs (default=0.5)
            temp_init (float): Initial temperature of the system in K
                (default is 300)
            temp_bath (float): Carry out Langevin NVT dynamics at the specified
                temperature. If set to None, NVE dynamics are performed
                instead (default=None)
            reset (bool): Whether dynamics should be restarted with new initial
                conditions (default=False)
            interval (int): Data is stored every interval steps (default=1)
        """
        from ase import Atoms
      
        # If a previous dynamics run has been performed, don't reinitialize
        # velocities unless explicitly requested via reset=True
        if not self.dynamics or reset:
            self._init_velocities(temp_init=temp_init)

        
        self.dynamics = Langevin(
            atoms=self.molecule,
            timestep=time_step * units.fs,
            temperature_K=temp_bath,
            friction=1.0 / (100.0 * units.fs),
            loginterval=interval
        )

        # Create monitors for logfile and a trajectory file
        logfile = os.path.join(self.working_dir, "%s.log" % name)
        trajfile = os.path.join(self.working_dir, "%s.traj" % name)
        logger = MDLogger(
            self.dynamics,
            self.molecule,
            logfile,
            stress=False,
            peratom=False,
            header=True,
            mode="a",
        )
        trajectory = Trajectory(trajfile, "w", self.molecule)

        # Attach monitors to trajectory
        self.dynamics.attach(logger, interval=interval)
        self.dynamics.attach(trajectory.write, interval=interval)

    def _init_velocities(
        self, temp_init=300, remove_translation=True, remove_rotation=True
    ):
        """
        Initialize velocities for molecular dynamics
        Args:
            temp_init (float): Initial temperature in Kelvin (default 300)
            remove_translation (bool): Remove translation components of
                velocity (default True)
            remove_rotation (bool): Remove rotation components of velocity
                (default True)
        """
        MaxwellBoltzmannDistribution(self.molecule, temperature_K=temp_init)

        if remove_translation:
            Stationary(self.molecule)
        if remove_rotation:
            ZeroRotation(self.molecule)

    def run_md(self, steps):
        """
        Perform a molecular dynamics simulation using the settings specified
        upon initializing the class.
        Args:
            steps (int): Number of simulation steps performed
        """
        if not self.dynamics:
            raise AttributeError(
                "Dynamics need to be initialized using the" " 'setup_md' function"
            )

        self.dynamics.run(steps)
 
    def optimize(self, fmax=1.0, steps=1000): 
        """
        Optimize a molecular geometry using the Quasi Newton optimizer in ase
        (BFGS + line search)
        Args:
            fmax (float): Maximum residual force change (default 1.0)
            steps (int): Maximum number of steps (default 1000)
        """
        name = "optimization"
        optimize_file = os.path.join(self.working_dir, name)
   
        optimizer = BFGS(
            self.molecule,
            trajectory="%s.traj" % optimize_file,
            logfile="%s.log" % optimize_file,
            restart="%s.pkl" % optimize_file,
        )
        import time
        import logging
        start_time = time.time()
        optimizer.run(fmax, steps)
        end_time = time.time()
        logging.debug(print('running time:',end_time - start_time))
        # Save final geometry in xyz format
        self.save_molecule(name)

    
    def compute_normal_modes(self, write_jmol=True):
        """
        Use ase calculator to compute numerical frequencies for the molecule
        Args:
            write_jmol (bool): Write frequencies to input file for
                visualization in jmol (default=True)
        """
        freq_file = os.path.join(self.working_dir, "normal_modes")

        # Compute frequencies
        frequencies = Vibrations(self.molecule, name=freq_file)
        frequencies.run()

        # Print a summary
        frequencies.summary()

        # Write jmol file if requested
        if write_jmol:
            frequencies.write_jmol()
            
    def convert_to_mdanalysis(self, ase_traj: str=None, slices=slice(0,None,1), save_format="dcd"):
        assert os.path.exists(ase_traj), "ASE traj does not exist... Run MD or Optimization first!"
        traj = Trajectory(ase_traj)[slices] #Trajectory ==> ase.io.trajectory.SlicedTrajectory
        coords = np.stack([atom.positions for atom in traj], axis=0) #Frames, L, 3
        ase_molecule = os.path.join(self.working_dir, self.molecule_path) #xyz file
        
        u = mda.Universe(ase_molecule) #universe
        u.load_new(coords) #overwrite coords
        ase_traj_name = os.path.split(ase_traj)[-1].split(".")[0] #filename w/o/ extenstion...
        mda_traj_name = ase_traj_name + "." + save_format #[filename].dcd
        
        with mda.Writer(mda_traj_name, u.atoms.n_atoms) as w:
            for ts in u.trajectory:
                w.write(u.atoms)        
        
        return u #mda.Universe

    @staticmethod
    def convert_to_mdanalysis_static(ase_traj: str=None, slices=slice(0,None,1), save_format="dcd", working_dir=None, molecule_path=None):
        assert os.path.exists(ase_traj), "ASE traj does not exist... Run MD or Optimization first!"
        traj = Trajectory(ase_traj)[slices] #Trajectory ==> ase.io.trajectory.SlicedTrajectory
        coords = np.stack([atom.positions for atom in traj], axis=0) #Frames, L, 3
        ase_molecule = os.path.join(working_dir, molecule_path) #xyz file
        
        u = mda.Universe(ase_molecule) #universe
        u.load_new(coords) #overwrite coords
        ase_traj_name = os.path.split(ase_traj)[-1].split(".")[0] #filename w/o/ extenstion...
        mda_traj_name = ase_traj_name + "." + save_format #[filename].dcd
        
        with mda.Writer(mda_traj_name, u.atoms.n_atoms) as w:
            for ts in u.trajectory:
                w.write(u.atoms)        
        
        return u #mda.Universe


