# This file is part of the espaloma package.
# =============================================================================
# IMPORTS
# =============================================================================
import random

import numpy as np
import pandas as pd
import torch
import contextlib

import resff

OFFSETS = {
    1: -0.500607632585,
    6: -37.8302333826,
    7: -54.5680045287,
    8: -75.0362229210,
}

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================


@contextlib.contextmanager
def make_temp_directory():
    import tempfile, shutil

    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)


def sum_offsets(elements):
    return sum([OFFSETS[element] for element in elements])


def from_csv(path, toolkit="rdkit", smiles_col=-1, y_cols=[-2], seed=2666):
    """Read csv from file."""

    def _from_csv():
        df = pd.read_csv(path)
        df_smiles = df.iloc[:, smiles_col]
        df_y = df.iloc[:, y_cols]

        if toolkit == "rdkit":
            from rdkit import Chem

            mols = [Chem.MolFromSmiles(smiles) for smiles in df_smiles]
            gs = [resff.HomogeneousGraph(mol) for mol in mols]

        elif toolkit == "openeye":
            from openeye import oechem

            mols = [
                oechem.OESmilesToMol(oechem.OEGraphMol(), smiles)
                for smiles in df_smiles
            ]
            gs = [resff.HomogeneousGraph(mol) for mol in mols]

        ds = list(zip(gs, list(torch.tensor(df_y.values))))

        random.seed(seed)
        random.shuffle(ds)

        return ds

    return _from_csv


def normalize(ds):
    """Get mean and std."""

    gs, ys = tuple(zip(*ds))
    y_mean = np.mean(ys)
    y_std = np.std(ys)

    def norm(y):
        return (y - y_mean) / y_std

    def unnorm(y):
        return y * y_std + y_mean

    return y_mean, y_std, norm, unnorm


def split(ds, partition):
    """Split the dataset according to some partition."""
    n_data = len(ds)

    # get the actual size of partition
    partition = [int(n_data * x / sum(partition)) for x in partition]

    ds_batched = []
    idx = 0
    for p_size in partition:
        ds_batched.append(ds[idx : idx + p_size])
        idx += p_size

    return ds_batched


def batch(ds, batch_size, seed=2666):
    """Batch graphs and values after shuffling."""
    import dgl

    # get the numebr of data
    n_data_points = len(ds)
    n_batches = n_data_points // batch_size  # drop the rest

    random.seed(seed)
    random.shuffle(ds)
    gs, ys = tuple(zip(*ds))

    gs_batched = [
        dgl.batch(gs[idx * batch_size : (idx + 1) * batch_size])
        for idx in range(n_batches)
    ]

    ys_batched = [
        torch.stack(ys[idx * batch_size : (idx + 1) * batch_size], dim=0)
        for idx in range(n_batches)
    ]

    return list(zip(gs_batched, ys_batched))


def collate_fn(graphs):
    import dgl

    return resff.HomogeneousGraph(dgl.batch(graphs))

def infer_mol_from_coordinates(
    coordinates,
    species,
    smiles_ref=None,
    coordinates_unit="angstrom",
):

    # local import
    from rdkit import Chem
    from openmm import unit
    from openmm.unit import Quantity

    if isinstance(coordinates_unit, str):
        coordinates_unit = getattr(unit, coordinates_unit)
    coordinates = Quantity(coordinates, coordinates_unit).value_in_unit(unit.angstrom)

    mol = Chem.RWMol()  
    
    if all(isinstance(symbol, str) for symbol in species):
        [
        mol.AddAtom(Chem.Atom(symbol))
        for symbol in species
        ]

    elif all(isinstance(symbol, int) for symbol in species):
        [
        mol.AddAtom(Chem.Atom(Chem.GetPeriodicTable().GetElementSymbol(symbol)))
        for symbol in species
        ]

    else:
        raise RuntimeError("The species can only be all strings or all integers.")

    conf = Chem.Conformer()  
    from rdkit.Geometry import Point3D

    for i in range(mol.GetNumAtoms()):
        x,y,z = coordinates[i]
        conf.SetAtomPosition(i,Point3D(x,y,z))
    mol.AddConformer(conf)  


    bond1 = Chem.BondType.SINGLE  
    bond2 = Chem.BondType.AROMATIC 

    for i in range(0, mol.GetNumAtoms(), 12):
        for j in range(i, i + 6):
            mol.AddBond(j, j + 6, bond1)  # Add bond1 between atoms j and j+6
        for j in range(i, i + 6):
            mol.AddBond(j, (j + 1) % 6 + i, bond2)  # Add bond2 between atoms j and (j+1)%6+i


    for bond in mol.GetBonds():
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()
        print(f"Atom {atom1_idx} to Atom {atom2_idx}: Bond type {bond_type}")

    #Chem.GetBonds(mol)

    Chem.GetSSSR(mol)
    Chem.Kekulize(mol)

    for atom_idx in range(mol.GetNumAtoms()):
        atom_pos = conf.GetAtomPosition(atom_idx)
        print(f"Atom {atom_idx + 1}: x={atom_pos.x:.3f}, y={atom_pos.y:.3f}, z={atom_pos.z:.3f}")


    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        atom_symbol = atom.GetSymbol()
        atom_degree = atom.GetDegree()
        atom_valence = atom.GetTotalValence()
        print(f"Atom {atom_idx + 1}: Symbol={atom_symbol}, Degree={atom_degree}, Valence={atom_valence}")
    
    if smiles_ref is not None:
        smiles_can = Chem.MolToSmiles(mol)
        mol_ref = Chem.MolFromSmiles(smiles_ref)
        mol_ref=Chem.AddHs(mol_ref)
        smiles_ref = Chem.MolToSmiles(mol_ref)

        assert (smiles_ref == smiles_can), "SMILES different. Input is %s, ref is %s" % (
            smiles_can,
            smiles_ref
        )
    from openff.toolkit.topology import Molecule

    _mol = Molecule.from_rdkit(mol, allow_undefined_stereo=True)
    g =resff.Graph(_mol)

    return g

