#!/bin/bash

# Configuration
SDF_DIR="ResFF/data/raw_data/OpenFF-Industry-Benchmark/02-chunks" # Input molecule directory
DEST_DIR="./opt-results" # Output directory
RANGE_START=0 
RANGE_END=250

for i in $(seq $RANGE_START $RANGE_END); do
    SDF_FILE="${SDF_DIR}/01-processed-qm-${i}.sdf"
    
    if [[ -f "$SDF_FILE" ]]; then
        echo "Processing $SDF_FILE..."
        python3 - <<EOF
import os
from rdkit import Chem
from rdkit.Chem import rdmolfiles

sdf_file = "$SDF_FILE"
dest_dir = "$DEST_DIR"

# Read molecule from SDF
supplier = Chem.SDMolSupplier(sdf_file,removeHs=False)
if not supplier:
    print(f"Error reading {sdf_file}")
    exit()

for mol in supplier:
    if mol is None:
        continue
    mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else "unknown"
    
    mol_dir = os.path.join(dest_dir, mol_name)
    os.makedirs(mol_dir, exist_ok=True)
    
    optimization_file = os.path.join(mol_dir, "optimization.xyz")
    if os.path.exists(optimization_file):
        print(f"Skipping {mol_name} as optimization.xyz already exists.")
        continue

    # Save molecule
    mol_sdf_path = os.path.join(mol_dir, f"{mol_name}.sdf")
    writer = rdmolfiles.SDWriter(mol_sdf_path)
    writer.write(mol)
    writer.close()
    
    # Run run.py 
    os.system(f"taskset -c 30 python3 run.py {mol_name}")
EOF
    else
        echo "File $SDF_FILE does not exist. Skipping..."
    fi
done
