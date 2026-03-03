# Benchmarks used in ResFF

-   Gen2 Optimization(B3LYP-D3(BJ)/DZVP): GEN2-OPTIMIZATION-DATASET-OPENFF-DEFAULT.hdf5

-   TorsionNet-500(B3LYP/6-31G(d)): Torsion_Net.sdf

-   Torsion Scan(CCSD(T)/CBS): QM_MM_Gas_Phase_Torsion_Scan_Individual_Results_with_CCSD_T_CBS_baseline.sdf

-   S66x8(CCSD(T)/CBS): S66X8.sdf

-   OpenFF Industry Benchmark(B3LYP-D3BJ/DZVP): ./OpenFF-Industry-Benchmark.zip


The energy units of TorsionNet-500, Torsion Scan, S66x8 and OpenFF Industry Benchmark are kcal/mol, and the coordinate units are angstroms. The energy unit of Gen2 Optimization is hartree, and the coordinate unit is bohr.

We recomputed TorsionNet-500, Torsion Scan and S66x8 at the ωB97M-D3(BJ)/def2-TZVPPD level as the SPICE dataset in ./wb97m.