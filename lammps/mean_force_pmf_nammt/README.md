# Na-MMT fixed-separation PMF simulations

This folder contains LAMMPS scripts for fixed-separation Na-montmorillonite (Na-MMT) simulations used in potential of mean force (PMF) analysis.

The workflow applies prescribed z-direction displacements to control the interlayer separation between two MMT platelets. For each displacement, the system is pre-equilibrated using NPT and then propagated using NVT production. The output includes energies, pressure components, center-of-mass coordinates, and force-related quantities that can be post-processed to estimate relative PMF/free-energy profiles.

## Files

- `NaMMT_SemiInf.in`  
  Template LAMMPS input script for one fixed-separation Na-MMT simulation.

- `Na.loop`  
  Bash loop script that creates separate folders for different displacement indices and generates `Na.in` for each simulation.

## Input data file

The LAMMPS data file is not included because it is system-specific and may contain large data.

To use this workflow, provide a compatible LAMMPS data file and update the `read_data` line in the LAMMPS input script if needed:

```lammps
read_data       WaterSemiInfNaMMT.data
