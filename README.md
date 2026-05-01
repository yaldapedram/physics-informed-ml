# Physics-informed ML — selected code examples

This repository contains small, self-contained examples extracted from my PhD research codebase on multiscale modeling and physics-informed machine learning for clay platelet interactions. 

The purpose of this repository is to showcase clean, reproducible parts of the workflow, including geometry generation, periodic image handling, PMF-style data processing, pairwise-distance preprocessing, and lightweight training demos.


## What this repo demonstrates 
**Research relevance**
- A reproducible pipeline from **simulation outputs → structured datasets → simple training loops**
- Physics-informed choices (e.g., **Morse baseline + kernel/GPR correction**) to improve data efficiency and interpretability
- Practical validation steps for periodic systems (replication / neighbor images) to avoid missing cross-boundary interactions

**Industry relevance**
- Modular Python package structure (`src/`) with scripts that run end-to-end (`scripts/`)
- Data I/O utilities and sanity checks that make experiments debuggable and repeatable
- Training demos in PyTorch that illustrate model design, optimization loops, and parameter tracking

## Repository layout

- `src/geometry/` — geometry builders for semiperiodic and two-sheet platelet configurations
- `src/pairwise/` — utilities for loading and saving pairwise-distance datasets
- `src/pmf/` — utilities for loading PMF-style curves, such as distance/free-energy data
- `src/training/` — lightweight training utilities used by demo scripts
- `scripts/` — runnable demo scripts for geometry validation, PMF post-processing, and lightweight training
- `data/sample/` — small sample files used by the demo scripts
- `lammps/mean_force_pmf_nammt/` — LAMMPS workflow template for fixed-separation Na-MMT simulations used in PMF analysis
- `tabulated_pair_potentials/` — example tabulated pair-potential files for coarse-grained simulations
  
## Demo 1 — Semiperiodic sheet + two-sheet configuration
Builds a semiperiodic sheet (central + edge sites), then constructs a two-sheet 3D configuration (tilt + vertical separation).

```bash
python -m scripts.coarse_graining_geometry_demo

## Demo 2 — LAMMPS-to-PMF workflow

The folder `lammps/mean_force_pmf_nammt/` contains a LAMMPS workflow template for fixed-separation Na-MMT simulations used in potential of mean force (PMF) analysis.

The workflow applies prescribed z-direction displacements between MMT platelets, runs NPT pre-equilibration followed by NVT production, and outputs thermodynamic, center-of-mass, and force-related quantities. These outputs can be post-processed across multiple separations to estimate a relative PMF/free-energy profile.

The full LAMMPS structure/data file is not included because it is system-specific and may contain large or unpublished research data.

A sample PMF-style dataset is included in `data/sample/`, and `scripts/pmf_from_lammps_out_demo.py` demonstrates how LAMMPS-style output can be converted into a relative PMF profile.
