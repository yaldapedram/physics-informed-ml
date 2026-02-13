# Physics-informed ML — selected code examples

This repository contains small, self-contained examples extracted from my PhD research codebase (multiscale modeling / physics-informed ML for clay platelet interactions). The goal is to show **clean, reproducible pieces** of the workflow—geometry generation, periodic “image” handling, PMF-style data handling, pairwise distance preprocessing, and lightweight training demos—without publishing the full research repo or any confidential data.

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
- `src/geometry/` — geometry builders for a semiperiodic sheet and a two-sheet configuration
- `src/pairwise/` — utilities for loading/saving pairwise distance datasets
- `src/pmf/` — utilities for loading “PMF-style” curves (distance vs free energy)
- `src/training/` — small training utilities / helpers used by demo scripts
- `scripts/` — runnable demos (plotting, geometry validation, training)

## Demo 1 — Semiperiodic sheet + two-sheet configuration
Builds a semiperiodic sheet (central + edge sites), then constructs a two-sheet 3D configuration (tilt + vertical separation).

```bash
python -m scripts.coarse_graining_geometry_demo
