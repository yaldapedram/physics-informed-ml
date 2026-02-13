# Physics-informed ML — selected code examples

This repository contains small, self-contained examples extracted from my PhD research codebase.

## Demo: Semiperiodic sheet geometry
Generates a 2D semiperiodic sheet with edge particles, then builds a two-sheet 3D configuration (tilt angle + vertical separation) and visualizes both.

## Sample PMF-style dataset (d-spacing vs energy)
I include a small example dataset extracted from PMF post-processing:

- `data/sample/distance_Semi_extrapolation.out`

**Format (2 columns):**
1. `d_spacing` (same length unit as used in the simulations / analysis)
2. `free_energy` (relative; shifted by an arbitrary constant)

### Quick plot
```bash
python -m scripts.plot_pmf_demo data/sample/distance_Semi_extrapolation.out


### Run locally
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.coarse_graining_geometry_demo
