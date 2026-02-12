# Physics-informed ML — selected code examples

This repository contains small, self-contained examples extracted from my PhD research codebase.

## Demo: Semiperiodic sheet geometry
Generates a 2D semiperiodic sheet with edge particles, then builds a two-sheet 3D configuration (tilt angle + vertical separation) and visualizes both.

### Run locally
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.coarse_graining_geometry_demo
