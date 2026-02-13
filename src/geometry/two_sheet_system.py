"""
Build a simple two-sheet (two-platelet) configuration from a 2D semiperiodic sheet.

This is a small, self-contained helper to keep demo scripts clean.
- Sheet 1: z = 0
- Sheet 2: shifted in z by z_dist, and shifted in x by x_shift = z_dist / tan(theta)

Returns separate arrays for central/edge particles for each sheet.
"""

from __future__ import annotations

import numpy as np


def build_two_sheet_system(
    central_xy: np.ndarray,
    edge_xy: np.ndarray,
    theta_deg: float,
    z_dist: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    central_xy : (N, 2) array
    edge_xy    : (M, 2) array
    theta_deg  : tilt angle (degrees)
    z_dist     : vertical separation between sheets

    Returns
    -------
    flat_central  : (N, 3) sheet1 central
    flat_edge     : (M, 3) sheet1 edge
    flat2_central : (N, 3) sheet2 central
    flat2_edge    : (M, 3) sheet2 edge
    """
    if central_xy.ndim != 2 or central_xy.shape[1] != 2:
        raise ValueError("central_xy must be shape (N, 2)")
    if edge_xy.ndim != 2 or edge_xy.shape[1] != 2:
        raise ValueError("edge_xy must be shape (M, 2)")

    theta = np.deg2rad(theta_deg)
    # Avoid division by zero for theta ~ 0
    if np.isclose(np.tan(theta), 0.0):
        raise ValueError("theta_deg leads to tan(theta)=0; cannot compute x_shift.")

    x_shift = z_dist / np.tan(theta)

    flat_central = np.column_stack((central_xy, np.zeros(len(central_xy))))
    flat_edge = np.column_stack((edge_xy, np.zeros(len(edge_xy))))

    flat2_central = flat_central.copy()
    flat2_central[:, 2] += z_dist
    flat2_central[:, 0] -= x_shift

    flat2_edge = flat_edge.copy()
    flat2_edge[:, 2] += z_dist
    flat2_edge[:, 0] -= x_shift

    return flat_central, flat_edge, flat2_central, flat2_edge
