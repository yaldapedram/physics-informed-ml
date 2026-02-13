"""
Demo: replicate neighbor images in Y to mimic periodic interactions.

Why this exists:
- LAMMPS is periodic (PBC). Interactions can cross the box boundary.
- For post-processing (pairwise distances / interactions), we replicate the base system
  by shifting it by +/- N * Ly in Y, so interactions are not "missed".

This script:
1) Builds a semiperiodic sheet (central + edges) in 2D
2) Builds a two-sheet 3D configuration (tilt angle + vertical separation)
3) Replicates the full two-sheet system in Y (image copies)
4) Filters a "primary" Y-window to avoid double counting
5) Visualizes (optional) so you can sanity-check

Run:
python -m scripts.replicate_neighbors_demo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from src.geometry.semiperiodic_sheet import build_semiperiodic_sheet
from src.geometry.two_sheet_system import build_two_sheet_system


def replicate_in_y(points: np.ndarray, y_offsets: list[float]) -> np.ndarray:
    """
    Stack image copies of `points` by shifting in y.
    points: (N, 3)
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be shape (N, 3)")

    copies = []
    for y in y_offsets:
        displacement = np.array([0.0, float(y), 0.0])
        copies.append(points + displacement)
    return np.vstack(copies)


def filter_primary_y(points: np.ndarray, y_min: float, y_max: float) -> np.ndarray:
    """
    Keep only points with y in (y_min, y_max).
    """
    mask = (points[:, 1] > y_min) & (points[:, 1] < y_max)
    return points[mask]


def main():
    # -----------------------------
    # Parameters (edit these)
    # -----------------------------
    theta_deg = 61.0
    z_dist = 5.44

    # sheet geometry
    radius = 5.19
    w_max = 51.50
    h_max = 50.0
    edge_offset = 2.595

    # Replication in y (Ly should match your "box length" in y for the CG system)
    # If you know Ly exactly, set it here.
    Ly = 53.9346
    n_images = 3
    y_offsets = [k * Ly for k in range(-n_images, n_images + 1)]

    # Primary window in y (chosen to avoid double counting in your earlier workflow)
    y_min_1, y_max_1 = -1.0, 36.06

    # -----------------------------
    # Build 2D sheet
    # -----------------------------
    central_xy, edge_xy = build_semiperiodic_sheet(
        radius=radius, w_max=w_max, h_max=h_max, edge_offset=edge_offset
    )

    # -----------------------------
    # Build 3D two-sheet system
    # -----------------------------
    flat_central, flat_edge, flat2_central, flat2_edge = build_two_sheet_system(
        central_xy=central_xy,
        edge_xy=edge_xy,
        theta_deg=theta_deg,
        z_dist=z_dist,
    )

    # Combine both sheets for replication
    p1_cen = np.vstack((flat_central, flat2_central))
    p1_ed = np.vstack((flat_edge, flat2_edge))

    # -----------------------------
    # Replicate neighbor images in Y
    # -----------------------------
    mirrored_cen = replicate_in_y(p1_cen, y_offsets)
    mirrored_ed = replicate_in_y(p1_ed, y_offsets)

    # -----------------------------
    # Filter "primary" region
    # -----------------------------
    p1_cen_filtered = filter_primary_y(p1_cen, y_min=y_min_1, y_max=y_max_1)
    p1_ed_filtered = filter_primary_y(p1_ed, y_min=y_min_1, y_max=y_max_1)

    # -----------------------------
    # Quick 2D sanity plot (XY)
    # -----------------------------
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(p1_cen_filtered[:, 0], p1_cen_filtered[:, 1], s=20, alpha=0.9, label="Central (primary)")
    ax.scatter(p1_ed_filtered[:, 0], p1_ed_filtered[:, 1], s=20, alpha=0.9, label="Edge (primary)")

    ax.xaxis.set_major_locator(MultipleLocator(edge_offset))
    ax.yaxis.set_major_locator(MultipleLocator(np.sqrt(3) * radius / 2))
    ax.grid(which="major", linestyle="--", alpha=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Primary Y-window (filtered) — used to avoid double counting")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 3D sanity plot (optional, can be slow if large)
    # -----------------------------
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection="3d")

    # Show a subset if you want faster plotting:
    # idx = np.random.choice(len(mirrored_cen), size=min(5000, len(mirrored_cen)), replace=False)
    # ax.scatter(mirrored_cen[idx,0], mirrored_cen[idx,1], mirrored_cen[idx,2], s=5, alpha=0.3, label="Central images")

    ax.scatter(mirrored_cen[:, 0], mirrored_cen[:, 1], mirrored_cen[:, 2], s=6, alpha=0.25, label="Central images")
    ax.scatter(mirrored_ed[:, 0], mirrored_ed[:, 1], mirrored_ed[:, 2], s=6, alpha=0.25, label="Edge images")

    ax.scatter(p1_cen_filtered[:, 0], p1_cen_filtered[:, 1], p1_cen_filtered[:, 2], s=10, alpha=0.9, label="Central primary")
    ax.scatter(p1_ed_filtered[:, 0], p1_ed_filtered[:, 1], p1_ed_filtered[:, 2], s=10, alpha=0.9, label="Edge primary")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Neighbor images in Y + primary region")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Print counts
    # -----------------------------
    print("Base central (two sheets):", len(p1_cen))
    print("Base edge (two sheets):", len(p1_ed))
    print("Mirrored central total:", len(mirrored_cen))
    print("Mirrored edge total:", len(mirrored_ed))
    print("Primary central:", len(p1_cen_filtered))
    print("Primary edge:", len(p1_ed_filtered))


if __name__ == "__main__":
    main()
