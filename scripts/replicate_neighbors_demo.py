"""
Replicate neighbor images in Y (same logic as original notebook/script).

This script reproduces the same steps you used:
1) Build 2D sheet (central + edge)
2) Build 3D two-sheet system (tilt + z separation)
3) Replicate in Y using multiples of Ly: [-3Ly, ..., +3Ly]
4) Filter a primary Y-window (y_min_1, y_max_1)
5) Plot the same sanity figures

Run:
python -m scripts.replicate_neighbors_same_as_original_demo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from src.geometry.semiperiodic_sheet import build_semiperiodic_sheet
from src.geometry.two_sheet_system import build_two_sheet_system


def replicate_in_y(points: np.ndarray, y_offsets: list[float]) -> np.ndarray:
    """Replicate points by shifting in y for each value in y_offsets."""
    copies = []
    for y in y_offsets:
        displacement = np.array([0.0, float(y), 0.0])
        copies.append(points + displacement)
    return np.vstack(copies)


def main():
    # -----------------------------
    # Parameters (same as your code)
    # -----------------------------
    theta_deg = 61.0
    z_dist = 5.44

    radius = 5.19
    w_max = 51.50
    h_max = 50.0
    edge_offset = 2.595

    # same as your code
    Ly = 53.9346
    y_offsets = [-3 * Ly, -2 * Ly, -1 * Ly, 0.0, 1 * Ly, 2 * Ly, 3 * Ly]

    y_min_1, y_max_1 = -1.0, 36.06

    # -----------------------------
    # Build 2D sheet (same result)
    # -----------------------------
    central_xy, edge_xy = build_semiperiodic_sheet(
        radius=radius, w_max=w_max, h_max=h_max, edge_offset=edge_offset
    )

    # -----------------------------
    # Build 3D two-sheet system
    # -----------------------------
    Flat_central, Flat_edge, Flat2_central, Flat2_edge = build_two_sheet_system(
        central_xy=central_xy,
        edge_xy=edge_xy,
        theta_deg=theta_deg,
        z_dist=z_dist,
    )

    # Same as your code: stack two sheets
    p1_cen = np.vstack((Flat_central, Flat2_central))
    p1_ed = np.vstack((Flat_edge, Flat2_edge))

    # -----------------------------
    # Replicate in Y (same as yours)
    # -----------------------------
    FlatPlateMirr_cen = replicate_in_y(p1_cen, y_offsets)
    FlatPlateMirr_ed = replicate_in_y(p1_ed, y_offsets)

    # -----------------------------
    # Plot: base (2D scatter like yours)
    # -----------------------------
    plt.scatter(p1_cen[:, 0], p1_cen[:, 1])
    plt.scatter(p1_ed[:, 0], p1_ed[:, 1])
    plt.title("Base two-sheet system (XY)")
    plt.show()

    # -----------------------------
    # Plot: mirrored copies (3D like yours)
    # -----------------------------
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(FlatPlateMirr_cen[:, 0], FlatPlateMirr_cen[:, 1], FlatPlateMirr_cen[:, 2], s=10, alpha=0.5)
    ax.scatter(FlatPlateMirr_ed[:, 0], FlatPlateMirr_ed[:, 1], FlatPlateMirr_ed[:, 2], s=10, alpha=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Mirrored Platelets in Y (3D)")
    plt.show()

    # -----------------------------
    # Filter primary region (same)
    # -----------------------------
    mask_cen = (p1_cen[:, 1] > y_min_1) & (p1_cen[:, 1] < y_max_1)
    p1_cen_filtered = p1_cen[mask_cen]

    mask_ed = (p1_ed[:, 1] > y_min_1) & (p1_ed[:, 1] < y_max_1)
    p1_ed_filtered = p1_ed[mask_ed]

    # -----------------------------
    # Plot: filtered region (same style)
    # -----------------------------
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.scatter(p1_cen_filtered[:, 0], p1_cen_filtered[:, 1], s=30, alpha=0.9, label="Filtered center")
    ax.scatter(p1_ed_filtered[:, 0], p1_ed_filtered[:, 1], s=30, alpha=0.9, label="Filtered edge")

    ax.xaxis.set_major_locator(MultipleLocator(edge_offset))
    ax.yaxis.set_major_locator(MultipleLocator(np.sqrt(3) * radius / 2))
    ax.grid(which="major", color="#CCCCCC", linestyle="--")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Filtered particles in base system (primary Y-window)")
    ax.legend()
    plt.show()

    # -----------------------------
    # Plot: mirrored + filtered highlighted (same idea)
    # -----------------------------
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(FlatPlateMirr_cen[:, 0], FlatPlateMirr_cen[:, 1], FlatPlateMirr_cen[:, 2] + 10, s=10, alpha=0.5)
    ax.scatter(FlatPlateMirr_ed[:, 0], FlatPlateMirr_ed[:, 1], FlatPlateMirr_ed[:, 2] + 10, s=10, alpha=0.5)

    ax.scatter(p1_cen_filtered[:, 0], p1_cen_filtered[:, 1], p1_cen_filtered[:, 2], s=10, alpha=0.7)
    ax.scatter(p1_ed_filtered[:, 0], p1_ed_filtered[:, 1], p1_ed_filtered[:, 2], s=10, alpha=0.7)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Mirrored neighbors + primary region")
    plt.show()

    print("Base central (two sheets):", len(p1_cen))
    print("Base edge (two sheets):", len(p1_ed))
    print("Mirrored central total:", len(FlatPlateMirr_cen))
    print("Mirrored edge total:", len(FlatPlateMirr_ed))
    print("Primary central:", len(p1_cen_filtered))
    print("Primary edge:", len(p1_ed_filtered))


if __name__ == "__main__":
    main()
