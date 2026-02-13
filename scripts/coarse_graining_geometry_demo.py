import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from src.geometry.semiperiodic_sheet import build_semiperiodic_sheet


def main():
    # -----------------------------
    # Parameters (edit these)
    # -----------------------------
    theta_deg = 61.0
    z_dist = 5.44

    radius = 5.19
    w_max = 51.50
    h_max = 50.0

    edge_offset = 2.595

    # -----------------------------
    # Build 2D sheet + edges
    # -----------------------------
    central_coord, edge_coord = build_semiperiodic_sheet(
        radius=radius, w_max=w_max, h_max=h_max, edge_offset=edge_offset
    )

    # -----------------------------
    # 2D Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(central_coord[:, 0], central_coord[:, 1], "o", markersize=3, label="Central")
    ax.plot(edge_coord[:, 0], edge_coord[:, 1], "o", markersize=3, label="Edge")

    # grid spacing (matches your original style)
    ax.xaxis.set_major_locator(MultipleLocator(edge_offset))
    ax.yaxis.set_major_locator(MultipleLocator(np.sqrt(3) * radius / 2))
    ax.grid(which="major", linestyle="--", alpha=0.5)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    plt.show()

    # -----------------------------
    # Build 3D two-sheet system
    # -----------------------------
    theta = np.deg2rad(theta_deg)
    x_shift = z_dist / np.tan(theta)

    flat_central = np.column_stack((central_coord, np.zeros(len(central_coord))))
    flat_edge = np.column_stack((edge_coord, np.zeros(len(edge_coord))))

    flat2_central = flat_central.copy()
    flat2_central[:, 2] += z_dist
    flat2_central[:, 0] -= x_shift

    flat2_edge = flat_edge.copy()
    flat2_edge[:, 2] += z_dist
    flat2_edge[:, 0] -= x_shift

    # -----------------------------
    # 3D Plot
    # -----------------------------
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection="3d")

    ax.scatter(flat_central[:, 0], flat_central[:, 1], flat_central[:, 2], s=8, label="Central")
    ax.scatter(flat2_central[:, 0], flat2_central[:, 1], flat2_central[:, 2], s=8)

    ax.scatter(flat_edge[:, 0], flat_edge[:, 1], flat_edge[:, 2], s=8, label="Edge")
    ax.scatter(flat2_edge[:, 0], flat2_edge[:, 1], flat2_edge[:, 2], s=8)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    plt.show()

    # -----------------------------
    # Counts
    # -----------------------------
    print("Central atoms:", len(flat_central) + len(flat2_central))
    print("Edge atoms:", len(flat_edge) + len(flat2_edge))
    print("Total atoms:", len(flat_central) + len(flat_edge) + len(flat2_central) + len(flat2_edge))


if __name__ == "__main__":
    main()
