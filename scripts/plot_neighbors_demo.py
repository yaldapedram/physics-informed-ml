import numpy as np
import matplotlib.pyplot as plt

from src.geometry.semiperiodic_sheet import build_semiperiodic_sheet


def main():
    # --- same params as your other scripts ---
    radius = 5.19
    w_max = 51.50
    h_max = 50.0
    edge_offset = 2.595

    Ly = 53.9346        # your y-period length
    cutoff = 25.0

    central, edge = build_semiperiodic_sheet(radius, w_max, h_max, edge_offset)

    # choose a point to inspect (pick something near the top boundary)
    p = central[np.argmax(central[:, 1])]  # highest-y central point

    offsets = [-Ly, 0.0, +Ly]

    fig, ax = plt.subplots(figsize=(12, 7))

    for dy in offsets:
        # base + neighbor shifted copies
        ax.scatter(central[:, 0], central[:, 1] + dy, s=8, alpha=0.5, label=f"central dy={dy:g}")
        ax.scatter(edge[:, 0], edge[:, 1] + dy, s=8, alpha=0.5, label=f"edge dy={dy:g}")

    # highlight the chosen point and cutoff radius
    ax.scatter([p[0]], [p[1]], s=80, marker="x", label="chosen point (dy=0)")
    circle = plt.Circle((p[0], p[1]), cutoff, fill=False)
    ax.add_patch(circle)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Base sheet + y-neighbors (dy = 0, ±Ly) with cutoff circle")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(ncols=2, fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
