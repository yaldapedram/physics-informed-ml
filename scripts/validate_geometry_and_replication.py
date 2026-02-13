import numpy as np
from scipy.spatial import distance

# Reuse your existing builder
from src.geometry.semiperiodic_sheet import build_semiperiodic_sheet


def build_two_sheets(
    theta_deg: float,
    z_dist: float,
    radius: float,
    w_max: float,
    h_max: float,
    edge_offset: float,
):
    """
    Build two sheets (central + edge) in 3D:
    Sheet 1 at z=0
    Sheet 2 at z=z_dist and shifted in x by x_shift = z_dist/tan(theta)
    """
    theta = np.deg2rad(theta_deg)
    x_shift = z_dist / np.tan(theta)

    central_2d, edge_2d = build_semiperiodic_sheet(
        radius=radius, w_max=w_max, h_max=h_max, edge_offset=edge_offset
    )

    # Sheet 1
    s1_cen = np.column_stack((central_2d, np.zeros(len(central_2d))))
    s1_ed = np.column_stack((edge_2d, np.zeros(len(edge_2d))))

    # Sheet 2
    s2_cen = s1_cen.copy()
    s2_cen[:, 2] += z_dist
    s2_cen[:, 0] -= x_shift

    s2_ed = s1_ed.copy()
    s2_ed[:, 2] += z_dist
    s2_ed[:, 0] -= x_shift

    return s1_cen, s1_ed, s2_cen, s2_ed, x_shift


def replicate_in_y(points: np.ndarray, y_offsets):
    """Replicate a set of 3D points by shifting in y."""
    blocks = []
    for y in y_offsets:
        disp = np.array([0.0, float(y), 0.0])
        blocks.append(points + disp)
    return np.vstack(blocks)


def count_distances_within_cutoff(A: np.ndarray, B: np.ndarray, cutoff: float) -> int:
    """Count pair distances <= cutoff between two point clouds."""
    dmat = distance.cdist(A, B)
    return int(np.sum(dmat <= cutoff))


def main():
    # -----------------------------
    # Edit these to match your system
    # -----------------------------
    theta_deg = 61.0
    z_dist = 5.44

    radius = 5.19
    w_max = 51.50
    h_max = 50.0
    edge_offset = 2.595

    cutoff = 25.0

    # This should be the periodic length in y (your number)
    Ly = 53.9346

    # -----------------------------
    # Build geometry
    # -----------------------------
    s1_cen, s1_ed, s2_cen, s2_ed, x_shift = build_two_sheets(
        theta_deg, z_dist, radius, w_max, h_max, edge_offset
    )

    print("\n=== Two-sheet sanity ===")
    print(f"Sheet1 central: {len(s1_cen)}  edge: {len(s1_ed)}")
    print(f"Sheet2 central: {len(s2_cen)}  edge: {len(s2_ed)}")
    print(f"z-dist target: {z_dist}")
    print(f"Sheet1 z range: [{s1_cen[:,2].min():.4f}, {s1_cen[:,2].max():.4f}]")
    print(f"Sheet2 z range: [{s2_cen[:,2].min():.4f}, {s2_cen[:,2].max():.4f}]")
    print(f"x_shift used (sheet2 moved by -x_shift): {x_shift:.6f}")
    print(f"mean x shift (s2 - s1): {(s2_cen[:,0].mean() - s1_cen[:,0].mean()):.6f}")

    # Combine central+edge (often you care about each separately, but this is a quick check)
    s1_all = np.vstack((s1_cen, s1_ed))
    s2_all = np.vstack((s2_cen, s2_ed))

    print("\n=== Replication sanity ===")
    print(f"Ly = {Ly} (y-period length)")
    print("We will test neighbor layers 0..3 and see if distance counts converge.")

    # -----------------------------
    # Convergence test: neighbor layers
    # -----------------------------
    base_count = count_distances_within_cutoff(s1_all, s2_all, cutoff)
    print(f"\nDistances within cutoff={cutoff} WITHOUT replication: {base_count}")

    prev = None
    for layers in range(0, 4):
        y_offsets = [k * Ly for k in range(-layers, layers + 1)]
        s2_rep = replicate_in_y(s2_all, y_offsets)

        rep_count = count_distances_within_cutoff(s1_all, s2_rep, cutoff)

        print(f"layers={layers}  offsets={len(y_offsets):2d}  distances kept: {rep_count}")

        if prev is not None and rep_count == prev:
            print("  -> unchanged from previous layer (good sign: converging)")
        prev = rep_count

    print("\nIf 'distances kept' stops changing as layers increase, replication is sufficient.")
    print("If it keeps increasing at layers=3, increase to 4-5 layers and re-check.\n")


if __name__ == "__main__":
    main()
