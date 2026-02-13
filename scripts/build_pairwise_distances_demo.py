import os
import numpy as np

from src.geometry.semiperiodic_sheet import build_semiperiodic_sheet
from src.geometry.two_sheet_system import build_two_sheet_system
from src.geometry.replication import replicate_in_y, filter_primary_y  # if you already made these helpers
from src.pairwise.build_pairwise_distances import (
    build_pairwise_distance_dicts,
    save_distance_dict,
    check_keys_match,
)


def load_z_spacings_from_out(path: str) -> np.ndarray:
    arr = np.genfromtxt(path)
    z = arr[:, 0]
    z = z[np.argsort(z)]
    return z


def main():
    # -----------------------------
    # Inputs
    # -----------------------------
    pmf_path = "data/sample/distance_Semi_extrapolation.out"  # adjust name if you renamed it

    # Geometry params (same as your demo)
    theta_deg = 61.0
    z_dist = 5.44
    radius = 5.19
    w_max = 51.50
    h_max = 50.0
    edge_offset = 2.595

    # Replication params (must match your CG box length in y)
    Ly = 53.9346
    n_images = 3
    y_offsets = [k * Ly for k in range(-n_images, n_images + 1)]

    # Primary window in y (same as you used)
    y_min_1, y_max_1 = -1.0, 36.06

    # Pairwise cutoff
    cutoff = 25.0

    # -----------------------------
    # Load z-spacings
    # -----------------------------
    z_S = load_z_spacings_from_out(pmf_path)
    print(f"Loaded {len(z_S)} z-spacings from {pmf_path}")

    # -----------------------------
    # Build 2D sheet
    # -----------------------------
    central_xy, edge_xy = build_semiperiodic_sheet(
        radius=radius, w_max=w_max, h_max=h_max, edge_offset=edge_offset
    )

    # -----------------------------
    # Build 3D two-sheet system
    # -----------------------------
    flat_cen, flat_ed, flat2_cen, flat2_ed = build_two_sheet_system(
        central_xy=central_xy,
        edge_xy=edge_xy,
        theta_deg=theta_deg,
        z_dist=z_dist,
    )

    # base two-sheet system
    p1_cen = np.vstack((flat_cen, flat2_cen))
    p1_ed = np.vstack((flat_ed, flat2_ed))

    # -----------------------------
    # Replicate neighbors in y
    # -----------------------------
    mirrored_cen = replicate_in_y(p1_cen, y_offsets)
    mirrored_ed = replicate_in_y(p1_ed, y_offsets)

    # -----------------------------
    # Filter primary window
    # -----------------------------
    p1_cen_filtered = filter_primary_y(p1_cen, y_min=y_min_1, y_max=y_max_1)
    p1_ed_filtered = filter_primary_y(p1_ed, y_min=y_min_1, y_max=y_max_1)

    # -----------------------------
    # Build pairwise distance dicts
    # -----------------------------
    cen_dis, ed_cen_dis, cen_ed_dis, ed_dis = build_pairwise_distance_dicts(
        z_spacings=z_S,
        p1_cen_filtered=p1_cen_filtered,
        p1_ed_filtered=p1_ed_filtered,
        mirrored_cen=mirrored_cen,
        mirrored_ed=mirrored_ed,
        cutoff=cutoff,
        verbose=True,
    )

    # -----------------------------
    # Save outputs
    # -----------------------------
    os.makedirs("outputs/pairwise", exist_ok=True)

    out_cc = "outputs/pairwise/PairDis_CC.npy"
    out_ec = "outputs/pairwise/PairDis_EC.npy"
    out_ce = "outputs/pairwise/PairDis_CE.npy"
    out_ee = "outputs/pairwise/PairDis_EE.npy"

    save_distance_dict(out_cc, cen_dis)
    save_distance_dict(out_ec, ed_cen_dis)
    save_distance_dict(out_ce, cen_ed_dis)
    save_distance_dict(out_ee, ed_dis)

    # -----------------------------
    # Sanity check keys
    # -----------------------------
    loaded = np.load(out_cc, allow_pickle=True).item()
    missing_in_dict, missing_in_out = check_keys_match(z_S, loaded)
    print("Missing in dict:", missing_in_dict)
    print("Extra keys in dict:", missing_in_out)

    print("Done. Saved:")
    print(" ", out_cc)
    print(" ", out_ec)
    print(" ", out_ce)
    print(" ", out_ee)


if __name__ == "__main__":
    main()
