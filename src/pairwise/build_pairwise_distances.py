import numpy as np
from scipy.spatial import distance


def _as_1d_distances(dist_matrix: np.ndarray, cutoff: float) -> np.ndarray:
    """
    Flatten all pair distances <= cutoff into a 1D array (same style as your original code).
    """
    return dist_matrix[dist_matrix <= cutoff]


def build_pairwise_distance_dicts(
    z_spacings: np.ndarray,
    p1_cen_filtered: np.ndarray,
    p1_ed_filtered: np.ndarray,
    mirrored_cen: np.ndarray,
    mirrored_ed: np.ndarray,
    cutoff: float = 25.0,
    round_key: int = 5,
    verbose: bool = True,
):
    """
    Reproduce your storage format:
      dict[round(z, round_key)] -> 1D np.ndarray of kept distances

    Returns:
      cen_dis (CC), ed_cen_dis (EC), cen_ed_dis (CE), ed_dis (EE)
    """
    cen_dis = {}      # CC
    ed_cen_dis = {}   # EC
    cen_ed_dis = {}   # CE
    ed_dis = {}       # EE

    for i, spacing in enumerate(z_spacings):
        key = round(float(spacing), round_key)

        # shift mirrored images in z for this spacing
        cen_m = mirrored_cen.copy()
        cen_m[:, 2] += spacing

        ed_m = mirrored_ed.copy()
        ed_m[:, 2] += spacing

        # CC: central vs central
        dist_cc = distance.cdist(p1_cen_filtered, cen_m)
        final_cc = _as_1d_distances(dist_cc, cutoff)
        cen_dis[key] = final_cc

        # CE: central vs edge
        dist_ce = distance.cdist(p1_cen_filtered, ed_m)
        final_ce = _as_1d_distances(dist_ce, cutoff)
        cen_ed_dis[key] = final_ce

        # EC: edge vs central
        dist_ec = distance.cdist(p1_ed_filtered, cen_m)
        final_ec = _as_1d_distances(dist_ec, cutoff)
        ed_cen_dis[key] = final_ec

        # EE: edge vs edge
        dist_ee = distance.cdist(p1_ed_filtered, ed_m)
        final_ee = _as_1d_distances(dist_ee, cutoff)
        ed_dis[key] = final_ee

        if verbose:
            print(f"[{i:03d}] z={spacing:.5f}  CC={len(final_cc)}  CE={len(final_ce)}  EC={len(final_ec)}  EE={len(final_ee)}")

    return cen_dis, ed_cen_dis, cen_ed_dis, ed_dis


def save_distance_dict(path: str, d: dict):
    """
    Save dict-of-arrays in .npy (pickle) format, like your original workflow.
    """
    np.save(path, d, allow_pickle=True)


def check_keys_match(z_spacings: np.ndarray, loaded_dict: dict, round_key: int = 5):
    """
    Sanity check: make sure every z in z_spacings has a corresponding key in dict, and vice versa.
    """
    z_out = {round(float(x), round_key) for x in z_spacings}
    z_npy = {round(float(k), round_key) for k in loaded_dict.keys()}
    missing_in_dict = z_out - z_npy
    missing_in_out = z_npy - z_out
    return missing_in_dict, missing_in_out
