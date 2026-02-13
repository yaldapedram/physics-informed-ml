import numpy as np
import torch


def load_pmf_out(file_path: str, device: torch.device):
    """
    Loads a two-column .out file: [d_spacing, energy]
    Sorts by d_spacing.

    Returns
    -------
    d_spacing : torch.Tensor shape (K,)
    energy    : torch.Tensor shape (K,)
    """
    arr = np.loadtxt(file_path, delimiter=" ")
    idx = np.argsort(arr[:, 0])
    arr = arr[idx]
    d = torch.tensor(arr[:, 0], dtype=torch.float32, device=device)
    e = torch.tensor(arr[:, 1], dtype=torch.float32, device=device)
    return d, e
