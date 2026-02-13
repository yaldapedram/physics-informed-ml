import numpy as np
import torch


def load_pairwise_distances_dict_npy(file_path: str, device: torch.device):
    """
    Loads dictionary-based pairwise distances saved as .npy.
    Expected format: dict[d_spacing] = array_of_distances

    Returns
    -------
    d_spacing : torch.Tensor shape (K,)
    distances : list[torch.Tensor] length K, each tensor is (Ni,)
    """
    data = np.load(file_path, allow_pickle=True).item()
    if not isinstance(data, dict):
        raise ValueError(f"Loaded data is not a dictionary: {type(data)} from {file_path}")

    sorted_keys = sorted(data.keys())
    d_spacing = torch.tensor(sorted_keys, dtype=torch.float32, device=device)
    distances = [torch.tensor(data[k], dtype=torch.float32, device=device) for k in sorted_keys]
    return d_spacing, distances
