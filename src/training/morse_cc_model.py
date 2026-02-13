import torch
import torch.nn as nn


class MorseCCModel(nn.Module):
    """
    Your CC-only Morse training, structured and reusable.

    Assumes:
    - processed_data[dkey]["center_center"] is list[tensor] per d-spacing
    - processed_data[dkey]["actual"] is tensor energies per d-spacing
    """
    def __init__(self, device: torch.device):
        super().__init__()

        self.morse_cc = nn.Parameter(
            torch.tensor([-89.33799, 0.71781504, -6.6749525], device=device)
        )

    @staticmethod
    def morse_potential(r, D_e, a, r_e):
        delta = a * (r - r_e)
        exp_term = torch.exp(-delta)
        return D_e * (1.0 - exp_term) ** 2 - D_e

    def forward(self, processed_data: dict) -> torch.Tensor:
        total_loss = 0.0

        for dkey in processed_data.keys():
            dist_cc_list = processed_data[dkey]["center_center"]   # list of tensors
            actual = processed_data[dkey]["actual"]                # tensor (K,)

            morse_cc_list = []
            for dist_tensor in dist_cc_list:
                morse_vals = self.morse_potential(dist_tensor, *self.morse_cc)
                morse_cc_list.append(morse_vals.sum())

            morse_cc_tensor = torch.stack(morse_cc_list)           # (K,)
            predicted = 2.0 * morse_cc_tensor                       # 

            #weighting
            w = 1.0 / (torch.abs(actual) + 1.0)
            mse_loss = torch.sum(w * (predicted - actual) ** 2)

            total_loss = total_loss + mse_loss

        return total_loss
