"""
Demo: CC energy model = 2 * (Morse_CC + GPR_CC)

Inputs
- Pairwise distances .npy: dict[d_spacing -> 1D array of pairwise distances]
- PMF/energy .out: two columns: d_spacing  energy

Example
python -m scripts.train_cc_gpr_demo \
  --pairwise data/sample/PairDis_cc.npy \
  --pmf data/sample/distance_cc.out \
  --epochs 200 \
  --optimizer lbfgs \
  --plot

Notes
- This is a demo script meant to be readable + portable (no absolute paths).
- It supports multiple datasets by passing multiple --pairwise/--pmf pairs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pairwise",
        nargs="+",
        required=True,
        help="One or more .npy files (dict: d->distances). Same count/order as --pmf.",
    )
    p.add_argument(
        "--pmf",
        nargs="+",
        required=True,
        help="One or more .out files (2 cols: d energy). Same count/order as --pairwise.",
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--num-s", type=int, default=25, help="Number of S-points for GPR basis.")
    p.add_argument("--s-max", type=float, default=25.0, help="Max S-point distance.")
    p.add_argument("--optimizer", choices=["lbfgs", "adam"], default="lbfgs")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--lbfgs-max-iter", type=int, default=20)
    p.add_argument("--lbfgs-history", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="", help="cuda / cpu. Default: auto.")
    p.add_argument("--plot", action="store_true", help="Plot training loss at end.")
    p.add_argument("--save-params", type=str, default="", help="Optional .pt path to save parameters.")
    return p.parse_args()


# ---------------------------
# Data utilities
# ---------------------------

@dataclass
class DatasetTensors:
    d_spacing: torch.Tensor                 # (M,)
    distances_cc: List[torch.Tensor]        # length M, each (Ni,)
    actual_energy: torch.Tensor             # (M,)


def _load_pairwise_dict(path: Path) -> Dict[float, np.ndarray]:
    data = np.load(str(path), allow_pickle=True).item()
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a dict. Got: {type(data)}")
    # Ensure numpy arrays
    out: Dict[float, np.ndarray] = {}
    for k, v in data.items():
        out[float(k)] = np.asarray(v, dtype=np.float32).ravel()
    return out


def load_dataset(pairwise_npy: Path, pmf_out: Path, device: torch.device) -> DatasetTensors:
    # Pairwise distances
    pair_dict = _load_pairwise_dict(pairwise_npy)
    d_sorted = sorted(pair_dict.keys())
    d_pair = torch.tensor(d_sorted, dtype=torch.float32, device=device)
    dist_list = [torch.tensor(pair_dict[d], dtype=torch.float32, device=device) for d in d_sorted]

    # PMF/energies
    arr = np.loadtxt(str(pmf_out), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"{pmf_out} must be 2 columns: d energy")
    arr = arr[np.argsort(arr[:, 0])]
    d_energy = torch.tensor(arr[:, 0], dtype=torch.float32, device=device)
    e_energy = torch.tensor(arr[:, 1], dtype=torch.float32, device=device)

    # Align lengths by truncating to min length (simple + robust for demo)
    m = min(d_pair.shape[0], d_energy.shape[0])
    if d_pair.shape[0] != d_energy.shape[0]:
        print(f"[warn] length mismatch for {pairwise_npy.name} vs {pmf_out.name} -> truncating to {m}")

    d_pair = d_pair[:m]
    dist_list = dist_list[:m]
    d_energy = d_energy[:m]
    e_energy = e_energy[:m]

    # Sanity: warn if d-spacings are not close (but continue)
    if not torch.allclose(d_pair, d_energy, rtol=1e-4, atol=1e-4):
        max_abs = torch.max(torch.abs(d_pair - d_energy)).item()
        print(f"[warn] d-spacing values differ (max abs diff ~ {max_abs:.3e}). Continuing for demo.")

    return DatasetTensors(d_spacing=d_pair, distances_cc=dist_list, actual_energy=e_energy)


# ---------------------------
# Model
# ---------------------------

def morse_potential(r: torch.Tensor, D_e: torch.Tensor, a: torch.Tensor, r_e: torch.Tensor) -> torch.Tensor:
    delta = a * (r - r_e)
    exp_term = torch.exp(-delta)
    return D_e * (1.0 - exp_term) ** 2 - D_e


def gaussian_kernel_cutoff(distances: torch.Tensor, s_points: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
    """
    distances: (N,)
    s_points : (S,)
    returns  : (N, S)
    Kernel: exp(- (d-s)^2 / (2 l^2)) - exp(- (3l)^2 / (2 l^2)) with hard cutoff |d-s|>3l -> 0
    """
    l = torch.clamp(l, min=1e-3)
    diff2 = (distances.unsqueeze(1) - s_points.unsqueeze(0)) ** 2
    kernel = torch.exp(-diff2 / (2.0 * l**2))

    cutoff_value = torch.exp(-((3.0 * l) ** 2) / (2.0 * l**2))
    kernel = kernel - cutoff_value

    mask = torch.abs(distances.unsqueeze(1) - s_points.unsqueeze(0)) > 3.0 * l
    kernel = torch.where(mask, torch.zeros((), device=distances.device), kernel)
    return kernel


class GPRCCModel(nn.Module):
    """
    E_pred(d) = 2 * sum_{pairs} Morse(r) + 2 * sum_{pairs} (K(r,S;l) @ coeffs)
    """

    def __init__(
        self,
        s_points: torch.Tensor,
        morse_cc: torch.Tensor,
        l_init: float,
        coeffs_init: torch.Tensor,
    ):
        super().__init__()
        self.register_buffer("s_points", s_points)  # (S,)
        self.register_buffer("morse_cc", morse_cc)  # (3,) fixed in this demo

        self.l_cc = nn.Parameter(torch.tensor(float(l_init), device=s_points.device))
        self.coeffs_cc = nn.Parameter(coeffs_init.clone().to(s_points.device))

    def forward(self, datasets: List[DatasetTensors]) -> torch.Tensor:
        total = torch.zeros((), device=self.l_cc.device)

        for ds in datasets:
            # Precompute Morse sum per spacing
            morse_sums = []
            gpr_sums = []

            for dist in ds.distances_cc:
                morse_sum = morse_potential(dist, self.morse_cc[0], self.morse_cc[1], self.morse_cc[2]).sum()
                K = gaussian_kernel_cutoff(dist, self.s_points, self.l_cc)  # (N,S)
                gpr_sum = (K @ self.coeffs_cc).sum()
                morse_sums.append(morse_sum)
                gpr_sums.append(gpr_sum)

            morse_sums_t = torch.stack(morse_sums)  # (M,)
            gpr_sums_t = torch.stack(gpr_sums)      # (M,)

            pred = 2.0 * morse_sums_t + 2.0 * gpr_sums_t

            # Weighted MSE (same idea as your code, but contained)
            w = 1.0 / (torch.abs(ds.actual_energy) + 1e-3)
            w = torch.clamp(w, min=1.0 / 300.0, max=1.0)
            loss = torch.mean(w * (pred - ds.actual_energy) ** 2)

            total = total + loss

        return total


# ---------------------------
# Training
# ---------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()

    if len(args.pairwise) != len(args.pmf):
        raise SystemExit("ERROR: --pairwise and --pmf must have the same number of files (paired).")

    if args.device.strip():
        device = torch.device(args.device.strip())
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    set_seed(args.seed)

    # Load datasets
    datasets: List[DatasetTensors] = []
    for p_path, e_path in zip(args.pairwise, args.pmf):
        ds = load_dataset(Path(p_path), Path(e_path), device=device)
        datasets.append(ds)
        print(f"Loaded: {Path(p_path).name} + {Path(e_path).name}  (M={len(ds.d_spacing)})")

    # S-points
    s_points = torch.linspace(0.0, float(args.s_max), int(args.num_s), device=device)

    # Fixed Morse params (demo)
    morse_cc = torch.tensor([-89.81815, 0.7167118, -6.1022387], dtype=torch.float32, device=device)

    # Initial params (use your values, but keep them together)
    l_init = 1.1372
    coeffs_init = torch.tensor(
        [
            4.1929865e00, 4.9637661e00, -2.1399536e00, 1.3515205e00,
            -4.7577569e-01, 2.9874527e-01, -2.8380176e-01, 3.0190304e-01,
            -3.5991228e-01, 4.1833746e-01, -3.6897525e-01, 2.2087468e-01,
            -8.4270284e-02, 5.7198666e-03, 2.9485710e-02, -4.2849217e-02,
            4.7438167e-02, -4.3131039e-02, 3.6231603e-02, -2.9810885e-02,
            2.1291079e-02, -1.2015426e-02, 3.8713149e-03, 3.7167082e-04,
            -8.3351095e-04
        ],
        dtype=torch.float32,
        device=device,
    )

    model = GPRCCModel(
        s_points=s_points,
        morse_cc=morse_cc,
        l_init=l_init,
        coeffs_init=coeffs_init,
    ).to(device)

    loss_hist: List[float] = []

    if args.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
        for epoch in range(1, args.epochs + 1):
            opt.zero_grad()
            loss = model(datasets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_val = float(loss.detach().cpu().item())
            loss_hist.append(loss_val)

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:4d}/{args.epochs}  loss={loss_val:.6e}  l_cc={model.l_cc.item():.6f}  ||c||={torch.norm(model.coeffs_cc).item():.4f}")

    else:  # lbfgs
        opt = torch.optim.LBFGS(
            model.parameters(),
            lr=float(args.lr),
            max_iter=int(args.lbfgs_max_iter),
            history_size=int(args.lbfgs_history),
        )

        def closure() -> torch.Tensor:
            opt.zero_grad()
            loss = model(datasets)
            loss.backward()
            return loss

        for epoch in range(1, args.epochs + 1):
            loss = opt.step(closure)
            loss_val = float(loss.detach().cpu().item())
            loss_hist.append(loss_val)

            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:4d}/{args.epochs}  loss={loss_val:.6e}  l_cc={model.l_cc.item():.6f}  ||c||={torch.norm(model.coeffs_cc).item():.4f}")

    # Save params (optional)
    if args.save_params.strip():
        out_path = Path(args.save_params)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "l_cc": model.l_cc.detach().cpu(),
                "coeffs_cc": model.coeffs_cc.detach().cpu(),
                "s_points": model.s_points.detach().cpu(),
                "morse_cc": model.morse_cc.detach().cpu(),
            },
            str(out_path),
        )
        print(f"Saved params -> {out_path}")

    # Plot (optional)
    if args.plot:
        plt.figure(figsize=(8, 5))
        plt.plot(loss_hist, marker="o", linestyle="-")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training loss ({args.optimizer})")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    print("\nFinal parameters:")
    print(f"l_cc: {model.l_cc.item():.8f}")
    print(f"coeffs_cc (first 10): {model.coeffs_cc.detach().cpu().numpy()[:10]}")
    print(f"coeffs_norm: {torch.norm(model.coeffs_cc).item():.6f}")


if __name__ == "__main__":
    main()
