"""
Demo: train Morse CC parameters using precomputed pairwise distances + PMF energies.

This script is a cleaned, portable version of your original training snippet:
- no hard-coded /Volumes/... paths
- takes local repo files or CLI arguments
- keeps your model logic (including factor-2 and weighting)

Run example (repo-root):
python -m scripts.train_morse_cc_demo \
  --pairwise data/sample/PairDis_CC.npy \
  --pmf data/sample/distance_sample.out \
  --epochs 500 \
  --lr 1e-6
"""

import argparse
import torch
import matplotlib.pyplot as plt

from src.pairwise.io import load_pairwise_distances_dict_npy
from src.pmf.io import load_pmf_out
from src.training.morse_cc_model import MorseCCModel


def build_processed_data(pairwise_path: str, pmf_path: str, device: torch.device):
    processed_data = {}
    dataset_key = "dataset_1"
    processed_data[dataset_key] = {}

    d_spacing_npy, dist_cc = load_pairwise_distances_dict_npy(pairwise_path, device=device)
    d_spacing_out, energy = load_pmf_out(pmf_path, device=device)

    # Match lengths (your approach)
    len_out = d_spacing_out.shape[0]
    len_npy = d_spacing_npy.shape[0]
    if len_out != len_npy:
        min_size = min(len_out, len_npy)
        d_spacing_out = d_spacing_out[:min_size]
        energy = energy[:min_size]
        d_spacing_npy = d_spacing_npy[:min_size]
        dist_cc = dist_cc[:min_size]
        print(f"⚠ Truncated to match lengths: {min_size}")

    # Check spacing agreement (loose tolerance)
    if not torch.allclose(d_spacing_out, d_spacing_npy, rtol=1e-4, atol=1e-4):
        print("⚠ Warning: d-spacing mismatch even after truncation (check your keys / rounding).")

    processed_data[dataset_key]["d_spacing"] = d_spacing_npy
    processed_data[dataset_key]["center_center"] = dist_cc
    processed_data[dataset_key]["actual"] = energy

    return processed_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairwise", required=True, help="Path to PairDis_CC.npy (dict[d]=distances)")
    parser.add_argument("--pmf", required=True, help="Path to distance_*.out (two cols: d, E)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--clip", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processed_data = build_processed_data(args.pairwise, args.pmf, device=device)

    model = MorseCCModel(device=device).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_values = []

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        loss = model(processed_data)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        loss_values.append(float(loss.detach().cpu().numpy()))

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{args.epochs}, Loss: {loss_values[-1]:.6f}")

        if epoch % 50 == 0:
            print("\n🔹 Optimized morse_cc:", model.morse_cc.detach().cpu().numpy())

    # Plot loss
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, args.epochs + 1), loss_values, marker="o", linestyle="-")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss curve (Morse CC training demo)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n🚀 Final morse_cc:", model.morse_cc.detach().cpu().numpy())


if __name__ == "__main__":
    main()
