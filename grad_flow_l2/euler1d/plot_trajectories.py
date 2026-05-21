"""Plot 1D Euler primitive-channel trajectory heatmaps."""

from __future__ import annotations

import argparse
import os

import torch

try:
    from ..heat_data import load_dataset_splits
except ImportError:
    from grad_flow_l2.heat_data import load_dataset_splits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot Euler1D trajectory heatmaps")
    p.add_argument("--dataset-path", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="grad_flow_l2/euler1d/plots")
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--dpi", type=int, default=160)
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    splits = load_dataset_splits(args.dataset_path, map_location="cpu")
    split = splits[args.split]
    traj = split["u_traj"]
    meta = splits.get("meta", {})
    n_samples = min(int(args.n_samples), int(traj.shape[0]))
    if n_samples <= 0:
        raise ValueError(f"Split {args.split!r} is empty")

    n_steps = int(traj.shape[1] - 1)
    n_x = int(traj.shape[-1])
    t_final = float(meta.get("t_final", n_steps))
    domain_length = float(meta.get("domain_length", 1.0))
    extent = [0.0, domain_length, 0.0, t_final]
    names = list(meta.get("state_names", ["rho", "u", "p"]))
    os.makedirs(args.output_dir, exist_ok=True)

    for sample_id in range(n_samples):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
        for c, ax in enumerate(axes):
            field = traj[sample_id, :, c, :].numpy()
            cmap = "coolwarm" if names[c] == "u" else "viridis"
            im = ax.imshow(field, origin="lower", aspect="auto", extent=extent, cmap=cmap)
            ax.set_title(names[c])
            ax.set_xlabel("x")
            if c == 0:
                ax.set_ylabel("t")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"Euler1D {args.split} sample {sample_id} | {n_steps} steps, n_x={n_x}")
        out_path = os.path.join(args.output_dir, f"euler1d_{args.split}_sample_{sample_id:02d}_heatmaps.png")
        fig.savefig(out_path, dpi=int(args.dpi))
        plt.close(fig)
        print(out_path)


if __name__ == "__main__":
    main(parse_args())
