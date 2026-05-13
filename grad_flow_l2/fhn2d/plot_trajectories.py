"""Plot reference periodic 2D FitzHugh-Nagumo trajectories from a saved dataset."""

from __future__ import annotations

import argparse
import os
from typing import Sequence

import numpy as np
import torch

try:
    from ..heat_data import load_dataset_splits
except ImportError:
    from grad_flow_l2.heat_data import load_dataset_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot FHN2D reference trajectory snapshots")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--n-snapshots", type=int, default=5)
    parser.add_argument("--snapshot-times", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="grad_flow_l2/fhn2d/plots")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _parse_snapshot_times(raw: str, t_final: float, n_snapshots: int) -> list[float]:
    if raw.strip():
        vals = [float(tok.strip()) for tok in raw.split(",") if tok.strip()]
    else:
        vals = np.linspace(0.0, float(t_final), int(n_snapshots)).tolist()
    if not vals:
        raise ValueError("At least one snapshot time is required")
    for val in vals:
        if val < 0.0 or val > float(t_final):
            raise ValueError(f"Snapshot time must be in [0,{t_final}], got {val}")
    return vals


def _snapshot_indices(times: np.ndarray, requested: Sequence[float]) -> list[int]:
    out = []
    for t in requested:
        idx = int(np.argmin(np.abs(times - float(t))))
        if idx not in out:
            out.append(idx)
    return out


def _plot_sample(
    sample_id: int,
    f: torch.Tensor,
    traj: torch.Tensor,
    times: np.ndarray,
    snapshot_indices: Sequence[int],
    output_dir: str,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    n_cols = len(snapshot_indices)
    fig, axes = plt.subplots(
        2,
        n_cols + 1,
        figsize=(3.0 * (n_cols + 1), 5.5),
        squeeze=False,
        constrained_layout=True,
    )

    force = f[sample_id].cpu().numpy()
    im_f = axes[0, 0].imshow(force, origin="lower", cmap="coolwarm", extent=[0.0, 1.0, 0.0, 1.0])
    axes[0, 0].set_title("I(x,y)")
    axes[1, 0].axis("off")

    u_scale = max(float(traj[sample_id, :, 0].abs().max().item()), 1e-8)
    v_scale = max(float(traj[sample_id, :, 1].abs().max().item()), 1e-8)
    im_u = None
    im_v = None
    for col, idx in enumerate(snapshot_indices, start=1):
        t = float(times[idx])
        im_u = axes[0, col].imshow(
            traj[sample_id, idx, 0].cpu().numpy(),
            origin="lower",
            cmap="coolwarm",
            vmin=-u_scale,
            vmax=u_scale,
            extent=[0.0, 1.0, 0.0, 1.0],
        )
        axes[0, col].set_title(f"u, t={t:g}")
        im_v = axes[1, col].imshow(
            traj[sample_id, idx, 1].cpu().numpy(),
            origin="lower",
            cmap="coolwarm",
            vmin=-v_scale,
            vmax=v_scale,
            extent=[0.0, 1.0, 0.0, 1.0],
        )
        axes[1, col].set_title(f"v, t={t:g}")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im_f, ax=[axes[0, 0]], fraction=0.046, pad=0.02)
    if im_u is not None:
        fig.colorbar(im_u, ax=axes[0, 1:], fraction=0.015, pad=0.01)
    if im_v is not None:
        fig.colorbar(im_v, ax=axes[1, 1:], fraction=0.015, pad=0.01)
    out_path = os.path.join(output_dir, f"sample_{sample_id:04d}_snapshots.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_sample_grid(
    traj: torch.Tensor,
    channel: int,
    channel_name: str,
    times: np.ndarray,
    snapshot_indices: Sequence[int],
    sample_ids: Sequence[int],
    output_dir: str,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    n_rows = len(sample_ids)
    n_cols = len(snapshot_indices)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.1 * n_rows), squeeze=False, constrained_layout=True)
    scale = max(float(traj[sample_ids, :, channel].abs().max().item()), 1e-8)
    last_im = None
    for row, sample_id in enumerate(sample_ids):
        for col, idx in enumerate(snapshot_indices):
            last_im = axes[row, col].imshow(
                traj[sample_id, idx, channel].cpu().numpy(),
                origin="lower",
                cmap="coolwarm",
                vmin=-scale,
                vmax=scale,
            )
            if row == 0:
                axes[row, col].set_title(f"t={float(times[idx]):g}")
            if col == 0:
                axes[row, col].set_ylabel(f"sample {sample_id}")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
    out_path = os.path.join(output_dir, f"grid_{channel_name}_snapshots.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {out_path}")


def main(args: argparse.Namespace) -> None:
    splits = load_dataset_splits(args.dataset_path, map_location="cpu")
    split = splits[args.split]
    meta = splits.get("meta", {})
    traj = split["u_traj"]
    f = split["f"]
    n_steps = int(traj.shape[1] - 1)
    t_final = float(meta.get("t_final", float(n_steps)))
    times = np.linspace(0.0, t_final, n_steps + 1)
    snapshot_times = _parse_snapshot_times(args.snapshot_times, t_final=t_final, n_snapshots=args.n_snapshots)
    snapshot_indices = _snapshot_indices(times, snapshot_times)
    total = int(traj.shape[0])
    sample_ids = torch.linspace(0, total - 1, min(max(1, args.n_samples), total)).long().tolist()

    os.makedirs(args.output_dir, exist_ok=True)
    print(
        f"Dataset={args.dataset_path}, split={args.split}, samples={total}, "
        f"shape={tuple(traj.shape)}, t_final={t_final:g}, snapshots={[float(times[i]) for i in snapshot_indices]}"
    )
    for sample_id in sample_ids:
        _plot_sample(sample_id, f=f, traj=traj, times=times, snapshot_indices=snapshot_indices, output_dir=args.output_dir, dpi=args.dpi)
    _plot_sample_grid(traj, 0, "u", times, snapshot_indices, sample_ids, args.output_dir, args.dpi)
    _plot_sample_grid(traj, 1, "v", times, snapshot_indices, sample_ids, args.output_dir, args.dpi)


if __name__ == "__main__":
    main(parse_args())
