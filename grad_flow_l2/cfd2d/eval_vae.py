"""Evaluation entrypoint for periodic 2D compressible NS latent VAE checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from argparse import Namespace

import torch
from torch.utils.data import DataLoader

try:
    from ..cfd2d.cfd_data import build_cfd2d_step_dataset, build_cfd2d_trajectory_dataset_from_split
    from ..cfd2d.eval import (
        _evaluate_rollout_curves,
        _parse_snapshot_times,
        _plot_curve,
        _plot_samples,
        _save_curve_csv,
        _torch_load_checkpoint,
    )
    from ..heat_data import load_dataset_splits
    from ..ns2d_per.train_vae import PeriodicLatentVAETrainer2D
    from .train_vae import _build_model
except ImportError:
    from grad_flow_l2.cfd2d.cfd_data import build_cfd2d_step_dataset, build_cfd2d_trajectory_dataset_from_split
    from grad_flow_l2.cfd2d.eval import (
        _evaluate_rollout_curves,
        _parse_snapshot_times,
        _plot_curve,
        _plot_samples,
        _save_curve_csv,
        _torch_load_checkpoint,
    )
    from grad_flow_l2.heat_data import load_dataset_splits
    from grad_flow_l2.ns2d_per.train_vae import PeriodicLatentVAETrainer2D
    from grad_flow_l2.cfd2d.train_vae import _build_model


# Primitive variable names: rho, vx, vy, p
CHANNEL_NAMES = ["rho", "vx", "vy", "p"]

# vx and vy are zero-mean → symmetric diverging colormap
# rho and p are positive-definite → sequential colormap with data-range limits
_CHANNEL_CMAP     = {"rho": "viridis", "vx": "RdBu_r", "vy": "RdBu_r", "p": "plasma"}
_CHANNEL_SYMMETRIC = {"rho": False,    "vx": True,      "vy": True,      "p": False}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate periodic 2D compressible NS latent VAE checkpoint")
    parser.add_argument("--dataset-path",   type=str, required=True)
    parser.add_argument("--checkpoint",     type=str, required=True)
    parser.add_argument("--split",          type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-dir",     type=str, default="grad_flow_l2/cfd2d/outputs_vae/eval")
    parser.add_argument("--batch-size",     type=int, default=64)
    parser.add_argument("--num-workers",    type=int, default=0)
    parser.add_argument("--n-plot-samples", type=int,   default=4)
    parser.add_argument("--snapshot-times", type=str,   default="")
    parser.add_argument("--delta-clip",     type=float, default=1.0,
                        help="Clip predicted increment per step. Set 0 to disable.")
    parser.add_argument("--cpu",            action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    device  = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.dirname(args.checkpoint)
    with open(os.path.join(run_dir, "args.json"), "r", encoding="utf-8") as f:
        train_args = Namespace(**json.load(f))

    splits = load_dataset_splits(args.dataset_path, map_location="cpu")
    split  = splits[args.split]
    meta   = splits.get("meta", {})
    n_x    = int(split["u0"].shape[-2])
    n_y    = int(split["u0"].shape[-1])
    n_steps = int(split["u_traj"].shape[1] - 1)
    t_final = float(meta.get("t_final", float(n_steps)))
    dt      = t_final / float(n_steps)
    area    = 1.0 / float(n_x * n_y)

    model = _build_model(n_x=n_x, n_y=n_y, dt=dt, args=train_args).to(device)
    checkpoint = _torch_load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    step_loader = DataLoader(build_cfd2d_step_dataset(split),
                             batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    traj_loader = DataLoader(build_cfd2d_trajectory_dataset_from_split(split),
                             batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    trainer = PeriodicLatentVAETrainer2D(
        model=model, dt=dt,
        h_x=1.0 / float(n_x), h_y=1.0 / float(n_y),
        beta_kl=getattr(train_args, "beta_kl", checkpoint.get("beta_kl", 1e-4)),
        lambda_rec=getattr(train_args, "lambda_rec", checkpoint.get("lambda_rec", 1.0)),
        device=device, output_dir=None, show_epoch_pbar=False,
    )
    delta_clip = float(args.delta_clip) if float(args.delta_clip) > 0.0 else None
    trainer.delta_clip = delta_clip

    metrics = trainer.validate(step_loader, traj_loader=None)
    curves  = _evaluate_rollout_curves(model, traj_loader, device=device, dt=dt, area=area,
                                        delta_clip=delta_clip)
    metrics["rollout_rel_l2"]        = curves["rollout_rel_mean"]
    metrics["rollout_rel_l2_median"] = curves["rollout_rel_median"]
    for c, name in enumerate(CHANNEL_NAMES):
        metrics[f"rollout_rel_l2_{name}"] = float(curves["rel_curve_mean"][:, c].mean())

    print(f"Device: {device}")
    print(f"Split: {args.split}, n={int(split['u0'].shape[0])}, grid=({n_x},{n_y}), steps={n_steps}, dt={dt:.6f}")
    print(f"delta_clip: {delta_clip}")
    print("Step metrics:", {k: v for k, v in metrics.items() if "rollout" not in k})
    print("Rollout rel L2 per channel:")
    for c, name in enumerate(CHANNEL_NAMES):
        print(f"  {name}: {metrics[f'rollout_rel_l2_{name}']:.4f}")
    print(f"  mean:  {metrics['rollout_rel_l2']:.4f}")
    _save_curve_csv(curves, dt, os.path.join(args.output_dir, f"{args.split}_rollout_error_curve.csv"))
    _plot_curve(curves, dt, os.path.join(args.output_dir, f"{args.split}_rollout_error_curve.png"))
    _plot_samples(model, split, device, dt, t_final,
                  _parse_snapshot_times(args.snapshot_times, t_final),
                  args.n_plot_samples,
                  os.path.join(args.output_dir, f"{args.split}_sample_comparisons"),
                  delta_clip=delta_clip)


if __name__ == "__main__":
    main(parse_args())
