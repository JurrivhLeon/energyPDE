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
        _add_rollout_std_metrics,
        _add_rollout_max_metrics,
        FIELD_NAMES,
        _parse_snapshot_times,
        _plot_curve,
        _plot_samples,
        _save_curve_csv,
        _save_per_sample_errors_json,
        _torch_load_checkpoint,
        _channel_weights_from_args_or_checkpoint,
        _resolve_delta_clip,
        _truncate_split_for_eval,
    )
    from ..heat_data import load_dataset_splits
    from ..cfd2d.train_vae import LatentVAETrainer2D, _rollout_vae_mean
    from .train_vae import _build_model
except ImportError:
    from grad_flow_l2.cfd2d.cfd_data import build_cfd2d_step_dataset, build_cfd2d_trajectory_dataset_from_split
    from grad_flow_l2.cfd2d.eval import (
        _evaluate_rollout_curves,
        _add_rollout_std_metrics,
        _add_rollout_max_metrics,
        FIELD_NAMES,
        _parse_snapshot_times,
        _plot_curve,
        _plot_samples,
        _save_curve_csv,
        _save_per_sample_errors_json,
        _torch_load_checkpoint,
        _channel_weights_from_args_or_checkpoint,
        _resolve_delta_clip,
        _truncate_split_for_eval,
    )
    from grad_flow_l2.heat_data import load_dataset_splits
    from grad_flow_l2.cfd2d.train_vae import LatentVAETrainer2D, _rollout_vae_mean
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
    parser.add_argument("--max-snapshots",  type=int,   default=101,
                        help="Maximum trajectory snapshots to evaluate. Default: 51 for OOD datasets (T=50), otherwise all.")
    parser.add_argument("--delta-clip",     type=float, default=None,
                        help="Clip predicted increment per step. Default: checkpoint training value; set 0 to disable.")
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
    if int(split["u0"].shape[0]) == 0:
        raise ValueError(f"Requested split {args.split!r} is empty")
    meta   = splits.get("meta", {})
    original_n_steps = int(split["u_traj"].shape[1] - 1)
    original_t_final = float(meta.get("t_final", float(original_n_steps)))
    dt = original_t_final / float(original_n_steps)
    split, eval_snapshots = _truncate_split_for_eval(split, args.dataset_path, args.max_snapshots)
    n_x    = int(split["u0"].shape[-2])
    n_y    = int(split["u0"].shape[-1])
    n_steps = int(split["u_traj"].shape[1] - 1)
    t_final = dt * float(n_steps)
    area    = 1.0 / float(n_x * n_y)

    model = _build_model(n_x=n_x, n_y=n_y, dt=dt, args=train_args).to(device)
    checkpoint = _torch_load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    step_loader = DataLoader(build_cfd2d_step_dataset(split),
                             batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    traj_loader = DataLoader(build_cfd2d_trajectory_dataset_from_split(split),
                             batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    channel_weights = _channel_weights_from_args_or_checkpoint(train_args, checkpoint)
    trainer = LatentVAETrainer2D(
        model=model, dt=dt,
        h_x=1.0 / float(n_x), h_y=1.0 / float(n_y),
        beta_kl=getattr(train_args, "beta_kl", checkpoint.get("beta_kl", 1e-4)),
        lambda_rec=getattr(train_args, "lambda_rec", checkpoint.get("lambda_rec", 1.0)),
        channel_weights=channel_weights,
        alpha_min=getattr(train_args, "alpha_min", checkpoint.get("alpha_min", 1e-4)),
        alpha_max=getattr(train_args, "alpha_max", checkpoint.get("alpha_max", 0.5)),
        transition_noise_scale=getattr(train_args, "transition_noise_scale", checkpoint.get("transition_noise_scale", 1.0)),
        device=device, output_dir=None, show_epoch_pbar=False,
    )
    delta_clip = _resolve_delta_clip(args.delta_clip, checkpoint, default=1.0)
    trainer.delta_clip = delta_clip

    metrics = trainer.validate(step_loader, traj_loader=None)
    curves  = _evaluate_rollout_curves(model, traj_loader, device=device, dt=dt, area=area,
                                        delta_clip=delta_clip, rollout_fn=_rollout_vae_mean)
    metrics["rollout_rel_l2"] = curves["rollout_rel_mean"]
    metrics["rollout_rel_l2_median"] = curves["rollout_rel_median"]
    metrics["rollout_rel_h1"] = curves["rollout_rel_h1_mean"]
    metrics["rollout_rel_h1_median"] = curves["rollout_rel_h1_median"]
    for c, name in enumerate(CHANNEL_NAMES):
        metrics[f"rollout_rel_l2_{name}"] = float(curves["rollout_rel_l2_channels"][c])
        metrics[f"rollout_rel_h1_{name}"] = float(curves["rollout_rel_h1_channels"][c])
    for c, name in enumerate(FIELD_NAMES):
        metrics[f"rollout_rel_l2_{name}"] = float(curves["rollout_rel_l2_fields"][c])
        metrics[f"rollout_rel_h1_{name}"] = float(curves["rollout_rel_h1_fields"][c])
    _add_rollout_std_metrics(metrics, curves)
    _add_rollout_max_metrics(metrics, curves)

    print(f"Device: {device}")
    print(f"Split: {args.split}, n={int(split['u0'].shape[0])}, grid=({n_x},{n_y}), steps={n_steps}, dt={dt:.6f}")
    print(f"Evaluation snapshots: {eval_snapshots} / {original_n_steps + 1}")
    print(f"delta_clip: {delta_clip}")
    print("Step metrics:", {k: v for k, v in metrics.items() if "rollout" not in k})
    print("Rollout rel L2 per field:")
    for name in FIELD_NAMES:
        print(f"  {name}: {metrics[f'rollout_rel_l2_{name}']:.4f}")
    print(f"  mean:  {metrics['rollout_rel_l2']:.4f}")
    print(f"  components: vx={metrics['rollout_rel_l2_vx']:.4f}, vy={metrics['rollout_rel_l2_vy']:.4f}")
    print("Rollout rel H1 per field:")
    for name in FIELD_NAMES:
        print(f"  {name}: {metrics[f'rollout_rel_h1_{name}']:.4f}")
    print(f"  mean:  {metrics['rollout_rel_h1']:.4f}")
    print(f"  components: vx={metrics['rollout_rel_h1_vx']:.4f}, vy={metrics['rollout_rel_h1_vy']:.4f}")
    print(f"Overall relative L2 across time: {curves['overall_rel_l2']:.8e}")
    print(f"Overall relative H1 across time: {curves['overall_rel_h1']:.8e}")
    _save_curve_csv(curves, dt, os.path.join(args.output_dir, f"{args.split}_rollout_error_curve.csv"))
    _save_per_sample_errors_json(curves, os.path.join(args.output_dir, f"{args.split}_per_sample_errors.json"))
    _plot_curve(curves, dt, os.path.join(args.output_dir, f"{args.split}_rollout_error_curve.png"))
    _plot_samples(model, split, device, dt, t_final,
                  _parse_snapshot_times(args.snapshot_times, t_final),
                  args.n_plot_samples,
                  os.path.join(args.output_dir, f"{args.split}_sample_comparisons"),
                  delta_clip=delta_clip, rollout_fn=_rollout_vae_mean)

    summary = {
        "dataset_path": args.dataset_path,
        "checkpoint_path": args.checkpoint,
        "split": args.split,
        "n_x": n_x,
        "n_y": n_y,
        "n_steps": n_steps,
        "dt": dt,
        "evaluation_snapshots": eval_snapshots,
        "stored_snapshots": original_n_steps + 1,
        "delta_clip": delta_clip,
        "metrics": metrics,
        "overall_rel_l2": curves["overall_rel_l2"],
        "overall_rel_h1": curves["overall_rel_h1"],
        "overall_rel_l2_global_components": curves["overall_rel_l2_global_components"],
        "overall_rel_h1_global_components": curves["overall_rel_h1_global_components"],
        "rel_l2_curve_mean": curves["rel_curve_mean"].tolist(),
        "rel_h1_curve_mean": curves["rel_h1_curve_mean"].tolist(),
        "rel_l2_field_curve_mean": curves["rel_field_curve_mean"].tolist(),
        "rel_h1_field_curve_mean": curves["rel_h1_field_curve_mean"].tolist(),
        "channel_names": CHANNEL_NAMES,
        "field_names": FIELD_NAMES,
        "meta": meta,
    }
    summary_path = os.path.join(args.output_dir, f"{args.split}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved evaluation summary: {summary_path}")


if __name__ == "__main__":
    main(parse_args())
