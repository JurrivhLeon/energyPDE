"""Evaluate physical-space FNO checkpoints on 2D compressible NS data."""

from __future__ import annotations

import argparse
import json
import os
from argparse import Namespace

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from ..cfd2d.cfd_data import STATE_CHANNELS, build_cfd2d_step_dataset, build_cfd2d_trajectory_dataset_from_split
    from ..fno import FNO2D
    from ..heat_data import load_dataset_splits
    from ..latent_markov_trainer_mc import AverageMeter, channel_weighted_mse, rollout_latent_markov_2d
    from .eval import (
        CHANNEL_NAMES,
        FIELD_NAMES,
        _add_rollout_max_metrics,
        _add_rollout_std_metrics,
        _channel_weights_from_args_or_checkpoint,
        _evaluate_rollout_curves,
        _parse_snapshot_times,
        _plot_curve,
        _plot_samples,
        _resolve_delta_clip,
        _save_curve_csv,
        _save_per_sample_errors_json,
        _torch_load_checkpoint,
        _truncate_split_for_eval,
    )
except ImportError:
    from grad_flow_l2.cfd2d.cfd_data import STATE_CHANNELS, build_cfd2d_step_dataset, build_cfd2d_trajectory_dataset_from_split
    from grad_flow_l2.fno import FNO2D
    from grad_flow_l2.heat_data import load_dataset_splits
    from grad_flow_l2.latent_markov_trainer_mc import AverageMeter, channel_weighted_mse, rollout_latent_markov_2d
    from grad_flow_l2.cfd2d.eval import (
        CHANNEL_NAMES,
        FIELD_NAMES,
        _add_rollout_max_metrics,
        _add_rollout_std_metrics,
        _channel_weights_from_args_or_checkpoint,
        _evaluate_rollout_curves,
        _parse_snapshot_times,
        _plot_curve,
        _plot_samples,
        _resolve_delta_clip,
        _save_curve_csv,
        _save_per_sample_errors_json,
        _torch_load_checkpoint,
        _truncate_split_for_eval,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate periodic 2D compressible NS physical FNO checkpoint")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=str, default="grad_flow_l2/cfd2d/outputs_fno/eval")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-plot-samples", type=int, default=4)
    parser.add_argument("--snapshot-times", type=str, default="")
    parser.add_argument("--max-snapshots", type=int, default=76)
    parser.add_argument("--delta-clip", type=float, default=None)
    parser.add_argument("--cpu", action="store_true")

    # Fallback architecture args used only when args.json is unavailable.
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--fno-layers", type=int, default=6)
    parser.add_argument("--fno-modes-x", type=int, default=16)
    parser.add_argument("--fno-modes-y", type=int, default=16)
    parser.add_argument("--disable-fno-grid", action="store_true")
    parser.add_argument("--use-dt-channel", action="store_true")
    parser.add_argument("--disable-forcing-channel", action="store_true")
    parser.add_argument("--no-residual", action="store_true")
    parser.add_argument("--lift-noise-std", type=float, default=0.0, help="Fallback FNO lift-noise std when args.json is unavailable; inactive during eval.")
    parser.add_argument("--lift-noise-corr-length", type=float, default=1.0, help="Fallback Matern correlation length for FNO lift noise.")
    parser.add_argument("--lift-noise-decay-s", type=float, default=2.0, help="Fallback spectral decay exponent s for FNO lift noise.")
    return parser.parse_args()


def _build_model(n_x: int, n_y: int, dt: float, args, train_args) -> FNO2D:
    return FNO2D(
        n_x=n_x,
        n_y=n_y,
        state_channels=STATE_CHANNELS,
        forcing_channels=1,
        width=int(getattr(train_args, "width", args.width)),
        n_layers=int(getattr(train_args, "fno_layers", args.fno_layers)),
        modes_x=int(getattr(train_args, "fno_modes_x", args.fno_modes_x)),
        modes_y=int(getattr(train_args, "fno_modes_y", args.fno_modes_y)),
        use_forcing_channel=not bool(getattr(train_args, "disable_forcing_channel", args.disable_forcing_channel)),
        use_dt_channel=bool(getattr(train_args, "use_dt_channel", args.use_dt_channel)),
        use_grid_features=not bool(getattr(train_args, "disable_fno_grid", args.disable_fno_grid)),
        default_dt=dt,
        residual=not bool(getattr(train_args, "no_residual", args.no_residual)),
        lift_noise_std=float(getattr(train_args, "lift_noise_std", args.lift_noise_std)),
        lift_noise_corr_length=float(getattr(train_args, "lift_noise_corr_length", args.lift_noise_corr_length)),
        lift_noise_decay_s=float(getattr(train_args, "lift_noise_decay_s", args.lift_noise_decay_s)),
    )


@torch.no_grad()
def _evaluate_step(model, step_loader, device, dt, channel_weights):
    model.eval()
    meters = {k: AverageMeter() for k in ("val_loss", "val_loss_step")}
    weights = None if channel_weights is None else torch.as_tensor(channel_weights, dtype=torch.float32)
    for u_k, u_k1, f in step_loader:
        u_k, u_k1, f = u_k.to(device), u_k1.to(device), f.to(device)
        pred = model.predict_step(u_k, f, dt=dt)
        w = None if weights is None else weights.to(device=pred.device, dtype=pred.dtype)
        loss = channel_weighted_mse(pred, u_k1, w)
        bsz = int(u_k.shape[0])
        meters["val_loss"].update(loss.item(), bsz)
        meters["val_loss_step"].update(loss.item(), bsz)
    return {k: v.avg for k, v in meters.items()}


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.dirname(args.checkpoint_path)
    args_path = os.path.join(run_dir, "args.json")
    train_args = Namespace()
    if os.path.exists(args_path):
        with open(args_path, "r", encoding="utf-8") as f:
            train_args = Namespace(**json.load(f))
    splits = load_dataset_splits(args.dataset_path, map_location="cpu")
    split = splits[args.split]
    if int(split["u0"].shape[0]) == 0:
        raise ValueError(f"Requested split {args.split!r} is empty")
    meta = splits.get("meta", {})
    original_n_steps = int(split["u_traj"].shape[1] - 1)
    original_t_final = float(meta.get("t_final", float(original_n_steps)))
    dt = original_t_final / float(original_n_steps)
    split, eval_snapshots = _truncate_split_for_eval(split, args.dataset_path, args.max_snapshots)
    n_x = int(split["u0"].shape[-2])
    n_y = int(split["u0"].shape[-1])
    n_steps = int(split["u_traj"].shape[1] - 1)
    t_final = dt * float(n_steps)
    area = 1.0 / float(n_x * n_y)
    model = _build_model(n_x=n_x, n_y=n_y, dt=dt, args=args, train_args=train_args).to(device)
    checkpoint = _torch_load_checkpoint(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    step_loader = DataLoader(build_cfd2d_step_dataset(split), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    traj_loader = DataLoader(build_cfd2d_trajectory_dataset_from_split(split), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    channel_weights = _channel_weights_from_args_or_checkpoint(train_args, checkpoint)
    delta_clip = _resolve_delta_clip(args.delta_clip, checkpoint, default=10.0)
    metrics = _evaluate_step(model, step_loader, device=device, dt=dt, channel_weights=channel_weights)
    curves = _evaluate_rollout_curves(
        model,
        traj_loader,
        device=device,
        dt=dt,
        area=area,
        delta_clip=delta_clip,
        rollout_fn=rollout_latent_markov_2d,
    )
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
    _plot_samples(
        model,
        split,
        device,
        dt,
        t_final,
        _parse_snapshot_times(args.snapshot_times, t_final),
        args.n_plot_samples,
        os.path.join(args.output_dir, f"{args.split}_sample_comparisons"),
        delta_clip=delta_clip,
        rollout_fn=rollout_latent_markov_2d,
    )
    summary = {
        "dataset_path": args.dataset_path,
        "checkpoint_path": args.checkpoint_path,
        "split": args.split,
        "n_x": n_x,
        "n_y": n_y,
        "n_steps": n_steps,
        "dt": dt,
        "error_time_values": (
            (np.arange(curves["rel_curve_mean"].shape[0]) + 1) * float(dt)
        ).tolist(),
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
