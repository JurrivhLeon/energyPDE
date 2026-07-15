"""Evaluate physical-space FNO checkpoints on 1D Euler data."""

from __future__ import annotations

import argparse
import json
import os
from argparse import Namespace

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from ..fno import FNO1D
    from ..heat_data import load_dataset_splits
    from .common import CHANNEL_NAMES, STATE_CHANNELS, resolve_device, safe_torch_load
    from .euler_data import build_euler1d_step_dataset, build_euler1d_trajectory_dataset_from_split
    from .eval import (
        _add_rollout_max_metrics,
        _add_rollout_std_metrics,
        _parse_snapshot_times,
        _plot_curve,
        _plot_samples,
        _save_curve_csv,
        _save_per_sample_errors_json,
        _truncate_split,
        evaluate_rollout,
        evaluate_step,
    )
except ImportError:
    from grad_flow_l2.fno import FNO1D
    from grad_flow_l2.heat_data import load_dataset_splits
    from grad_flow_l2.euler1d.common import CHANNEL_NAMES, STATE_CHANNELS, resolve_device, safe_torch_load
    from grad_flow_l2.euler1d.euler_data import build_euler1d_step_dataset, build_euler1d_trajectory_dataset_from_split
    from grad_flow_l2.euler1d.eval import (
        _add_rollout_max_metrics,
        _add_rollout_std_metrics,
        _parse_snapshot_times,
        _plot_curve,
        _plot_samples,
        _save_curve_csv,
        _save_per_sample_errors_json,
        _truncate_split,
        evaluate_rollout,
        evaluate_step,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Euler1D physical FNO checkpoint")
    p.add_argument("--dataset-path", type=str, required=True)
    p.add_argument("--checkpoint-path", type=str, required=True)
    p.add_argument("--args-json", type=str, default=None)
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--output-dir", type=str, default="grad_flow_l2/euler1d/outputs_fno/eval")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--n-plot-samples", type=int, default=4)
    p.add_argument("--snapshot-times", type=str, default="")
    p.add_argument("--max-snapshots", type=int, default=None)
    p.add_argument("--delta-clip", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--cpu", action="store_true")

    # Fallback architecture args used only when args.json is unavailable.
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--fno-layers", type=int, default=6)
    p.add_argument("--fno-modes", type=int, default=64)
    p.add_argument("--disable-fno-grid", action="store_true")
    p.add_argument("--use-dt-channel", action="store_true")
    p.add_argument("--disable-forcing-channel", action="store_true")
    p.add_argument("--no-residual", action="store_true")
    p.add_argument("--lift-noise-std", type=float, default=0.0, help="Fallback FNO lift-noise std when args.json is unavailable; inactive during eval.")
    p.add_argument("--lift-noise-corr-length", type=float, default=1.0, help="Fallback Matern correlation length for FNO lift noise.")
    p.add_argument("--lift-noise-decay-s", type=float, default=2.0, help="Fallback spectral decay exponent s for FNO lift noise.")
    return p.parse_args()


def _load_train_args(args):
    args_json = args.args_json or os.path.join(os.path.dirname(args.checkpoint_path), "args.json")
    if not os.path.exists(args_json):
        return Namespace(), args_json
    with open(args_json, "r", encoding="utf-8") as f:
        return Namespace(**json.load(f)), args_json


def _build_model(n_x: int, dt: float, args, train_args) -> FNO1D:
    return FNO1D(
        n_x=n_x,
        state_channels=STATE_CHANNELS,
        forcing_channels=1,
        width=int(getattr(train_args, "width", args.width)),
        n_layers=int(getattr(train_args, "fno_layers", args.fno_layers)),
        modes=int(getattr(train_args, "fno_modes", args.fno_modes)),
        use_forcing_channel=not bool(getattr(train_args, "disable_forcing_channel", args.disable_forcing_channel)),
        use_dt_channel=bool(getattr(train_args, "use_dt_channel", args.use_dt_channel)),
        use_grid_features=not bool(getattr(train_args, "disable_fno_grid", args.disable_fno_grid)),
        default_dt=dt,
        residual=not bool(getattr(train_args, "no_residual", args.no_residual)),
        lift_noise_std=float(getattr(train_args, "lift_noise_std", args.lift_noise_std)),
        lift_noise_corr_length=float(getattr(train_args, "lift_noise_corr_length", args.lift_noise_corr_length)),
        lift_noise_decay_s=float(getattr(train_args, "lift_noise_decay_s", args.lift_noise_decay_s)),
    )


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = resolve_device(cpu=args.cpu, device=args.device)
    train_args, args_json = _load_train_args(args)
    splits = load_dataset_splits(args.dataset_path, map_location="cpu")
    split, eval_snapshots = _truncate_split(splits[args.split], args.max_snapshots)
    if int(split["u0"].shape[0]) == 0:
        raise ValueError(f"Split {args.split!r} is empty")
    meta = splits.get("meta", {})
    n_x = int(split["u0"].shape[-1])
    original_steps = int(splits[args.split]["u_traj"].shape[1] - 1)
    original_t_final = float(meta.get("t_final", original_steps))
    dt = float(meta.get("dataset_dt", original_t_final / float(original_steps)))
    n_steps = int(split["u_traj"].shape[1] - 1)
    t_final = dt * n_steps
    domain_length = float(meta.get("domain_length", getattr(train_args, "domain_length", 1.0)))
    h = domain_length / float(n_x)
    model = _build_model(n_x=n_x, dt=dt, args=args, train_args=train_args).to(device)
    ckpt = safe_torch_load(args.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    channel_weights = getattr(train_args, "channel_weights_used", None) or ckpt.get("channel_weights")
    if channel_weights is None:
        channel_weights = torch.ones(STATE_CHANNELS)
    delta_clip = ckpt.get("rollout_delta_clip", None) if args.delta_clip is None else args.delta_clip
    delta_clip = None if delta_clip is None or float(delta_clip) <= 0 else float(delta_clip)
    step_loader = DataLoader(build_euler1d_step_dataset(split), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    traj_loader = DataLoader(build_euler1d_trajectory_dataset_from_split(split), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    step_metrics = evaluate_step(model, step_loader, device, dt, channel_weights)
    curves = evaluate_rollout(model, traj_loader, device, dt, h, domain_length, delta_clip)
    metrics = dict(step_metrics)
    for c, name in enumerate(CHANNEL_NAMES):
        metrics[f"rollout_rel_l2_{name}"] = float(curves["rollout_rel_l2_channels"][c])
        metrics[f"rollout_rel_h1_{name}"] = float(curves["rollout_rel_h1_channels"][c])
    metrics["rollout_rel_l2"] = curves["rollout_rel_mean"]
    metrics["rollout_rel_l2_median"] = curves["rollout_rel_median"]
    metrics["rollout_rel_h1"] = curves["rollout_rel_h1_mean"]
    metrics["rollout_rel_h1_median"] = curves["rollout_rel_h1_median"]
    _add_rollout_std_metrics(metrics, curves)
    _add_rollout_max_metrics(metrics, curves)
    print(f"Device: {device}")
    print(f"Split={args.split}, n={int(split['u0'].shape[0])}, n_x={n_x}, steps={n_steps}, dt={dt}, L={domain_length}, delta_clip={delta_clip}")
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.8e}")
    _save_curve_csv(curves, dt, os.path.join(args.output_dir, f"{args.split}_rollout_error_curve.csv"))
    _save_per_sample_errors_json(curves, os.path.join(args.output_dir, f"{args.split}_per_sample_errors.json"))
    _plot_curve(curves, dt, os.path.join(args.output_dir, f"{args.split}_rollout_error_curve.png"))
    _plot_samples(
        model,
        split,
        device,
        dt,
        t_final,
        domain_length,
        _parse_snapshot_times(args.snapshot_times, t_final),
        args.n_plot_samples,
        os.path.join(args.output_dir, f"{args.split}_sample_comparisons"),
        delta_clip,
    )
    summary = {
        "dataset_path": args.dataset_path,
        "checkpoint_path": args.checkpoint_path,
        "args_json": args_json,
        "split": args.split,
        "n_x": n_x,
        "n_steps": n_steps,
        "dt": dt,
        "error_time_values": (
            (np.arange(curves["rel_curve_mean"].shape[0]) + 1) * float(dt)
        ).tolist(),
        "domain_length": domain_length,
        "delta_clip": delta_clip,
        "metrics": metrics,
        "overall_rel_l2": curves["overall_rel_l2"],
        "overall_rel_h1": curves["overall_rel_h1"],
        "overall_rel_l2_global_components": curves["overall_rel_l2_global_components"],
        "overall_rel_h1_global_components": curves["overall_rel_h1_global_components"],
        "rel_l2_curve_mean": curves["rel_curve_mean"].tolist(),
        "rel_h1_curve_mean": curves["rel_h1_curve_mean"].tolist(),
        "channel_names": CHANNEL_NAMES,
        "field_names": CHANNEL_NAMES,
        "meta": meta,
    }
    with open(os.path.join(args.output_dir, f"{args.split}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main(parse_args())
