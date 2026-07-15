"""Evaluate probabilistic latent VAE checkpoints on 1D Euler data."""

from __future__ import annotations

import argparse
import csv
import json
import os
from argparse import Namespace
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from ..heat_data import load_dataset_splits
    from .common import (
        CHANNEL_NAMES,
        STATE_CHANNELS,
        channel_weighted_mse,
        relative_l2_error_1d,
        resolve_device,
        rollout_vae_mean_1d,
        safe_torch_load,
    )
    from .euler_data import (
        build_euler1d_step_dataset,
        build_euler1d_trajectory_dataset_from_split,
    )
    from .train_vae import _build_model
except ImportError:
    from grad_flow_l2.heat_data import load_dataset_splits
    from grad_flow_l2.euler1d.common import (
        CHANNEL_NAMES,
        STATE_CHANNELS,
        channel_weighted_mse,
        relative_l2_error_1d,
        resolve_device,
        rollout_vae_mean_1d,
        safe_torch_load,
    )
    from grad_flow_l2.euler1d.euler_data import (
        build_euler1d_step_dataset,
        build_euler1d_trajectory_dataset_from_split,
    )
    from grad_flow_l2.euler1d.train_vae import _build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Euler1D latent VAE checkpoint")
    p.add_argument("--dataset-path", type=str, required=True)
    p.add_argument("--checkpoint-path", type=str, required=True)
    p.add_argument("--args-json", type=str, default=None)
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument(
        "--output-dir", type=str, default="grad_flow_l2/euler1d/outputs_vae/eval"
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--n-plot-samples", type=int, default=4)
    p.add_argument("--snapshot-times", type=str, default="")
    p.add_argument("--max-snapshots", type=int, default=None)
    p.add_argument("--delta-clip", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def _load_train_args(args):
    args_json = args.args_json or os.path.join(
        os.path.dirname(args.checkpoint_path), "args.json"
    )
    with open(args_json, "r", encoding="utf-8") as f:
        return Namespace(**json.load(f)), args_json


def _parse_snapshot_times(raw: str, t_final: float) -> List[float]:
    if not raw.strip():
        return [0.0, 0.25 * t_final, 0.5 * t_final, 0.75 * t_final, t_final]
    out = [float(x.strip()) for x in raw.split(",") if x.strip()]
    return [min(max(0.0, x), t_final) for x in out]


def _truncate_split(split: Dict[str, torch.Tensor], max_snapshots: Optional[int]):
    available = int(split["u_traj"].shape[1])
    if (
        max_snapshots is None
        or int(max_snapshots) <= 0
        or int(max_snapshots) >= available
    ):
        return split, available
    if int(max_snapshots) < 2:
        raise ValueError("max_snapshots must be >=2")
    out = dict(split)
    out["u_traj"] = split["u_traj"][:, : int(max_snapshots)]
    out["u0"] = out["u_traj"][:, 0].clone()
    return out, int(max_snapshots)


@torch.no_grad()
def evaluate_step(model, step_loader, device, dt, channel_weights):
    model.eval()
    total = 0.0
    count = 0
    ch_sum = torch.zeros(STATE_CHANNELS, dtype=torch.float64)
    ch_count = 0
    w = torch.as_tensor(channel_weights, dtype=torch.float32)
    for u_k, u_k1, f in step_loader:
        u_k, u_k1, f = u_k.to(device), u_k1.to(device), f.to(device)
        pred = model.predict_step(u_k, f, dt=dt, sample=False)
        loss = channel_weighted_mse(pred, u_k1, w.to(pred.device, pred.dtype))
        total += loss.item() * int(u_k.shape[0])
        count += int(u_k.shape[0])
        ch_sum += (pred - u_k1).square().mean(dim=(0, 2)).detach().cpu().double() * int(
            u_k.shape[0]
        )
        ch_count += int(u_k.shape[0])
    metrics = {"one_step_weighted_mse": total / max(1, count)}
    for c, name in enumerate(CHANNEL_NAMES):
        metrics[f"one_step_mse_{name}"] = float((ch_sum[c] / max(1, ch_count)).item())
    return metrics


@torch.no_grad()
def evaluate_rollout(model, traj_loader, device, dt, h, domain_length, delta_clip):
    num_batches = []
    den_batches = []
    l1_num_batches = []
    l1_den_batches = []
    mse_batches = []
    for batch in traj_loader:
        u0, f, ref = (
            batch["u0"].to(device),
            batch["f"].to(device),
            batch["u_traj"].to(device),
        )
        pred = rollout_vae_mean_1d(
            model, u0, f, n_steps=int(ref.shape[1] - 1), dt=dt, delta_clip=delta_clip
        )
        diff = pred - ref
        num = torch.sqrt(float(h) * diff.square().sum(dim=-1))
        den = torch.sqrt(float(h) * ref.square().sum(dim=-1))
        l1_num = float(h) * diff.abs().sum(dim=-1)
        l1_den = float(h) * ref.abs().sum(dim=-1)
        num_batches.append(num.detach().cpu())
        den_batches.append(den.detach().cpu())
        l1_num_batches.append(l1_num.detach().cpu())
        l1_den_batches.append(l1_den.detach().cpu())
        mse_batches.append(diff.square().mean(dim=-1).detach().cpu())
    num = torch.cat(num_batches, dim=0)[:, 1:]
    den = torch.cat(den_batches, dim=0)[:, 1:]
    l1_num = torch.cat(l1_num_batches, dim=0)[:, 1:]
    l1_den = torch.cat(l1_den_batches, dim=0)[:, 1:]
    mse = torch.cat(mse_batches, dim=0)[:, 1:]
    rel = num / (den + 1e-8)
    rel_l1 = l1_num / (l1_den + 1e-8)
    overall_rel_l2_samples = torch.sqrt(torch.sum(num.square(), dim=1)) / (
        torch.sqrt(torch.sum(den.square(), dim=1)) + 1e-8
    )
    overall_rel_l1_samples = torch.sum(l1_num, dim=1) / (
        torch.sum(l1_den, dim=1) + 1e-8
    )
    rollout_rel_l2_channels = (
        torch.nanmean(overall_rel_l2_samples, dim=0).numpy().astype(np.float64)
    )
    rollout_rel_l1_channels = (
        torch.nanmean(overall_rel_l1_samples, dim=0).numpy().astype(np.float64)
    )
    sample_rel_l2_mean = np.nanmean(overall_rel_l2_samples.numpy(), axis=1)
    sample_rel_l1_mean = np.nanmean(overall_rel_l1_samples.numpy(), axis=1)
    return {
        "rel_curve_mean": rel.mean(dim=0).numpy(),
        "rel_curve_median": rel.median(dim=0).values.numpy(),
        "rel_l1_curve_mean": rel_l1.mean(dim=0).numpy(),
        "rel_l1_curve_median": rel_l1.median(dim=0).values.numpy(),
        "mse_curve_mean": mse.mean(dim=0).numpy(),
        "mse_curve_median": mse.median(dim=0).values.numpy(),
        "rel_samples": rel.numpy().astype(np.float64),
        "rel_l1_samples": rel_l1.numpy().astype(np.float64),
        "overall_rel_l2_samples": overall_rel_l2_samples.numpy().astype(np.float64),
        "overall_rel_l1_samples": overall_rel_l1_samples.numpy().astype(np.float64),
        "rollout_rel_l2_channels": rollout_rel_l2_channels,
        "rollout_rel_l1_channels": rollout_rel_l1_channels,
        "rollout_rel_mean": float(np.nanmean(rollout_rel_l2_channels)),
        "rollout_rel_median": float(np.nanmedian(sample_rel_l2_mean)),
        "rollout_rel_l1_mean": float(np.nanmean(rollout_rel_l1_channels)),
        "rollout_rel_l1_median": float(np.nanmedian(sample_rel_l1_mean)),
        "overall_rel_l2": float(np.nanmean(rollout_rel_l2_channels)),
        "overall_rel_l1": float(np.nanmean(rollout_rel_l1_channels)),
        "overall_rel_l2_global_components": float(np.nanmean(sample_rel_l2_mean)),
        "overall_rel_l1_global_components": float(np.nanmean(sample_rel_l1_mean)),
    }


def _add_rollout_std_metrics(metrics, curves):
    rel_sample_overall = curves["overall_rel_l2_samples"]
    rel_l1_sample_overall = curves["overall_rel_l1_samples"]
    for c, name in enumerate(CHANNEL_NAMES):
        metrics[f"rollout_rel_l2_{name}_std"] = float(
            np.nanstd(rel_sample_overall[:, c])
        )
        metrics[f"rollout_rel_l1_{name}_std"] = float(
            np.nanstd(rel_l1_sample_overall[:, c])
        )
    metrics["rollout_rel_l2_std"] = float(
        np.nanstd(np.nanmean(rel_sample_overall, axis=1))
    )
    metrics["rollout_rel_l1_std"] = float(
        np.nanstd(np.nanmean(rel_l1_sample_overall, axis=1))
    )


def _add_rollout_max_metrics(metrics, curves):
    rel_curve = curves["rel_curve_mean"]
    rel_l1_curve = curves["rel_l1_curve_mean"]
    for c, name in enumerate(CHANNEL_NAMES):
        metrics[f"rollout_rel_l2_{name}_max"] = float(np.nanmax(rel_curve[:, c]))
        metrics[f"rollout_rel_l1_{name}_max"] = float(np.nanmax(rel_l1_curve[:, c]))
    metrics["rollout_rel_l2_max"] = float(np.nanmax(np.nanmean(rel_curve, axis=1)))
    metrics["rollout_rel_l1_max"] = float(np.nanmax(np.nanmean(rel_l1_curve, axis=1)))


def _stats_dict(values: np.ndarray) -> Dict[str, object]:
    out: Dict[str, object] = {
        name: values[..., c].tolist() for c, name in enumerate(CHANNEL_NAMES)
    }
    out["mean"] = np.nanmean(values, axis=-1).tolist()
    return out


def _save_per_sample_errors_json(curves, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rel_l2 = curves["rel_samples"]
    rel_l1 = curves["rel_l1_samples"]
    overall_l2 = curves["overall_rel_l2_samples"]
    overall_l1 = curves["overall_rel_l1_samples"]
    items = []
    for sample_idx in range(int(rel_l2.shape[0])):
        items.append(
            {
                "sample_index": sample_idx,
                "rel_l2": _stats_dict(rel_l2[sample_idx]),
                "rel_l1": _stats_dict(rel_l1[sample_idx]),
                "overall_rel_l2": _stats_dict(overall_l2[sample_idx]),
                "overall_rel_l1": _stats_dict(overall_l1[sample_idx]),
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)


def _save_curve_csv(curves, dt, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        header = ["step", "time"]
        for metric in ("mse", "rel_l2", "rel_l1"):
            for name in CHANNEL_NAMES:
                header += [f"{metric}_mean_{name}", f"{metric}_median_{name}"]
            header += [f"{metric}_mean_all", f"{metric}_median_all"]
        wr.writerow(header)
        T = curves["rel_curve_mean"].shape[0]
        for k in range(T):
            row = [k + 1, f"{(k + 1) * dt:.8f}"]
            for mean_key, med_key in (
                ("mse_curve_mean", "mse_curve_median"),
                ("rel_curve_mean", "rel_curve_median"),
                ("rel_l1_curve_mean", "rel_l1_curve_median"),
            ):
                mean = curves[mean_key][k]
                med = curves[med_key][k]
                for c in range(len(CHANNEL_NAMES)):
                    row += [mean[c], med[c]]
                row += [mean.mean(), med.mean()]
            wr.writerow(row)


def _plot_curve(curves, dt, path):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping curve plot: {exc}")
        return
    t = (np.arange(curves["rel_curve_mean"].shape[0]) + 1) * float(dt)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    specs = [
        ("rel_curve_mean", "relative L2"),
        ("rel_l1_curve_mean", "relative L1"),
    ]
    for ax, (key, title) in zip(axes, specs):
        data = curves[key]
        for c, name in enumerate(CHANNEL_NAMES):
            ax.plot(t, data[:, c], label=name)
        ax.plot(t, data.mean(axis=1), "k--", label="all")
        ax.set_title(title)
        ax.set_xlabel("time")
        ax.grid(alpha=0.3)
        ax.legend()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def _plot_samples(
    model,
    split,
    device,
    dt,
    t_final,
    domain_length,
    snapshot_times,
    n_plot_samples,
    output_dir,
    delta_clip,
):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping sample plots: {exc}")
        return
    os.makedirs(output_dir, exist_ok=True)
    total = int(split["u0"].shape[0])
    ids = (
        torch.linspace(0, total - 1, min(max(1, n_plot_samples), total)).long().tolist()
    )
    n_steps = int(split["u_traj"].shape[1] - 1)
    n_x = int(split["u0"].shape[-1])
    h = float(domain_length) / float(n_x)
    x = np.linspace(0.0, float(domain_length), n_x, endpoint=False)
    times = np.arange(n_steps + 1) * float(dt)
    extent = [0.0, float(domain_length), times[0], times[-1]]
    for sample_id in ids:
        u0 = split["u0"][sample_id : sample_id + 1].to(device)
        f = split["f"][sample_id : sample_id + 1].to(device)
        ref = split["u_traj"][sample_id]
        pred = rollout_vae_mean_1d(
            model, u0, f, n_steps=n_steps, dt=dt, delta_clip=delta_clip
        )[0].cpu()
        abs_err = (pred - ref).abs()
        rel_l2 = relative_l2_error_1d(pred.unsqueeze(0), ref.unsqueeze(0), h=h)[0]
        l1_num = float(h) * (pred - ref).abs().sum(dim=-1)
        l1_den = float(h) * ref.abs().sum(dim=-1)
        rel_l1 = l1_num / (l1_den + 1e-8)
        fig, axes = plt.subplots(
            4,
            STATE_CHANNELS,
            figsize=(13, 12),
            squeeze=False,
            constrained_layout=True,
        )
        for c, name in enumerate(CHANNEL_NAMES):
            ref_np = ref[:, c].numpy()
            pred_np = pred[:, c].numpy()
            err_np = abs_err[:, c].numpy()
            vmin = min(float(ref_np.min()), float(pred_np.min()))
            vmax = max(float(ref_np.max()), float(pred_np.max()))
            axes[0, c].imshow(
                ref_np,
                aspect="auto",
                origin="lower",
                extent=extent,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            axes[0, c].set_title(f"ref {name}")
            axes[1, c].imshow(
                pred_np,
                aspect="auto",
                origin="lower",
                extent=extent,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            axes[1, c].set_title(f"pred {name}")
            axes[2, c].imshow(
                err_np,
                aspect="auto",
                origin="lower",
                extent=extent,
                cmap="magma",
            )
            axes[2, c].set_title(f"abs error {name}")
            for row in range(3):
                axes[row, c].set_xlabel("x")
                axes[row, c].set_ylabel("t")
        for c, name in enumerate(CHANNEL_NAMES):
            axes[3, 0].plot(x, ref[0, c].numpy(), label=name)
            axes[3, 1].plot(times, rel_l2[:, c].numpy(), label=name)
            axes[3, 2].plot(times, rel_l1[:, c].numpy(), label=name)
        axes[3, 1].plot(times, rel_l2.mean(dim=1).numpy(), "k--", label="all")
        axes[3, 2].plot(times, rel_l1.mean(dim=1).numpy(), "k--", label="all")
        axes[3, 0].set_title("initial condition")
        axes[3, 0].set_xlabel("x")
        axes[3, 1].set_title("rollout relative L2")
        axes[3, 1].set_xlabel("t")
        axes[3, 2].set_title("rollout relative L1")
        axes[3, 2].set_xlabel("t")
        for ax in axes[3]:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.savefig(
            os.path.join(output_dir, f"sample_{sample_id:04d}_comparison.png"), dpi=150
        )
        plt.close(fig)


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
    domain_length = float(
        meta.get("domain_length", getattr(train_args, "domain_length", 1.0))
    )
    h = domain_length / float(n_x)
    bc = str(meta.get("boundary_condition", "periodic"))
    model = _build_model(n_x=n_x, dt=dt, boundary_condition=bc, args=train_args).to(
        device
    )
    ckpt = safe_torch_load(args.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    channel_weights = getattr(train_args, "channel_weights_used", None) or ckpt.get(
        "channel_weights"
    )
    if channel_weights is None:
        channel_weights = torch.ones(STATE_CHANNELS)
    delta_clip = (
        ckpt.get("rollout_delta_clip", None)
        if args.delta_clip is None
        else args.delta_clip
    )
    delta_clip = (
        None if delta_clip is None or float(delta_clip) <= 0 else float(delta_clip)
    )
    step_loader = DataLoader(
        build_euler1d_step_dataset(split),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    traj_loader = DataLoader(
        build_euler1d_trajectory_dataset_from_split(split),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    step_metrics = evaluate_step(model, step_loader, device, dt, channel_weights)
    curves = evaluate_rollout(
        model, traj_loader, device, dt, h, domain_length, delta_clip
    )
    metrics = dict(step_metrics)
    for c, name in enumerate(CHANNEL_NAMES):
        metrics[f"rollout_rel_l2_{name}"] = float(curves["rollout_rel_l2_channels"][c])
        metrics[f"rollout_rel_l1_{name}"] = float(curves["rollout_rel_l1_channels"][c])
    metrics["rollout_rel_l2"] = curves["rollout_rel_mean"]
    metrics["rollout_rel_l2_median"] = curves["rollout_rel_median"]
    metrics["rollout_rel_l1"] = curves["rollout_rel_l1_mean"]
    metrics["rollout_rel_l1_median"] = curves["rollout_rel_l1_median"]
    _add_rollout_std_metrics(metrics, curves)
    _add_rollout_max_metrics(metrics, curves)
    print(f"Device: {device}")
    print(
        f"Split={args.split}, n={int(split['u0'].shape[0])}, n_x={n_x}, steps={n_steps}, dt={dt}, L={domain_length}, delta_clip={delta_clip}"
    )
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.8e}")
    _save_curve_csv(
        curves,
        dt,
        os.path.join(args.output_dir, f"{args.split}_rollout_error_curve.csv"),
    )
    _save_per_sample_errors_json(
        curves,
        os.path.join(args.output_dir, f"{args.split}_per_sample_errors.json"),
    )
    _plot_curve(
        curves,
        dt,
        os.path.join(args.output_dir, f"{args.split}_rollout_error_curve.png"),
    )
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
        "overall_rel_l1": curves["overall_rel_l1"],
        "overall_rel_l2_global_components": curves["overall_rel_l2_global_components"],
        "overall_rel_l1_global_components": curves["overall_rel_l1_global_components"],
        "rel_l2_curve_mean": curves["rel_curve_mean"].tolist(),
        "rel_l1_curve_mean": curves["rel_l1_curve_mean"].tolist(),
        "channel_names": CHANNEL_NAMES,
        "field_names": CHANNEL_NAMES,
        "meta": meta,
    }
    with open(
        os.path.join(args.output_dir, f"{args.split}_summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main(parse_args())
