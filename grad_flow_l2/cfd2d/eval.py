"""Evaluation entrypoint for periodic 2D compressible NS hidden-space checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
from argparse import Namespace
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from ..cfd2d.cfd_data import build_cfd2d_step_dataset, build_cfd2d_trajectory_dataset_from_split
    from ..heat_data import load_dataset_splits
    from ..latent_markov_trainer_mc import LatentMarkovTrainer2D, rollout_latent_markov_2d
    from .train import _build_model   # uses latent_markov_mc
except ImportError:
    from grad_flow_l2.cfd2d.cfd_data import build_cfd2d_step_dataset, build_cfd2d_trajectory_dataset_from_split
    from grad_flow_l2.heat_data import load_dataset_splits
    from grad_flow_l2.latent_markov_trainer_mc import LatentMarkovTrainer2D, rollout_latent_markov_2d
    from grad_flow_l2.cfd2d.train import _build_model   # uses latent_markov_mc

# Primitive variable names: rho, vx, vy, p. Field statistics merge vx/vy into one vector velocity field.
CHANNEL_NAMES = ["rho", "vx", "vy", "p"]
FIELD_NAMES = ["rho", "velocity", "p"]

# vx and vy are zero-mean → symmetric diverging colormap
# rho and p are positive-definite → sequential colormap with data-range limits
_CHANNEL_CMAP     = {"rho": "viridis", "vx": "RdBu_r", "vy": "RdBu_r", "p": "plasma"}
_CHANNEL_SYMMETRIC = {"rho": False,    "vx": True,      "vy": True,      "p": False}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate periodic 2D compressible NS hidden-space checkpoint")
    parser.add_argument("--dataset-path",    type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--split",           type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-dir",      type=str, default="grad_flow_l2/cfd2d/outputs/eval")
    parser.add_argument("--batch-size",      type=int, default=64)
    parser.add_argument("--num-workers",     type=int, default=0)
    parser.add_argument("--n-plot-samples",  type=int,   default=4)
    parser.add_argument("--snapshot-times",  type=str,   default="")
    parser.add_argument("--max-snapshots",   type=int,   default=76,
                        help="Maximum trajectory snapshots to evaluate. Default: 61 for OOD datasets (T=60), otherwise all.")
    parser.add_argument("--delta-clip",      type=float, default=None,
                        help="Clip predicted increment per step. Default: checkpoint training value; set 0 to disable.")
    parser.add_argument("--cpu",             action="store_true")
    return parser.parse_args()


def _torch_load_checkpoint(checkpoint_path: str, map_location):
    try:
        return torch.load(checkpoint_path, map_location=map_location)
    except RuntimeError as exc:
        if "weights_only=True" not in str(exc) or "legacy .tar format" not in str(exc):
            raise
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def _parse_snapshot_times(raw: str, t_final: float) -> List[float]:
    if not raw.strip():
        return [0.0, 0.25 * t_final, 0.5 * t_final, 0.75 * t_final, t_final]
    vals = [float(tok.strip()) for tok in raw.split(",") if tok.strip()]
    for val in vals:
        if val < 0.0 or val > t_final:
            raise ValueError(f"Snapshot time must be in [0,{t_final}], got {val}")
    return vals


def _channel_weights_from_args_or_checkpoint(train_args: Namespace, checkpoint) -> Optional[torch.Tensor]:
    weights = getattr(train_args, "channel_weights_used", None)
    if weights is None:
        weights = checkpoint.get("channel_weights")
    return None if weights is None else torch.as_tensor(weights, dtype=torch.float32)


def _resolve_delta_clip(cli_value: Optional[float], checkpoint, default: Optional[float]) -> Optional[float]:
    value = checkpoint.get("rollout_delta_clip", default) if cli_value is None else cli_value
    return None if value is None or float(value) <= 0.0 else float(value)


def _truncate_split_for_eval(split: Dict[str, torch.Tensor], dataset_path: str,
                             max_snapshots: Optional[int]) -> tuple[Dict[str, torch.Tensor], int]:
    if max_snapshots is None:
        max_snapshots = 61 if "ood" in os.path.basename(dataset_path).lower() else 0
    max_snapshots = int(max_snapshots)
    available = int(split["u_traj"].shape[1])
    if max_snapshots <= 0 or max_snapshots >= available:
        return split, available
    if max_snapshots < 2:
        raise ValueError("max_snapshots must be >= 2 when enabled")
    truncated = dict(split)
    truncated["u_traj"] = split["u_traj"][:, :max_snapshots]
    truncated["u0"] = truncated["u_traj"][:, 0].clone()
    return truncated, max_snapshots


def _spectral_h1_squared_per_channel_2d(u: torch.Tensor) -> torch.Tensor:
    """Return spatial H1 squared norms with shape matching u[..., channel]."""
    n_x = int(u.shape[-2])
    n_y = int(u.shape[-1])
    u_hat = torch.fft.fft2(u, dim=(-2, -1), norm="ortho")
    real_dtype = u.real.dtype
    kx = 2.0 * torch.pi * torch.fft.fftfreq(n_x, d=1.0 / float(n_x), device=u.device).to(dtype=real_dtype)
    ky = 2.0 * torch.pi * torch.fft.fftfreq(n_y, d=1.0 / float(n_y), device=u.device).to(dtype=real_dtype)
    kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing="ij")
    weight = 1.0 + kx_grid.square() + ky_grid.square()
    power = u_hat.real.square() + u_hat.imag.square()
    return torch.sum(power * weight, dim=(-2, -1))


def _merge_velocity_norms(norms: torch.Tensor) -> torch.Tensor:
    """Merge vx/vy component norms into one vector-valued velocity norm."""
    velocity = torch.sqrt(norms[..., 1].square() + norms[..., 2].square())
    return torch.stack((norms[..., 0], velocity, norms[..., 3]), dim=-1)


@torch.no_grad()
def _evaluate_rollout_curves(model, traj_loader: DataLoader, device: str,
                               dt: float, area: float,
                               delta_clip: Optional[float] = None,
                               rollout_fn: Callable = rollout_latent_markov_2d) -> Dict[str, np.ndarray]:
    """
    Compute rollout relative L2 and H1 errors, excluding the initial snapshot.

    Curve metrics are per-snapshot relative errors for snapshots 1..T. Scalar
    rollout metrics first accumulate norms over time for each sample trajectory,
    then average the resulting ratios equally over samples.
    """
    num_batches = []
    den_batches = []
    field_num_batches = []
    field_den_batches = []
    h1_num_batches = []
    h1_den_batches = []
    h1_field_num_batches = []
    h1_field_den_batches = []
    for batch in traj_loader:
        u0 = batch["u0"].to(device)
        f = batch["f"].to(device)
        u_ref = batch["u_traj"].to(device)
        u_pred = rollout_fn(
            model,
            u0=u0,
            f=f,
            n_steps=int(u_ref.shape[1] - 1),
            dt=dt,
            delta_clip=delta_clip,
        )
        diff = u_pred - u_ref
        num = torch.sqrt(float(area) * diff.pow(2).sum(dim=(-2, -1)))
        den = torch.sqrt(float(area) * u_ref.pow(2).sum(dim=(-2, -1)))
        field_num = _merge_velocity_norms(num)
        field_den = _merge_velocity_norms(den)
        h1_diff_sq = _spectral_h1_squared_per_channel_2d(diff)
        h1_ref_sq = _spectral_h1_squared_per_channel_2d(u_ref)
        h1_num = torch.sqrt(h1_diff_sq)
        h1_den = torch.sqrt(h1_ref_sq)
        h1_field_num = _merge_velocity_norms(h1_num)
        h1_field_den = _merge_velocity_norms(h1_den)
        num_batches.append(num.detach().cpu())
        den_batches.append(den.detach().cpu())
        field_num_batches.append(field_num.detach().cpu())
        field_den_batches.append(field_den.detach().cpu())
        h1_num_batches.append(h1_num.detach().cpu())
        h1_den_batches.append(h1_den.detach().cpu())
        h1_field_num_batches.append(h1_field_num.detach().cpu())
        h1_field_den_batches.append(h1_field_den.detach().cpu())

    num = torch.cat(num_batches, dim=0)[:, 1:]
    den = torch.cat(den_batches, dim=0)[:, 1:]
    field_num = torch.cat(field_num_batches, dim=0)[:, 1:]
    field_den = torch.cat(field_den_batches, dim=0)[:, 1:]
    h1_num = torch.cat(h1_num_batches, dim=0)[:, 1:]
    h1_den = torch.cat(h1_den_batches, dim=0)[:, 1:]
    h1_field_num = torch.cat(h1_field_num_batches, dim=0)[:, 1:]
    h1_field_den = torch.cat(h1_field_den_batches, dim=0)[:, 1:]

    rel = num / (den + 1e-8)
    rel_h1 = h1_num / (h1_den + 1e-12)
    rel_field = field_num / (field_den + 1e-8)
    rel_h1_field = h1_field_num / (h1_field_den + 1e-12)

    overall_rel_l2_samples = torch.sqrt(torch.sum(num.square(), dim=1)) / (
        torch.sqrt(torch.sum(den.square(), dim=1)) + 1e-8
    )
    overall_rel_h1_samples = torch.sqrt(torch.sum(h1_num.square(), dim=1)) / (
        torch.sqrt(torch.sum(h1_den.square(), dim=1)) + 1e-12
    )
    overall_rel_l2_field_samples = torch.sqrt(torch.sum(field_num.square(), dim=1)) / (
        torch.sqrt(torch.sum(field_den.square(), dim=1)) + 1e-8
    )
    overall_rel_h1_field_samples = torch.sqrt(torch.sum(h1_field_num.square(), dim=1)) / (
        torch.sqrt(torch.sum(h1_field_den.square(), dim=1)) + 1e-12
    )

    rel_mean = torch.nanmean(rel, dim=0).numpy().astype(np.float64)
    rel_median = np.nanmedian(rel.numpy(), axis=0).astype(np.float64)
    rel_h1_mean = torch.nanmean(rel_h1, dim=0).numpy().astype(np.float64)
    rel_h1_median = np.nanmedian(rel_h1.numpy(), axis=0).astype(np.float64)
    rel_field_mean = torch.nanmean(rel_field, dim=0).numpy().astype(np.float64)
    rel_field_median = np.nanmedian(rel_field.numpy(), axis=0).astype(np.float64)
    rel_h1_field_mean = torch.nanmean(rel_h1_field, dim=0).numpy().astype(np.float64)
    rel_h1_field_median = np.nanmedian(rel_h1_field.numpy(), axis=0).astype(np.float64)
    rollout_rel_l2_channels = torch.nanmean(overall_rel_l2_samples, dim=0).numpy().astype(np.float64)
    rollout_rel_h1_channels = torch.nanmean(overall_rel_h1_samples, dim=0).numpy().astype(np.float64)
    rollout_rel_l2_fields = torch.nanmean(overall_rel_l2_field_samples, dim=0).numpy().astype(np.float64)
    rollout_rel_h1_fields = torch.nanmean(overall_rel_h1_field_samples, dim=0).numpy().astype(np.float64)
    sample_rel_l2_mean = np.nanmean(overall_rel_l2_field_samples.numpy(), axis=1)
    sample_rel_h1_mean = np.nanmean(overall_rel_h1_field_samples.numpy(), axis=1)
    return {
        "rel_curve_mean": rel_mean,
        "rel_curve_median": rel_median,
        "rel_h1_curve_mean": rel_h1_mean,
        "rel_h1_curve_median": rel_h1_median,
        "rel_field_curve_mean": rel_field_mean,
        "rel_field_curve_median": rel_field_median,
        "rel_h1_field_curve_mean": rel_h1_field_mean,
        "rel_h1_field_curve_median": rel_h1_field_median,
        "rel_samples": rel.numpy().astype(np.float64),
        "rel_h1_samples": rel_h1.numpy().astype(np.float64),
        "rel_field_samples": rel_field.numpy().astype(np.float64),
        "rel_h1_field_samples": rel_h1_field.numpy().astype(np.float64),
        "overall_rel_l2_samples": overall_rel_l2_samples.numpy().astype(np.float64),
        "overall_rel_h1_samples": overall_rel_h1_samples.numpy().astype(np.float64),
        "overall_rel_l2_field_samples": overall_rel_l2_field_samples.numpy().astype(np.float64),
        "overall_rel_h1_field_samples": overall_rel_h1_field_samples.numpy().astype(np.float64),
        "rollout_rel_l2_channels": rollout_rel_l2_channels,
        "rollout_rel_h1_channels": rollout_rel_h1_channels,
        "rollout_rel_l2_fields": rollout_rel_l2_fields,
        "rollout_rel_h1_fields": rollout_rel_h1_fields,
        "rollout_rel_mean": float(np.nanmean(rollout_rel_l2_fields)),
        "rollout_rel_median": float(np.nanmedian(sample_rel_l2_mean)),
        "rollout_rel_h1_mean": float(np.nanmean(rollout_rel_h1_fields)),
        "rollout_rel_h1_median": float(np.nanmedian(sample_rel_h1_mean)),
        "overall_rel_l2": float(np.nanmean(rollout_rel_l2_fields)),
        "overall_rel_h1": float(np.nanmean(rollout_rel_h1_fields)),
        "overall_rel_l2_global_components": float(np.nanmean(sample_rel_l2_mean)),
        "overall_rel_h1_global_components": float(np.nanmean(sample_rel_h1_mean)),
    }


def _add_rollout_std_metrics(metrics: Dict[str, float], curves: Dict[str, np.ndarray]) -> None:
    rel_sample_overall = curves["overall_rel_l2_samples"]              # (N, C)
    rel_h1_sample_overall = curves["overall_rel_h1_samples"]           # (N, C)
    rel_field_sample_overall = curves["overall_rel_l2_field_samples"]  # (N, 3)
    rel_h1_field_sample_overall = curves["overall_rel_h1_field_samples"]
    for c, name in enumerate(CHANNEL_NAMES):
        metrics[f"rollout_rel_l2_{name}_std"] = float(np.nanstd(rel_sample_overall[:, c]))
        metrics[f"rollout_rel_h1_{name}_std"] = float(np.nanstd(rel_h1_sample_overall[:, c]))
    for c, name in enumerate(FIELD_NAMES):
        metrics[f"rollout_rel_l2_{name}_std"] = float(np.nanstd(rel_field_sample_overall[:, c]))
        metrics[f"rollout_rel_h1_{name}_std"] = float(np.nanstd(rel_h1_field_sample_overall[:, c]))
    metrics["rollout_rel_l2_std"] = float(np.nanstd(np.nanmean(rel_field_sample_overall, axis=1)))
    metrics["rollout_rel_h1_std"] = float(np.nanstd(np.nanmean(rel_h1_field_sample_overall, axis=1)))


def _add_rollout_max_metrics(metrics: Dict[str, float], curves: Dict[str, np.ndarray]) -> None:
    rel_curve = curves["rel_curve_mean"]
    rel_h1_curve = curves["rel_h1_curve_mean"]
    rel_field_curve = curves["rel_field_curve_mean"]
    rel_h1_field_curve = curves["rel_h1_field_curve_mean"]
    for c, name in enumerate(CHANNEL_NAMES):
        metrics[f"rollout_rel_l2_{name}_max"] = float(np.nanmax(rel_curve[:, c]))
        metrics[f"rollout_rel_h1_{name}_max"] = float(np.nanmax(rel_h1_curve[:, c]))
    for c, name in enumerate(FIELD_NAMES):
        metrics[f"rollout_rel_l2_{name}_max"] = float(np.nanmax(rel_field_curve[:, c]))
        metrics[f"rollout_rel_h1_{name}_max"] = float(np.nanmax(rel_h1_field_curve[:, c]))
    metrics["rollout_rel_l2_max"] = float(np.nanmax(np.nanmean(rel_field_curve, axis=1)))
    metrics["rollout_rel_h1_max"] = float(np.nanmax(np.nanmean(rel_h1_field_curve, axis=1)))


def _stats_dict(component_values: np.ndarray, field_values: np.ndarray) -> Dict[str, object]:
    out: Dict[str, object] = {
        name: component_values[..., c].tolist()
        for c, name in enumerate(CHANNEL_NAMES)
    }
    out["velocity"] = field_values[..., 1].tolist()
    out["mean"] = np.nanmean(field_values, axis=-1).tolist()
    return out


def _save_per_sample_errors_json(curves: Dict[str, np.ndarray], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rel_l2 = curves["rel_samples"]
    rel_h1 = curves["rel_h1_samples"]
    rel_l2_field = curves["rel_field_samples"]
    rel_h1_field = curves["rel_h1_field_samples"]
    overall_l2 = curves["overall_rel_l2_samples"]
    overall_h1 = curves["overall_rel_h1_samples"]
    overall_l2_field = curves["overall_rel_l2_field_samples"]
    overall_h1_field = curves["overall_rel_h1_field_samples"]
    n_samples = int(rel_l2.shape[0])
    items = []
    for sample_idx in range(n_samples):
        items.append({
            "sample_index": sample_idx,
            "rel_l2": _stats_dict(rel_l2[sample_idx], rel_l2_field[sample_idx]),
            "rel_h1": _stats_dict(rel_h1[sample_idx], rel_h1_field[sample_idx]),
            "overall_rel_l2": _stats_dict(overall_l2[sample_idx], overall_l2_field[sample_idx]),
            "overall_rel_h1": _stats_dict(overall_h1[sample_idx], overall_h1_field[sample_idx]),
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)


def _save_curve_csv(curves: Dict[str, np.ndarray], dt: float, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rel_l2_mean = curves["rel_curve_mean"]
    rel_l2_median = curves["rel_curve_median"]
    rel_h1_mean = curves["rel_h1_curve_mean"]
    rel_h1_median = curves["rel_h1_curve_median"]
    rel_l2_field_mean = curves["rel_field_curve_mean"]
    rel_l2_field_median = curves["rel_field_curve_median"]
    rel_h1_field_mean = curves["rel_h1_field_curve_mean"]
    rel_h1_field_median = curves["rel_h1_field_curve_median"]
    n_steps, n_ch = rel_l2_mean.shape
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["step", "time"]
        for prefix in ("rel_l2", "rel_h1"):
            for name in CHANNEL_NAMES:
                header += [f"{prefix}_mean_{name}", f"{prefix}_median_{name}"]
            header += [f"{prefix}_mean_velocity", f"{prefix}_median_velocity"]
            header += [f"{prefix}_mean_all", f"{prefix}_median_all"]
        writer.writerow(header)
        for k in range(n_steps):
            row = [k + 1, f"{(k + 1) * dt:.8f}"]
            for c in range(n_ch):
                row += [rel_l2_mean[k, c], rel_l2_median[k, c]]
            row += [rel_l2_field_mean[k, 1], rel_l2_field_median[k, 1]]
            row += [rel_l2_field_mean[k].mean(), rel_l2_field_median[k].mean()]
            for c in range(n_ch):
                row += [rel_h1_mean[k, c], rel_h1_median[k, c]]
            row += [rel_h1_field_mean[k, 1], rel_h1_field_median[k, 1]]
            row += [rel_h1_field_mean[k].mean(), rel_h1_field_median[k].mean()]
            writer.writerow(row)


def _plot_curve(curves: Dict[str, np.ndarray], dt: float, path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping curve plot (matplotlib unavailable): {exc}")
        return

    t = (np.arange(curves["rel_curve_mean"].shape[0]) + 1) * float(dt)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=False)
    specs = (
        (axes[0], curves["rel_curve_mean"], curves["rel_field_curve_mean"], "relative L2", "Rollout Relative L2 Mean"),
        (axes[1], curves["rel_h1_curve_mean"], curves["rel_h1_field_curve_mean"], "relative H1", "Rollout Relative H1 Mean"),
    )
    for ax, data, field_data, ylabel, title in specs:
        for c, (name, col) in enumerate(zip(CHANNEL_NAMES, colors)):
            ax.plot(t, data[:, c], color=col, label=name)
        ax.plot(t, field_data[:, 1], color="tab:purple", linestyle=":", linewidth=1.5, label="velocity")
        ax.plot(t, field_data.mean(axis=1), color="black", linestyle="--", linewidth=1.5, label="rho/velocity/p mean")
        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def _plot_samples(model, split, device: str, dt: float, t_final: float,
                  snapshot_times: List[float], n_plot_samples: int, out_dir: str,
                  delta_clip: Optional[float] = None,
                  rollout_fn: Callable = rollout_latent_markov_2d) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping sample plots (matplotlib unavailable): {exc}"); return
    os.makedirs(out_dir, exist_ok=True)
    total      = int(split["u0"].shape[0])
    sample_ids = torch.linspace(0, total - 1, min(max(1, n_plot_samples), total)).long().tolist()
    n_steps    = int(split["u_traj"].shape[1] - 1)
    n_ch       = len(CHANNEL_NAMES)
    for sample_id in sample_ids:
        u0  = split["u0"][sample_id:sample_id + 1].to(device)
        f   = split["f"][sample_id:sample_id + 1].to(device)
        ref  = split["u_traj"][sample_id]                                   # (T+1, 4, nx, ny)
        pred = rollout_fn(model, u0=u0, f=f, n_steps=n_steps, dt=dt,
                          delta_clip=delta_clip)[0].cpu()
        cols = len(snapshot_times)

        # Colorbar limits from the reference trajectory (all time steps, per channel).
        # Using only ref ensures the reference always appears with full dynamic range;
        # predictions outside this range saturate, making large errors visually obvious.
        clim = {}
        for c, name in enumerate(CHANNEL_NAMES):
            ref_ch = ref[:, c]                         # (T+1, nx, ny)
            if _CHANNEL_SYMMETRIC[name]:
                vabs = max(float(ref_ch.abs().max()), 1e-8)
                clim[name] = (-vabs, vabs)
            else:
                clim[name] = (float(ref_ch.min()), float(ref_ch.max()))

        fig, axes = plt.subplots(2 * n_ch, cols,
                                 figsize=(3.0 * cols, 3.0 * 2 * n_ch),
                                 squeeze=False, constrained_layout=True)
        for j, t_snap in enumerate(snapshot_times):
            k = max(0, min(n_steps, int(round(float(t_snap) / float(t_final) * n_steps)) if t_final > 0 else 0))
            for c, name in enumerate(CHANNEL_NAMES):
                cmap = _CHANNEL_CMAP[name]
                vmin, vmax = clim[name]
                axes[2 * c,     j].imshow(ref[k,  c].numpy(), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
                axes[2 * c,     j].set_title(f"ref {name} t={t_snap:g}", fontsize=8)
                axes[2 * c + 1, j].imshow(pred[k, c].numpy(), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
                axes[2 * c + 1, j].set_title(f"pred {name} t={t_snap:g}", fontsize=8)
                for row in [2 * c, 2 * c + 1]:
                    axes[row, j].set_xticks([]); axes[row, j].set_yticks([])
        fig.savefig(os.path.join(out_dir, f"sample_{sample_id:04d}_comparison.png"), dpi=150)
        plt.close(fig)


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    device  = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.dirname(args.checkpoint_path)
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
    checkpoint = _torch_load_checkpoint(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    step_loader = DataLoader(build_cfd2d_step_dataset(split),
                             batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    traj_loader = DataLoader(build_cfd2d_trajectory_dataset_from_split(split),
                             batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    channel_weights = _channel_weights_from_args_or_checkpoint(train_args, checkpoint)
    trainer = LatentMarkovTrainer2D(model=model, dt=dt, h_x=1.0 / float(n_x), h_y=1.0 / float(n_y),
                                          lambda_spec=0.0, channel_weights=channel_weights, device=device,
                                          output_dir=None, show_epoch_pbar=False)
    delta_clip = _resolve_delta_clip(args.delta_clip, checkpoint, default=10.0)
    trainer.delta_clip = delta_clip

    metrics = trainer.validate(step_loader, traj_loader=None)
    curves  = _evaluate_rollout_curves(model, traj_loader, device=device, dt=dt, area=area,
                                        delta_clip=delta_clip)
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
                  delta_clip=delta_clip)

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
