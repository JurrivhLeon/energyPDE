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

# Primitive variable names: rho, vx, vy, p
CHANNEL_NAMES = ["rho", "vx", "vy", "p"]

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


@torch.no_grad()
def _evaluate_rollout_curves(model, traj_loader: DataLoader, device: str,
                               dt: float, area: float,
                               delta_clip: Optional[float] = None,
                               rollout_fn: Callable = rollout_latent_markov_2d) -> Dict[str, np.ndarray]:
    """
    Compute per-channel rollout relative L2 error curves.

    Returns
    -------
    rel_curve_mean   : (T+1, C)  – mean  over samples at each step, per channel
    rel_curve_median : (T+1, C)  – median over samples at each step, per channel
    rollout_rel_mean   : float   – mean of per-channel means (for model selection)
    rollout_rel_median : float   – median equivalent
    """
    rel_batches = []
    for batch in traj_loader:
        u0    = batch["u0"].to(device)
        f     = batch["f"].to(device)
        u_ref = batch["u_traj"].to(device)              # (B, T+1, C, H, W)
        u_pred = rollout_fn(model, u0=u0, f=f,
                            n_steps=int(u_ref.shape[1] - 1), dt=dt,
                            delta_clip=delta_clip)
        diff = u_pred - u_ref                           # (B, T+1, C, H, W)
        # Per-channel relative L2: sqrt(area * Σ_spatial diff²) / sqrt(area * Σ_spatial ref²)
        num = torch.sqrt(float(area) * diff.pow(2).sum(dim=(-2, -1)))   # (B, T+1, C)
        den = torch.sqrt(float(area) * u_ref.pow(2).sum(dim=(-2, -1))) # (B, T+1, C)
        rel_batches.append((num / (den + 1e-8)).detach().cpu())

    rel = torch.cat(rel_batches, dim=0)                 # (N, T+1, C)
    rel_mean   = torch.nanmean(rel, dim=0).numpy().astype(np.float64)          # (T+1, C)
    rel_median = np.nanmedian(rel.numpy(), axis=0).astype(np.float64)          # (T+1, C)
    # Aggregate: mean over channels, then mean/median over samples and time
    rel_all = rel.mean(dim=2)                           # (N, T+1)
    return {
        "rel_curve_mean":     rel_mean,                 # (T+1, C)
        "rel_curve_median":   rel_median,               # (T+1, C)
        "rollout_rel_mean":   float(np.nanmean(rel_mean)),
        "rollout_rel_median": float(np.nanmedian(rel_median)),
    }


def _save_curve_csv(curves: Dict[str, np.ndarray], dt: float, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rel_mean = curves["rel_curve_mean"]                 # (T+1, C)
    n_steps, n_ch = rel_mean.shape
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["step", "time"]
        for name in CHANNEL_NAMES:
            header += [f"rel_l2_mean_{name}", f"rel_l2_median_{name}"]
        header += ["rel_l2_mean_all", "rel_l2_median_all"]
        writer.writerow(header)
        rel_med = curves["rel_curve_median"]            # (T+1, C)
        for k in range(n_steps):
            row = [k, f"{k * dt:.8f}"]
            for c in range(n_ch):
                row += [rel_mean[k, c], rel_med[k, c]]
            row += [rel_mean[k].mean(), rel_med[k].mean()]
            writer.writerow(row)


def _plot_curve(curves: Dict[str, np.ndarray], dt: float, path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping curve plot (matplotlib unavailable): {exc}"); return

    rel_mean   = curves["rel_curve_mean"]               # (T+1, C)
    rel_median = curves["rel_curve_median"]             # (T+1, C)
    T_plus1, _ = rel_mean.shape
    t      = np.arange(T_plus1) * float(dt)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=False)
    for data, ax, title in [
        (rel_mean,   axes[0], "Rollout Relative L2 — mean over samples"),
        (rel_median, axes[1], "Rollout Relative L2 — median over samples"),
    ]:
        for c, (name, col) in enumerate(zip(CHANNEL_NAMES, colors)):
            ax.plot(t, data[:, c], color=col, label=name)
        ax.plot(t, data.mean(axis=1), color="black", linestyle="--",
                linewidth=1.5, label="all-ch mean")
        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel("relative L2")
        ax.legend()
        ax.grid(alpha=0.3)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


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
    n_x    = int(split["u0"].shape[-2])
    n_y    = int(split["u0"].shape[-1])
    n_steps = int(split["u_traj"].shape[1] - 1)
    t_final = float(meta.get("t_final", float(n_steps)))
    dt      = t_final / float(n_steps)
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
