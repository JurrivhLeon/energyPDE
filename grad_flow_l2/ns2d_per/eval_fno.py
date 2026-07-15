"""Evaluation and sample-visualization script for periodic 2D Navier-Stokes FNO models."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from argparse import Namespace
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

try:
    from ..fno import FNO2D
    from ..fno_trainer import rollout_fno_2d
    from ..heat_data import load_dataset_splits
    from .rollout_diagnostics import compute_rollout_diagnostics
except ImportError:
    from grad_flow_l2.fno import FNO2D
    from grad_flow_l2.fno_trainer import rollout_fno_2d
    from grad_flow_l2.heat_data import load_dataset_splits
    from grad_flow_l2.ns2d_per.rollout_diagnostics import compute_rollout_diagnostics


def set_seed(seed: int, seed_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda:
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and visualize periodic 2D Navier-Stokes physical FNO")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--n-plot-samples", type=int, default=6)
    parser.add_argument("--snapshot-times", type=str, default="0,2,4,6,8,10")
    parser.add_argument("--delta-clip", type=float, default=10.0, help="Use <=0 to disable rollout increment clipping.")
    parser.add_argument("--output-dir", type=str, default="grad_flow_l2/ns2d_per/outputs_fno/eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None, help="If set, truncate trajectories to this many steps.")

    # Fallback architecture args used only when args.json is unavailable.
    parser.add_argument("--state-channels", type=int, default=None)
    parser.add_argument("--forcing-channels", type=int, default=None)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--fno-layers", type=int, default=4)
    parser.add_argument("--fno-modes-x", type=int, default=16)
    parser.add_argument("--fno-modes-y", type=int, default=16)
    parser.add_argument("--disable-fno-grid", action="store_true")
    parser.add_argument("--use-dt-channel", action="store_true")
    parser.add_argument("--disable-forcing-channel", action="store_true")
    parser.add_argument("--no-residual", action="store_true")
    #parser.add_argument("--lift-noise-std", type=float, default=0.0, help="Fallback FNO lift-noise std when args.json is unavailable; inactive during eval.")
    #parser.add_argument("--lift-noise-corr-length", type=float, default=1.0, help="Fallback Matern correlation length for FNO lift noise.")
    #parser.add_argument("--lift-noise-decay-s", type=float, default=2.0, help="Fallback spectral decay exponent s for FNO lift noise.")
    return parser.parse_args()


def _load_checkpoint(checkpoint_path: str, map_location):
    try:
        return torch.load(checkpoint_path, map_location=map_location)
    except RuntimeError as exc:
        msg = str(exc)
        if "weights_only=True" not in msg or "legacy .tar format" not in msg:
            raise
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def _load_train_args(checkpoint_path: str) -> Namespace:
    args_path = os.path.join(os.path.dirname(checkpoint_path), "args.json")
    if not os.path.exists(args_path):
        return Namespace()
    with open(args_path, "r", encoding="utf-8") as f:
        return Namespace(**json.load(f))


def _time_metadata(meta: Dict, n_steps: int) -> tuple[float, float, float, np.ndarray]:
    t_start = float(meta.get("stored_t_start", meta.get("warmup_time", 0.0)))
    if "record_dt" in meta:
        dt = float(meta["record_dt"])
    elif "stored_time_horizon" in meta:
        dt = float(meta["stored_time_horizon"]) / float(n_steps)
    else:
        dt = float(meta.get("t_final", 1.0)) / float(n_steps)
    if dt <= 0.0:
        raise ValueError(f"Dataset record_dt/dt must be positive, got {dt}")
    t_end = float(meta.get("stored_t_final", t_start + dt * float(n_steps)))
    time_values = t_start + np.arange(n_steps + 1, dtype=np.float64) * dt
    if abs(time_values[-1] - t_end) > max(1e-8, 1e-6 * max(1.0, abs(t_end))):
        t_end = float(time_values[-1])
    return dt, t_start, t_end, time_values


def _parse_snapshot_times(raw: str, t_start: float, t_end: float) -> List[float]:
    vals = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = float(tok)
        if v < float(t_start) or v > float(t_end):
            raise ValueError(f"Snapshot time must be in [{t_start},{t_end}], got {v}")
        vals.append(v)
    if not vals:
        horizon = float(t_end) - float(t_start)
        vals = [t_start + frac * horizon for frac in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    return vals


def _infer_state_channels(split: dict) -> int:
    return 1 if split["u0"].dim() == 3 else int(split["u0"].shape[1])


def _infer_forcing_channels(split: dict) -> int:
    return 1 if split["f"].dim() == 3 else int(split["f"].shape[1])


def _build_model(n_x: int, n_y: int, dt: float, split: dict, args, train_args) -> FNO2D:
    state_channels = int(
        getattr(train_args, "state_channels_used", None)
        or getattr(train_args, "state_channels", None)
        or args.state_channels
        or _infer_state_channels(split)
    )
    forcing_channels = int(
        getattr(train_args, "forcing_channels_used", None)
        or getattr(train_args, "forcing_channels", None)
        or args.forcing_channels
        or _infer_forcing_channels(split)
    )
    return FNO2D(
        n_x=n_x,
        n_y=n_y,
        state_channels=state_channels,
        forcing_channels=forcing_channels,
        width=int(getattr(train_args, "width", args.width)),
        n_layers=int(getattr(train_args, "fno_layers", args.fno_layers)),
        modes_x=int(getattr(train_args, "fno_modes_x", args.fno_modes_x)),
        modes_y=int(getattr(train_args, "fno_modes_y", args.fno_modes_y)),
        use_forcing_channel=not bool(getattr(train_args, "disable_forcing_channel", args.disable_forcing_channel)),
        use_dt_channel=bool(getattr(train_args, "use_dt_channel", args.use_dt_channel)),
        use_grid_features=not bool(getattr(train_args, "disable_fno_grid", args.disable_fno_grid)),
        default_dt=dt,
        residual=not bool(getattr(train_args, "no_residual", args.no_residual)),
        #lift_noise_std=float(getattr(train_args, "lift_noise_std", args.lift_noise_std)),
        #lift_noise_corr_length=float(getattr(train_args, "lift_noise_corr_length", args.lift_noise_corr_length)),
        #lift_noise_decay_s=float(getattr(train_args, "lift_noise_decay_s", args.lift_noise_decay_s)),
    )


@torch.no_grad()
def _rollout(model: FNO2D, u0: torch.Tensor, f: torch.Tensor, n_steps: int, dt: float, delta_clip: float | None):
    return rollout_fno_2d(model, u0=u0, f=f, n_steps=n_steps, dt=dt, delta_clip=delta_clip)


@torch.no_grad()
def _evaluate_one_step_mse(model: FNO2D, split: Dict[str, torch.Tensor], device: str, dt: float) -> float:
    u_traj = split["u_traj"].to(device)
    f = split["f"].to(device)
    total_sq = 0.0
    n_elem = 0
    for k in range(int(u_traj.shape[1] - 1)):
        u_pred = model.predict_step(u_traj[:, k], f, dt=dt)
        u_ref = u_traj[:, k + 1]
        total_sq += F.mse_loss(u_pred, u_ref, reduction="sum").item()
        n_elem += int(np.prod(u_ref.shape))
    return total_sq / max(1, n_elem)


def _spectral_h1_norm_2d(u: torch.Tensor) -> torch.Tensor:
    n_x = int(u.shape[-2])
    n_y = int(u.shape[-1])
    u_hat = torch.fft.fft2(u, dim=(-2, -1), norm="ortho")
    real_dtype = u.real.dtype
    kx = 2.0 * torch.pi * torch.fft.fftfreq(n_x, d=1.0 / float(n_x), device=u.device).to(dtype=real_dtype)
    ky = 2.0 * torch.pi * torch.fft.fftfreq(n_y, d=1.0 / float(n_y), device=u.device).to(dtype=real_dtype)
    kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing="ij")
    weight = 1.0 + kx_grid.square() + ky_grid.square()
    power = u_hat.real.square() + u_hat.imag.square()
    return torch.sqrt(torch.sum(power * weight, dim=(-2, -1)))


def _state_reduce_dims(u: torch.Tensor) -> tuple[int, ...]:
    return (-3, -2, -1) if u.dim() == 5 else (-2, -1)


@torch.no_grad()
def _evaluate_rollout_curves(
    model: FNO2D,
    split: Dict[str, torch.Tensor],
    device: str,
    dt: float,
    area: float,
    delta_clip: float | None,
) -> Dict[str, np.ndarray]:
    u0 = split["u0"].to(device)
    f = split["f"].to(device)
    u_ref = split["u_traj"].to(device)
    u_pred = _rollout(model, u0=u0, f=f, n_steps=int(u_ref.shape[1] - 1), dt=dt, delta_clip=delta_clip)

    diff = u_pred - u_ref
    reduce_dims = _state_reduce_dims(u_ref)
    num = torch.sqrt(area * torch.sum(diff * diff, dim=reduce_dims))
    den = torch.sqrt(area * torch.sum(u_ref * u_ref, dim=reduce_dims))
    h1_num = _spectral_h1_norm_2d(diff)
    h1_den = _spectral_h1_norm_2d(u_ref)
    if h1_num.dim() == 3:
        h1_num = h1_num.sum(dim=-1)
        h1_den = h1_den.sum(dim=-1)
    num = num[:, 1:]
    den = den[:, 1:]
    h1_num = h1_num[:, 1:]
    h1_den = h1_den[:, 1:]
    rel = num / (den + 1e-8)
    rel_h1 = h1_num / (h1_den + 1e-12)
    overall_rel_l2_samples = (torch.sqrt(torch.sum(num.square(), dim=1)) / (torch.sqrt(torch.sum(den.square(), dim=1)) + 1e-8)).detach().cpu().numpy().astype(np.float64)
    overall_rel_h1_samples = (torch.sqrt(torch.sum(h1_num.square(), dim=1)) / (torch.sqrt(torch.sum(h1_den.square(), dim=1)) + 1e-12)).detach().cpu().numpy().astype(np.float64)
    rel_cpu = rel.detach().cpu()
    rel_h1_cpu = rel_h1.detach().cpu()
    rel_curve_mean = torch.nanmean(rel_cpu, dim=0).numpy().astype(np.float64)
    rel_curve_median = np.nanmedian(rel_cpu.numpy(), axis=0).astype(np.float64)
    rel_h1_curve_mean = torch.nanmean(rel_h1_cpu, dim=0).numpy().astype(np.float64)
    rel_h1_curve_median = np.nanmedian(rel_h1_cpu.numpy(), axis=0).astype(np.float64)
    rollout_rel_l2 = float(np.nanmean(overall_rel_l2_samples))
    rollout_rel_h1 = float(np.nanmean(overall_rel_h1_samples))
    diagnostics = compute_rollout_diagnostics(u_pred.detach().cpu(), u_ref.detach().cpu(), area=area)

    return {
        "rel_curve_mean": rel_curve_mean,
        "rel_curve_median": rel_curve_median,
        "rollout_rel_mean": rollout_rel_l2,
        "rollout_rel_median": float(np.nanmedian(overall_rel_l2_samples)),
        "rollout_rel_std": float(np.nanstd(overall_rel_l2_samples)),
        "rollout_rel_max": float(np.nanmax(rel_curve_mean)),
        "rel_h1_curve_mean": rel_h1_curve_mean,
        "rel_h1_curve_median": rel_h1_curve_median,
        "rollout_rel_h1": rollout_rel_h1,
        "rollout_rel_h1_median": float(np.nanmedian(overall_rel_h1_samples)),
        "rollout_rel_h1_std": float(np.nanstd(overall_rel_h1_samples)),
        "rollout_rel_h1_max": float(np.nanmax(rel_h1_curve_mean)),
        "rel_samples": rel_cpu.numpy().astype(np.float64),
        "rel_h1_samples": rel_h1_cpu.numpy().astype(np.float64),
        "overall_rel_l2_samples": overall_rel_l2_samples,
        "overall_rel_h1_samples": overall_rel_h1_samples,
        "overall_rel_l2": rollout_rel_l2,
        "overall_rel_h1": rollout_rel_h1,
        **diagnostics,
    }


def _sample_stats(values: np.ndarray) -> Dict[str, object]:
    val = np.asarray(values)
    return {"vorticity": val.tolist(), "mean": val.tolist()}


def _save_per_sample_errors_json(curves: Dict[str, np.ndarray], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rel_l2 = curves["rel_samples"]
    rel_h1 = curves["rel_h1_samples"]
    overall_l2 = curves["overall_rel_l2_samples"]
    overall_h1 = curves["overall_rel_h1_samples"]
    items = []
    for sample_idx in range(int(rel_l2.shape[0])):
        items.append({
            "sample_index": sample_idx,
            "rel_l2": _sample_stats(rel_l2[sample_idx]),
            "rel_h1": _sample_stats(rel_h1[sample_idx]),
            "overall_rel_l2": _sample_stats(overall_l2[sample_idx]),
            "overall_rel_h1": _sample_stats(overall_h1[sample_idx]),
        })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)


def _save_rollout_curve_csv(curves: Dict[str, np.ndarray], time_values: np.ndarray, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step",
            "time",
            "rel_l2_mean",
            "rel_l2_median",
            "rel_h1_mean",
            "rel_h1_median",
            "enstrophy_rel_mean",
            "palinstrophy_rel_mean",
            "spectrum_rel_mean",
        ])
        for k in range(len(curves["rel_curve_mean"])):
            writer.writerow([
                k,
                f"{float(time_values[k + 1]):.8f}",
                f"{float(curves['rel_curve_mean'][k]):.12e}",
                f"{float(curves['rel_curve_median'][k]):.12e}",
                f"{float(curves['rel_h1_curve_mean'][k]):.12e}",
                f"{float(curves['rel_h1_curve_median'][k]):.12e}",
                f"{float(curves['enstrophy_rel_curve_mean'][k]):.12e}",
                f"{float(curves['palinstrophy_rel_curve_mean'][k]):.12e}",
                f"{float(curves['spectrum_rel_curve_mean'][k]):.12e}",
            ])
    print(f"Saved rollout curve csv: {out_path}")


def _plot_rollout_curves(curves: Dict[str, np.ndarray], time_values: np.ndarray, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping curve plotting because matplotlib is unavailable: {exc}")
        return

    t = time_values[1 : 1 + curves["rel_curve_mean"].shape[0]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    panels = [
        (axes[0, 0], "rel_curve_mean", "relative L2", "L2", curves["rollout_rel_mean"], "tab:orange"),
        (axes[0, 1], "rel_h1_curve_mean", "relative H1", "H1", curves["rollout_rel_h1"], "tab:red"),
        (axes[1, 0], "enstrophy_rel_curve_mean", "relative enstrophy", "Enstrophy", curves["enstrophy_rel_error"], "tab:blue"),
        (axes[1, 1], "palinstrophy_rel_curve_mean", "relative palinstrophy", "Palinstrophy", curves["palinstrophy_rel_error"], "tab:purple"),
    ]
    for ax, key, ylabel, title, aggregate, color in panels:
        ax.plot(t, curves[key], linewidth=2, color=color)
        ax.set_title(f"FNO Rollout {title}\nagg={aggregate:.4e}")
        ax.set_xlabel("time")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved rollout curve plot: {out_path}")


@torch.no_grad()
def _plot_test_samples(
    model: FNO2D,
    split: Dict[str, torch.Tensor],
    device: str,
    dt: float,
    time_values: np.ndarray,
    snapshot_times: List[float],
    n_plot_samples: int,
    out_dir: str,
    delta_clip: float | None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping sample plotting because matplotlib is unavailable: {exc}")
        return

    os.makedirs(out_dir, exist_ok=True)
    u_traj = split["u_traj"]
    u0 = split["u0"]
    f = split["f"]
    total = int(u_traj.shape[0])
    n_plot = min(max(1, int(n_plot_samples)), total)
    sample_ids = torch.linspace(0, total - 1, n_plot).long().tolist()
    n_steps = int(u_traj.shape[1] - 1)
    n_x = int(u_traj.shape[-2])
    n_y = int(u_traj.shape[-1])
    n_cols = 1 + len(snapshot_times)
    error_snapshot_times = [t for t in snapshot_times if t > 0.0]
    error_start_col = 2 if snapshot_times and snapshot_times[0] <= 0.0 else 1
    area = (1.0 / float(n_x)) * (1.0 / float(n_y))

    for sample_id in sample_ids:
        u0_i = u0[sample_id : sample_id + 1].to(device)
        f_i = f[sample_id : sample_id + 1].to(device)
        u_ref_i = u_traj[sample_id]
        u_pred_i = _rollout(model, u0_i, f_i, n_steps=n_steps, dt=dt, delta_clip=delta_clip)[0].cpu()
        diff_i = u_pred_i - u_ref_i
        reduce_dims = (-3, -2, -1) if u_ref_i.dim() == 4 else (-2, -1)
        num_i = torch.sqrt(area * torch.sum(diff_i * diff_i, dim=reduce_dims))
        den_i = torch.sqrt(area * torch.sum(u_ref_i * u_ref_i, dim=reduce_dims))
        rel_curve_i = num_i / (den_i + 1e-8)
        rel_mean_i = float(rel_curve_i.mean().item())
        rel_final_i = float(rel_curve_i[-1].item())

        f_plot = f[sample_id]
        if f_plot.dim() == 3:
            f_plot = f_plot[0]
        state_ref = u_ref_i[:, 0] if u_ref_i.dim() == 4 else u_ref_i
        state_pred = u_pred_i[:, 0] if u_pred_i.dim() == 4 else u_pred_i
        state_scale = max(float(torch.max(torch.abs(state_ref)).item()), float(torch.max(torch.abs(state_pred)).item()), 1e-8)

        fig, axes = plt.subplots(3, n_cols, figsize=(3.1 * n_cols, 8.0), squeeze=False, constrained_layout=True)
        im_force = axes[0, 0].imshow(f_plot.cpu().numpy(), origin="lower", cmap="coolwarm", extent=[0.0, 1.0, 0.0, 1.0], aspect="auto")
        axes[0, 0].set_title("forcing")
        axes[0, 0].set_xticks([])
        axes[0, 0].set_yticks([])
        axes[1, 0].axis("off")
        axes[1, 0].text(0.5, 0.5, "pred", ha="center", va="center", fontsize=12)
        axes[2, 0].axis("off")
        axes[2, 0].text(0.5, 0.5, "abs error", ha="center", va="center", fontsize=12)
        if snapshot_times and snapshot_times[0] <= 0.0:
            axes[2, 1].axis("off")
            axes[2, 1].text(0.5, 0.5, "t=0 exact init", ha="center", va="center", fontsize=11)

        im_ref_last = im_force
        im_err_last = None
        for j, t_snap in enumerate(snapshot_times, start=1):
            k = int(np.argmin(np.abs(time_values - float(t_snap))))
            k = max(0, min(n_steps, k))
            t_label = float(time_values[k])
            u_ref_k = state_ref[k]
            u_pred_k = state_pred[k]
            err_k = torch.abs(u_pred_k - u_ref_k)
            rel_k = float(rel_curve_i[k].item())
            ax_ref = axes[0, j]
            ax_pred = axes[1, j]
            ax_err = axes[2, j]
            im_ref = ax_ref.imshow(u_ref_k.cpu().numpy(), origin="lower", cmap="coolwarm", vmin=-state_scale, vmax=state_scale, extent=[0.0, 1.0, 0.0, 1.0], aspect="auto")
            ax_ref.set_title(f"ref t={t_label:g}")
            ax_ref.set_xticks([])
            ax_ref.set_yticks([])
            ax_pred.imshow(u_pred_k.cpu().numpy(), origin="lower", cmap="coolwarm", vmin=-state_scale, vmax=state_scale, extent=[0.0, 1.0, 0.0, 1.0], aspect="auto")
            ax_pred.set_title(f"pred t={t_label:g}")
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])
            im_ref_last = im_ref
            if k > 0:
                im_err = ax_err.imshow(err_k.cpu().numpy(), origin="lower", cmap="magma", extent=[0.0, 1.0, 0.0, 1.0], aspect="auto")
                ax_err.set_xticks([])
                ax_err.set_yticks([])
                im_err_last = im_err
                ax_err.set_title(f"|err| t={t_label:g}\nrelL2={rel_k:.3e}")
            else:
                ax_err.axis("off")
                ax_err.text(0.5, 0.5, f"t={t_label:g} exact init", ha="center", va="center", fontsize=11)

        cbar_state = fig.colorbar(im_ref_last, ax=axes[0:2, 1:], fraction=0.015, pad=0.01)
        cbar_state.ax.set_ylabel("state value", rotation=90)
        if im_err_last is not None and len(error_snapshot_times) > 0:
            cbar_err = fig.colorbar(im_err_last, ax=axes[2, error_start_col:], fraction=0.015, pad=0.01)
            cbar_err.ax.set_ylabel("abs error", rotation=90)
        cbar_f = fig.colorbar(im_force, ax=[axes[0, 0]], fraction=0.046, pad=0.02)
        cbar_f.ax.set_ylabel("forcing", rotation=90)
        fig.suptitle(
            f"FNO sample {sample_id}: forcing + reference/prediction snapshots | "
            f"relL2 mean={rel_mean_i:.3e}, final={rel_final_i:.3e}",
            fontsize=13,
        )
        out_path = os.path.join(out_dir, f"sample_{sample_id:04d}_comparison.png")
        fig.savefig(out_path, dpi=180)
        plt.close(fig)

    print(f"Saved sample plots: {out_dir}")


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed, seed_cuda=(device == "cuda"))
    splits = load_dataset_splits(args.dataset_path, map_location="cpu")
    split = splits[args.split]
    meta = splits.get("meta", {})
    n_x = int(split["u0"].shape[-2])
    n_y = int(split["u0"].shape[-1])
    n_steps = int(split["u_traj"].shape[1] - 1)
    h_x = 1.0 / float(n_x)
    h_y = 1.0 / float(n_y)
    area = h_x * h_y
    dt, t_start, t_final, time_values = _time_metadata(meta, n_steps=n_steps)
    if args.max_steps is not None:
        n_steps = min(int(args.max_steps), n_steps)
        split = {**split, "u_traj": split["u_traj"][:, : n_steps + 1], "u0": split["u_traj"][:, 0].clone()}
        time_values = time_values[: n_steps + 1]
        t_final = float(time_values[-1])
    snapshot_times = _parse_snapshot_times(args.snapshot_times, t_start=t_start, t_end=t_final)
    delta_clip = args.delta_clip if args.delta_clip > 0.0 else None

    train_args = _load_train_args(args.checkpoint_path)
    model = _build_model(n_x, n_y, dt, split, args, train_args).to(device)
    ckpt = _load_checkpoint(args.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    print(f"Device: {device}")
    print(f"Loaded split={args.split} from {args.dataset_path}")
    print(f"Grid: n_x={n_x}, n_y={n_y}, n_steps={n_steps}, stored_time=[{t_start:.6f},{t_final:.6f}], dt={dt:.6f}")
    print(f"Delta clip (L-inf): {args.delta_clip:.6f}")
    print(f"Loaded checkpoint: {args.checkpoint_path}")

    step_mse = _evaluate_one_step_mse(model, split, device=device, dt=dt)
    curves = _evaluate_rollout_curves(model, split, device=device, dt=dt, area=area, delta_clip=delta_clip)
    print(f"Split one-step MSE: {step_mse:.8e}")
    print(f"Split rollout mean relative L2: {curves['rollout_rel_mean']:.8e}")
    print(f"Split rollout median relative L2: {curves['rollout_rel_median']:.8e}")
    print(f"Split rollout mean relative H1: {curves['rollout_rel_h1']:.8e}")
    print(f"Split rollout median relative H1: {curves['rollout_rel_h1_median']:.8e}")
    print(f"Split rollout enstrophy relative error: {curves['enstrophy_rel_error']:.8e}")
    print(f"Split rollout palinstrophy relative error: {curves['palinstrophy_rel_error']:.8e}")
    print(f"Split rollout spectrum relative error: {curves['spectrum_rel_error']:.8e}")
    print(f"Split rollout std relative L2: {curves['rollout_rel_std']:.8e}")
    print(f"Split rollout std relative H1: {curves['rollout_rel_h1_std']:.8e}")
    print(f"Split max rollout curve relative L2: {curves['rollout_rel_max']:.8e}")
    print(f"Split max rollout curve relative H1: {curves['rollout_rel_h1_max']:.8e}")
    print("Rollout accumulation by step (step, time, rel_l2_mean, rel_l2_median, rel_h1_mean, rel_h1_median):")
    for k in range(len(curves["rel_curve_mean"])):
        print(
            f"  {k + 1:03d}  {float(time_values[k + 1]):8.4f}  "
            f"{curves['rel_curve_mean'][k]:.8e}  {curves['rel_curve_median'][k]:.8e}  "
            f"{curves['rel_h1_curve_mean'][k]:.8e}  {curves['rel_h1_curve_median'][k]:.8e}"
        )

    curve_csv = os.path.join(args.output_dir, f"{args.split}_rollout_error_curve.csv")
    curve_png = os.path.join(args.output_dir, f"{args.split}_rollout_error_curve.png")
    _save_rollout_curve_csv(curves, time_values=time_values, out_path=curve_csv)
    _plot_rollout_curves(curves, time_values=time_values, out_path=curve_png)
    per_sample_path = os.path.join(args.output_dir, f"{args.split}_per_sample_errors.json")
    _save_per_sample_errors_json(curves, out_path=per_sample_path)
    print(f"Saved per-sample errors: {per_sample_path}")

    sample_dir = os.path.join(args.output_dir, f"{args.split}_sample_comparisons")
    _plot_test_samples(
        model=model,
        split=split,
        device=device,
        dt=dt,
        time_values=time_values,
        snapshot_times=snapshot_times,
        n_plot_samples=args.n_plot_samples,
        out_dir=sample_dir,
        delta_clip=delta_clip,
    )

    summary = {
        "dataset_path": args.dataset_path,
        "checkpoint_path": args.checkpoint_path,
        "split": args.split,
        "n_x": n_x,
        "n_y": n_y,
        "n_steps": n_steps,
        "dt": dt,
        "t_start": t_start,
        "t_final": t_final,
        "time_values": time_values.tolist(),
        "error_time_values": time_values[1 : 1 + len(curves["rel_curve_mean"])].tolist(),
        "delta_clip": args.delta_clip,
        "step_mse": step_mse,
        "rollout_rel_l2": curves["rollout_rel_mean"],
        "rollout_rel_l2_median": curves["rollout_rel_median"],
        "rollout_rel_l2_std": curves["rollout_rel_std"],
        "rollout_rel_l2_max": curves["rollout_rel_max"],
        "overall_rel_l2": curves["overall_rel_l2"],
        "rollout_rel_h1": curves["rollout_rel_h1"],
        "rollout_rel_h1_median": curves["rollout_rel_h1_median"],
        "rollout_rel_h1_std": curves["rollout_rel_h1_std"],
        "rollout_rel_h1_max": curves["rollout_rel_h1_max"],
        "overall_rel_h1": curves["overall_rel_h1"],
        "enstrophy_rel_error": curves["enstrophy_rel_error"],
        "palinstrophy_rel_error": curves["palinstrophy_rel_error"],
        "spectrum_rel_error": curves["spectrum_rel_error"],
        "field_names": ["vorticity"],
        "rel_curve_mean": curves["rel_curve_mean"].tolist(),
        "rel_curve_median": curves["rel_curve_median"].tolist(),
        "rel_h1_curve_mean": curves["rel_h1_curve_mean"].tolist(),
        "rel_h1_curve_median": curves["rel_h1_curve_median"].tolist(),
        "enstrophy_rel_curve_mean": curves["enstrophy_rel_curve_mean"].tolist(),
        "palinstrophy_rel_curve_mean": curves["palinstrophy_rel_curve_mean"].tolist(),
        "spectrum_rel_curve_mean": curves["spectrum_rel_curve_mean"].tolist(),
        "snapshot_times": snapshot_times,
        "max_steps": args.max_steps,
        "seed": int(args.seed),
        "meta": meta,
    }
    summary_path = os.path.join(args.output_dir, f"{args.split}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved evaluation summary: {summary_path}")
    print(f"Saved evaluation artifacts to: {args.output_dir}")


if __name__ == "__main__":
    main(parse_args())
