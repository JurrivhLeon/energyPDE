"""
Evaluation entrypoint for periodic 2D Navier-Stokes latent VAE checkpoints.
"""

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
from torch.utils.data import DataLoader

try:
    from ..heat_data import load_dataset_splits
    from .ns2d_data import (
        build_navier_stokes2d_periodic_step_dataset,
        build_navier_stokes2d_periodic_trajectory_dataset_from_split,
    )
    from .train_vae import (
        PeriodicLatentVAETrainer2D,
        _build_model,
        _unpack_traj_batch,
        rollout_vae_latent_mean,
        rollout_vae_mean,
    )
    from .rollout_diagnostics import compute_rollout_diagnostics
except ImportError:
    from grad_flow_l2.heat_data import load_dataset_splits
    from grad_flow_l2.ns2d_per.ns2d_data import (
        build_navier_stokes2d_periodic_step_dataset,
        build_navier_stokes2d_periodic_trajectory_dataset_from_split,
    )
    from grad_flow_l2.ns2d_per.train_vae import (
        PeriodicLatentVAETrainer2D,
        _build_model,
        _unpack_traj_batch,
        rollout_vae_latent_mean,
        rollout_vae_mean,
    )
    from grad_flow_l2.ns2d_per.rollout_diagnostics import compute_rollout_diagnostics


def _torch_load_checkpoint(checkpoint_path: str, map_location):
    try:
        return torch.load(checkpoint_path, map_location=map_location)
    except RuntimeError as exc:
        msg = str(exc)
        if "weights_only=True" not in msg or "legacy .tar format" not in msg:
            raise
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def set_seed(seed: int, seed_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda:
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate periodic 2D Navier-Stokes latent VAE checkpoint")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt or final_model.pt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--n-plot-samples", type=int, default=20)
    parser.add_argument("--snapshot-times", type=str, default="4,8,12,16,20")
    parser.add_argument("--output-dir", type=str, default="grad_flow_l2/ns2d_per/outputs_vae/eval")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rollout-mode",
        type=str,
        default="latent",
        choices=["physical", "latent"],
        help="Rollout path: physical re-encodes each decoded state; latent encodes u0 once and advances in latent space.",
    )
    parser.add_argument(
        "--delta-clip",
        type=float,
        default=10.0,
        help="Optional L-infinity clip on decoded rollout increments. Use <=0 to disable.",
    )
    parser.add_argument(
        "--state-clip",
        type=float,
        default=20.0,
        help="Optional L-infinity clip on decoded rollout states. Use <=0 to disable.",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="If set, truncate trajectories to this many steps (e.g. 30 keeps snapshots T=0..30).",
    )
    return parser.parse_args()


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
        # Prefer the actual stored cadence for indexing; keep metadata visible in
        # the summary so mismatches are easy to diagnose.
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
        vals = [t_start + frac * horizon for frac in (0.2, 0.4, 0.6, 0.8, 1.0)]
    return vals




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


@torch.no_grad()
def _evaluate_rollout_curves(
    model,
    traj_loader: DataLoader,
    device: str,
    dt: float,
    area: float,
    delta_clip: float = 0.0,
    state_clip: float = 0.0,
    rollout_mode: str = "latent",
) -> Dict[str, np.ndarray]:
    rel_batches = []
    rel_h1_batches = []
    overall_rel_l2_batches = []
    overall_rel_h1_batches = []
    l2_num_sum = torch.zeros((), device=device)
    l2_den_sum = torch.zeros((), device=device)
    h1_num_sum = torch.zeros((), device=device)
    h1_den_sum = torch.zeros((), device=device)
    pred_batches = []
    ref_batches = []
    for batch in traj_loader:
        u0, f, u_ref = _unpack_traj_batch(batch)
        u0 = u0.to(device)
        f = f.to(device)
        u_ref = u_ref.to(device)
        n_steps = int(u_ref.shape[1] - 1)
        if rollout_mode == "latent":
            u_pred = rollout_vae_latent_mean(
                model,
                u0=u0,
                f=f,
                n_steps=n_steps,
                dt=dt,
                state_clip=state_clip,
            )
        else:
            u_pred = rollout_vae_mean(
                model,
                u0=u0,
                f=f,
                n_steps=n_steps,
                dt=dt,
                delta_clip=delta_clip,
                state_clip=state_clip,
            )
        pred_batches.append(u_pred.detach().cpu())
        ref_batches.append(u_ref.detach().cpu())
        diff = u_pred - u_ref
        num = torch.sqrt(area * torch.sum(diff * diff, dim=(-2, -1)))
        den = torch.sqrt(area * torch.sum(u_ref * u_ref, dim=(-2, -1)))
        h1_num = _spectral_h1_norm_2d(diff)
        h1_den = _spectral_h1_norm_2d(u_ref)
        rel_batches.append((num / (den + 1e-8)).detach().cpu())
        rel_h1_batches.append((h1_num / (h1_den + 1e-12)).detach().cpu())
        overall_rel_l2_batches.append((num.sum(dim=1) / (den.sum(dim=1) + 1e-8)).detach().cpu())
        overall_rel_h1_batches.append((h1_num.sum(dim=1) / (h1_den.sum(dim=1) + 1e-12)).detach().cpu())
        l2_num_sum += num.sum()
        l2_den_sum += den.sum()
        h1_num_sum += h1_num.sum()
        h1_den_sum += h1_den.sum()

    rel = torch.cat(rel_batches, dim=0)
    rel_h1 = torch.cat(rel_h1_batches, dim=0)
    overall_rel_l2_samples = torch.cat(overall_rel_l2_batches, dim=0).numpy().astype(np.float64)
    overall_rel_h1_samples = torch.cat(overall_rel_h1_batches, dim=0).numpy().astype(np.float64)
    rel_curve_mean = torch.nanmean(rel, dim=0).numpy().astype(np.float64)
    rel_curve_median = np.nanmedian(rel.numpy(), axis=0).astype(np.float64)
    rel_h1_curve_mean = torch.nanmean(rel_h1, dim=0).numpy().astype(np.float64)
    rel_h1_curve_median = np.nanmedian(rel_h1.numpy(), axis=0).astype(np.float64)
    rollout_rel_l2 = float((l2_num_sum / (l2_den_sum + 1e-8)).item())
    rollout_rel_h1 = float((h1_num_sum / (h1_den_sum + 1e-12)).item())
    diagnostics = compute_rollout_diagnostics(
        torch.cat(pred_batches, dim=0),
        torch.cat(ref_batches, dim=0),
        area=area,
    )

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
        "rel_samples": rel.numpy().astype(np.float64),
        "rel_h1_samples": rel_h1.numpy().astype(np.float64),
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
        writer.writerow(
            [
                "step",
                "time",
                "rel_l2_mean",
                "rel_l2_median",
                "rel_h1_mean",
                "rel_h1_median",
                "enstrophy_rel_mean",
                "palinstrophy_rel_mean",
                "spectrum_rel_mean",
            ]
        )
        for k in range(len(curves["rel_curve_mean"])):
            writer.writerow(
                [
                    k,
                    f"{float(time_values[k]):.8f}",
                    f"{float(curves['rel_curve_mean'][k]):.12e}",
                    f"{float(curves['rel_curve_median'][k]):.12e}",
                    f"{float(curves['rel_h1_curve_mean'][k]):.12e}",
                    f"{float(curves['rel_h1_curve_median'][k]):.12e}",
                    f"{float(curves['enstrophy_rel_curve_mean'][k]):.12e}",
                    f"{float(curves['palinstrophy_rel_curve_mean'][k]):.12e}",
                    f"{float(curves['spectrum_rel_curve_mean'][k]):.12e}",
                ]
            )
    print(f"Saved rollout curve csv: {out_path}")


def _plot_rollout_curves(curves: Dict[str, np.ndarray], time_values: np.ndarray, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping curve plotting because matplotlib is unavailable: {exc}")
        return

    t = time_values[: curves["rel_curve_mean"].shape[0]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    panels = [
        (axes[0, 0], "rel_curve_mean", "relative L2", "L2", curves["rollout_rel_mean"], "tab:orange"),
        (axes[0, 1], "rel_h1_curve_mean", "relative H1", "H1", curves["rollout_rel_h1"], "tab:red"),
        (axes[1, 0], "enstrophy_rel_curve_mean", "relative enstrophy", "Enstrophy", curves["enstrophy_rel_error"], "tab:blue"),
        (axes[1, 1], "palinstrophy_rel_curve_mean", "relative palinstrophy", "Palinstrophy", curves["palinstrophy_rel_error"], "tab:purple"),
    ]
    for ax, key, ylabel, title, aggregate, color in panels:
        ax.plot(t, curves[key], linewidth=2, color=color)
        ax.set_title(f"VAE Rollout {title}\nagg={aggregate:.4e}")
        ax.set_xlabel("time")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved rollout curve plot: {out_path}")


@torch.no_grad()
def _plot_sample_trajectories(
    model,
    split: Dict[str, torch.Tensor],
    device: str,
    dt: float,
    time_values: np.ndarray,
    area: float,
    snapshot_times: List[float],
    n_plot_samples: int,
    out_dir: str,
    delta_clip: float = 0.0,
    state_clip: float = 0.0,
    rollout_mode: str = "physical",
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
    n_cols = 1 + len(snapshot_times)

    for sample_id in sample_ids:
        u0_i = u0[sample_id : sample_id + 1].to(device)
        f_i = f[sample_id : sample_id + 1].to(device)
        u_ref_i = u_traj[sample_id]
        if rollout_mode == "latent":
            u_pred_i = rollout_vae_latent_mean(
                model,
                u0_i,
                f_i,
                n_steps=n_steps,
                dt=dt,
                state_clip=state_clip,
            )[0].cpu()
        else:
            u_pred_i = rollout_vae_mean(
                model,
                u0_i,
                f_i,
                n_steps=n_steps,
                dt=dt,
                delta_clip=delta_clip,
                state_clip=state_clip,
            )[0].cpu()

        diff_i = u_pred_i - u_ref_i
        num_i = torch.sqrt(area * torch.sum(diff_i * diff_i, dim=(-2, -1)))
        den_i = torch.sqrt(area * torch.sum(u_ref_i * u_ref_i, dim=(-2, -1)))
        rel_curve_i = num_i / (den_i + 1e-8)
        rel_mean_i = float(rel_curve_i.mean().item())
        rel_final_i = float(rel_curve_i[-1].item())

        f_plot = f[sample_id].cpu().numpy()
        state_scale = max(float(torch.max(torch.abs(u_ref_i)).item()), float(torch.max(torch.abs(u_pred_i)).item()), 1e-8)

        fig, axes = plt.subplots(3, n_cols, figsize=(3.1 * n_cols, 8.0), squeeze=False, constrained_layout=True)
        im_force = axes[0, 0].imshow(f_plot, origin="lower", cmap="coolwarm", extent=[0.0, 1.0, 0.0, 1.0])
        axes[0, 0].set_title("forcing")
        axes[0, 0].set_xticks([])
        axes[0, 0].set_yticks([])
        axes[1, 0].axis("off")
        axes[1, 0].text(0.5, 0.5, "pred", ha="center", va="center", fontsize=12)
        axes[2, 0].axis("off")
        axes[2, 0].text(0.5, 0.5, "abs error", ha="center", va="center", fontsize=12)

        im_ref_last = im_force
        im_err_last = None
        for j, t_snap in enumerate(snapshot_times, start=1):
            k = int(np.argmin(np.abs(time_values - float(t_snap))))
            k = max(0, min(n_steps, k))
            t_label = float(time_values[k])

            u_ref_k = u_ref_i[k]
            u_pred_k = u_pred_i[k]
            err_k = torch.abs(u_pred_k - u_ref_k)
            rel_k = float(rel_curve_i[k].item())

            im_ref = axes[0, j].imshow(
                u_ref_k.cpu().numpy(),
                origin="lower",
                cmap="coolwarm",
                vmin=-state_scale,
                vmax=state_scale,
                extent=[0.0, 1.0, 0.0, 1.0],
            )
            axes[0, j].set_title(f"ref t={t_label:g}")
            axes[0, j].set_xticks([])
            axes[0, j].set_yticks([])

            axes[1, j].imshow(
                u_pred_k.cpu().numpy(),
                origin="lower",
                cmap="coolwarm",
                vmin=-state_scale,
                vmax=state_scale,
                extent=[0.0, 1.0, 0.0, 1.0],
            )
            axes[1, j].set_title(f"pred t={t_label:g}")
            axes[1, j].set_xticks([])
            axes[1, j].set_yticks([])

            im_err = axes[2, j].imshow(
                err_k.cpu().numpy(),
                origin="lower",
                cmap="magma",
                extent=[0.0, 1.0, 0.0, 1.0],
            )
            axes[2, j].set_title(f"|err| t={t_label:g}\nrelL2={rel_k:.3e}")
            axes[2, j].set_xticks([])
            axes[2, j].set_yticks([])
            im_ref_last = im_ref
            im_err_last = im_err

        cbar_state = fig.colorbar(im_ref_last, ax=axes[0:2, 1:], fraction=0.015, pad=0.01)
        cbar_state.ax.set_ylabel("state value", rotation=90)
        if im_err_last is not None:
            cbar_err = fig.colorbar(im_err_last, ax=axes[2, 1:], fraction=0.015, pad=0.01)
            cbar_err.ax.set_ylabel("abs error", rotation=90)
        cbar_f = fig.colorbar(im_force, ax=[axes[0, 0]], fraction=0.046, pad=0.02)
        cbar_f.ax.set_ylabel("forcing", rotation=90)

        fig.suptitle(
            f"VAE sample {sample_id}: deterministic rollout snapshots | "
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
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    run_dir = os.path.dirname(args.checkpoint)
    args_path = os.path.join(run_dir, "args.json")
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"Training args not found next to checkpoint: {args_path}")
    with open(args_path, "r", encoding="utf-8") as f:
        train_args = Namespace(**json.load(f))

    splits = load_dataset_splits(args.dataset_path, map_location="cpu")
    split = splits[args.split]
    meta = splits.get("meta", {})
    n_x = int(split["u0"].shape[1])
    n_y = int(split["u0"].shape[2])
    n_steps = int(split["u_traj"].shape[1] - 1)
    dt, t_start, t_final, time_values = _time_metadata(meta, n_steps=n_steps)
    if args.max_steps is not None:
        n_steps = min(int(args.max_steps), n_steps)
        split["u_traj"] = split["u_traj"][:, : n_steps + 1]
        time_values = time_values[: n_steps + 1]
        t_final = float(time_values[-1])
    h_x = 1.0 / float(n_x)
    h_y = 1.0 / float(n_y)
    area = h_x * h_y
    snapshot_times = _parse_snapshot_times(args.snapshot_times, t_start=t_start, t_end=t_final)

    model = _build_model(n_x=n_x, n_y=n_y, dt=dt, args=train_args).to(device)
    checkpoint = _torch_load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    step_loader = DataLoader(
        build_navier_stokes2d_periodic_step_dataset(split),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
    )
    traj_loader = DataLoader(
        build_navier_stokes2d_periodic_trajectory_dataset_from_split(split),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
    )
    trainer = PeriodicLatentVAETrainer2D(
        model=model,
        dt=dt,
        h_x=h_x,
        h_y=h_y,
        beta_kl=getattr(train_args, "beta_kl", checkpoint.get("beta_kl", 1e-4)),
        lambda_rec=getattr(train_args, "lambda_rec", checkpoint.get("lambda_rec", 1.0)),
        alpha_min=getattr(train_args, "alpha_min", checkpoint.get("alpha_min", 1e-4)),
        alpha_max=getattr(train_args, "alpha_max", checkpoint.get("alpha_max", 0.5)),
        device=device,
        output_dir=None,
        show_epoch_pbar=False,
        rollout_mode=args.rollout_mode,
    )
    clips_enabled = (args.delta_clip is not None and float(args.delta_clip) > 0.0) or (
        args.state_clip is not None and float(args.state_clip) > 0.0
    )
    metrics = trainer.validate(step_loader, traj_loader=None if clips_enabled else traj_loader)
    curves = _evaluate_rollout_curves(
        model,
        traj_loader,
        device=device,
        dt=dt,
        area=area,
        delta_clip=args.delta_clip,
        state_clip=args.state_clip,
        rollout_mode=args.rollout_mode,
    )
    metrics["rollout_rel_l2"] = curves["rollout_rel_mean"]
    metrics["rollout_rel_l2_median"] = curves["rollout_rel_median"]
    metrics["rollout_rel_l2_std"] = curves["rollout_rel_std"]
    metrics["rollout_rel_l2_max"] = curves["rollout_rel_max"]
    metrics["overall_rel_l2"] = curves["overall_rel_l2"]
    metrics["rollout_rel_h1"] = curves["rollout_rel_h1"]
    metrics["rollout_rel_h1_median"] = curves["rollout_rel_h1_median"]
    metrics["rollout_rel_h1_std"] = curves["rollout_rel_h1_std"]
    metrics["rollout_rel_h1_max"] = curves["rollout_rel_h1_max"]
    metrics["overall_rel_h1"] = curves["overall_rel_h1"]
    metrics["enstrophy_rel_error"] = curves["enstrophy_rel_error"]
    metrics["palinstrophy_rel_error"] = curves["palinstrophy_rel_error"]
    metrics["spectrum_rel_error"] = curves["spectrum_rel_error"]

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(
        f"Split: {args.split}, n={int(split['u0'].shape[0])}, grid=({n_x},{n_y}), "
        f"steps={n_steps}, dt={dt:.6f}, stored_time=[{t_start:.6f},{t_final:.6f}]"
    )
    print(f"Rollout mode: {args.rollout_mode}")
    print(f"Delta clip: {args.delta_clip:.6f}")
    print(f"State clip: {args.state_clip:.6f}")
    print("Metrics:", metrics)
    print(f"Split rollout mean relative H1: {curves['rollout_rel_h1']:.8e}")
    print(f"Split rollout median relative H1: {curves['rollout_rel_h1_median']:.8e}")
    print(f"Split rollout enstrophy relative error: {curves['enstrophy_rel_error']:.8e}")
    print(f"Split rollout palinstrophy relative error: {curves['palinstrophy_rel_error']:.8e}")
    print(f"Split rollout spectrum relative error: {curves['spectrum_rel_error']:.8e}")
    print(f"Split rollout std relative L2: {curves['rollout_rel_std']:.8e}")
    print(f"Split rollout std relative H1: {curves['rollout_rel_h1_std']:.8e}")
    print(f"Split max rollout curve relative L2: {curves['rollout_rel_max']:.8e}")
    print(f"Split max rollout curve relative H1: {curves['rollout_rel_h1_max']:.8e}")
    print(
        "Rollout accumulation by step "
        "(step, time, rel_l2_mean, rel_l2_median, rel_h1_mean, rel_h1_median):"
    )
    for k in range(len(curves["rel_curve_mean"])):
        print(
            f"  {k:03d}  {float(time_values[k]):8.4f}  "
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
    _plot_sample_trajectories(
        model=model,
        split=split,
        device=device,
        dt=dt,
        time_values=time_values,
        area=area,
        snapshot_times=snapshot_times,
        n_plot_samples=args.n_plot_samples,
        out_dir=sample_dir,
        delta_clip=args.delta_clip,
        state_clip=args.state_clip,
        rollout_mode=args.rollout_mode,
    )

    summary = {
        "dataset_path": args.dataset_path,
        "checkpoint_path": args.checkpoint,
        "split": args.split,
        "n_x": n_x,
        "n_y": n_y,
        "n_steps": n_steps,
        "dt": dt,
        "t_start": t_start,
        "t_final": t_final,
        "time_values": time_values.tolist(),
        "metrics": metrics,
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
        "deterministic_dynamics": True,
        "rollout_mode": args.rollout_mode,
        "delta_clip": float(args.delta_clip),
        "state_clip": float(args.state_clip),
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
