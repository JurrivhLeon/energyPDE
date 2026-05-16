"""
Evaluation script for forced damped-driven 1D KdV latent VAE models.

Reports:
1) One-step split MSE.
2) Transition uncertainty proxy from the prior amplitude head.
3) Rollout error accumulation by step (MSE and relative L2).
4) Per-sample reference/prediction comparison plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from ..heat_data import build_step_dataset, build_trajectory_dataset_from_split, load_dataset_splits
    from ..latent_flow_VAE import (
        FNOLatentTransition1D,
        LatentVAE1D,
        StateDecoder1D,
        TransitionAmplitudeHead1D,
        VariationalStateEncoder1D,
    )
    from ..utils import compute_relative_l2_error, pad_dirichlet_1d
    from .train_vae import rollout_vae_mean
except ImportError:
    from grad_flow_l2.heat_data import build_step_dataset, build_trajectory_dataset_from_split, load_dataset_splits
    from grad_flow_l2.latent_flow_VAE import (
        FNOLatentTransition1D,
        LatentVAE1D,
        StateDecoder1D,
        TransitionAmplitudeHead1D,
        VariationalStateEncoder1D,
    )
    from grad_flow_l2.utils import compute_relative_l2_error, pad_dirichlet_1d
    from grad_flow_l2.kdv_1d.train_vae import rollout_vae_mean


def _parse_snapshot_times(raw: str, t_final: float) -> List[float]:
    vals = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        val = float(tok)
        if val < 0.0:
            raise ValueError(f"Snapshot time must be nonnegative, got {val}")
        if val <= float(t_final):
            vals.append(val)
    return vals or [0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0, min(5.0, float(t_final))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained latent VAE KdV model")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="grad_flow_l2/kdv_1d/datasets/kdv_forced_periodic_L32_snx4096_nx512_dt0p1_solverdt0p01_gamma0p1.pt",
        help="Path to cached KdV dataset (.pt)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="grad_flow_l2/kdv_1d/outputs_vae/run_latest/best_model.pt",
        help="Path to trained VAE checkpoint (.pt)",
    )
    parser.add_argument(
        "--args-json",
        type=str,
        default=None,
        help="Optional training args.json. Defaults to args.json next to the checkpoint if present.",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--enc-blocks", type=int, default=4)
    parser.add_argument("--dec-blocks", type=int, default=4)
    parser.add_argument("--fno-width", type=int, default=None)
    parser.add_argument("--fno-layers", type=int, default=6)
    parser.add_argument("--fno-modes", type=int, default=16)
    parser.add_argument("--disable-fno-grid", action="store_true")
    parser.add_argument("--use-dt-channel", action="store_true")
    parser.add_argument(
        "--disable-forcing-channel",
        action="store_true",
        help="Disable the static forcing channel.",
    )
    parser.add_argument("--disable-u-grad-feature", action="store_true")
    parser.add_argument("--amp-head-hidden", type=int, default=32)
    parser.add_argument("--noise-corr-length", type=float, default=1.0)
    parser.add_argument("--noise-decay-s", type=float, default=2.0)
    parser.add_argument("--alpha-min", type=float, default=1e-4)
    parser.add_argument("--alpha-max", type=float, default=0.5)
    parser.add_argument("--alpha-init", type=float, default=0.075)

    parser.add_argument("--n-plot-samples", type=int, default=8)
    parser.add_argument("--snapshot-times", type=str, default="0.2,0.4,0.6,0.8,1.0,2.0,3.0,4.0,5.0")
    parser.add_argument("--output-dir", type=str, default="grad_flow_l2/kdv_1d/outputs_vae/eval")
    args = parser.parse_args()
    return _merge_training_args(args)


def _merge_training_args(args: argparse.Namespace) -> argparse.Namespace:
    args_json = args.args_json
    if args_json is None:
        candidate = os.path.join(os.path.dirname(args.checkpoint_path), "args.json")
        if os.path.exists(candidate):
            args_json = candidate
    if args_json is None or not os.path.exists(args_json):
        return args

    with open(args_json, "r", encoding="utf-8") as f:
        train_args = json.load(f)
    for key in (
        "hidden_channels",
        "latent_channels",
        "enc_blocks",
        "dec_blocks",
        "fno_width",
        "fno_layers",
        "fno_modes",
        "disable_fno_grid",
        "use_dt_channel",
        "disable_forcing_channel",
        "disable_u_grad_feature",
        "amp_head_hidden",
        "noise_corr_length",
        "noise_decay_s",
        "alpha_min",
        "alpha_max",
        "alpha_init",
    ):
        if key in train_args:
            setattr(args, key, train_args[key])
    args.args_json = args_json
    return args


def _build_model(n_x: int, dt: float, boundary_condition: str, args: argparse.Namespace) -> LatentVAE1D:
    use_forcing_channel = not args.disable_forcing_channel
    fno_width = args.hidden_channels if args.fno_width is None else args.fno_width

    encoder = VariationalStateEncoder1D(
        n_x=n_x,
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
        n_blocks=args.enc_blocks,
        use_grad_features=not args.disable_u_grad_feature,
        boundary_condition=boundary_condition,
    )
    decoder = StateDecoder1D(
        n_x=n_x,
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
        n_blocks=args.dec_blocks,
        boundary_condition=boundary_condition,
    )
    transition = FNOLatentTransition1D(
        n_x=n_x,
        latent_channels=args.latent_channels,
        width=fno_width,
        n_layers=args.fno_layers,
        modes=args.fno_modes,
        use_forcing_channel=use_forcing_channel,
        use_dt_channel=args.use_dt_channel,
        use_grid_features=not args.disable_fno_grid,
        default_dt=dt,
    )
    alpha_min = float(getattr(args, "alpha_min", 1e-4))
    alpha_max = float(getattr(args, "alpha_max", 0.5))
    alpha_init = float(getattr(args, "alpha_init", 0.075))
    if not alpha_min < alpha_init < alpha_max:
        raise ValueError("alpha_init must satisfy alpha_min < alpha_init < alpha_max")
    alpha_init_unit = (alpha_init - alpha_min) / (alpha_max - alpha_min)
    alpha_init_logit = float(np.log(alpha_init_unit / (1.0 - alpha_init_unit)))
    amplitude_head = TransitionAmplitudeHead1D(
        n_x=n_x,
        latent_channels=args.latent_channels,
        hidden_channels=args.amp_head_hidden,
        use_forcing_channel=use_forcing_channel,
        boundary_condition=boundary_condition,
        alpha_init_logit=alpha_init_logit,
    )
    return LatentVAE1D(
        encoder=encoder,
        decoder=decoder,
        transition=transition,
        amplitude_head=amplitude_head,
        noise_corr_length=args.noise_corr_length,
        noise_decay_s=args.noise_decay_s,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )


@torch.no_grad()
def evaluate_one_step(
    model: LatentVAE1D,
    step_loader: DataLoader,
    device: str,
    dt: float,
) -> Dict[str, float]:
    model.eval()
    sq_sum = 0.0
    n_elem = 0
    alpha_sum = 0.0
    alpha_sq_sum = 0.0
    alpha_count = 0
    for u_k, u_k1, f in step_loader:
        u_k = u_k.to(device)
        u_k1 = u_k1.to(device)
        f = f.to(device)
        stats = model.predict_step(u_k, f, dt=dt, sample=False, return_stats=True)
        pred = stats["u_next"]
        alpha = stats["alpha"].detach()
        sq_sum += F.mse_loss(pred, u_k1, reduction="sum").item()
        n_elem += int(np.prod(u_k1.shape))
        alpha_sum += alpha.sum().item()
        alpha_sq_sum += alpha.square().sum().item()
        alpha_count += int(alpha.numel())

    alpha_mean = alpha_sum / max(1, alpha_count)
    alpha_var = alpha_sq_sum / max(1, alpha_count) - alpha_mean * alpha_mean
    return {
        "one_step_mse": sq_sum / max(1, n_elem),
        "alpha_mean": alpha_mean,
        "alpha_std": float(max(0.0, alpha_var) ** 0.5),
    }


@torch.no_grad()
def evaluate_rollout_curves(
    model: LatentVAE1D,
    traj_loader: DataLoader,
    device: str,
    dt: float,
    h: float,
    n_steps: int,
) -> Dict[str, np.ndarray]:
    model.eval()
    mse_curves = []
    rel_curves = []

    for batch in traj_loader:
        if isinstance(batch, dict):
            u0 = batch["u0"]
            f = batch["f"]
            u_ref = batch["u_traj"]
        else:
            u0, f, u_ref = batch
        u0 = u0.to(device)
        f = f.to(device)
        u_ref = u_ref.to(device)

        u_pred = rollout_vae_mean(model, u0=u0, f=f, n_steps=n_steps, dt=dt)
        mse_t = torch.mean((u_pred - u_ref) ** 2, dim=-1)
        rel_t = compute_relative_l2_error(u_pred, u_ref, h=h)
        mse_curves.append(mse_t.cpu().numpy())
        rel_curves.append(rel_t.cpu().numpy())

    mse_arr = np.concatenate(mse_curves, axis=0) if mse_curves else np.zeros((0, n_steps + 1), dtype=np.float64)
    rel_arr = np.concatenate(rel_curves, axis=0) if rel_curves else np.zeros((0, n_steps + 1), dtype=np.float64)
    return {
        "mse_curve_mean": np.nanmean(mse_arr, axis=0) if mse_arr.size else np.zeros(n_steps + 1, dtype=np.float64),
        "rel_curve_mean": np.nanmean(rel_arr, axis=0) if rel_arr.size else np.zeros(n_steps + 1, dtype=np.float64),
        "mse_curve_median": np.nanmedian(mse_arr, axis=0) if mse_arr.size else np.zeros(n_steps + 1, dtype=np.float64),
        "rel_curve_median": np.nanmedian(rel_arr, axis=0) if rel_arr.size else np.zeros(n_steps + 1, dtype=np.float64),
    }


def _save_metrics_csv(metrics: Dict[str, float], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, f"{float(value):.12e}"])
    print(f"Saved metrics csv: {out_path}")


def _save_curve_csv(
    mse_mean: np.ndarray,
    rel_mean: np.ndarray,
    mse_median: np.ndarray,
    rel_median: np.ndarray,
    dt: float,
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "time", "mse_mean", "mse_median", "rel_l2_mean", "rel_l2_median"])
        for k in range(len(mse_mean)):
            writer.writerow(
                [
                    k,
                    f"{k * float(dt):.8f}",
                    f"{float(mse_mean[k]):.12e}",
                    f"{float(mse_median[k]):.12e}",
                    f"{float(rel_mean[k]):.12e}",
                    f"{float(rel_median[k]):.12e}",
                ]
            )
    print(f"Saved error-curve csv: {out_path}")


def _plot_error_curve(
    mse_mean: np.ndarray,
    rel_mean: np.ndarray,
    mse_median: np.ndarray,
    rel_median: np.ndarray,
    dt: float,
    out_path: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping curve plot because matplotlib is unavailable: {exc}")
        return

    times = np.arange(mse_mean.shape[0], dtype=np.float64) * float(dt)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
    axes[0, 0].plot(times, mse_mean, linewidth=2, label="mean")
    axes[0, 0].plot(times, mse_median, linewidth=2, linestyle="--", label="median")
    axes[0, 0].set_title("Rollout MSE by Time")
    axes[0, 0].set_xlabel("time")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()
    axes[0, 1].plot(times, rel_mean, linewidth=2, color="tab:orange", label="mean")
    axes[0, 1].plot(times, rel_median, linewidth=2, linestyle="--", color="tab:red", label="median")
    axes[0, 1].set_title("Rollout Relative L2 by Time")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylabel("relative L2")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved error-curve plot: {out_path}")


def _display_fields(u: torch.Tensor, boundary_condition: str, domain_length: float) -> tuple[torch.Tensor, torch.Tensor]:
    if boundary_condition == "dirichlet":
        n_x = int(u.shape[-1])
        h = 1.0 / float(n_x + 1)
        x = torch.linspace(0.0, 1.0, n_x + 2)
        return pad_dirichlet_1d(u), x
    n_x = int(u.shape[-1])
    x = torch.arange(n_x, dtype=u.dtype) * (float(domain_length) / float(n_x))
    return u, x


@torch.no_grad()
def _plot_sample_comparisons(
    model: LatentVAE1D,
    split: Dict[str, torch.Tensor],
    device: str,
    dt: float,
    t_final: float,
    h: float,
    boundary_condition: str,
    domain_length: float,
    n_plot_samples: int,
    snapshot_times: Sequence[float],
    out_dir: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping sample plots because matplotlib is unavailable: {exc}")
        return

    u_traj = split["u_traj"]
    u0 = split["u0"]
    f = split["f"]
    total = int(u_traj.shape[0])
    n_plot = min(max(1, int(n_plot_samples)), total)
    sample_ids = torch.linspace(0, total - 1, n_plot).long().tolist()
    n_steps = int(u_traj.shape[1] - 1)
    t_grid = np.arange(n_steps + 1, dtype=np.float64) * float(dt)
    os.makedirs(out_dir, exist_ok=True)

    for sample_id in sample_ids:
        u0_i = u0[sample_id : sample_id + 1].to(device)
        f_i = f[sample_id : sample_id + 1].to(device)
        u_ref = u_traj[sample_id]
        u_pred = rollout_vae_mean(model, u0=u0_i, f=f_i, n_steps=n_steps, dt=dt)[0].cpu()

        u_ref_plot, x = _display_fields(u_ref, boundary_condition, domain_length=domain_length)
        u_pred_plot, _ = _display_fields(u_pred, boundary_condition, domain_length=domain_length)
        err_plot = torch.abs(u_pred_plot - u_ref_plot)
        state_scale = max(float(u_ref_plot.abs().max()), float(u_pred_plot.abs().max()), 1e-8)
        err_scale = max(float(err_plot.max()), 1e-8)

        rel_curve = compute_relative_l2_error(u_pred, u_ref, h=h)
        rel_mean = float(rel_curve.mean().item())
        rel_final = float(rel_curve[-1].item())

        fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0), squeeze=False, constrained_layout=True)

        axes[0, 0].plot(x.numpy(), u_ref_plot[0].numpy(), color="black", linewidth=1.8)
        axes[0, 0].set_title("initial state")
        axes[0, 0].set_xlabel("x")
        axes[0, 0].set_ylabel("u")
        axes[0, 0].grid(alpha=0.25)

        colors = ["black", "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:brown", "tab:pink"]
        for j, t_snap in enumerate(snapshot_times):
            k = int(round((float(t_snap) / float(t_final)) * n_steps)) if t_final > 0 else 0
            k = max(0, min(n_steps, k))
            t_actual = t_grid[k]
            color = colors[j % len(colors)]
            axes[0, 1].plot(x.numpy(), u_ref_plot[k].numpy(), color=color, linewidth=1.7, label=f"true t={t_actual:g}")
            axes[0, 1].plot(
                x.numpy(),
                u_pred_plot[k].numpy(),
                color=color,
                linewidth=1.7,
                linestyle="--",
                label=f"pred t={t_actual:g}",
            )
        axes[0, 1].set_title("snapshots")
        axes[0, 1].set_xlabel("x")
        axes[0, 1].set_ylabel("u")
        axes[0, 1].grid(alpha=0.25)
        axes[0, 1].legend(loc="best", fontsize=8, ncol=2)

        axes[0, 2].plot(t_grid, rel_curve.numpy(), color="tab:red", linewidth=1.8)
        axes[0, 2].set_title("relative L2")
        axes[0, 2].set_xlabel("t")
        axes[0, 2].set_ylabel("rel L2")
        axes[0, 2].grid(alpha=0.25)

        im_ref = axes[1, 0].imshow(
            u_ref_plot.numpy(),
            aspect="auto",
            origin="lower",
            extent=[0.0, float(domain_length), 0.0, t_grid[-1]],
            cmap="RdBu_r",
            vmin=-state_scale,
            vmax=state_scale,
        )
        axes[1, 0].set_title("true u(x,t)")
        axes[1, 0].set_xlabel("x")
        axes[1, 0].set_ylabel("t")

        im_pred = axes[1, 1].imshow(
            u_pred_plot.numpy(),
            aspect="auto",
            origin="lower",
            extent=[0.0, float(domain_length), 0.0, t_grid[-1]],
            cmap="RdBu_r",
            vmin=-state_scale,
            vmax=state_scale,
        )
        axes[1, 1].set_title("predicted u(x,t)")
        axes[1, 1].set_xlabel("x")
        axes[1, 1].set_ylabel("t")

        im_err = axes[1, 2].imshow(
            err_plot.numpy(),
            aspect="auto",
            origin="lower",
            extent=[0.0, float(domain_length), 0.0, t_grid[-1]],
            cmap="magma",
            vmin=0.0,
            vmax=err_scale,
        )
        axes[1, 2].set_title("|error|")
        axes[1, 2].set_xlabel("x")
        axes[1, 2].set_ylabel("t")

        fig.colorbar(im_ref, ax=axes[1, 0], fraction=0.046, pad=0.04, label="u")
        fig.colorbar(im_pred, ax=axes[1, 1], fraction=0.046, pad=0.04, label="u")
        fig.colorbar(im_err, ax=axes[1, 2], fraction=0.046, pad=0.04, label="abs error")
        fig.suptitle(
            f"KdV VAE sample {sample_id}: mean rel L2={rel_mean:.3e}, final rel L2={rel_final:.3e}",
            fontsize=13,
        )
        out_path = os.path.join(out_dir, f"sample_{sample_id:04d}_comparison.png")
        fig.savefig(out_path, dpi=180)
        plt.close(fig)

    print(f"Saved sample comparison plots in: {out_dir}")


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if args.args_json is not None:
        print(f"Loaded architecture args: {args.args_json}")

    splits = load_dataset_splits(args.dataset_path, map_location="cpu")
    split = splits[args.split]
    meta = splits.get("meta", {})
    n_x = int(split["u0"].shape[1])
    n_steps = int(split["u_traj"].shape[1] - 1)
    meta_t_final = float(meta.get("t_final", 1.0))
    boundary_condition = str(meta.get("boundary_condition", "periodic" if meta.get("periodic", False) else "dirichlet"))
    domain_length = float(meta.get("domain_length", 1.0))
    h_default = domain_length / float(n_x) if boundary_condition == "periodic" else 1.0 / float(n_x + 1)
    h = float(meta.get("h", h_default))
    dt = float(meta.get("dataset_dt", meta_t_final / float(n_steps)))
    t_final = float(dt * n_steps)
    gamma = float(meta.get("gamma", float("nan")))
    print(f"Loaded split={args.split} from {args.dataset_path}")
    print(
        f"Grid: n_x={n_x}, n_steps={n_steps}, t_final={t_final:.6f}, "
        f"h={h:.6f}, dt={dt:.6f}, gamma={gamma:.6f}, "
        f"boundary_condition={boundary_condition}, L={domain_length:.6f}"
    )

    model = _build_model(n_x=n_x, dt=dt, boundary_condition=boundary_condition, args=args).to(device)
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint_path}")
    if "dt" in ckpt:
        print(f"Checkpoint dt={float(ckpt['dt']):.6f}")

    step_loader = DataLoader(
        build_step_dataset(split),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    traj_loader = DataLoader(
        build_trajectory_dataset_from_split(split),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    one_step_metrics = evaluate_one_step(model, step_loader=step_loader, device=device, dt=dt)
    curves = evaluate_rollout_curves(model, traj_loader=traj_loader, device=device, dt=dt, h=h, n_steps=n_steps)
    mse_curve = curves["mse_curve_mean"]
    rel_curve = curves["rel_curve_mean"]
    mse_curve_median = curves["mse_curve_median"]
    rel_curve_median = curves["rel_curve_median"]

    metrics = {
        **one_step_metrics,
        "rollout_final_mse": float(mse_curve[-1]),
        "rollout_final_rel_l2": float(rel_curve[-1]),
        "rollout_mean_rel_l2": float(rel_curve.mean()),
        "rollout_final_mse_median": float(mse_curve_median[-1]),
        "rollout_final_rel_l2_median": float(rel_curve_median[-1]),
        "rollout_median_rel_l2": float(np.nanmedian(rel_curve_median)),
    }
    print("VAE evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.8e}")
    print("Rollout accumulation (step, mse, rel_l2):")
    for k in range(n_steps + 1):
        print(f"  {k:03d}  {mse_curve[k]:.8e}  {rel_curve[k]:.8e}")

    _save_metrics_csv(metrics, os.path.join(args.output_dir, f"{args.split}_metrics.csv"))
    _save_curve_csv(
        mse_curve,
        rel_curve,
        mse_curve_median,
        rel_curve_median,
        dt=dt,
        out_path=os.path.join(args.output_dir, f"{args.split}_rollout_error_curve.csv"),
    )
    _plot_error_curve(
        mse_curve,
        rel_curve,
        mse_curve_median,
        rel_curve_median,
        dt=dt,
        out_path=os.path.join(args.output_dir, f"{args.split}_rollout_error_curve.png"),
    )
    _plot_sample_comparisons(
        model=model,
        split=split,
        device=device,
        dt=dt,
        t_final=t_final,
        h=h,
        boundary_condition=boundary_condition,
        domain_length=domain_length,
        n_plot_samples=args.n_plot_samples,
        snapshot_times=_parse_snapshot_times(args.snapshot_times, t_final=t_final),
        out_dir=os.path.join(args.output_dir, f"{args.split}_sample_comparisons"),
    )


if __name__ == "__main__":
    main(parse_args())
