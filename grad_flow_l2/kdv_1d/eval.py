"""
Evaluate and visualize deterministic latent FNO Markov baselines on forced damped-driven 1D KdV data.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

try:
    from ..heat_data import load_dataset_splits
    from ..utils import compute_relative_l2_error
    from .train import _build_model
except ImportError:
    from grad_flow_l2.heat_data import load_dataset_splits
    from grad_flow_l2.utils import compute_relative_l2_error
    from grad_flow_l2.kdv_1d.train import _build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate deterministic latent FNO Markov baseline on 1D KdV")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="grad_flow_l2/kdv_1d/datasets/kdv_forced_periodic_L32_snx4096_nx512_dt0p1_solverdt0p01_gamma0p1.pt",
        help="Path to cached KdV dataset (.pt)",
    )
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--n-plot-samples", type=int, default=8)
    parser.add_argument("--snapshot-times", type=str, default="0.2,0.4,0.6,0.8,1.0,2.0,3.0,4.0,5.0")
    parser.add_argument("--output-dir", type=str, default="grad_flow_l2/kdv_1d/outputs_sv/eval")
    parser.add_argument("--cpu", action="store_true")

    # Must match training architecture.
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--enc-blocks", type=int, default=4)
    parser.add_argument("--dec-blocks", type=int, default=4)
    parser.add_argument("--fno-width", type=int, default=None)
    parser.add_argument("--fno-layers", type=int, default=6)
    parser.add_argument("--fno-modes", type=int, default=64)
    parser.add_argument("--disable-fno-grid", action="store_true")
    parser.add_argument("--use-dt-channel", action="store_true")
    parser.add_argument(
        "--disable-forcing-channel",
        action="store_true",
        help="Disable the static forcing channel.",
    )
    parser.add_argument("--disable-u-grad-feature", action="store_true")
    return parser.parse_args()


def _parse_snapshot_times(raw: str, t_final: float) -> List[float]:
    vals = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = float(tok)
        if v < 0.0:
            raise ValueError(f"Snapshot time must be nonnegative, got {v}")
        if v <= float(t_final):
            vals.append(v)
    if not vals:
        vals = [0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0, min(5.0, float(t_final))]
    return vals


def _load_checkpoint(checkpoint_path: str, map_location: str | torch.device) -> Dict[str, object]:
    try:
        return torch.load(checkpoint_path, map_location=map_location)
    except RuntimeError as exc:
        msg = str(exc)
        if "weights_only=True" not in msg or "legacy .tar format" not in msg:
            raise
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


@torch.no_grad()
def _rollout(model, u0: torch.Tensor, f: torch.Tensor, n_steps: int, dt: float) -> torch.Tensor:
    states = [u0]
    u = u0
    for _ in range(n_steps):
        u = model.predict_step(u, f, dt=dt)
        states.append(u)
    return torch.stack(states, dim=1)


@torch.no_grad()
def _evaluate_one_step_mse(model, split: Dict[str, torch.Tensor], device: str, dt: float, batch_size: int = 64) -> float:
    u_traj = split["u_traj"]
    f = split["f"]
    total_sq = 0.0
    n_elem = 0
    n_samples = int(u_traj.shape[0])
    n_steps = int(u_traj.shape[1] - 1)
    for start in range(0, n_samples, batch_size):
        end = min(n_samples, start + batch_size)
        f_b = f[start:end].to(device)
        for k in range(n_steps):
            u_k = u_traj[start:end, k].to(device)
            u_k1 = u_traj[start:end, k + 1].to(device)
            u_pred = model.predict_step(u_k, f_b, dt=dt)
            total_sq += F.mse_loss(u_pred, u_k1, reduction="sum").item()
            n_elem += int(np.prod(u_k1.shape))
    return total_sq / max(1, n_elem)


@torch.no_grad()
def _evaluate_rollout_curves(
    model,
    split: Dict[str, torch.Tensor],
    device: str,
    dt: float,
    h: float,
    batch_size: int = 64,
) -> Dict[str, object]:
    u0_all = split["u0"]
    f_all = split["f"]
    u_ref_all = split["u_traj"]
    n_samples = int(u0_all.shape[0])
    n_steps = int(u_ref_all.shape[1] - 1)
    mse_curves = []
    rel_curves = []
    rel_per_sample_mean = []

    for start in range(0, n_samples, batch_size):
        end = min(n_samples, start + batch_size)
        u0 = u0_all[start:end].to(device)
        f = f_all[start:end].to(device)
        u_ref = u_ref_all[start:end].to(device)
        u_pred = _rollout(model, u0=u0, f=f, n_steps=n_steps, dt=dt)
        mse_t = torch.mean((u_pred - u_ref) ** 2, dim=-1)
        rel_t = compute_relative_l2_error(u_pred, u_ref, h=h)
        mse_np = mse_t.cpu().numpy()
        rel_np = rel_t.cpu().numpy()
        mse_curves.append(mse_np)
        rel_curves.append(rel_np)
        rel_per_sample_mean.extend(rel_np.mean(axis=1).tolist())

    mse_arr = np.concatenate(mse_curves, axis=0) if mse_curves else np.zeros((0, n_steps + 1), dtype=np.float64)
    rel_arr = np.concatenate(rel_curves, axis=0) if rel_curves else np.zeros((0, n_steps + 1), dtype=np.float64)
    mse_curve = np.nanmean(mse_arr, axis=0) if mse_arr.size else np.zeros(n_steps + 1, dtype=np.float64)
    rel_curve = np.nanmean(rel_arr, axis=0) if rel_arr.size else np.zeros(n_steps + 1, dtype=np.float64)
    mse_curve_median = np.nanmedian(mse_arr, axis=0) if mse_arr.size else np.zeros(n_steps + 1, dtype=np.float64)
    rel_curve_median = np.nanmedian(rel_arr, axis=0) if rel_arr.size else np.zeros(n_steps + 1, dtype=np.float64)
    rel_per_sample_mean_arr = np.asarray(rel_per_sample_mean, dtype=np.float64)
    return {
        "mse_curve_mean": mse_curve,
        "rel_curve_mean": rel_curve,
        "mse_curve_median": mse_curve_median,
        "rel_curve_median": rel_curve_median,
        "rollout_rel_mean": float(np.nanmean(rel_per_sample_mean_arr)),
        "rollout_rel_median": float(np.nanmedian(rel_per_sample_mean_arr)),
    }


def _save_rollout_curve_csv(
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
                    f"{k * dt:.8f}",
                    f"{float(mse_mean[k]):.12e}",
                    f"{float(mse_median[k]):.12e}",
                    f"{float(rel_mean[k]):.12e}",
                    f"{float(rel_median[k]):.12e}",
                ]
            )
    print(f"Saved rollout curve csv: {out_path}")


def _plot_rollout_curves(
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
        print(f"Skipping curve plotting because matplotlib is unavailable: {exc}")
        return
    t = np.arange(mse_mean.shape[0], dtype=np.float64) * float(dt)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
    axes[0, 0].plot(t, mse_mean, linewidth=2, label="mean")
    axes[0, 0].plot(t, mse_median, linewidth=2, linestyle="--", label="median")
    axes[0, 0].set_title("Rollout MSE Accumulation")
    axes[0, 0].set_xlabel("time")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()
    axes[0, 1].plot(t, rel_mean, linewidth=2, color="tab:orange", label="mean")
    axes[0, 1].plot(t, rel_median, linewidth=2, linestyle="--", color="tab:red", label="median")
    axes[0, 1].set_title("Rollout Relative L2 Accumulation")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylabel("relative L2")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved rollout curve plot: {out_path}")


@torch.no_grad()
def _plot_samples(
    model,
    split: Dict[str, torch.Tensor],
    device: str,
    dt: float,
    t_final: float,
    h: float,
    domain_length: float,
    snapshot_times: List[float],
    n_plot_samples: int,
    out_dir: str,
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
    n_x = int(u_traj.shape[-1])
    x = np.arange(n_x, dtype=np.float64) * (float(domain_length) / float(n_x))
    t_grid = np.arange(n_steps + 1, dtype=np.float64) * float(dt)

    for sample_id in sample_ids:
        u0_i = u0[sample_id : sample_id + 1].to(device)
        f_i = f[sample_id : sample_id + 1].to(device)
        u_ref_i = u_traj[sample_id]
        u_pred_i = _rollout(model, u0_i, f_i, n_steps=n_steps, dt=dt)[0].cpu()
        rel_curve_i = compute_relative_l2_error(u_pred_i, u_ref_i, h=h)
        rel_mean_i = float(rel_curve_i.mean().item())
        rel_final_i = float(rel_curve_i[-1].item())

        state_scale = max(float(u_ref_i.abs().max().item()), float(u_pred_i.abs().max().item()), 1e-8)
        err_i = torch.abs(u_pred_i - u_ref_i)
        err_scale = max(float(err_i.max().item()), 1e-8)

        fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0), squeeze=False, constrained_layout=True)

        axes[0, 0].plot(x, u_ref_i[0].numpy(), color="black", linewidth=1.8)
        axes[0, 0].set_title("initial state")
        axes[0, 0].set_xlabel("x")
        axes[0, 0].set_ylabel("u")
        axes[0, 0].grid(alpha=0.25)

        axes[0, 1].plot(t_grid, rel_curve_i.numpy(), color="tab:red", linewidth=1.8)
        axes[0, 1].set_title("relative L2")
        axes[0, 1].set_xlabel("t")
        axes[0, 1].set_ylabel("rel L2")
        axes[0, 1].grid(alpha=0.25)

        colors = ["black", "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:brown", "tab:pink"]
        for j, t_snap in enumerate(snapshot_times):
            k = int(round((float(t_snap) / float(t_final)) * n_steps)) if t_final > 0 else 0
            k = max(0, min(n_steps, k))
            t_actual = t_grid[k]
            color = colors[j % len(colors)]
            axes[0, 2].plot(x, u_ref_i[k].numpy(), color=color, linewidth=1.7, label=f"true t={t_actual:g}")
            axes[0, 2].plot(
                x,
                u_pred_i[k].numpy(),
                color=color,
                linewidth=1.7,
                linestyle="--",
                label=f"pred t={t_actual:g}",
            )
        axes[0, 2].set_title("snapshots")
        axes[0, 2].set_xlabel("x")
        axes[0, 2].set_ylabel("u")
        axes[0, 2].grid(alpha=0.25)
        axes[0, 2].legend(loc="best", fontsize=8, ncol=2)

        im_ref = axes[1, 0].imshow(
            u_ref_i.numpy(),
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
            u_pred_i.numpy(),
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
            err_i.numpy(),
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
            f"KdV sample {sample_id}: mean rel L2={rel_mean_i:.3e}, final rel L2={rel_final_i:.3e}",
            fontsize=13,
        )
        out_path = os.path.join(out_dir, f"sample_{sample_id:04d}.png")
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"Saved sample plot: {out_path}")


def main(args: argparse.Namespace) -> None:
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_dataset_splits(args.dataset_path, map_location="cpu")
    split = splits[args.split]
    meta = splits.get("meta", {})
    n_x = int(split["u0"].shape[-1])
    n_steps = int(split["u_traj"].shape[1] - 1)
    meta_t_final = float(meta.get("t_final", 1.0))
    dt = float(meta.get("dataset_dt", meta_t_final / float(n_steps)))
    t_final = float(dt * n_steps)
    boundary_condition = str(meta.get("boundary_condition", "periodic" if meta.get("periodic", False) else "dirichlet"))
    domain_length = float(meta.get("domain_length", 1.0))
    h_default = domain_length / float(n_x) if boundary_condition == "periodic" else 1.0 / float(n_x + 1)
    h = float(meta.get("h", h_default))
    gamma = float(meta.get("gamma", float("nan")))

    model = _build_model(n_x=n_x, dt=dt, boundary_condition=boundary_condition, args=args).to(device)
    ckpt = _load_checkpoint(args.checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {device}")
    print(f"Loaded dataset: {args.dataset_path} split={args.split}")
    print(f"Loaded checkpoint: {args.checkpoint_path}")
    print(
        f"Grid from data: n_x={n_x}, n_steps={n_steps}, h={h:.8f}, "
        f"dt={dt:.8f}, bc={boundary_condition}, L={domain_length:.8f}, gamma={gamma:.8f}"
    )

    one_step_mse = _evaluate_one_step_mse(model, split, device=device, dt=dt)
    curves = _evaluate_rollout_curves(model, split, device=device, dt=dt, h=h)
    metrics = {
        "split": args.split,
        "one_step_mse": one_step_mse,
        "rollout_rel_l2": curves["rollout_rel_mean"],
        "rollout_rel_l2_median": curves["rollout_rel_median"],
        "rollout_mse_mean_final": float(curves["mse_curve_mean"][-1]),
        "rollout_rel_l2_mean_final": float(curves["rel_curve_mean"][-1]),
        "rollout_mse_median_final": float(curves["mse_curve_median"][-1]),
        "rollout_rel_l2_median_final": float(curves["rel_curve_median"][-1]),
        "checkpoint_epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "checkpoint_metrics": ckpt.get("metrics", {}) if isinstance(ckpt, dict) else {},
        "dataset_meta": meta,
    }

    summary_path = os.path.join(args.output_dir, f"{args.split}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved summary: {summary_path}")

    _save_rollout_curve_csv(
        curves["mse_curve_mean"],
        curves["rel_curve_mean"],
        curves["mse_curve_median"],
        curves["rel_curve_median"],
        dt=dt,
        out_path=os.path.join(args.output_dir, f"{args.split}_rollout_curves.csv"),
    )
    _plot_rollout_curves(
        curves["mse_curve_mean"],
        curves["rel_curve_mean"],
        curves["mse_curve_median"],
        curves["rel_curve_median"],
        dt=dt,
        out_path=os.path.join(args.output_dir, f"{args.split}_rollout_curves.png"),
    )
    _plot_samples(
        model,
        split,
        device=device,
        dt=dt,
        t_final=t_final,
        h=h,
        domain_length=domain_length,
        snapshot_times=_parse_snapshot_times(args.snapshot_times, t_final=t_final),
        n_plot_samples=args.n_plot_samples,
        out_dir=os.path.join(args.output_dir, f"{args.split}_sample_comparisons"),
    )

    print(f"Split one-step MSE: {one_step_mse:.8e}")
    print(f"Split rollout mean relative L2: {curves['rollout_rel_mean']:.8e}")
    print(f"Split rollout median relative L2: {curves['rollout_rel_median']:.8e}")


if __name__ == "__main__":
    main(parse_args())
