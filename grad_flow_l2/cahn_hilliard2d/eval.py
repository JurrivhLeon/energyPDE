"""
Evaluation and visualization script for 2D Cahn-Hilliard hidden-space models.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

try:
    from ..cahn_hilliard2d_solver import compute_ch_free_energy_2d, prepare_ch2d_spectral_cache
    from ..heat_data import load_dataset_splits
    from .model import build_cahn_hilliard2d_model
except ImportError:
    from grad_flow_l2.cahn_hilliard2d_solver import compute_ch_free_energy_2d, prepare_ch2d_spectral_cache
    from grad_flow_l2.heat_data import load_dataset_splits
    from grad_flow_l2.cahn_hilliard2d.model import build_cahn_hilliard2d_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and visualize 2D Cahn-Hilliard hidden-space model")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to cached dataset (.pt)")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--n-plot-samples", type=int, default=6)
    parser.add_argument("--snapshot-times", type=str, default="0,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--output-dir", type=str, default="grad_flow_l2/cahn_hilliard2d/outputs/eval")
    parser.add_argument("--energy-tol", type=float, default=1e-8, help="Tolerance for energy-increase violation checks")
    parser.add_argument("--cpu", action="store_true")

    # Must match training architecture.
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--enc-blocks", type=int, default=4)
    parser.add_argument("--dec-blocks", type=int, default=4)
    parser.add_argument("--prox-blocks", type=int, default=6)
    parser.add_argument("--prox-simulator-type", type=str, default="cnn", choices=["cnn", "fno"])
    parser.add_argument("--fno-modes-x", type=int, default=16)
    parser.add_argument("--fno-modes-y", type=int, default=16)
    parser.add_argument("--disable-fno-grid", action="store_true")
    parser.add_argument("--energy-layers", type=int, default=4)
    parser.add_argument("--use-dt-channel", action="store_true")
    parser.add_argument("--disable-forcing-channel", action="store_true")
    parser.add_argument("--disable-z-grad-feature", action="store_true")
    parser.add_argument("--disable-u-grad-feature", action="store_true")
    return parser.parse_args()


def _parse_snapshot_times(raw: str, t_final: float) -> List[float]:
    vals = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = float(tok)
        if v < 0.0 or v > float(t_final):
            raise ValueError(f"Snapshot time must be in [0,{t_final}], got {v}")
        vals.append(v)
    if not vals:
        vals = [0.0, 0.2 * t_final, 0.4 * t_final, 0.6 * t_final, 0.8 * t_final, t_final]
    return vals


def _build_model(
    n_x: int,
    n_y: int,
    h_x: float,
    h_y: float,
    dt: float,
    args: argparse.Namespace,
) -> HiddenGradientFlowModel2D:
    return build_cahn_hilliard2d_model(n_x=n_x, n_y=n_y, h_x=h_x, h_y=h_y, dt=dt, args=args)


@torch.no_grad()
def _rollout(model: HiddenGradientFlowModel2D, u0: torch.Tensor, f: torch.Tensor, n_steps: int, dt: float) -> torch.Tensor:
    states = [u0]
    u = u0
    for _ in range(n_steps):
        u = model.predict_step(u, f, dt=dt)
        states.append(u)
    return torch.stack(states, dim=1)


def _extract_epsilon(split: Dict[str, torch.Tensor], meta: Dict, n_samples: int, device: str) -> torch.Tensor:
    eps = split.get("epsilon")
    if eps is not None:
        if eps.dim() != 1 or int(eps.shape[0]) != n_samples:
            raise ValueError("split['epsilon'] must have shape (n_samples,)")
        return eps.to(device)
    default_eps = float(meta.get("epsilon", meta.get("epsilon_min", 0.04)))
    return torch.full((n_samples,), default_eps, device=device, dtype=torch.float32)


@torch.no_grad()
def _evaluate_one_step_mse(model: HiddenGradientFlowModel2D, split: Dict[str, torch.Tensor], device: str, dt: float) -> float:
    u_traj = split["u_traj"].to(device)
    f = split["f"].to(device)
    total_sq = 0.0
    n_elem = 0
    n_steps = int(u_traj.shape[1] - 1)
    for k in range(n_steps):
        u_k = u_traj[:, k]
        u_k1 = u_traj[:, k + 1]
        u_pred = model.predict_step(u_k, f, dt=dt)
        total_sq += F.mse_loss(u_pred, u_k1, reduction="sum").item()
        n_elem += int(np.prod(u_k1.shape))
    return total_sq / max(1, n_elem)


@torch.no_grad()
def _evaluate_rollout_stats(
    model: HiddenGradientFlowModel2D,
    split: Dict[str, torch.Tensor],
    meta: Dict,
    device: str,
    dt: float,
    h_x: float,
    h_y: float,
    energy_tol: float,
) -> Dict[str, np.ndarray | float]:
    u0 = split["u0"].to(device)
    f = split["f"].to(device)
    u_ref = split["u_traj"].to(device)
    n_samples = int(u_ref.shape[0])
    n_steps = int(u_ref.shape[1] - 1)
    n_x = int(u_ref.shape[2])
    n_y = int(u_ref.shape[3])
    area = float(h_x) * float(h_y)

    u_pred = _rollout(model, u0=u0, f=f, n_steps=n_steps, dt=dt)

    mse_curve = torch.mean((u_pred - u_ref) ** 2, dim=(0, 2, 3))
    diff = u_pred - u_ref
    num = torch.sqrt(area * torch.sum(diff * diff, dim=(-2, -1)))
    den = torch.sqrt(area * torch.sum(u_ref * u_ref, dim=(-2, -1)))
    rel_curve = torch.mean(num / (den + 1e-8), dim=0)

    mass_ref = area * torch.sum(u_ref, dim=(-2, -1))
    mass_pred = area * torch.sum(u_pred, dim=(-2, -1))
    mass_abs_err_curve = torch.mean(torch.abs(mass_pred - mass_ref), dim=0)
    mass_drift_curve = torch.mean(torch.abs(mass_pred - mass_pred[:, :1]), dim=0)

    eps = _extract_epsilon(split, meta=meta, n_samples=n_samples, device=device).to(dtype=u_ref.dtype)
    eps_flat = eps.view(n_samples, 1).expand(n_samples, n_steps + 1).reshape(-1)
    cache = prepare_ch2d_spectral_cache(
        n_x=n_x,
        n_y=n_y,
        h_x=h_x,
        h_y=h_y,
        device=device,
        dtype=u_ref.dtype,
    )
    e_ref = compute_ch_free_energy_2d(
        u_ref.reshape(-1, n_x, n_y),
        epsilon=eps_flat,
        h_x=h_x,
        h_y=h_y,
        cache=cache,
    ).reshape(n_samples, n_steps + 1)
    e_pred = compute_ch_free_energy_2d(
        u_pred.reshape(-1, n_x, n_y),
        epsilon=eps_flat,
        h_x=h_x,
        h_y=h_y,
        cache=cache,
    ).reshape(n_samples, n_steps + 1)
    energy_ref_curve = torch.mean(e_ref, dim=0)
    energy_pred_curve = torch.mean(e_pred, dim=0)
    energy_gap_curve = torch.mean(e_pred - e_ref, dim=0)
    energy_violation_rate = float(torch.mean((e_pred[:, 1:] > e_pred[:, :-1] + float(energy_tol)).float()).item())

    return {
        "mse_curve": mse_curve.cpu().numpy().astype(np.float64),
        "rel_curve": rel_curve.cpu().numpy().astype(np.float64),
        "mass_abs_err_curve": mass_abs_err_curve.cpu().numpy().astype(np.float64),
        "mass_drift_curve": mass_drift_curve.cpu().numpy().astype(np.float64),
        "energy_ref_curve": energy_ref_curve.cpu().numpy().astype(np.float64),
        "energy_pred_curve": energy_pred_curve.cpu().numpy().astype(np.float64),
        "energy_gap_curve": energy_gap_curve.cpu().numpy().astype(np.float64),
        "rollout_rel_l2_mean": float(rel_curve.mean().item()),
        "rollout_rel_l2_final": float(rel_curve[-1].item()),
        "mass_abs_err_final": float(mass_abs_err_curve[-1].item()),
        "mass_drift_final": float(mass_drift_curve[-1].item()),
        "energy_violation_rate": energy_violation_rate,
        "energy_drop_mean": float((e_pred[:, 0] - e_pred[:, -1]).mean().item()),
    }


def _save_rollout_curve_csv(curves: Dict[str, np.ndarray], dt: float, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "time",
                "mse",
                "rel_l2",
                "mass_abs_err",
                "mass_drift",
                "energy_ref",
                "energy_pred",
                "energy_gap",
            ]
        )
        n_points = int(curves["mse_curve"].shape[0])
        for k in range(n_points):
            writer.writerow(
                [
                    k,
                    f"{k * dt:.8f}",
                    f"{curves['mse_curve'][k]:.12e}",
                    f"{curves['rel_curve'][k]:.12e}",
                    f"{curves['mass_abs_err_curve'][k]:.12e}",
                    f"{curves['mass_drift_curve'][k]:.12e}",
                    f"{curves['energy_ref_curve'][k]:.12e}",
                    f"{curves['energy_pred_curve'][k]:.12e}",
                    f"{curves['energy_gap_curve'][k]:.12e}",
                ]
            )
    print(f"Saved rollout curve csv: {out_path}")


def _plot_rollout_curves(curves: Dict[str, np.ndarray], dt: float, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping curve plotting because matplotlib is unavailable: {exc}")
        return

    t = np.arange(curves["mse_curve"].shape[0], dtype=np.float64) * float(dt)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)

    ax = axes[0, 0]
    ax.plot(t, curves["mse_curve"], linewidth=2)
    ax.set_title("Rollout MSE")
    ax.set_xlabel("time")
    ax.set_ylabel("MSE")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, curves["rel_curve"], linewidth=2, color="tab:orange")
    ax.set_title("Rollout Relative L2")
    ax.set_xlabel("time")
    ax.set_ylabel("relative L2")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, curves["mass_abs_err_curve"], linewidth=2, label="|mass_pred - mass_ref|")
    ax.plot(t, curves["mass_drift_curve"], linewidth=2, label="|mass_pred - mass_pred(t0)|")
    ax.set_title("Mass Diagnostics")
    ax.set_xlabel("time")
    ax.set_ylabel("mass error")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, curves["energy_ref_curve"], linewidth=2, label="E_ref")
    ax.plot(t, curves["energy_pred_curve"], linewidth=2, label="E_pred")
    ax.plot(t, curves["energy_gap_curve"], linewidth=2, linestyle="--", label="E_pred - E_ref")
    ax.set_title("Free Energy Diagnostics")
    ax.set_xlabel("time")
    ax.set_ylabel("energy")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved rollout curve plot: {out_path}")


@torch.no_grad()
def _plot_test_samples(
    model: HiddenGradientFlowModel2D,
    split: Dict[str, torch.Tensor],
    device: str,
    dt: float,
    t_final: float,
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
    n_cols = len(snapshot_times)

    for sample_id in sample_ids:
        u0_i = u0[sample_id : sample_id + 1].to(device)
        f_i = f[sample_id : sample_id + 1].to(device)
        u_ref_i = u_traj[sample_id]  # (K+1, n_x, n_y)
        u_pred_i = _rollout(model, u0_i, f_i, n_steps=n_steps, dt=dt)[0].cpu()
        area = (1.0 / float(u_ref_i.shape[-2] + 1)) * (1.0 / float(u_ref_i.shape[-1] + 1))
        diff_i = u_pred_i - u_ref_i
        num_i = torch.sqrt(area * torch.sum(diff_i * diff_i, dim=(-2, -1)))
        den_i = torch.sqrt(area * torch.sum(u_ref_i * u_ref_i, dim=(-2, -1)))
        rel_curve_i = num_i / (den_i + 1e-8)
        rel_mean_i = float(rel_curve_i.mean().item())
        rel_final_i = float(rel_curve_i[-1].item())
        mass_ref_i = area * torch.sum(u_ref_i, dim=(-2, -1))
        mass_pred_i = area * torch.sum(u_pred_i, dim=(-2, -1))
        mass_drift_i = float(torch.max(torch.abs(mass_pred_i - mass_pred_i[0])).item())

        scale = max(float(torch.max(torch.abs(u_ref_i)).item()), float(torch.max(torch.abs(u_pred_i)).item()), 1e-8)
        fig, axes = plt.subplots(
            3,
            n_cols,
            figsize=(3.2 * n_cols, 8.0),
            squeeze=False,
            constrained_layout=True,
        )

        im_ref_last = None
        im_err_last = None
        for j, t_snap in enumerate(snapshot_times):
            frac = 0.0 if t_final <= 0 else float(t_snap) / float(t_final)
            k = int(round(frac * n_steps))
            k = max(0, min(n_steps, k))

            u_ref_k = u_ref_i[k]
            u_pred_k = u_pred_i[k]
            err_k = torch.abs(u_pred_k - u_ref_k)
            rel_k = float(rel_curve_i[k].item())

            ax_ref = axes[0, j]
            ax_pred = axes[1, j]
            ax_err = axes[2, j]

            im_ref = ax_ref.imshow(
                u_ref_k.cpu().numpy(),
                origin="lower",
                cmap="coolwarm",
                vmin=-scale,
                vmax=scale,
                extent=[0.0, 1.0, 0.0, 1.0],
                aspect="auto",
            )
            ax_ref.set_title(f"ref t={t_snap:g}")
            ax_ref.set_xticks([])
            ax_ref.set_yticks([])

            ax_pred.imshow(
                u_pred_k.cpu().numpy(),
                origin="lower",
                cmap="coolwarm",
                vmin=-scale,
                vmax=scale,
                extent=[0.0, 1.0, 0.0, 1.0],
                aspect="auto",
            )
            ax_pred.set_title(f"pred t={t_snap:g}")
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])

            im_err = ax_err.imshow(
                err_k.cpu().numpy(),
                origin="lower",
                cmap="magma",
                extent=[0.0, 1.0, 0.0, 1.0],
                aspect="auto",
            )
            ax_err.set_title(f"|err| t={t_snap:g}\nrelL2={rel_k:.3e}")
            ax_err.set_xticks([])
            ax_err.set_yticks([])

            im_ref_last = im_ref
            im_err_last = im_err

        if im_ref_last is not None:
            cbar_state = fig.colorbar(im_ref_last, ax=axes[0:2, :], fraction=0.015, pad=0.01)
            cbar_state.ax.set_ylabel("state value", rotation=90)
        if im_err_last is not None:
            cbar_err = fig.colorbar(im_err_last, ax=axes[2, :], fraction=0.015, pad=0.01)
            cbar_err.ax.set_ylabel("abs error", rotation=90)

        fig.suptitle(
            f"Sample {sample_id}: CH reference vs prediction | "
            f"relL2 mean={rel_mean_i:.3e}, final={rel_final_i:.3e}, "
            f"max mass drift={mass_drift_i:.3e}",
            fontsize=13,
        )
        out_path = os.path.join(out_dir, f"sample_{sample_id:04d}_comparison.png")
        fig.savefig(out_path, dpi=180)
        plt.close(fig)

    print(f"Saved sample plots: {out_dir}")


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    splits = load_dataset_splits(args.dataset_path, map_location="cpu")
    split = splits[args.split]
    meta = splits.get("meta", {})
    n_x = int(split["u0"].shape[1])
    n_y = int(split["u0"].shape[2])
    n_steps = int(split["u_traj"].shape[1] - 1)
    t_final = float(meta.get("t_final", 1.0))
    h_x = float(meta.get("h_x", 1.0 / float(n_x + 1)))
    h_y = float(meta.get("h_y", 1.0 / float(n_y + 1)))
    dt = t_final / float(n_steps)
    snapshot_times = _parse_snapshot_times(args.snapshot_times, t_final=t_final)

    print(f"Loaded split={args.split} from {args.dataset_path}")
    print(
        f"Grid: n_x={n_x}, n_y={n_y}, n_steps={n_steps}, "
        f"t_final={t_final:.6f}, dt={dt:.6f}, h_x={h_x:.6f}, h_y={h_y:.6f}"
    )

    model = _build_model(n_x=n_x, n_y=n_y, h_x=h_x, h_y=h_y, dt=dt, args=args).to(device)
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint_path}")

    step_mse = _evaluate_one_step_mse(model, split, device=device, dt=dt)
    curves = _evaluate_rollout_stats(
        model=model,
        split=split,
        meta=meta,
        device=device,
        dt=dt,
        h_x=h_x,
        h_y=h_y,
        energy_tol=args.energy_tol,
    )
    print(f"Split one-step MSE: {step_mse:.8e}")
    print(f"Split rollout mean relative L2: {curves['rollout_rel_l2_mean']:.8e}")
    print(f"Split rollout final relative L2: {curves['rollout_rel_l2_final']:.8e}")
    print(f"Final mean |mass_pred - mass_ref|: {curves['mass_abs_err_final']:.8e}")
    print(f"Final mean |mass_pred - mass_pred(t0)|: {curves['mass_drift_final']:.8e}")
    print(f"Predicted energy violation rate: {curves['energy_violation_rate']:.6f}")
    print(f"Predicted mean energy drop E(t0)-E(tK): {curves['energy_drop_mean']:.8e}")
    print("Rollout accumulation by step (step, time, mse, rel_l2, mass_err, mass_drift, E_ref, E_pred):")
    for k in range(len(curves["mse_curve"])):
        print(
            f"  {k:03d}  {k * dt:8.4f}  "
            f"{curves['mse_curve'][k]:.8e}  "
            f"{curves['rel_curve'][k]:.8e}  "
            f"{curves['mass_abs_err_curve'][k]:.8e}  "
            f"{curves['mass_drift_curve'][k]:.8e}  "
            f"{curves['energy_ref_curve'][k]:.8e}  "
            f"{curves['energy_pred_curve'][k]:.8e}"
        )

    curve_csv = os.path.join(args.output_dir, f"{args.split}_rollout_diagnostics.csv")
    curve_png = os.path.join(args.output_dir, f"{args.split}_rollout_diagnostics.png")
    _save_rollout_curve_csv(curves=curves, dt=dt, out_path=curve_csv)
    _plot_rollout_curves(curves=curves, dt=dt, out_path=curve_png)

    sample_dir = os.path.join(args.output_dir, f"{args.split}_sample_comparisons")
    _plot_test_samples(
        model=model,
        split=split,
        device=device,
        dt=dt,
        t_final=t_final,
        snapshot_times=snapshot_times,
        n_plot_samples=args.n_plot_samples,
        out_dir=sample_dir,
    )


if __name__ == "__main__":
    main(parse_args())
