"""
Energy-head-only evaluation for 2D Cahn-Hilliard latent models.

This script performs autoregressive prediction by repeatedly solving the
approximate proximal problem in latent space:

    z_{k+1} \approx argmin_z { (1 / (2 eta)) ||z - z_k||^2 + E_psi(z; c) }.

Because the script uses the energy head only, the proximal center is taken to
be the previous latent state z_k. The optimization is solved approximately by
fixed-step gradient descent in latent space.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Tuple

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
    parser = argparse.ArgumentParser(description="Energy-head-only evaluation for 2D Cahn-Hilliard models")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to cached dataset (.pt)")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for rollout evaluation")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print a running evaluation summary every N batches.",
    )
    parser.add_argument("--n-plot-samples", type=int, default=6)
    parser.add_argument("--snapshot-times", type=str, default="0,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--output-dir", type=str, default="grad_flow_l2/cahn_hilliard2d/outputs/energy_head_eval")
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--eta", type=float, default=None, help="Proximal step size. Defaults to dataset dt.")
    parser.add_argument("--prox-steps", type=int, default=20, help="Inner gradient-descent steps per proximal solve")
    parser.add_argument("--prox-lr", type=float, default=0.1, help="Inner optimizer step size for proximal solve")

    # Must match training architecture so the checkpoint can be loaded.
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


def _load_model_state(model: HiddenGradientFlowModel2D, checkpoint_path: str, device: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict, strict=True)


def _extract_epsilon(split: Dict[str, torch.Tensor], meta: Dict, n_samples: int, device: str) -> torch.Tensor:
    eps = split.get("epsilon")
    if eps is not None:
        if eps.dim() != 1 or int(eps.shape[0]) != n_samples:
            raise ValueError("split['epsilon'] must have shape (n_samples,)")
        return eps.to(device)
    default_eps = float(meta.get("epsilon", meta.get("epsilon_min", 0.04)))
    return torch.full((n_samples,), default_eps, device=device, dtype=torch.float32)


def _batch_slices(n: int, batch_size: int) -> List[Tuple[int, int]]:
    return [(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]


def _proximal_energy_step(
    model: HiddenGradientFlowModel2D,
    z_center: torch.Tensor,
    f: torch.Tensor,
    eta: float,
    prox_steps: int,
    prox_lr: float,
) -> torch.Tensor:
    """
    Approximately solve:
        argmin_z 0.5/eta ||z - z_center||^2 + E(z; f)

    We use z_center as both the proximal center and warm start. This is the
    natural energy-only autoregressive update when no separate step predictor
    is available.
    """
    if eta <= 0:
        raise ValueError("eta must be > 0")
    if prox_steps < 1:
        raise ValueError("prox_steps must be >= 1")
    if prox_lr <= 0:
        raise ValueError("prox_lr must be > 0")

    z = z_center.detach()
    for _ in range(int(prox_steps)):
        z = z.detach().requires_grad_(True)
        energy = model.latent_energy(z, f)
        quad = 0.5 / float(eta) * torch.sum((z - z_center) ** 2, dim=(1, 2, 3))
        objective = energy + quad
        grad = torch.autograd.grad(objective.sum(), z, create_graph=False, retain_graph=False)[0]
        z = (z - float(prox_lr) * grad).detach()
    return z


@torch.no_grad()
def _decode(model: HiddenGradientFlowModel2D, z: torch.Tensor) -> torch.Tensor:
    return model.decode(z)


def _rollout_energy_head_only(
    model: HiddenGradientFlowModel2D,
    u0: torch.Tensor,
    f: torch.Tensor,
    n_steps: int,
    eta: float,
    prox_steps: int,
    prox_lr: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return physical-state and latent rollouts.
    """
    z = model.encode(u0)
    u_states = [u0]
    z_states = [z]
    for _ in range(n_steps):
        z = _proximal_energy_step(model, z_center=z, f=f, eta=eta, prox_steps=prox_steps, prox_lr=prox_lr)
        u = _decode(model, z)
        u_states.append(u)
        z_states.append(z)
    return torch.stack(u_states, dim=1), torch.stack(z_states, dim=1)


def _relative_l2_error_2d(u_pred: torch.Tensor, u_ref: torch.Tensor, area: float) -> torch.Tensor:
    diff = u_pred - u_ref
    num = torch.sqrt(float(area) * torch.sum(diff * diff, dim=(-2, -1)))
    den = torch.sqrt(float(area) * torch.sum(u_ref * u_ref, dim=(-2, -1)))
    return num / (den + 1e-8)


def _compute_energy_and_mass_curves(
    model: HiddenGradientFlowModel2D,
    u_pred: torch.Tensor,
    u_ref: torch.Tensor,
    f: torch.Tensor,
    eps: torch.Tensor,
    h_x: float,
    h_y: float,
    device: str,
) -> Dict[str, np.ndarray]:
    n_samples, n_steps1, n_x, n_y = u_pred.shape
    area = float(h_x) * float(h_y)
    eps_flat = eps.to(device=device, dtype=u_pred.dtype).view(n_samples, 1).expand(n_samples, n_steps1).reshape(-1)
    cache = prepare_ch2d_spectral_cache(
        n_x=n_x,
        n_y=n_y,
        h_x=h_x,
        h_y=h_y,
        device=device,
        dtype=u_pred.dtype,
    )
    u_pred_dev = u_pred.to(device)
    u_ref_dev = u_ref.to(device)
    f_dev = f.to(device)
    u_pred_flat = u_pred_dev.reshape(-1, n_x, n_y)
    u_ref_flat = u_ref_dev.reshape(-1, n_x, n_y)
    phys_energy_pred = compute_ch_free_energy_2d(
        u_pred_flat,
        epsilon=eps_flat,
        h_x=h_x,
        h_y=h_y,
        cache=cache,
    ).reshape(n_samples, n_steps1)
    phys_energy_ref = compute_ch_free_energy_2d(
        u_ref_flat,
        epsilon=eps_flat,
        h_x=h_x,
        h_y=h_y,
        cache=cache,
    ).reshape(n_samples, n_steps1)

    latent_energy_pred = []
    with torch.no_grad():
        for k in range(n_steps1):
            z_k = model.encode(u_pred_dev[:, k])
            latent_energy_pred.append(model.latent_energy(z_k, f_dev).detach().cpu())
    latent_energy_pred_all = torch.stack(latent_energy_pred, dim=1)

    mass_pred = area * torch.sum(u_pred.cpu(), dim=(-2, -1))
    mass_ref = area * torch.sum(u_ref.cpu(), dim=(-2, -1))

    return {
        "phys_energy_pred": phys_energy_pred.mean(dim=0).cpu().numpy().astype(np.float64),
        "phys_energy_ref": phys_energy_ref.mean(dim=0).cpu().numpy().astype(np.float64),
        "latent_energy_pred": latent_energy_pred_all.mean(dim=0).cpu().numpy().astype(np.float64),
        "mass_pred": mass_pred.mean(dim=0).cpu().numpy().astype(np.float64),
        "mass_ref": mass_ref.mean(dim=0).cpu().numpy().astype(np.float64),
        "phys_energy_pred_all": phys_energy_pred.cpu().numpy().astype(np.float64),
        "phys_energy_ref_all": phys_energy_ref.cpu().numpy().astype(np.float64),
        "latent_energy_pred_all": latent_energy_pred_all.numpy().astype(np.float64),
        "mass_pred_all": mass_pred.cpu().numpy().astype(np.float64),
        "mass_ref_all": mass_ref.cpu().numpy().astype(np.float64),
    }


def _save_curve_csv(curves: Dict[str, np.ndarray], dt: float, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "time", "mse", "rel_l2", "phys_energy_pred", "phys_energy_ref", "latent_energy_pred", "mass_pred", "mass_ref"])
        n_points = int(curves["mse_curve"].shape[0])
        for k in range(n_points):
            writer.writerow(
                [
                    k,
                    f"{k * dt:.8f}",
                    f"{curves['mse_curve'][k]:.12e}",
                    f"{curves['rel_curve'][k]:.12e}",
                    f"{curves['phys_energy_pred'][k]:.12e}",
                    f"{curves['phys_energy_ref'][k]:.12e}",
                    f"{curves['latent_energy_pred'][k]:.12e}",
                    f"{curves['mass_pred'][k]:.12e}",
                    f"{curves['mass_ref'][k]:.12e}",
                ]
            )
    print(f"Saved diagnostics csv: {out_path}")


def _plot_curves(curves: Dict[str, np.ndarray], dt: float, out_path: str) -> None:
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
    ax.plot(t, curves["phys_energy_pred"], linewidth=2, label="phys E pred")
    ax.plot(t, curves["phys_energy_ref"], linewidth=2, label="phys E ref")
    ax.plot(t, curves["latent_energy_pred"], linewidth=2, linestyle="--", label="latent E pred")
    ax.set_title("Energy Curves")
    ax.set_xlabel("time")
    ax.set_ylabel("energy")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, curves["mass_pred"], linewidth=2, label="mass pred")
    ax.plot(t, curves["mass_ref"], linewidth=2, label="mass ref")
    ax.set_title("Mass Curves")
    ax.set_xlabel("time")
    ax.set_ylabel("mass")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved curve plot: {out_path}")

def _plot_samples(
    model: HiddenGradientFlowModel2D,
    split: Dict[str, torch.Tensor],
    device: str,
    dt: float,
    t_final: float,
    snapshot_times: List[float],
    n_plot_samples: int,
    eta: float,
    prox_steps: int,
    prox_lr: float,
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
        u_ref_i = u_traj[sample_id].to(device)
        u_pred_i, _ = _rollout_energy_head_only(
            model=model,
            u0=u0_i,
            f=f_i,
            n_steps=n_steps,
            eta=eta,
            prox_steps=prox_steps,
            prox_lr=prox_lr,
        )
        u_pred_i = u_pred_i[0].cpu()
        u_ref_i = u_ref_i.cpu()
        area = (1.0 / float(u_ref_i.shape[-2] + 1)) * (1.0 / float(u_ref_i.shape[-1] + 1))
        diff_i = u_pred_i - u_ref_i
        rel_curve_i = torch.sqrt(area * torch.sum(diff_i * diff_i, dim=(-2, -1))) / (
            torch.sqrt(area * torch.sum(u_ref_i * u_ref_i, dim=(-2, -1))) + 1e-8
        )
        rel_mean_i = float(rel_curve_i.mean().item())
        rel_final_i = float(rel_curve_i[-1].item())

        scale = max(float(torch.max(torch.abs(u_ref_i)).item()), float(torch.max(torch.abs(u_pred_i)).item()), 1e-8)
        fig, axes = plt.subplots(3, n_cols, figsize=(3.2 * n_cols, 8.0), squeeze=False, constrained_layout=True)

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
                u_ref_k.numpy(),
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
                u_pred_k.numpy(),
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
                err_k.numpy(),
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
            f"Energy-head-only rollout | sample {sample_id} | "
            f"relL2 mean={rel_mean_i:.3e}, final={rel_final_i:.3e}",
            fontsize=13,
        )
        out_path = os.path.join(out_dir, f"sample_{sample_id:04d}_comparison.png")
        fig.savefig(out_path, dpi=180)
        plt.close(fig)

    print(f"Saved sample plots: {out_dir}")


def _evaluate_rollouts(
    model: HiddenGradientFlowModel2D,
    split: Dict[str, torch.Tensor],
    meta: Dict,
    device: str,
    dt: float,
    eta: float,
    prox_steps: int,
    prox_lr: float,
    h_x: float,
    h_y: float,
    batch_size: int,
    progress_every: int,
) -> Dict[str, np.ndarray | float]:
    u0_all = split["u0"].to(device)
    f_all = split["f"].to(device)
    u_ref_all = split["u_traj"].to(device)
    n_samples = int(u_ref_all.shape[0])
    n_steps = int(u_ref_all.shape[1] - 1)
    area = float(h_x) * float(h_y)

    pred_batches = []
    latent_batches = []
    batch_ranges = _batch_slices(n_samples, batch_size)
    total_batches = len(batch_ranges)
    for batch_idx, (start, end) in enumerate(batch_ranges, start=1):
        u0 = u0_all[start:end]
        f = f_all[start:end]
        u_pred_b, z_pred_b = _rollout_energy_head_only(
            model=model,
            u0=u0,
            f=f,
            n_steps=n_steps,
            eta=eta,
            prox_steps=prox_steps,
            prox_lr=prox_lr,
        )
        pred_batches.append(u_pred_b.detach().cpu())
        latent_batches.append(z_pred_b.detach().cpu())

        if progress_every > 0 and (batch_idx == 1 or batch_idx % progress_every == 0 or batch_idx == total_batches):
            u_pred_so_far = torch.cat(pred_batches, dim=0)
            u_ref_so_far = u_ref_all[: u_pred_so_far.shape[0]].cpu()
            diff_so_far = u_pred_so_far - u_ref_so_far
            num_so_far = torch.sqrt(area * torch.sum(diff_so_far * diff_so_far, dim=(-2, -1)))
            den_so_far = torch.sqrt(area * torch.sum(u_ref_so_far * u_ref_so_far, dim=(-2, -1)))
            rel_so_far = num_so_far / (den_so_far + 1e-8)
            batch_rel_final = float(rel_so_far[:, -1].mean().item())
            batch_rel_mean = float(rel_so_far.mean().item())
            print(
                f"[energy-eval] batch {batch_idx:03d}/{total_batches:03d} "
                f"samples={start}-{end - 1} "
                f"running_relL2_mean={batch_rel_mean:.6e} "
                f"running_relL2_final={batch_rel_final:.6e}",
                flush=True,
            )

    u_pred = torch.cat(pred_batches, dim=0)
    z_pred = torch.cat(latent_batches, dim=0)

    mse_curve = torch.mean((u_pred - u_ref_all.cpu()) ** 2, dim=(0, 2, 3))
    diff = u_pred - u_ref_all.cpu()
    num = torch.sqrt(area * torch.sum(diff * diff, dim=(-2, -1)))
    den = torch.sqrt(area * torch.sum(u_ref_all.cpu() * u_ref_all.cpu(), dim=(-2, -1)))
    rel_curve = torch.mean(num / (den + 1e-8), dim=0)

    eps = _extract_epsilon(split, meta=meta, n_samples=n_samples, device=device)
    curves = _compute_energy_and_mass_curves(
        model=model,
        u_pred=u_pred,
        u_ref=u_ref_all.cpu(),
        f=f_all.cpu(),
        eps=eps.cpu(),
        h_x=h_x,
        h_y=h_y,
        device=device,
    )

    # Predicted latent rollout energy should be non-increasing for a good proximal solver.
    latent_energy_pred = curves["latent_energy_pred_all"]
    energy_violation_rate = float(np.mean(latent_energy_pred[:, 1:] > latent_energy_pred[:, :-1] + 1e-8))

    return {
        "mse_curve": mse_curve.numpy().astype(np.float64),
        "rel_curve": rel_curve.numpy().astype(np.float64),
        **curves,
        "energy_violation_rate": energy_violation_rate,
        "rollout_rel_l2_mean": float(rel_curve.mean().item()),
        "rollout_rel_l2_final": float(rel_curve[-1].item()),
        "u_pred": u_pred,
        "z_pred": z_pred,
    }


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
    eta = float(dt if args.eta is None else args.eta)
    snapshot_times = _parse_snapshot_times(args.snapshot_times, t_final=t_final)

    print(f"Loaded split={args.split} from {args.dataset_path}")
    print(
        f"Grid: n_x={n_x}, n_y={n_y}, n_steps={n_steps}, "
        f"t_final={t_final:.6f}, dt={dt:.6f}, eta={eta:.6f}"
    )

    model = _build_model(n_x=n_x, n_y=n_y, h_x=h_x, h_y=h_y, dt=dt, args=args).to(device)
    _load_model_state(model, args.checkpoint_path, device=device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint_path}")

    curves = _evaluate_rollouts(
        model=model,
        split=split,
        meta=meta,
        device=device,
        dt=dt,
        eta=eta,
        prox_steps=args.prox_steps,
        prox_lr=args.prox_lr,
        h_x=h_x,
        h_y=h_y,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
    )

    print(f"Split rollout mean relative L2: {curves['rollout_rel_l2_mean']:.8e}")
    print(f"Split rollout final relative L2: {curves['rollout_rel_l2_final']:.8e}")
    print(f"Energy violation rate (latent): {curves['energy_violation_rate']:.6f}")
    print("Rollout accumulation by step (step, time, mse, rel_l2, phys_E_pred, phys_E_ref, latent_E_pred):")
    for k in range(len(curves["mse_curve"])):
        print(
            f"  {k:03d}  {k * dt:8.4f}  "
            f"{curves['mse_curve'][k]:.8e}  "
            f"{curves['rel_curve'][k]:.8e}  "
            f"{curves['phys_energy_pred'][k]:.8e}  "
            f"{curves['phys_energy_ref'][k]:.8e}  "
            f"{curves['latent_energy_pred'][k]:.8e}"
        )

    curve_csv = os.path.join(args.output_dir, f"{args.split}_energy_head_rollout.csv")
    curve_png = os.path.join(args.output_dir, f"{args.split}_energy_head_rollout.png")
    _save_curve_csv(curves=curves, dt=dt, out_path=curve_csv)
    _plot_curves(curves=curves, dt=dt, out_path=curve_png)

    sample_dir = os.path.join(args.output_dir, f"{args.split}_sample_comparisons")
    _plot_samples(
        model=model,
        split=split,
        device=device,
        dt=dt,
        t_final=t_final,
        snapshot_times=snapshot_times,
        n_plot_samples=args.n_plot_samples,
        eta=eta,
        prox_steps=args.prox_steps,
        prox_lr=args.prox_lr,
        out_dir=sample_dir,
    )


if __name__ == "__main__":
    main(parse_args())
