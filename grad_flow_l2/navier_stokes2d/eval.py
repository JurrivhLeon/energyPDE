"""
Evaluation and sample-visualization script for 2D Navier-Stokes hidden-space models.
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
    from ..heat_data import load_dataset_splits
    from ..grad_flow2d import (
        EnergyHead2D,
        FNOProximalStepSimulator2D,
        HiddenGradientFlowModel2D,
        ProximalStepSimulator2D,
        StateDecoder2D,
        StateEncoder2D,
    )
    from ..navier_stokes2d_solver import to_interior_field_2d
except ImportError:
    from grad_flow_l2.heat_data import load_dataset_splits
    from grad_flow_l2.grad_flow2d import (
        EnergyHead2D,
        FNOProximalStepSimulator2D,
        HiddenGradientFlowModel2D,
        ProximalStepSimulator2D,
        StateDecoder2D,
        StateEncoder2D,
    )
    from grad_flow_l2.navier_stokes2d_solver import to_interior_field_2d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and visualize 2D Navier-Stokes hidden-space model")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to cached dataset (.pt)")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--n-plot-samples", type=int, default=6)
    parser.add_argument("--snapshot-times", type=str, default="0,2,4,6,8,10")
    parser.add_argument("--output-dir", type=str, default="grad_flow_l2/navier_stokes2d/outputs/eval")
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
    parser.add_argument("--energy-head-type", type=str, default="local", choices=["local", "fno"])
    parser.add_argument("--energy-fno-modes-x", type=int, default=16)
    parser.add_argument("--energy-fno-modes-y", type=int, default=16)
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


def _remap_checkpoint_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map legacy local energy-head parameter names to the current ones."""

    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("energy_head.backbone."):
            new_key = key.replace("energy_head.backbone.", "energy_head.local_backbone.", 1)
        elif key.startswith("energy_head.density_head."):
            new_key = key.replace("energy_head.density_head.", "energy_head.local_density_head.", 1)
        remapped[new_key] = value
    return remapped


def _build_model(n_x: int, n_y: int, h_x: float, h_y: float, dt: float, args: argparse.Namespace) -> HiddenGradientFlowModel2D:
    use_forcing_channel = not args.disable_forcing_channel
    encoder = StateEncoder2D(
        n_x=n_x,
        n_y=n_y,
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
        n_blocks=args.enc_blocks,
        use_grad_features=not args.disable_u_grad_feature,
    )
    decoder = StateDecoder2D(
        n_x=n_x,
        n_y=n_y,
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
        n_blocks=args.dec_blocks,
    )
    if args.prox_simulator_type == "cnn":
        prox_step = ProximalStepSimulator2D(
            n_x=n_x,
            n_y=n_y,
            latent_channels=args.latent_channels,
            hidden_channels=args.hidden_channels,
            n_blocks=args.prox_blocks,
            use_forcing_channel=use_forcing_channel,
            use_dt_channel=args.use_dt_channel,
            default_dt=dt,
        )
    elif args.prox_simulator_type == "fno":
        prox_step = FNOProximalStepSimulator2D(
            n_x=n_x,
            n_y=n_y,
            latent_channels=args.latent_channels,
            width=args.hidden_channels,
            n_layers=args.prox_blocks,
            modes_x=args.fno_modes_x,
            modes_y=args.fno_modes_y,
            use_forcing_channel=use_forcing_channel,
            use_dt_channel=args.use_dt_channel,
            use_grid_features=not args.disable_fno_grid,
            default_dt=dt,
        )
    else:
        raise ValueError(f"Unsupported prox-simulator-type: {args.prox_simulator_type}")
    energy_head = EnergyHead2D(
        n_x=n_x,
        n_y=n_y,
        h_x=h_x,
        h_y=h_y,
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
        n_layers=args.energy_layers,
        use_forcing_channel=use_forcing_channel,
        use_grad_norm_feature=not args.disable_z_grad_feature,
        head_type=args.energy_head_type,
        energy_fno_modes_x=args.energy_fno_modes_x,
        energy_fno_modes_y=args.energy_fno_modes_y,
    )
    return HiddenGradientFlowModel2D(
        encoder=encoder,
        decoder=decoder,
        prox_step=prox_step,
        energy_head=energy_head,
    )


@torch.no_grad()
def _rollout(model: HiddenGradientFlowModel2D, u0: torch.Tensor, f: torch.Tensor, n_steps: int, dt: float) -> torch.Tensor:
    states = [u0]
    u = u0
    for _ in range(n_steps):
        u = model.predict_step(u, f, dt=dt)
        states.append(u)
    return torch.stack(states, dim=1)


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
def _evaluate_rollout_rel_l2(
    model: HiddenGradientFlowModel2D,
    split: Dict[str, torch.Tensor],
    device: str,
    dt: float,
    area: float,
) -> float:
    u0 = split["u0"].to(device)
    f = split["f"].to(device)
    u_ref = split["u_traj"].to(device)
    n_steps = int(u_ref.shape[1] - 1)
    u_pred = _rollout(model, u0=u0, f=f, n_steps=n_steps, dt=dt)
    diff = u_pred - u_ref
    num = torch.sqrt(area * torch.sum(diff * diff, dim=(-2, -1)))
    den = torch.sqrt(area * torch.sum(u_ref * u_ref, dim=(-2, -1)))
    rel = num / (den + 1e-8)
    return float(rel.mean().item())


@torch.no_grad()
def _evaluate_rollout_curves(
    model: HiddenGradientFlowModel2D,
    split: Dict[str, torch.Tensor],
    device: str,
    dt: float,
    area: float,
) -> Dict[str, np.ndarray]:
    u0 = split["u0"].to(device)
    f = split["f"].to(device)
    u_ref = split["u_traj"].to(device)
    n_steps = int(u_ref.shape[1] - 1)

    u_pred = _rollout(model, u0=u0, f=f, n_steps=n_steps, dt=dt)
    mse_curve = torch.mean((u_pred - u_ref) ** 2, dim=(0, 2, 3)).cpu().numpy()  # (K+1,)

    diff = u_pred - u_ref
    num = torch.sqrt(area * torch.sum(diff * diff, dim=(-2, -1)))  # (N,K+1)
    den = torch.sqrt(area * torch.sum(u_ref * u_ref, dim=(-2, -1)))  # (N,K+1)
    rel = (num / (den + 1e-8)).mean(dim=0).cpu().numpy()  # (K+1,)

    return {
        "mse_curve": mse_curve.astype(np.float64),
        "rel_curve": rel.astype(np.float64),
    }


def _save_rollout_curve_csv(
    mse_curve: np.ndarray,
    rel_curve: np.ndarray,
    dt: float,
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "time", "mse", "rel_l2"])
        for k, (m, r) in enumerate(zip(mse_curve.tolist(), rel_curve.tolist())):
            writer.writerow([k, f"{k * dt:.8f}", f"{m:.12e}", f"{r:.12e}"])
    print(f"Saved rollout curve csv: {out_path}")


def _plot_rollout_curves(
    mse_curve: np.ndarray,
    rel_curve: np.ndarray,
    dt: float,
    out_path: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping curve plotting because matplotlib is unavailable: {exc}")
        return

    t = np.arange(mse_curve.shape[0], dtype=np.float64) * float(dt)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
    ax1, ax2 = axes[0, 0], axes[0, 1]

    ax1.plot(t, mse_curve, linewidth=2)
    ax1.set_title("Rollout MSE Accumulation")
    ax1.set_xlabel("time")
    ax1.set_ylabel("MSE")
    ax1.grid(alpha=0.3)

    ax2.plot(t, rel_curve, linewidth=2, color="tab:orange")
    ax2.set_title("Rollout Relative L2 Accumulation")
    ax2.set_xlabel("time")
    ax2.set_ylabel("relative L2")
    ax2.grid(alpha=0.3)

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
    n_x = int(u_traj.shape[2])
    n_y = int(u_traj.shape[3])
    n_cols = 1 + len(snapshot_times)

    for sample_id in sample_ids:
        u0_i = u0[sample_id : sample_id + 1].to(device)
        f_i = f[sample_id : sample_id + 1].to(device)
        u_ref_i = u_traj[sample_id]  # (K+1, n_x, n_y)
        u_pred_i = _rollout(model, u0_i, f_i, n_steps=n_steps, dt=dt)[0].cpu()
        area = (1.0 / float(n_x + 1)) * (1.0 / float(n_y + 1))
        diff_i = u_pred_i - u_ref_i
        num_i = torch.sqrt(area * torch.sum(diff_i * diff_i, dim=(-2, -1)))
        den_i = torch.sqrt(area * torch.sum(u_ref_i * u_ref_i, dim=(-2, -1)))
        rel_curve_i = num_i / (den_i + 1e-8)
        rel_mean_i = float(rel_curve_i.mean().item())
        rel_final_i = float(rel_curve_i[-1].item())

        f_int = to_interior_field_2d(f[sample_id], n_x=n_x, n_y=n_y).squeeze(0)
        state_scale = max(float(torch.max(torch.abs(u_ref_i)).item()), float(torch.max(torch.abs(u_pred_i)).item()), 1e-8)

        fig, axes = plt.subplots(
            3,
            n_cols,
            figsize=(3.1 * n_cols, 8.0),
            squeeze=False,
            constrained_layout=True,
        )

        # Row 0 col 0: forcing
        im_force = axes[0, 0].imshow(
            f_int.cpu().numpy(),
            origin="lower",
            cmap="coolwarm",
            extent=[0.0, 1.0, 0.0, 1.0],
            aspect="auto",
        )
        axes[0, 0].set_title("forcing")
        axes[0, 0].set_xticks([])
        axes[0, 0].set_yticks([])
        axes[1, 0].axis("off")
        axes[1, 0].text(0.5, 0.5, "pred", ha="center", va="center", fontsize=12)
        axes[2, 0].axis("off")
        axes[2, 0].text(0.5, 0.5, "|pred-ref|", ha="center", va="center", fontsize=12)

        im_ref_last = im_force
        im_err_last = None
        for j, t_snap in enumerate(snapshot_times, start=1):
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
                vmin=-state_scale,
                vmax=state_scale,
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
                vmin=-state_scale,
                vmax=state_scale,
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

        cbar_state = fig.colorbar(im_ref_last, ax=axes[0:2, 1:], fraction=0.015, pad=0.01)
        cbar_state.ax.set_ylabel("state value", rotation=90)
        if im_err_last is not None:
            cbar_err = fig.colorbar(im_err_last, ax=axes[2, 1:], fraction=0.015, pad=0.01)
            cbar_err.ax.set_ylabel("abs error", rotation=90)
        cbar_f = fig.colorbar(im_force, ax=[axes[0, 0]], fraction=0.046, pad=0.02)
        cbar_f.ax.set_ylabel("forcing", rotation=90)

        fig.suptitle(
            f"Sample {sample_id}: forcing + reference/prediction snapshots | "
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
    print(f"Device: {device}")

    splits = load_dataset_splits(args.dataset_path, map_location="cpu")
    split = splits[args.split]
    meta = splits.get("meta", {})
    n_x = int(split["u0"].shape[1])
    n_y = int(split["u0"].shape[2])
    n_steps = int(split["u_traj"].shape[1] - 1)
    t_final = float(meta.get("t_final", 1.0))
    h_x = 1.0 / float(n_x + 1)
    h_y = 1.0 / float(n_y + 1)
    area = h_x * h_y
    dt = t_final / float(n_steps)
    snapshot_times = _parse_snapshot_times(args.snapshot_times, t_final=t_final)

    print(f"Loaded split={args.split} from {args.dataset_path}")
    print(f"Grid: n_x={n_x}, n_y={n_y}, n_steps={n_steps}, t_final={t_final:.6f}, dt={dt:.6f}")

    model = _build_model(n_x=n_x, n_y=n_y, h_x=h_x, h_y=h_y, dt=dt, args=args).to(device)
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(_remap_checkpoint_state_dict(ckpt["model_state_dict"]), strict=True)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint_path}")

    step_mse = _evaluate_one_step_mse(model, split, device=device, dt=dt)
    rollout_rel = _evaluate_rollout_rel_l2(model, split, device=device, dt=dt, area=area)
    curves = _evaluate_rollout_curves(model, split, device=device, dt=dt, area=area)
    mse_curve = curves["mse_curve"]
    rel_curve = curves["rel_curve"]
    print(f"Split one-step MSE: {step_mse:.8e}")
    print(f"Split rollout mean relative L2: {rollout_rel:.8e}")
    print("Rollout accumulation by step (step, time, mse, rel_l2):")
    for k, (m, r) in enumerate(zip(mse_curve.tolist(), rel_curve.tolist())):
        print(f"  {k:03d}  {k * dt:8.4f}  {m:.8e}  {r:.8e}")

    curve_csv = os.path.join(args.output_dir, f"{args.split}_rollout_error_curve.csv")
    curve_png = os.path.join(args.output_dir, f"{args.split}_rollout_error_curve.png")
    _save_rollout_curve_csv(mse_curve=mse_curve, rel_curve=rel_curve, dt=dt, out_path=curve_csv)
    _plot_rollout_curves(mse_curve=mse_curve, rel_curve=rel_curve, dt=dt, out_path=curve_png)

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
