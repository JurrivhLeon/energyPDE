"""
Evaluation script for hidden-space Burgers models.

Reports:
1) One-step split MSE.
2) Rollout error accumulation by step (MSE and relative L2).
3) Per-sample comparison plots (reference vs prediction + snapshots).
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from ..heat_data import build_step_dataset, build_trajectory_dataset_from_split, load_dataset_splits
    from ..generator import (
        HiddenGradientFlowModel1D,
        LatentEnergyHead1D,
        LatentGradientStep1D,
        LatentStateDecoder1D,
        LatentStateEncoder1D,
    )
    from ..utils import compute_relative_l2_error, pad_dirichlet_1d, rollout_model
except ImportError:
    from grad_flow_l2.heat_data import build_step_dataset, build_trajectory_dataset_from_split, load_dataset_splits
    from grad_flow_l2.generator import (
        HiddenGradientFlowModel1D,
        LatentEnergyHead1D,
        LatentGradientStep1D,
        LatentStateDecoder1D,
        LatentStateEncoder1D,
    )
    from grad_flow_l2.utils import compute_relative_l2_error, pad_dirichlet_1d, rollout_model


def _parse_snapshot_times(raw: str) -> List[float]:
    vals = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = float(tok)
        if v < 0.0 or v > 1.0:
            raise ValueError(f"Snapshot time must be in [0,1], got {v}")
        vals.append(v)
    if not vals:
        vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    return vals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained hidden-space Burgers model")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="grad_flow_l2/datasets/burgers_l2_nu0p01_nx100_steps10.pt",
        help="Path to cached dataset file (.pt)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="grad_flow_l2/burgers/outputs/run_20260305_001234/best_model.pt",
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")

    # Model architecture (must match training config).
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--enc-blocks", type=int, default=4)
    parser.add_argument("--dec-blocks", type=int, default=4)
    parser.add_argument("--latent-blocks", type=int, default=6)
    parser.add_argument("--energy-layers", type=int, default=4)
    parser.add_argument("--use-dt-channel", action="store_true")
    parser.add_argument("--disable-forcing-channel", action="store_true")
    parser.add_argument("--disable-zx-feature", action="store_true")

    # Visualization/output.
    parser.add_argument("--n-plot-samples", type=int, default=8)
    parser.add_argument("--snapshot-times", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--output-dir", type=str, default="grad_flow_l2/burgers/outputs/eval")
    return parser.parse_args()


def _build_model(
    n_x: int,
    h: float,
    dt: float,
    args: argparse.Namespace,
) -> HiddenGradientFlowModel1D:
    use_forcing_channel = not args.disable_forcing_channel
    encoder = LatentStateEncoder1D(
        n_x=n_x,
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
        n_blocks=args.enc_blocks,
        use_ux_feature=True,
    )
    latent_step = LatentGradientStep1D(
        n_x=n_x,
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
        n_blocks=args.latent_blocks,
        use_forcing_channel=use_forcing_channel,
        use_dt_channel=args.use_dt_channel,
        default_dt=dt,
    )
    decoder = LatentStateDecoder1D(
        n_x=n_x,
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
        n_blocks=args.dec_blocks,
    )
    latent_energy_head = LatentEnergyHead1D(
        n_x=n_x,
        h=h,
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
        n_layers=args.energy_layers,
        use_forcing_channel=use_forcing_channel,
        use_zx_norm_feature=not args.disable_zx_feature,
    )
    return HiddenGradientFlowModel1D(
        encoder=encoder,
        latent_step=latent_step,
        decoder=decoder,
        latent_energy_head=latent_energy_head,
    )


@torch.no_grad()
def evaluate_one_step_mse(
    model: HiddenGradientFlowModel1D,
    step_loader: DataLoader,
    device: str,
    dt: float,
) -> float:
    model.eval()
    sq_sum = 0.0
    n_elem = 0
    for batch in step_loader:
        u_k, u_k1, f = batch
        u_k = u_k.to(device)
        u_k1 = u_k1.to(device)
        f = f.to(device)
        pred = model.predict_step(u_k, f, dt=dt)
        sq_sum += F.mse_loss(pred, u_k1, reduction="sum").item()
        n_elem += int(np.prod(u_k1.shape))
    return sq_sum / max(1, n_elem)


@torch.no_grad()
def evaluate_rollout_curves(
    model: HiddenGradientFlowModel1D,
    traj_loader: DataLoader,
    device: str,
    dt: float,
    h: float,
    n_steps: int,
) -> Dict[str, np.ndarray]:
    model.eval()
    mse_sum = np.zeros(n_steps + 1, dtype=np.float64)
    rel_sum = np.zeros(n_steps + 1, dtype=np.float64)
    n_samples = 0

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

        u_pred = rollout_model(model, u0=u0, f=f, n_steps=n_steps, dt=dt)
        mse_t = torch.mean((u_pred - u_ref) ** 2, dim=-1)
        rel_t = compute_relative_l2_error(u_pred, u_ref, h=h)

        mse_sum += mse_t.sum(dim=0).cpu().numpy()
        rel_sum += rel_t.sum(dim=0).cpu().numpy()
        n_samples += int(u0.shape[0])

    mse_curve = mse_sum / max(1, n_samples)
    rel_curve = rel_sum / max(1, n_samples)
    return {"mse_curve": mse_curve, "rel_curve": rel_curve}


def _plot_error_curve(
    mse_curve: np.ndarray,
    rel_curve: np.ndarray,
    out_path: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping curve plot because matplotlib is unavailable: {exc}")
        return

    steps = np.arange(mse_curve.shape[0], dtype=np.int64)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
    ax1, ax2 = axes[0, 0], axes[0, 1]

    ax1.plot(steps, mse_curve, linewidth=2)
    ax1.set_title("Rollout MSE by Step")
    ax1.set_xlabel("step k")
    ax1.set_ylabel("MSE(u_k^pred, u_k^ref)")
    ax1.grid(alpha=0.3)

    ax2.plot(steps, rel_curve, linewidth=2, color="tab:orange")
    ax2.set_title("Rollout Relative L2 by Step")
    ax2.set_xlabel("step k")
    ax2.set_ylabel("relative L2")
    ax2.grid(alpha=0.3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved error-curve plot: {out_path}")


def _save_curve_csv(
    mse_curve: np.ndarray,
    rel_curve: np.ndarray,
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "mse", "rel_l2"])
        for k, (m, r) in enumerate(zip(mse_curve.tolist(), rel_curve.tolist())):
            writer.writerow([k, f"{m:.12e}", f"{r:.12e}"])
    print(f"Saved error-curve csv: {out_path}")


@torch.no_grad()
def _plot_sample_comparisons(
    model: HiddenGradientFlowModel1D,
    split: Dict[str, torch.Tensor],
    device: str,
    dt: float,
    n_plot_samples: int,
    snapshot_times: Sequence[float],
    out_dir: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping sample plots because matplotlib is unavailable: {exc}")
        return

    u_traj = split["u_traj"]  # (N, K+1, n_x)
    u0 = split["u0"]          # (N, n_x)
    f = split["f"]            # (N, n_x) or (N, n_x+2)
    total = int(u_traj.shape[0])
    n_plot = min(max(1, int(n_plot_samples)), total)
    idx = torch.linspace(0, total - 1, n_plot).long().tolist()

    n_steps = int(u_traj.shape[1] - 1)
    n_x = int(u_traj.shape[2])
    h = 1.0 / float(n_x + 1)
    x_full = torch.linspace(0.0, 1.0, n_x + 2)
    x_interior = torch.linspace(h, 1.0 - h, n_x)

    os.makedirs(out_dir, exist_ok=True)

    for i_t in idx:
        u0_i = u0[i_t : i_t + 1].to(device)
        f_i = f[i_t : i_t + 1].to(device)
        u_ref_i = u_traj[i_t]
        u_pred_i = rollout_model(model, u0=u0_i, f=f_i, n_steps=n_steps, dt=dt)[0].cpu()

        u_ref_full = pad_dirichlet_1d(u_ref_i)
        u_pred_full = pad_dirichlet_1d(u_pred_i)
        err_full = torch.abs(u_pred_full - u_ref_full)

        vmin = min(float(u_ref_full.min()), float(u_pred_full.min()))
        vmax = max(float(u_ref_full.max()), float(u_pred_full.max()))

        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 3)
        ax_f = fig.add_subplot(gs[0, 0])
        ax_ref = fig.add_subplot(gs[0, 1])
        ax_pred = fig.add_subplot(gs[0, 2])
        ax_snap = fig.add_subplot(gs[1, :])

        if f.shape[1] == n_x + 2:
            ax_f.plot(x_full.numpy(), f[i_t].cpu().numpy(), linewidth=2)
            ax_f.set_title(f"sample {i_t}: forcing f(x) (full grid)")
        else:
            ax_f.plot(x_interior.numpy(), f[i_t].cpu().numpy(), linewidth=2)
            ax_f.set_title(f"sample {i_t}: forcing f(x) (interior)")
        ax_f.set_xlabel("x")
        ax_f.set_ylabel("f")
        ax_f.grid(alpha=0.3)

        im_ref = ax_ref.imshow(
            u_ref_full.numpy(),
            aspect="auto",
            origin="lower",
            extent=[0.0, 1.0, 0.0, 1.0],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax_ref.set_title("reference u(x,t)")
        ax_ref.set_xlabel("x")
        ax_ref.set_ylabel("t")
        fig.colorbar(im_ref, ax=ax_ref, fraction=0.046, pad=0.04)

        im_pred = ax_pred.imshow(
            u_pred_full.numpy(),
            aspect="auto",
            origin="lower",
            extent=[0.0, 1.0, 0.0, 1.0],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax_pred.set_title("predicted u(x,t)")
        ax_pred.set_xlabel("x")
        ax_pred.set_ylabel("t")
        fig.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

        snapshot_colors = ["k", "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:brown", "tab:pink"]
        for j, t_snap in enumerate(snapshot_times):
            k = int(round(float(t_snap) * n_steps))
            k = max(0, min(n_steps, k))
            c = snapshot_colors[j % len(snapshot_colors)]
            ax_snap.plot(
                x_full.numpy(),
                u_ref_full[k].numpy(),
                color=c,
                linewidth=2,
                label=f"ref t={t_snap:.2f}",
            )
            ax_snap.plot(
                x_full.numpy(),
                u_pred_full[k].numpy(),
                color=c,
                linewidth=2,
                linestyle="--",
                label=f"pred t={t_snap:.2f}",
            )
        ax_snap.set_title("snapshot comparison (solid=ref, dashed=pred)")
        ax_snap.set_xlabel("x")
        ax_snap.set_ylabel("u")
        ax_snap.grid(alpha=0.3)
        ax_snap.legend(loc="best", fontsize=8, ncol=2)

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"sample_{i_t:04d}_comparison.png")
        fig.savefig(out_path, dpi=180)
        plt.close(fig)

        fig_err, ax_err = plt.subplots(1, 1, figsize=(6, 3.5))
        im_err = ax_err.imshow(
            err_full.numpy(),
            aspect="auto",
            origin="lower",
            extent=[0.0, 1.0, 0.0, 1.0],
            cmap="magma",
        )
        ax_err.set_title(f"sample {i_t}: |pred-ref|")
        ax_err.set_xlabel("x")
        ax_err.set_ylabel("t")
        fig_err.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)
        fig_err.tight_layout()
        err_path = os.path.join(out_dir, f"sample_{i_t:04d}_abs_error.png")
        fig_err.savefig(err_path, dpi=180)
        plt.close(fig_err)

    print(f"Saved sample comparison plots in: {out_dir}")


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    splits = load_dataset_splits(args.dataset_path, map_location="cpu")
    split = splits[args.split]
    meta = splits.get("meta", {})
    n_x = int(split["u0"].shape[1])
    n_steps = int(split["u_traj"].shape[1] - 1)
    t_final = float(meta.get("t_final", 1.0))
    h = 1.0 / float(n_x + 1)
    dt_from_data = t_final / float(n_steps)
    print(f"Loaded split={args.split} from {args.dataset_path}")
    print(f"Grid: n_x={n_x}, n_steps={n_steps}, t_final={t_final:.6f}, h={h:.6f}, dt={dt_from_data:.6f}")

    model = _build_model(n_x=n_x, h=h, dt=dt_from_data, args=args).to(device)

    ckpt = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint_path}")
    if "dt" in ckpt:
        print(f"Checkpoint dt={float(ckpt['dt']):.6f}")

    step_ds = build_step_dataset(split)
    traj_ds = build_trajectory_dataset_from_split(split)
    step_loader = DataLoader(step_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    traj_loader = DataLoader(traj_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    split_mse = evaluate_one_step_mse(model, step_loader, device=device, dt=dt_from_data)
    curves = evaluate_rollout_curves(
        model,
        traj_loader=traj_loader,
        device=device,
        dt=dt_from_data,
        h=h,
        n_steps=n_steps,
    )
    mse_curve = curves["mse_curve"]
    rel_curve = curves["rel_curve"]

    print(f"One-step split MSE: {split_mse:.8e}")
    print("Rollout accumulation (step, mse, rel_l2):")
    for k in range(n_steps + 1):
        print(f"  {k:03d}  {mse_curve[k]:.8e}  {rel_curve[k]:.8e}")

    curve_csv = os.path.join(args.output_dir, f"{args.split}_rollout_error_curve.csv")
    curve_png = os.path.join(args.output_dir, f"{args.split}_rollout_error_curve.png")
    _save_curve_csv(mse_curve, rel_curve, out_path=curve_csv)
    _plot_error_curve(mse_curve, rel_curve, out_path=curve_png)

    snapshot_times = _parse_snapshot_times(args.snapshot_times)
    sample_plot_dir = os.path.join(args.output_dir, f"{args.split}_sample_comparisons")
    _plot_sample_comparisons(
        model=model,
        split=split,
        device=device,
        dt=dt_from_data,
        n_plot_samples=args.n_plot_samples,
        snapshot_times=snapshot_times,
        out_dir=sample_plot_dir,
    )


if __name__ == "__main__":
    main(parse_args())
