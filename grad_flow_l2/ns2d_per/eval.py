"""
Evaluation and sample-visualization script for periodic 2D Navier-Stokes hidden-space models.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

try:
    from ..heat_data import load_dataset_splits
    from ..latent_markov import (
        FNOProximalStepSimulator2D,
        LatentMarkovModel2D,
        ProximalStepSimulator2D,
        StateDecoder2D,
        StateEncoder2D,
    )
    from ..latent_markov_trainer import relative_spectral_hs_error_2d
except ImportError:
    from grad_flow_l2.heat_data import load_dataset_splits
    from grad_flow_l2.latent_markov import (
        FNOProximalStepSimulator2D,
        LatentMarkovModel2D,
        ProximalStepSimulator2D,
        StateDecoder2D,
        StateEncoder2D,
    )
    from grad_flow_l2.latent_markov_trainer import relative_spectral_hs_error_2d


def set_seed(seed: int, seed_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda:
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and visualize periodic 2D Navier-Stokes hidden-space model")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to cached periodic dataset (.pt)")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--n-plot-samples", type=int, default=6)
    parser.add_argument("--snapshot-times", type=str, default="0,2,4,6,8,10")
    parser.add_argument(
        "--delta-clip",
        type=float,
        default=10.0,
        help="L-infinity clip applied to the predicted increment delta = u_tilde - u_t before accumulation.",
    )
    parser.add_argument("--output-dir", type=str, default="grad_flow_l2/ns2d_per/outputs/eval")
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--use-dt-channel", action="store_true")
    parser.add_argument("--disable-forcing-channel", action="store_true")
    parser.add_argument("--disable-u-grad-feature", action="store_true")
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


def _load_checkpoint(checkpoint_path: str, map_location: str | torch.device) -> Dict[str, object]:
    """Load checkpoints across PyTorch versions, including legacy tar saves.

    PyTorch 2.6 changed ``torch.load`` to default to ``weights_only=True``.
    That fails for older checkpoints serialized in the legacy tar format, so
    we retry with ``weights_only=False`` only for that specific compatibility
    case. This should only be used for trusted local checkpoints.
    """

    try:
        return torch.load(checkpoint_path, map_location=map_location)
    except RuntimeError as exc:
        msg = str(exc)
        if "weights_only=True" not in msg or "legacy .tar format" not in msg:
            raise
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def _checkpoint_state_for_model(model: LatentMarkovModel2D, checkpoint_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in checkpoint_state.items() if not k.startswith("energy_head.")}


def _build_model(n_x: int, n_y: int, h_x: float, h_y: float, dt: float, args: argparse.Namespace) -> LatentMarkovModel2D:
    boundary_condition = "periodic"
    use_forcing_channel = not args.disable_forcing_channel
    encoder = StateEncoder2D(
        n_x=n_x,
        n_y=n_y,
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
        n_blocks=args.enc_blocks,
        use_grad_features=not args.disable_u_grad_feature,
        boundary_condition=boundary_condition,
    )
    decoder = StateDecoder2D(
        n_x=n_x,
        n_y=n_y,
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
        n_blocks=args.dec_blocks,
        boundary_condition=boundary_condition,
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
            boundary_condition=boundary_condition,
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
            boundary_condition=boundary_condition,
        )
    else:
        raise ValueError(f"Unsupported prox-simulator-type: {args.prox_simulator_type}")
    return LatentMarkovModel2D(
        encoder=encoder,
        decoder=decoder,
        transition=prox_step,
    )


@torch.no_grad()
def _rollout(
    model: LatentMarkovModel2D,
    u0: torch.Tensor,
    f: torch.Tensor,
    n_steps: int,
    dt: float,
    delta_clip: float = 10.0,
) -> torch.Tensor:
    states = [u0]
    u = u0
    for _ in range(n_steps):
        u_tilde = model.predict_step(u, f, dt=dt)
        delta = u_tilde - u
        if delta_clip is not None and float(delta_clip) > 0.0:
            delta = torch.clamp(delta, min=-float(delta_clip), max=float(delta_clip))
        u = u + delta
        finite = torch.isfinite(u).flatten(1).all(dim=1)
        u = torch.where(finite[:, None, None], u, states[-1])
        states.append(u)
    return torch.stack(states, dim=1)


@torch.no_grad()
def _evaluate_one_step_mse(model: LatentMarkovModel2D, split: Dict[str, torch.Tensor], device: str, dt: float) -> float:
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
    model: LatentMarkovModel2D,
    split: Dict[str, torch.Tensor],
    device: str,
    dt: float,
    area: float,
    delta_clip: float,
) -> float:
    u0 = split["u0"].to(device)
    f = split["f"].to(device)
    u_ref = split["u_traj"].to(device)
    n_steps = int(u_ref.shape[1] - 1)
    u_pred = _rollout(
        model,
        u0=u0,
        f=f,
        n_steps=n_steps,
        dt=dt,
        delta_clip=delta_clip,
    )
    diff = u_pred - u_ref
    num = torch.sqrt(area * torch.sum(diff * diff, dim=(-2, -1)))
    den = torch.sqrt(area * torch.sum(u_ref * u_ref, dim=(-2, -1)))
    rel = num / (den + 1e-8)
    return float(rel.mean().item())


@torch.no_grad()
def _evaluate_rollout_curves(
    model: LatentMarkovModel2D,
    split: Dict[str, torch.Tensor],
    device: str,
    dt: float,
    area: float,
    delta_clip: float,
) -> Dict[str, np.ndarray]:
    u0 = split["u0"].to(device)
    f = split["f"].to(device)
    u_ref = split["u_traj"].to(device)
    n_steps = int(u_ref.shape[1] - 1)

    u_pred = _rollout(
        model,
        u0=u0,
        f=f,
        n_steps=n_steps,
        dt=dt,
        delta_clip=delta_clip,
    )
    diff = u_pred - u_ref
    num = torch.sqrt(area * torch.sum(diff * diff, dim=(-2, -1)))
    den = torch.sqrt(area * torch.sum(u_ref * u_ref, dim=(-2, -1)))
    rel = num / (den + 1e-8)
    rel_curve_mean = torch.nanmean(rel, dim=0).cpu().numpy()
    rel_curve_median = np.nanmedian(rel.cpu().numpy(), axis=0)
    rel_per_sample_mean = torch.nanmean(rel, dim=1).cpu().numpy()
    rel_h1 = relative_spectral_hs_error_2d(u_pred, u_ref, s=1.0)
    rel_h1_curve_mean = torch.nanmean(rel_h1, dim=0).cpu().numpy()
    rel_h1_curve_median = np.nanmedian(rel_h1.cpu().numpy(), axis=0)
    rel_h1_per_sample_mean = torch.nanmean(rel_h1, dim=1).cpu().numpy()

    return {
        "rel_curve_mean": rel_curve_mean.astype(np.float64),
        "rel_curve_median": rel_curve_median.astype(np.float64),
        "rollout_rel_mean": float(np.nanmean(rel_per_sample_mean)),
        "rollout_rel_median": float(np.nanmedian(rel_per_sample_mean)),
        "rel_h1_curve_mean": rel_h1_curve_mean.astype(np.float64),
        "rel_h1_curve_median": rel_h1_curve_median.astype(np.float64),
        "rollout_rel_h1": float(np.nanmean(rel_h1_per_sample_mean)),
        "rollout_rel_h1_median": float(np.nanmedian(rel_h1_per_sample_mean)),
    }


def _save_rollout_curve_csv(
    rel_curve_mean: np.ndarray,
    rel_curve_median: np.ndarray,
    rel_h1_curve_mean: np.ndarray,
    rel_h1_curve_median: np.ndarray,
    time_values: np.ndarray,
    out_path: str,
) -> None:
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
            ]
        )
        for k in range(len(rel_curve_mean)):
            writer.writerow(
                [
                    k,
                    f"{float(time_values[k]):.8f}",
                    f"{float(rel_curve_mean[k]):.12e}",
                    f"{float(rel_curve_median[k]):.12e}",
                    f"{float(rel_h1_curve_mean[k]):.12e}",
                    f"{float(rel_h1_curve_median[k]):.12e}",
                ]
            )
    print(f"Saved rollout curve csv: {out_path}")


def _plot_rollout_curves(
    rel_curve_mean: np.ndarray,
    rel_curve_median: np.ndarray,
    rel_h1_curve_mean: np.ndarray,
    rel_h1_curve_median: np.ndarray,
    time_values: np.ndarray,
    out_path: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping curve plotting because matplotlib is unavailable: {exc}")
        return

    t = time_values[: rel_curve_mean.shape[0]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
    ax1, ax2 = axes[0, 0], axes[0, 1]

    ax1.plot(t, rel_curve_mean, linewidth=2, color="tab:orange", label="mean")
    ax1.plot(t, rel_curve_median, linewidth=2, linestyle="--", color="tab:green", label="median")
    ax1.set_title("Rollout Relative L2 Accumulation")
    ax1.set_xlabel("time")
    ax1.set_ylabel("relative L2")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(t, rel_h1_curve_mean, linewidth=2, color="tab:red", label="mean")
    ax2.plot(t, rel_h1_curve_median, linewidth=2, linestyle="--", color="tab:purple", label="median")
    ax2.set_title("Rollout Relative H1 Accumulation")
    ax2.set_xlabel("time")
    ax2.set_ylabel("relative H1")
    ax2.legend()
    ax2.grid(alpha=0.3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved rollout curve plot: {out_path}")


@torch.no_grad()
def _plot_test_samples(
    model: LatentMarkovModel2D,
    split: Dict[str, torch.Tensor],
    device: str,
    dt: float,
    time_values: np.ndarray,
    snapshot_times: List[float],
    n_plot_samples: int,
    out_dir: str,
    delta_clip: float,
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
    error_snapshot_times = [t for t in snapshot_times if t > 0.0]
    error_start_col = 2 if snapshot_times and snapshot_times[0] <= 0.0 else 1
    area = (1.0 / float(n_x)) * (1.0 / float(n_y))

    for sample_id in sample_ids:
        u0_i = u0[sample_id : sample_id + 1].to(device)
        f_i = f[sample_id : sample_id + 1].to(device)
        u_ref_i = u_traj[sample_id]
        u_pred_i = _rollout(
            model,
            u0_i,
            f_i,
            n_steps=n_steps,
            dt=dt,
            delta_clip=delta_clip,
        )[0].cpu()
        diff_i = u_pred_i - u_ref_i
        num_i = torch.sqrt(area * torch.sum(diff_i * diff_i, dim=(-2, -1)))
        den_i = torch.sqrt(area * torch.sum(u_ref_i * u_ref_i, dim=(-2, -1)))
        rel_curve_i = num_i / (den_i + 1e-8)
        rel_mean_i = float(rel_curve_i.mean().item())
        rel_final_i = float(rel_curve_i[-1].item())

        f_plot = f[sample_id].cpu().numpy()
        state_scale = max(float(torch.max(torch.abs(u_ref_i)).item()), float(torch.max(torch.abs(u_pred_i)).item()), 1e-8)

        fig, axes = plt.subplots(
            3,
            n_cols,
            figsize=(3.1 * n_cols, 8.0),
            squeeze=False,
            constrained_layout=True,
        )

        im_force = axes[0, 0].imshow(
            f_plot,
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
            ax_ref.set_title(f"ref t={t_label:g}")
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
            ax_pred.set_title(f"pred t={t_label:g}")
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])

            im_ref_last = im_ref
            if k > 0:
                im_err = ax_err.imshow(
                    err_k.cpu().numpy(),
                    origin="lower",
                    cmap="magma",
                    extent=[0.0, 1.0, 0.0, 1.0],
                    aspect="auto",
                )
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
    set_seed(args.seed, seed_cuda=(device == "cuda"))
    print(f"Device: {device}")

    splits = load_dataset_splits(args.dataset_path, map_location="cpu")
    split = splits[args.split]
    meta = splits.get("meta", {})
    n_x = int(split["u0"].shape[1])
    n_y = int(split["u0"].shape[2])
    n_steps = int(split["u_traj"].shape[1] - 1)
    h_x = 1.0 / float(n_x)
    h_y = 1.0 / float(n_y)
    area = h_x * h_y
    dt, t_start, t_final, time_values = _time_metadata(meta, n_steps=n_steps)
    snapshot_times = _parse_snapshot_times(args.snapshot_times, t_start=t_start, t_end=t_final)

    print(f"Loaded split={args.split} from {args.dataset_path}")
    print(
        f"Grid: n_x={n_x}, n_y={n_y}, n_steps={n_steps}, "
        f"stored_time=[{t_start:.6f},{t_final:.6f}], dt={dt:.6f}"
    )
    print(f"Delta clip (L-inf): {args.delta_clip:.6f}")

    model = _build_model(n_x=n_x, n_y=n_y, h_x=h_x, h_y=h_y, dt=dt, args=args).to(device)
    ckpt = _load_checkpoint(args.checkpoint_path, map_location=device)
    model.load_state_dict(_checkpoint_state_for_model(model, ckpt["model_state_dict"]), strict=True)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint_path}")

    step_mse = _evaluate_one_step_mse(model, split, device=device, dt=dt)
    rollout_rel = _evaluate_rollout_rel_l2(
        model,
        split,
        device=device,
        dt=dt,
        area=area,
        delta_clip=args.delta_clip,
    )
    curves = _evaluate_rollout_curves(
        model,
        split,
        device=device,
        dt=dt,
        area=area,
        delta_clip=args.delta_clip,
    )
    rel_curve_mean = curves["rel_curve_mean"]
    rel_curve_median = curves["rel_curve_median"]
    print(f"Split one-step MSE: {step_mse:.8e}")
    print(f"Split rollout mean relative L2: {rollout_rel:.8e}")
    print(f"Split rollout median relative L2: {curves['rollout_rel_median']:.8e}")
    print(f"Split rollout mean relative H1: {curves['rollout_rel_h1']:.8e}")
    print(f"Split rollout median relative H1: {curves['rollout_rel_h1_median']:.8e}")
    print(
        "Rollout accumulation by step "
        "(step, time, rel_l2_mean, rel_l2_median, rel_h1_mean, rel_h1_median):"
    )
    for k in range(len(rel_curve_mean)):
        print(
            f"  {k:03d}  {float(time_values[k]):8.4f}  "
            f"{rel_curve_mean[k]:.8e}  {rel_curve_median[k]:.8e}  "
            f"{curves['rel_h1_curve_mean'][k]:.8e}  {curves['rel_h1_curve_median'][k]:.8e}"
        )

    curve_csv = os.path.join(args.output_dir, f"{args.split}_rollout_error_curve.csv")
    curve_png = os.path.join(args.output_dir, f"{args.split}_rollout_error_curve.png")
    _save_rollout_curve_csv(
        rel_curve_mean=curves["rel_curve_mean"],
        rel_curve_median=curves["rel_curve_median"],
        rel_h1_curve_mean=curves["rel_h1_curve_mean"],
        rel_h1_curve_median=curves["rel_h1_curve_median"],
        time_values=time_values,
        out_path=curve_csv,
    )
    _plot_rollout_curves(
        rel_curve_mean=curves["rel_curve_mean"],
        rel_curve_median=curves["rel_curve_median"],
        rel_h1_curve_mean=curves["rel_h1_curve_mean"],
        rel_h1_curve_median=curves["rel_h1_curve_median"],
        time_values=time_values,
        out_path=curve_png,
    )

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
        delta_clip=args.delta_clip,
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
        "delta_clip": args.delta_clip,
        "step_mse": step_mse,
        "rollout_rel_l2": rollout_rel,
        "rollout_rel_l2_median": curves["rollout_rel_median"],
        "rollout_rel_h1": curves["rollout_rel_h1"],
        "rollout_rel_h1_median": curves["rollout_rel_h1_median"],
        "rel_curve_mean": curves["rel_curve_mean"].tolist(),
        "rel_curve_median": curves["rel_curve_median"].tolist(),
        "rel_h1_curve_mean": curves["rel_h1_curve_mean"].tolist(),
        "rel_h1_curve_median": curves["rel_h1_curve_median"].tolist(),
        "snapshot_times": snapshot_times,
        "seed": int(args.seed),
        "meta": meta,
    }
    with open(os.path.join(args.output_dir, f"{args.split}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved evaluation summary to: {args.output_dir}")


if __name__ == "__main__":
    main(parse_args())
