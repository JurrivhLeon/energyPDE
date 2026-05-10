"""
Generate reference KS trajectory plots over a requested time interval.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Sequence

import numpy as np
import torch

try:
    from ..burgers_data import spectral_truncate_periodic_field_1d
    from .ks_data import sample_iid_initial_conditions, sample_matern_initial_conditions, solve_ks_trajectory_etdrk4
except ImportError:
    from grad_flow_l2.burgers_data import spectral_truncate_periodic_field_1d
    from grad_flow_l2.ks1d.ks_data import (
        sample_iid_initial_conditions,
        sample_matern_initial_conditions,
        solve_ks_trajectory_etdrk4,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot unforced periodic 1D KS reference trajectories")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--n-x", type=int, default=512, help="Stored/plot grid size")
    parser.add_argument("--solver-n-x", type=int, default=1024, help="Reference solver grid size")
    parser.add_argument("--domain-length", type=float, default=32.0 * math.pi)
    parser.add_argument("--solver-dt", type=float, default=0.01)
    parser.add_argument("--dataset-dt", type=float, default=1.0)
    parser.add_argument("--t-final", type=float, default=50.0)
    parser.add_argument("--warmup-time", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--u0-sampler", type=str, default="matern", choices=["matern", "iid"])
    parser.add_argument("--u0-std", type=float, default=0.1)
    parser.add_argument("--matern-smoothness", type=float, default=1.5)
    parser.add_argument("--matern-length-scale", type=float, default=10.0)
    parser.add_argument(
        "--plot-max-modes",
        type=int,
        default=64,
        help="Retain this many positive Fourier modes in plotted trajectories. Use <=0 to disable plot-only smoothing.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="grad_flow_l2/ks1d/plots")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _integer_steps(t_final: float, dataset_dt: float) -> int:
    ratio = float(t_final) / float(dataset_dt)
    n_steps = int(round(ratio))
    if n_steps < 1 or abs(ratio - n_steps) > 1e-10:
        raise ValueError("t_final must be a positive integer multiple of dataset_dt")
    return n_steps


def _snapshot_indices(times: np.ndarray, requested: Sequence[float]) -> list[int]:
    out = []
    for t in requested:
        idx = int(np.argmin(np.abs(times - float(t))))
        if idx not in out:
            out.append(idx)
    return out


def _lowpass_for_plot(field: torch.Tensor, max_modes: int) -> torch.Tensor:
    """
    Smooth only the plotted field by truncating high spatial Fourier modes.

    Args:
        field: (..., n_x)
        max_modes: number of nonnegative rfft modes to retain, including zero.
    """
    if max_modes <= 0:
        return field
    n_x = int(field.shape[-1])
    n_freq = n_x // 2 + 1
    keep = min(int(max_modes), n_freq)
    if keep >= n_freq:
        return field
    field_hat = torch.fft.rfft(field, dim=-1)
    field_hat[..., keep:] = 0.0
    return torch.fft.irfft(field_hat, n=n_x, dim=-1)


def _sample_initial_conditions(args: argparse.Namespace, dtype: torch.dtype) -> torch.Tensor:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.u0_sampler == "matern":
        return sample_matern_initial_conditions(
            n_x=args.solver_n_x,
            n_samples=args.n_samples,
            domain_length=args.domain_length,
            smoothness=args.matern_smoothness,
            length_scale=args.matern_length_scale,
            std=args.u0_std,
            zero_mean=True,
            device=args.device,
            dtype=dtype,
        )
    return sample_iid_initial_conditions(
        n_x=args.solver_n_x,
        n_samples=args.n_samples,
        std=args.u0_std,
        zero_mean=True,
        device=args.device,
        dtype=dtype,
    )


def _plot_sample_heatmap(
    traj: torch.Tensor,
    sample_id: int,
    x: np.ndarray,
    times: np.ndarray,
    output_dir: str,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    arr = traj[sample_id].cpu().numpy()
    scale = max(float(np.max(np.abs(arr))), 1e-8)
    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    im = ax.imshow(
        arr.T,
        aspect="auto",
        origin="lower",
        interpolation="bilinear",
        extent=[float(times[0]), float(times[-1]), float(x[0]), float(x[-1] + (x[1] - x[0]))],
        cmap="RdBu_r",
        vmin=-scale,
        vmax=scale,
    )
    ax.set_title(f"KS trajectory {sample_id}")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    fig.colorbar(im, ax=ax, label="u")
    out_path = os.path.join(output_dir, f"trajectory_{sample_id:02d}_heatmap.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_sample_snapshots(
    traj: torch.Tensor,
    sample_id: int,
    x: np.ndarray,
    times: np.ndarray,
    output_dir: str,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    snap_idx = _snapshot_indices(times, [0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    for idx in snap_idx:
        ax.plot(x, traj[sample_id, idx].cpu().numpy(), linewidth=1.7, label=f"t={times[idx]:g}")
    ax.set_title(f"KS snapshots {sample_id}")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    out_path = os.path.join(output_dir, f"trajectory_{sample_id:02d}_snapshots.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_trajectory_grid(traj: torch.Tensor, times: np.ndarray, domain_length: float, output_dir: str, dpi: int) -> None:
    import matplotlib.pyplot as plt

    n_samples = int(traj.shape[0])
    n_cols = min(2, n_samples)
    n_rows = int(math.ceil(n_samples / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 3.2 * n_rows), squeeze=False, constrained_layout=True)
    scale = max(float(traj.abs().max().item()), 1e-8)
    last_im = None
    for i in range(n_rows * n_cols):
        ax = axes[i // n_cols, i % n_cols]
        if i >= n_samples:
            ax.axis("off")
            continue
        last_im = ax.imshow(
            traj[i].cpu().numpy().T,
            aspect="auto",
            origin="lower",
            interpolation="bilinear",
            extent=[float(times[0]), float(times[-1]), 0.0, float(domain_length)],
            cmap="RdBu_r",
            vmin=-scale,
            vmax=scale,
        )
        ax.set_title(f"sample {i}")
        ax.set_xlabel("t")
        ax.set_ylabel("x")
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), label="u")
    out_path = os.path.join(output_dir, "trajectory_grid_t0_50.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {out_path}")


def main(args: argparse.Namespace) -> None:
    if args.n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if args.solver_n_x < args.n_x:
        raise ValueError("solver_n_x must be >= n_x")
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but torch.cuda.is_available() is False")

    n_steps = _integer_steps(args.t_final, args.dataset_dt)
    os.makedirs(args.output_dir, exist_ok=True)
    dtype = torch.float64
    u0 = _sample_initial_conditions(args, dtype=dtype)
    traj_solver = solve_ks_trajectory_etdrk4(
        u0=u0,
        n_steps=n_steps,
        domain_length=args.domain_length,
        solver_dt=args.solver_dt,
        dataset_dt=args.dataset_dt,
        warmup_time=args.warmup_time,
    )
    traj = spectral_truncate_periodic_field_1d(traj_solver, target_n_x=args.n_x).to(dtype=torch.float32).cpu()
    traj_plot = _lowpass_for_plot(traj, max_modes=args.plot_max_modes)
    times = np.arange(n_steps + 1, dtype=np.float64) * float(args.dataset_dt)
    x = np.arange(args.n_x, dtype=np.float64) * (float(args.domain_length) / float(args.n_x))

    for sample_id in range(int(args.n_samples)):
        _plot_sample_heatmap(traj_plot, sample_id, x=x, times=times, output_dir=args.output_dir, dpi=args.dpi)
        _plot_sample_snapshots(traj_plot, sample_id, x=x, times=times, output_dir=args.output_dir, dpi=args.dpi)
    _plot_trajectory_grid(traj_plot, times=times, domain_length=args.domain_length, output_dir=args.output_dir, dpi=args.dpi)


if __name__ == "__main__":
    main(parse_args())
