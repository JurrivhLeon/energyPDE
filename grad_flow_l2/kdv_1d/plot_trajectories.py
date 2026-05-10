"""
Generate reference damped-driven KdV trajectory plots.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Sequence

import numpy as np
import torch

try:
    from ..burgers_data import sample_periodic_field_mixed_1d, spectral_truncate_periodic_field_1d
    from .kdv_data import (
        sample_damped_sinusoidal_initial_conditions,
        sample_matern_forcing_1d,
        sample_matern_initial_conditions,
        solve_kdv_trajectory_etdrk4,
    )
except ImportError:
    from grad_flow_l2.burgers_data import sample_periodic_field_mixed_1d, spectral_truncate_periodic_field_1d
    from grad_flow_l2.kdv_1d.kdv_data import (
        sample_matern_forcing_1d,
        sample_matern_initial_conditions,
        sample_damped_sinusoidal_initial_conditions,
        solve_kdv_trajectory_etdrk4,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot forced damped-driven periodic 1D KdV reference trajectories")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--n-x", type=int, default=512, help="Stored/plot grid size")
    parser.add_argument("--solver-n-x", type=int, default=4096, help="Reference solver grid size")
    parser.add_argument("--domain-length", type=float, default=32.0)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--solver-dt", type=float, default=0.01)
    parser.add_argument("--dataset-dt", type=float, default=0.1)
    parser.add_argument("--t-final", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--u0-sampler", type=str, default="matern", choices=["matern", "sinusoidal"])
    parser.add_argument("--u0-std", type=float, default=0.5)
    parser.add_argument("--matern-smoothness", type=float, default=1.5)
    parser.add_argument("--matern-length-scale", type=float, default=0.5)
    parser.add_argument("--u0-sinusoidal-max-modes", type=int, default=8)
    parser.add_argument("--u0-sinusoidal-decay", type=float, default=2.0)
    parser.add_argument("--forcing-amplitude", type=float, default=0.5)
    parser.add_argument("--forcing-sampler", type=str, default="matern", choices=["matern", "mixed"])
    parser.add_argument("--forcing-matern-smoothness", type=float, default=2.5)
    parser.add_argument("--forcing-matern-length-scale", type=float, default=0.5)
    parser.add_argument("--forcing-length-scale-min", type=float, default=0.15)
    parser.add_argument("--forcing-length-scale-max", type=float, default=0.8)
    parser.add_argument("--forcing-max-modes", type=int, default=6)
    parser.add_argument(
        "--plot-max-modes",
        type=int,
        default=128,
        help="Retain this many positive Fourier modes in plotted trajectories. Use <=0 to disable plot-only smoothing.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="grad_flow_l2/kdv_1d/plots")
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
    if args.u0_sampler == "sinusoidal":
        return sample_damped_sinusoidal_initial_conditions(
            n_x=args.solver_n_x,
            n_samples=args.n_samples,
            domain_length=args.domain_length,
            max_modes=args.u0_sinusoidal_max_modes,
            amplitude=args.u0_std,
            mode_decay=args.u0_sinusoidal_decay,
            zero_mean=True,
            device=args.device,
            dtype=dtype,
        )
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


def _sample_forcing(args: argparse.Namespace, dtype: torch.dtype) -> torch.Tensor:
    if args.forcing_sampler == "matern":
        return sample_matern_forcing_1d(
            n_x=args.solver_n_x,
            n_samples=args.n_samples,
            domain_length=args.domain_length,
            smoothness=args.forcing_matern_smoothness,
            length_scale=args.forcing_matern_length_scale,
            amplitude=args.forcing_amplitude,
            zero_mean=True,
            device=args.device,
            dtype=dtype,
        )
    return sample_periodic_field_mixed_1d(
        n_points=args.solver_n_x,
        n_samples=args.n_samples,
        amplitude=args.forcing_amplitude,
        length_scale_range=(args.forcing_length_scale_min, args.forcing_length_scale_max),
        max_modes=args.forcing_max_modes,
        zero_mean=True,
        device=args.device,
    ).to(dtype=dtype)


def _plot_sample_heatmap(
    traj: torch.Tensor,
    forcing: torch.Tensor,
    sample_id: int,
    x: np.ndarray,
    times: np.ndarray,
    gamma: float,
    output_dir: str,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    arr = traj[sample_id].cpu().numpy()
    scale = max(float(np.max(np.abs(arr))), 1e-8)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True, gridspec_kw={"width_ratios": [4, 1]})
    im = axes[0].imshow(
        arr.T,
        aspect="auto",
        origin="lower",
        interpolation="bilinear",
        extent=[float(times[0]), float(times[-1]), float(x[0]), float(x[-1] + (x[1] - x[0]))],
        cmap="RdBu_r",
        vmin=-scale,
        vmax=scale,
    )
    axes[0].set_title(f"KdV trajectory {sample_id}, gamma={gamma:g}")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("x")
    axes[1].plot(forcing[sample_id].cpu().numpy(), x, color="black", linewidth=1.5)
    axes[1].set_title("f(x)")
    axes[1].set_xlabel("f")
    axes[1].set_ylabel("x")
    axes[1].grid(alpha=0.25)
    fig.colorbar(im, ax=axes[0], label="u")
    out_path = os.path.join(output_dir, f"trajectory_{sample_id:02d}_heatmap.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_sample_snapshots(
    traj: torch.Tensor,
    forcing: torch.Tensor,
    sample_id: int,
    x: np.ndarray,
    times: np.ndarray,
    output_dir: str,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    snap_idx = _snapshot_indices(times, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    for idx in snap_idx:
        ax.plot(x, traj[sample_id, idx].cpu().numpy(), linewidth=1.7, label=f"u, t={times[idx]:g}")
    ax.plot(x, forcing[sample_id].cpu().numpy(), color="black", linewidth=1.2, linestyle=":", label="f(x)")
    ax.set_title(f"KdV snapshots {sample_id}")
    ax.set_xlabel("x")
    ax.set_ylabel("u / f")
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
    t_final = float(times[-1]) if len(times) else 0.0
    t_tag = f"{t_final:g}".replace(".", "p")
    out_path = os.path.join(output_dir, f"trajectory_grid_t0_{t_tag}.png")
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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dtype = torch.float64
    u0 = _sample_initial_conditions(args, dtype=dtype)
    forcing = _sample_forcing(args, dtype=dtype)
    traj_solver = solve_kdv_trajectory_etdrk4(
        u0=u0,
        f=forcing,
        n_steps=n_steps,
        domain_length=args.domain_length,
        gamma=args.gamma,
        solver_dt=args.solver_dt,
        dataset_dt=args.dataset_dt,
    )
    traj = spectral_truncate_periodic_field_1d(traj_solver, target_n_x=args.n_x).to(dtype=torch.float32).cpu()
    forcing_plot = spectral_truncate_periodic_field_1d(forcing, target_n_x=args.n_x).to(dtype=torch.float32).cpu()
    traj_plot = _lowpass_for_plot(traj, max_modes=args.plot_max_modes)
    times = np.arange(n_steps + 1, dtype=np.float64) * float(args.dataset_dt)
    x = np.arange(args.n_x, dtype=np.float64) * (float(args.domain_length) / float(args.n_x))

    print(
        f"KdV trajectories: samples={args.n_samples}, L={args.domain_length}, "
        f"solver_n_x={args.solver_n_x}, n_x={args.n_x}, solver_dt={args.solver_dt}, "
        f"dataset_dt={args.dataset_dt}, T={args.t_final}, gamma={args.gamma}, "
        f"u0_sampler={args.u0_sampler}, forcing_sampler={args.forcing_sampler}"
    )
    print(
        f"u_abs_max={traj.abs().max().item():.4e}, f_abs_max={forcing_plot.abs().max().item():.4e}, "
        f"finite={bool(torch.isfinite(traj).all())}"
    )

    for sample_id in range(int(args.n_samples)):
        _plot_sample_heatmap(
            traj_plot,
            forcing_plot,
            sample_id,
            x=x,
            times=times,
            gamma=args.gamma,
            output_dir=args.output_dir,
            dpi=args.dpi,
        )
        _plot_sample_snapshots(
            traj_plot,
            forcing_plot,
            sample_id,
            x=x,
            times=times,
            output_dir=args.output_dir,
            dpi=args.dpi,
        )
    _plot_trajectory_grid(traj_plot, times=times, domain_length=args.domain_length, output_dir=args.output_dir, dpi=args.dpi)


if __name__ == "__main__":
    main(parse_args())
