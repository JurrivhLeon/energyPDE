"""
Data generation for the periodic damped-driven 1D KdV equation:

    u_t + 6 u u_x + u_xxx = -gamma u + f(x),  x in [0, L].

Equivalently,

    u_t = -3 (u^2)_x - u_xxx - gamma u + f(x).

The reference solver is Fourier pseudospectral in space with ETDRK4 time
stepping. Datasets follow the shared grad_flow_l2 split format:
    split["f"]      : (n_samples, n_x)
    split["u0"]     : (n_samples, n_x)
    split["u_traj"] : (n_samples, n_steps+1, n_x)
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    from ..burgers_data import sample_periodic_field_mixed_1d, spectral_truncate_periodic_field_1d
    from ..heat_data import save_dataset_splits
except ImportError:
    from grad_flow_l2.burgers_data import sample_periodic_field_mixed_1d, spectral_truncate_periodic_field_1d
    from grad_flow_l2.heat_data import save_dataset_splits


DATASET_VERSION = 1


def remove_mean_1d(u: torch.Tensor) -> torch.Tensor:
    return u - u.mean(dim=-1, keepdim=True)


def sample_matern_initial_conditions(
    n_x: int,
    n_samples: int,
    domain_length: float,
    smoothness: float = 1.5,
    length_scale: float = 0.5,
    std: float = 0.5,
    zero_mean: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample smooth periodic Matern-like Gaussian fields by spectral synthesis."""
    if smoothness <= 0.0:
        raise ValueError("smoothness must be > 0")
    if length_scale <= 0.0:
        raise ValueError("length_scale must be > 0")
    k = 2.0 * math.pi * torch.fft.rfftfreq(n_x, d=float(domain_length) / float(n_x), device=device).to(dtype=dtype)
    amp = (1.0 + (float(length_scale) * k).square()).pow(-0.5 * (float(smoothness) + 0.5))
    if zero_mean:
        amp[0] = 0.0

    real = torch.randn(n_samples, amp.shape[0], device=device, dtype=dtype)
    imag = torch.randn(n_samples, amp.shape[0], device=device, dtype=dtype)
    imag[:, 0] = 0.0
    if n_x % 2 == 0:
        imag[:, -1] = 0.0
    spectrum = torch.complex(real, imag) * amp.unsqueeze(0)
    u0 = torch.fft.irfft(spectrum, n=n_x, dim=-1) * (n_x ** 0.5)
    if zero_mean:
        u0 = remove_mean_1d(u0)
    current_std = u0.std(dim=-1, keepdim=True).clamp_min(1e-8)
    return float(std) * u0 / current_std


def sample_matern_forcing_1d(
    n_x: int,
    n_samples: int,
    domain_length: float,
    smoothness: float = 2.5,
    length_scale: float = 0.5,
    amplitude: float = 0.5,
    zero_mean: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample periodic Matérn-like Gaussian forcing fields.

    The returned fields are normalized per sample so that ``amplitude`` sets
    the maximum absolute forcing scale, matching the mixed periodic sampler.
    """
    f = sample_matern_initial_conditions(
        n_x=n_x,
        n_samples=n_samples,
        domain_length=domain_length,
        smoothness=smoothness,
        length_scale=length_scale,
        std=1.0,
        zero_mean=zero_mean,
        device=device,
        dtype=dtype,
    )
    return float(amplitude) * f / f.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)


def sample_damped_sinusoidal_initial_conditions(
    n_x: int,
    n_samples: int,
    domain_length: float,
    max_modes: int = 8,
    amplitude: float = 0.5,
    mode_decay: float = 2.0,
    zero_mean: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample smooth periodic sinusoidal initial fields.

    Each sample is a random sine/cosine series with coefficient scale
    proportional to k^{-mode_decay}; larger ``mode_decay`` damps
    high-frequency Fourier features more strongly.
    """
    if max_modes < 1:
        raise ValueError("max_modes must be >= 1")
    if mode_decay < 0.0:
        raise ValueError("mode_decay must be >= 0")
    x = torch.arange(n_x, device=device, dtype=dtype) * (float(domain_length) / float(n_x))
    fields = torch.zeros(n_samples, n_x, device=device, dtype=dtype)
    for k_int in range(1, int(max_modes) + 1):
        k = torch.tensor(float(k_int), device=device, dtype=dtype)
        phase_s = 2.0 * math.pi * torch.rand(n_samples, 1, device=device, dtype=dtype)
        phase_c = 2.0 * math.pi * torch.rand(n_samples, 1, device=device, dtype=dtype)
        scale = k.pow(-float(mode_decay))
        amp_s = torch.randn(n_samples, 1, device=device, dtype=dtype) * scale
        amp_c = torch.randn(n_samples, 1, device=device, dtype=dtype) * scale
        arg = 2.0 * math.pi * k * x.view(1, -1) / float(domain_length)
        fields = fields + amp_s * torch.sin(arg + phase_s) + amp_c * torch.cos(arg + phase_c)
    if zero_mean:
        fields = remove_mean_1d(fields)
    return float(amplitude) * fields / fields.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)


def prepare_etdrk4_cache(
    n_x: int,
    domain_length: float,
    dt: float,
    gamma: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    contour_points: int = 16,
) -> Dict[str, torch.Tensor]:
    """Prepare Fourier wave numbers, ETDRK4 coefficients, and dealias mask."""
    if n_x < 2:
        raise ValueError("n_x must be >= 2")
    if domain_length <= 0.0:
        raise ValueError("domain_length must be > 0")
    if dt <= 0.0:
        raise ValueError("dt must be > 0")
    if gamma < 0.0:
        raise ValueError("gamma must be >= 0")
    if contour_points < 4:
        raise ValueError("contour_points must be >= 4")

    k_real = 2.0 * math.pi * torch.fft.fftfreq(n_x, d=float(domain_length) / float(n_x), device=device).to(dtype=dtype)
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    k = k_real.to(complex_dtype)
    linear = -float(gamma) + 1j * k.pow(3)
    dt_linear = float(dt) * linear
    e = torch.exp(dt_linear)
    e2 = torch.exp(0.5 * dt_linear)

    j = torch.arange(1, contour_points + 1, device=device, dtype=dtype)
    roots = torch.exp(1j * math.pi * (j - 0.5) / float(contour_points)).to(complex_dtype)
    lr = dt_linear.unsqueeze(-1) + roots.view(1, -1)
    q = float(dt) * torch.mean((torch.exp(lr / 2.0) - 1.0) / lr, dim=-1)
    f1 = float(dt) * torch.mean(
        (-4.0 - lr + torch.exp(lr) * (4.0 - 3.0 * lr + lr.square())) / lr.pow(3),
        dim=-1,
    )
    f2 = float(dt) * torch.mean((2.0 + lr + torch.exp(lr) * (-2.0 + lr)) / lr.pow(3), dim=-1)
    f3 = float(dt) * torch.mean(
        (-4.0 - 3.0 * lr - lr.square() + torch.exp(lr) * (4.0 - lr)) / lr.pow(3),
        dim=-1,
    )

    mode = torch.fft.fftfreq(n_x, d=1.0 / float(n_x), device=device).abs()
    dealias_mask = (mode <= float(n_x) / 3.0).to(complex_dtype)
    dealias_mask[0] = 1.0

    return {
        "k": k,
        "e": e,
        "e2": e2,
        "q": q,
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "dealias_mask": dealias_mask,
    }


def _rhs_hat(u_hat: torch.Tensor, forcing_hat: torch.Tensor, cache: Dict[str, torch.Tensor]) -> torch.Tensor:
    n_x = int(u_hat.shape[-1])
    u = torch.fft.ifft(u_hat, dim=-1).real
    u2_hat = torch.fft.fft(u.square(), dim=-1)
    out = -3.0j * cache["k"].view(1, -1) * u2_hat
    out = out * cache["dealias_mask"].view(1, -1)
    out[:, 0] = out[:, 0] + forcing_hat[:, 0]
    out[:, 1:] = out[:, 1:] + forcing_hat[:, 1:]
    if n_x % 2 == 0:
        out[:, n_x // 2] = 0.0
    return out


def etdrk4_step(u_hat: torch.Tensor, forcing_hat: torch.Tensor, cache: Dict[str, torch.Tensor]) -> torch.Tensor:
    n1 = _rhs_hat(u_hat, forcing_hat, cache)
    a = cache["e2"].view(1, -1) * u_hat + cache["q"].view(1, -1) * n1
    n2 = _rhs_hat(a, forcing_hat, cache)
    b = cache["e2"].view(1, -1) * u_hat + cache["q"].view(1, -1) * n2
    n3 = _rhs_hat(b, forcing_hat, cache)
    c = cache["e2"].view(1, -1) * a + cache["q"].view(1, -1) * (2.0 * n3 - n1)
    n4 = _rhs_hat(c, forcing_hat, cache)
    u_next = (
        cache["e"].view(1, -1) * u_hat
        + cache["f1"].view(1, -1) * n1
        + 2.0 * cache["f2"].view(1, -1) * (n2 + n3)
        + cache["f3"].view(1, -1) * n4
    )
    if int(u_next.shape[-1]) % 2 == 0:
        u_next[:, int(u_next.shape[-1]) // 2] = 0.0
    return u_next


def _integer_ratio(numer: float, denom: float, name: str) -> int:
    ratio = float(numer) / float(denom)
    out = int(round(ratio))
    if out < 0 or abs(ratio - out) > 1e-10:
        raise ValueError(f"{name} must be an integer multiple of solver_dt")
    return out


def solve_kdv_trajectory_etdrk4(
    u0: torch.Tensor,
    f: torch.Tensor,
    n_steps: int,
    domain_length: float = 32.0,
    gamma: float = 0.1,
    solver_dt: float = 0.01,
    dataset_dt: float = 0.1,
    warmup_time: float = 0.0,
    contour_points: int = 16,
) -> torch.Tensor:
    """Return recorded states [u_0, ..., u_K] after optional static-forcing warm-up."""
    if u0.dim() == 1:
        u0 = u0.unsqueeze(0)
        f = f.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    if u0.dim() != 2 or f.dim() != 2:
        raise ValueError("u0 and f must have shape (n_x,) or (batch,n_x)")
    if u0.shape != f.shape:
        raise ValueError(f"u0 and f must have identical shape, got {tuple(u0.shape)} and {tuple(f.shape)}")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if warmup_time < 0.0:
        raise ValueError("warmup_time must be >= 0")

    record_stride = _integer_ratio(dataset_dt, solver_dt, "dataset_dt")
    warmup_steps = _integer_ratio(warmup_time, solver_dt, "warmup_time")
    n_x = int(u0.shape[-1])
    cache = prepare_etdrk4_cache(
        n_x=n_x,
        domain_length=domain_length,
        dt=solver_dt,
        gamma=gamma,
        device=str(u0.device),
        dtype=u0.dtype,
        contour_points=contour_points,
    )

    u_hat = torch.fft.fft(u0, dim=-1)
    forcing_hat = torch.fft.fft(f, dim=-1).to(u_hat.dtype)
    for step in range(1, warmup_steps + 1):
        u_hat = etdrk4_step(u_hat, forcing_hat, cache)
        if not torch.isfinite(u_hat).all():
            raise FloatingPointError(f"Non-finite KdV Fourier state during warm-up at solver step {step}")

    states = [torch.fft.ifft(u_hat, dim=-1).real]
    for record_idx in range(1, n_steps + 1):
        for substep in range(1, record_stride + 1):
            u_hat = etdrk4_step(u_hat, forcing_hat, cache)
            if not torch.isfinite(u_hat).all():
                step = warmup_steps + (record_idx - 1) * record_stride + substep
                raise FloatingPointError(f"Non-finite KdV Fourier state at solver step {step}")
        u = torch.fft.ifft(u_hat, dim=-1).real
        if not torch.isfinite(u).all():
            raise FloatingPointError(f"Non-finite KdV physical state at recorded step {record_idx}")
        states.append(u)

    traj = torch.stack(states, dim=1)
    if squeeze:
        return traj.squeeze(0)
    return traj


def _slice_split(data: Dict[str, torch.Tensor], start: int, end: int) -> Dict[str, torch.Tensor]:
    return {
        "f": data["f"][start:end].clone(),
        "u0": data["u0"][start:end].clone(),
        "u_traj": data["u_traj"][start:end].clone(),
    }


def generate_kdv_dataset_splits(
    n_x: int = 512,
    solver_n_x: int = 4096,
    n_steps: int = 20,
    n_train: int = 1600,
    n_val: int = 400,
    n_test: int = 0,
    domain_length: float = 20.0,
    gamma: float = 0.02,
    solver_dt: float = 0.001,
    dataset_dt: float = 0.25,
    warmup_time: float = 5.0,
    seed: int = 42,
    u0_sampler: str = "sinusoidal",
    u0_std: float = 0.5,
    matern_smoothness: float = 1.5,
    matern_length_scale: float = 0.5,
    u0_sinusoidal_max_modes: int = 8,
    u0_sinusoidal_decay: float = 2.0,
    forcing_amplitude: float = 0.5,
    forcing_sampler: str = "matern",
    forcing_matern_smoothness: float = 2.5,
    forcing_matern_length_scale: float = 1.0,
    forcing_length_scale_min: float = 0.15,
    forcing_length_scale_max: float = 0.8,
    forcing_max_modes: int = 6,
    zero_mean: bool = True,
    solve_batch_size: int = 32,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    output_dtype: torch.dtype = torch.float32,
    show_progress: bool = True,
) -> Dict[str, Dict[str, torch.Tensor]]:
    rng_state_torch = torch.random.get_rng_state()
    rng_state_numpy = np.random.get_state()
    torch.manual_seed(seed)
    np.random.seed(seed)

    total = int(n_train + n_val + n_test)
    solver_n_x = int(solver_n_x)
    n_x = int(n_x)
    if solver_n_x < n_x:
        raise ValueError("solver_n_x must be >= n_x")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    record_stride = _integer_ratio(dataset_dt, solver_dt, "dataset_dt")
    warmup_steps = _integer_ratio(warmup_time, solver_dt, "warmup_time")
    solve_batch_size = max(1, int(solve_batch_size))
    solve_device = torch.device(device)
    if solve_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but torch.cuda.is_available() is False")

    sampler = u0_sampler.strip().lower()
    if sampler == "matern":
        u0_solver = sample_matern_initial_conditions(
            n_x=solver_n_x,
            n_samples=total,
            domain_length=domain_length,
            smoothness=matern_smoothness,
            length_scale=matern_length_scale,
            std=u0_std,
            zero_mean=zero_mean,
            device=str(solve_device),
            dtype=dtype,
        )
    elif sampler == "mixed":
        u0_solver = sample_periodic_field_mixed_1d(
            n_points=solver_n_x,
            n_samples=total,
            amplitude=u0_std,
            zero_mean=zero_mean,
            device=str(solve_device),
        ).to(dtype=dtype)
    elif sampler == "sinusoidal":
        u0_solver = sample_damped_sinusoidal_initial_conditions(
            n_x=solver_n_x,
            n_samples=total,
            domain_length=domain_length,
            max_modes=u0_sinusoidal_max_modes,
            amplitude=u0_std,
            mode_decay=u0_sinusoidal_decay,
            zero_mean=zero_mean,
            device=str(solve_device),
            dtype=dtype,
        )
    else:
        raise ValueError("u0_sampler must be one of {'matern','mixed','sinusoidal'}")

    force_sampler = forcing_sampler.strip().lower()
    if force_sampler == "matern":
        f_solver = sample_matern_forcing_1d(
            n_x=solver_n_x,
            n_samples=total,
            domain_length=domain_length,
            smoothness=forcing_matern_smoothness,
            length_scale=forcing_matern_length_scale,
            amplitude=forcing_amplitude,
            zero_mean=zero_mean,
            device=str(solve_device),
            dtype=dtype,
        )
    elif force_sampler == "mixed":
        f_solver = sample_periodic_field_mixed_1d(
            n_points=solver_n_x,
            n_samples=total,
            amplitude=forcing_amplitude,
            length_scale_range=(forcing_length_scale_min, forcing_length_scale_max),
            max_modes=forcing_max_modes,
            zero_mean=zero_mean,
            device=str(solve_device),
        ).to(dtype=dtype)
    else:
        raise ValueError("forcing_sampler must be one of {'matern','mixed'}")

    traj_chunks = []
    chunk_starts = list(range(0, total, solve_batch_size))
    iterator = chunk_starts
    pbar = None
    if show_progress and tqdm is not None:
        iterator = tqdm(chunk_starts, desc="solve KdV chunks", dynamic_ncols=True)
        pbar = iterator
    for start in iterator:
        end = min(total, start + solve_batch_size)
        traj_solver = solve_kdv_trajectory_etdrk4(
            u0=u0_solver[start:end],
            f=f_solver[start:end],
            n_steps=n_steps,
            domain_length=domain_length,
            gamma=gamma,
            solver_dt=solver_dt,
            dataset_dt=dataset_dt,
            warmup_time=warmup_time,
        ).to(dtype=dtype)
        traj_chunks.append(spectral_truncate_periodic_field_1d(traj_solver, target_n_x=n_x).to(dtype=output_dtype))
        if pbar is not None:
            pbar.set_postfix(samples=f"{end}/{total}")

    u_traj = torch.cat(traj_chunks, dim=0)
    u0 = u_traj[:, 0].clone()
    f_stored = spectral_truncate_periodic_field_1d(f_solver, target_n_x=n_x).to(dtype=output_dtype)
    h = float(domain_length) / float(n_x)

    all_data = {"f": f_stored.cpu(), "u0": u0.cpu(), "u_traj": u_traj.cpu()}
    train_end = int(n_train)
    val_end = int(n_train + n_val)
    splits: Dict[str, Dict[str, torch.Tensor]] = {
        "train": _slice_split(all_data, 0, train_end),
        "val": _slice_split(all_data, train_end, val_end),
        "test": _slice_split(all_data, val_end, total),
        "meta": {
            "dataset_version": DATASET_VERSION,
            "equation": "damped_driven_kdv_1d_periodic",
            "equation_form": "u_t + 6 u u_x + u_xxx = -gamma u + f(x)",
            "forcing_mode": "static_spatial",
            "forcing_sampler": force_sampler,
            "boundary_condition": "periodic",
            "periodic": True,
            "domain_length": float(domain_length),
            "gamma": float(gamma),
            "n_x": int(n_x),
            "solver_n_x": int(solver_n_x),
            "n_steps": int(n_steps),
            "t_final": float(n_steps) * float(dataset_dt),
            "dataset_dt": float(dataset_dt),
            "solver_dt": float(solver_dt),
            "save_stride": int(record_stride),
            "warmup_time": float(warmup_time),
            "warmup_steps": int(warmup_steps),
            "record_start_time": 0.0,
            "prewarm_start_time": -float(warmup_time),
            "solve_batch_size": int(solve_batch_size),
            "generation_device": str(solve_device),
            "h": float(h),
            "n_train": int(n_train),
            "n_val": int(n_val),
            "n_test": int(n_test),
            "seed": int(seed),
            "u0_sampler": sampler,
            "u0_std": float(u0_std),
            "matern_smoothness": float(matern_smoothness),
            "matern_length_scale": float(matern_length_scale),
            "u0_sinusoidal_max_modes": int(u0_sinusoidal_max_modes),
            "u0_sinusoidal_decay": float(u0_sinusoidal_decay),
            "forcing_amplitude": float(forcing_amplitude),
            "forcing_matern_smoothness": float(forcing_matern_smoothness),
            "forcing_matern_length_scale": float(forcing_matern_length_scale),
            "forcing_length_scale_min": float(forcing_length_scale_min),
            "forcing_length_scale_max": float(forcing_length_scale_max),
            "forcing_max_modes": int(forcing_max_modes),
            "zero_mean": bool(zero_mean),
            "solver_dtype": str(dtype).replace("torch.", ""),
            "storage_dtype": str(output_dtype).replace("torch.", ""),
        },
    }

    torch.random.set_rng_state(rng_state_torch)
    np.random.set_state(rng_state_numpy)
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cached 1D damped-driven KdV dataset splits")
    parser.add_argument("--n-x", type=int, default=512, help="Stored/model spatial grid size")
    parser.add_argument("--solver-n-x", type=int, default=4096, help="Reference solver spatial grid size")
    parser.add_argument("--n-steps", type=int, default=None, help="Compatibility override for stored training steps")
    parser.add_argument("--train-t-final", type=float, default=5.0, help="Train/val recording horizon after warm-up")
    parser.add_argument("--ood-t-final", type=float, default=25.0, help="OOD test recording horizon after warm-up")
    parser.add_argument("--domain-length", type=float, default=20.0)
    parser.add_argument("--gamma", type=float, default=0.02)
    parser.add_argument("--dataset-dt", type=float, default=0.25)
    parser.add_argument("--solver-dt", type=float, default=0.001)
    parser.add_argument("--warmup-time", type=float, default=5.0)
    parser.add_argument("--solve-batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--n-train", type=int, default=1600)
    parser.add_argument("--n-val", type=int, default=400)
    parser.add_argument("--n-test", type=int, default=0)
    parser.add_argument("--ood-n-test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ood-seed", type=int, default=4242)

    parser.add_argument("--u0-sampler", type=str, default="sinusoidal", choices=["matern", "mixed", "sinusoidal"])
    parser.add_argument("--u0-std", type=float, default=0.5)
    parser.add_argument("--matern-smoothness", type=float, default=1.5)
    parser.add_argument("--matern-length-scale", type=float, default=0.5)
    parser.add_argument("--u0-sinusoidal-max-modes", type=int, default=8)
    parser.add_argument("--u0-sinusoidal-decay", type=float, default=2.0)
    parser.add_argument("--forcing-amplitude", type=float, default=0.5)
    parser.add_argument("--forcing-sampler", type=str, default="matern", choices=["matern", "mixed"])
    parser.add_argument("--forcing-matern-smoothness", type=float, default=2.5)
    parser.add_argument("--forcing-matern-length-scale", type=float, default=1.0)
    parser.add_argument("--forcing-length-scale-min", type=float, default=0.15)
    parser.add_argument("--forcing-length-scale-max", type=float, default=0.8)
    parser.add_argument("--forcing-max-modes", type=int, default=6)
    parser.add_argument("--disable-zero-mean", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm chunk progress bar")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="grad_flow_l2/kdv_1d/datasets/kdv_trainval_L20_snx4096_nx512_dt0p25_T5_warm5_gamma0p02_sine_decay2_matern25_force.pt",
    )
    parser.add_argument(
        "--ood-dataset-path",
        type=str,
        default="grad_flow_l2/kdv_1d/datasets/kdv_ood_L20_snx4096_nx512_dt0p25_T25_warm5_gamma0p02_sine_decay2_matern25_force.pt",
    )
    return parser.parse_args()


def _print_split_stats(splits: Dict[str, Dict[str, torch.Tensor]]) -> None:
    print("Dataset meta:", splits.get("meta", {}))
    for split_name in ("train", "val", "test"):
        split = splits[split_name]
        u0 = split["u0"]
        u_traj = split["u_traj"]
        f = split["f"]
        if u0.shape[0] == 0:
            print(f"{split_name}: empty split")
            continue
        print(
            f"{split_name}: "
            f"f={tuple(f.shape)}, u0={tuple(u0.shape)}, u_traj={tuple(u_traj.shape)}, "
            f"u_mean_abs_max={u_traj.mean(dim=-1).abs().max().item():.4e}, "
            f"f_mean_abs_max={f.mean(dim=-1).abs().max().item():.4e}, "
            f"u_abs_max={u_traj.abs().max().item():.4e}, "
            f"f_abs_max={f.abs().max().item():.4e}, finite={bool(torch.isfinite(u_traj).all())}"
        )


def main(args: argparse.Namespace) -> None:
    train_steps = _integer_ratio(args.train_t_final, args.dataset_dt, "train_t_final")
    if args.n_steps is not None:
        train_steps = int(args.n_steps)
    splits = generate_kdv_dataset_splits(
        n_x=args.n_x,
        solver_n_x=args.solver_n_x,
        n_steps=train_steps,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        domain_length=args.domain_length,
        gamma=args.gamma,
        solver_dt=args.solver_dt,
        dataset_dt=args.dataset_dt,
        warmup_time=args.warmup_time,
        seed=args.seed,
        u0_sampler=args.u0_sampler,
        u0_std=args.u0_std,
        matern_smoothness=args.matern_smoothness,
        matern_length_scale=args.matern_length_scale,
        u0_sinusoidal_max_modes=args.u0_sinusoidal_max_modes,
        u0_sinusoidal_decay=args.u0_sinusoidal_decay,
        forcing_amplitude=args.forcing_amplitude,
        forcing_sampler=args.forcing_sampler,
        forcing_matern_smoothness=args.forcing_matern_smoothness,
        forcing_matern_length_scale=args.forcing_matern_length_scale,
        forcing_length_scale_min=args.forcing_length_scale_min,
        forcing_length_scale_max=args.forcing_length_scale_max,
        forcing_max_modes=args.forcing_max_modes,
        zero_mean=not args.disable_zero_mean,
        solve_batch_size=args.solve_batch_size,
        device=args.device,
        dtype=torch.float64,
        output_dtype=torch.float32,
        show_progress=not args.no_progress,
    )
    out_dir = os.path.dirname(args.dataset_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_dataset_splits(splits, args.dataset_path)
    print(f"Saved KdV train/val dataset splits to: {args.dataset_path}")
    _print_split_stats(splits)

    if args.ood_dataset_path:
        ood_steps = _integer_ratio(args.ood_t_final, args.dataset_dt, "ood_t_final")
        ood_splits = generate_kdv_dataset_splits(
            n_x=args.n_x,
            solver_n_x=args.solver_n_x,
            n_steps=ood_steps,
            n_train=0,
            n_val=0,
            n_test=args.ood_n_test,
            domain_length=args.domain_length,
            gamma=args.gamma,
            solver_dt=args.solver_dt,
            dataset_dt=args.dataset_dt,
            warmup_time=args.warmup_time,
            seed=args.ood_seed,
            u0_sampler=args.u0_sampler,
            u0_std=args.u0_std,
            matern_smoothness=args.matern_smoothness,
            matern_length_scale=args.matern_length_scale,
            u0_sinusoidal_max_modes=args.u0_sinusoidal_max_modes,
            u0_sinusoidal_decay=args.u0_sinusoidal_decay,
            forcing_amplitude=args.forcing_amplitude,
            forcing_sampler=args.forcing_sampler,
            forcing_matern_smoothness=args.forcing_matern_smoothness,
            forcing_matern_length_scale=args.forcing_matern_length_scale,
            forcing_length_scale_min=args.forcing_length_scale_min,
            forcing_length_scale_max=args.forcing_length_scale_max,
            forcing_max_modes=args.forcing_max_modes,
            zero_mean=not args.disable_zero_mean,
            solve_batch_size=args.solve_batch_size,
            device=args.device,
            dtype=torch.float64,
            output_dtype=torch.float32,
            show_progress=not args.no_progress,
        )
        ood_out_dir = os.path.dirname(args.ood_dataset_path)
        if ood_out_dir:
            os.makedirs(ood_out_dir, exist_ok=True)
        save_dataset_splits(ood_splits, args.ood_dataset_path)
        print(f"Saved KdV OOD dataset splits to: {args.ood_dataset_path}")
        _print_split_stats(ood_splits)


if __name__ == "__main__":
    main(parse_args())
