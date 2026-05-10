"""
Data generation for the unforced periodic 1D Kuramoto-Sivashinsky equation:

    u_t + u u_x + u_xx + u_xxxx = 0,  x in [0, L].

The reference solver is Fourier pseudospectral in space with ETDRK4 time
stepping. Datasets follow the existing grad_flow_l2 split format:
    split["f"]      : (n_samples, n_x) zero compatibility channel
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
    from ..burgers_data import spectral_truncate_periodic_field_1d
    from ..heat_data import save_dataset_splits
except ImportError:
    from grad_flow_l2.burgers_data import spectral_truncate_periodic_field_1d
    from grad_flow_l2.heat_data import save_dataset_splits


DATASET_VERSION = 2


def remove_mean_1d(u: torch.Tensor) -> torch.Tensor:
    return u - u.mean(dim=-1, keepdim=True)


def sample_iid_initial_conditions(
    n_x: int,
    n_samples: int,
    std: float = 0.1,
    zero_mean: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample paper-style gridwise IID normal initial fields."""
    u0 = float(std) * torch.randn(n_samples, n_x, device=device, dtype=dtype)
    return remove_mean_1d(u0) if zero_mean else u0


def sample_matern_initial_conditions(
    n_x: int,
    n_samples: int,
    domain_length: float,
    smoothness: float = 1.5,
    length_scale: float = 4.0,
    std: float = 0.1,
    zero_mean: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample smooth periodic Matérn-like Gaussian fields by spectral synthesis.

    The spectral envelope is proportional to
        (1 + (ell * k)^2)^(-(nu + 1/2) / 2)
    in amplitude for a 1D Matérn covariance.
    """
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


def prepare_etdrk4_cache(
    n_x: int,
    domain_length: float,
    dt: float,
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
    if contour_points < 4:
        raise ValueError("contour_points must be >= 4")

    k = 2.0 * math.pi * torch.fft.fftfreq(n_x, d=float(domain_length) / float(n_x), device=device).to(dtype=dtype)
    linear = k.square() - k.pow(4)
    dt_linear = float(dt) * linear
    e = torch.exp(dt_linear).to(torch.complex64 if dtype == torch.float32 else torch.complex128)
    e2 = torch.exp(0.5 * dt_linear).to(e.dtype)

    j = torch.arange(1, contour_points + 1, device=device, dtype=dtype)
    roots = torch.exp(1j * math.pi * (j - 0.5) / float(contour_points)).to(e.dtype)
    lr = dt_linear.to(e.dtype).unsqueeze(-1) + roots.view(1, -1)
    # Kassam-Trefethen contour averages are real for a real diagonal
    # linear operator. Keeping the quadrature's complex roundoff/quadrature
    # residue breaks Hermitian symmetry and can destabilize long rollouts.
    q = float(dt) * torch.mean((torch.exp(lr / 2.0) - 1.0) / lr, dim=-1).real.to(e.dtype)
    f1 = float(dt) * torch.mean(
        (-4.0 - lr + torch.exp(lr) * (4.0 - 3.0 * lr + lr.square())) / lr.pow(3),
        dim=-1,
    ).real.to(e.dtype)
    f2 = (
        float(dt)
        * torch.mean((2.0 + lr + torch.exp(lr) * (-2.0 + lr)) / lr.pow(3), dim=-1).real.to(e.dtype)
    )
    f3 = float(dt) * torch.mean(
        (-4.0 - 3.0 * lr - lr.square() + torch.exp(lr) * (4.0 - lr)) / lr.pow(3),
        dim=-1,
    ).real.to(e.dtype)

    mode = torch.fft.fftfreq(n_x, d=1.0 / float(n_x), device=device).abs()
    dealias_mask = (mode <= float(n_x) / 3.0).to(e.dtype)
    dealias_mask[0] = 1.0

    return {
        "k": k.to(e.dtype),
        "e": e,
        "e2": e2,
        "q": q,
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "dealias_mask": dealias_mask,
    }


def _nonlinear_hat(u_hat: torch.Tensor, cache: Dict[str, torch.Tensor]) -> torch.Tensor:
    n_x = int(u_hat.shape[-1])
    u = torch.fft.ifft(u_hat, dim=-1).real
    u2_hat = torch.fft.fft(u.square(), dim=-1)
    out = -0.5j * cache["k"].view(1, -1) * u2_hat
    out = out * cache["dealias_mask"].view(1, -1)
    out[:, 0] = 0.0
    if n_x % 2 == 0:
        out[:, n_x // 2] = 0.0
    return out


def etdrk4_step(u_hat: torch.Tensor, cache: Dict[str, torch.Tensor]) -> torch.Tensor:
    n1 = _nonlinear_hat(u_hat, cache)
    a = cache["e2"].view(1, -1) * u_hat + cache["q"].view(1, -1) * n1
    n2 = _nonlinear_hat(a, cache)
    b = cache["e2"].view(1, -1) * u_hat + cache["q"].view(1, -1) * n2
    n3 = _nonlinear_hat(b, cache)
    c = cache["e2"].view(1, -1) * a + cache["q"].view(1, -1) * (2.0 * n3 - n1)
    n4 = _nonlinear_hat(c, cache)
    u_next = (
        cache["e"].view(1, -1) * u_hat
        + cache["f1"].view(1, -1) * n1
        + 2.0 * cache["f2"].view(1, -1) * (n2 + n3)
        + cache["f3"].view(1, -1) * n4
    )
    u_next[:, 0] = 0.0
    if int(u_next.shape[-1]) % 2 == 0:
        u_next[:, int(u_next.shape[-1]) // 2] = 0.0
    return u_next


def _integer_ratio(numer: float, denom: float, name: str) -> int:
    ratio = float(numer) / float(denom)
    out = int(round(ratio))
    if out < 0 or abs(ratio - out) > 1e-10:
        raise ValueError(f"{name} must be an integer multiple of solver_dt")
    return out


def solve_ks_trajectory_etdrk4(
    u0: torch.Tensor,
    n_steps: int,
    domain_length: float = 32.0 * math.pi,
    solver_dt: float = 0.01,
    dataset_dt: float = 1.0,
    warmup_time: float = 100.0,
    contour_points: int = 16,
    project_zero_mean: bool = True,
) -> torch.Tensor:
    """
    Warm up from u0, then return recorded states [u_0, ..., u_K].
    """
    if u0.dim() == 1:
        u0 = u0.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    if u0.dim() != 2:
        raise ValueError("u0 must have shape (n_x,) or (batch,n_x)")
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
        device=str(u0.device),
        dtype=u0.dtype,
        contour_points=contour_points,
    )

    u = remove_mean_1d(u0) if project_zero_mean else u0
    u_hat = torch.fft.fft(u, dim=-1)
    u_hat[:, 0] = 0.0
    for step in range(1, warmup_steps + 1):
        u_hat = etdrk4_step(u_hat, cache)
        if not torch.isfinite(u_hat).all():
            raise FloatingPointError(f"Non-finite KS Fourier state during warm-up at solver step {step}")

    states = [torch.fft.ifft(u_hat, dim=-1).real]
    for record_idx in range(1, n_steps + 1):
        for substep in range(1, record_stride + 1):
            u_hat = etdrk4_step(u_hat, cache)
            if not torch.isfinite(u_hat).all():
                step = warmup_steps + (record_idx - 1) * record_stride + substep
                raise FloatingPointError(f"Non-finite KS Fourier state at solver step {step}")
        u = torch.fft.ifft(u_hat, dim=-1).real
        if not torch.isfinite(u).all():
            raise FloatingPointError(f"Non-finite KS physical state at recorded step {record_idx}")
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


def generate_ks_dataset_splits(
    n_x: int = 256,
    solver_n_x: int = 1024,
    n_steps: int = 20,
    n_train: int = 1500,
    n_val: int = 300,
    n_test: int = 200,
    domain_length: float = 32.0 * math.pi,
    solver_dt: float = 0.01,
    dataset_dt: float = 1.0,
    warmup_time: float = 100.0,
    seed: int = 42,
    u0_sampler: str = "matern",
    u0_std: float = 0.1,
    matern_smoothness: float = 1.5,
    matern_length_scale: float = 4.0,
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
    elif sampler == "iid":
        u0_solver = sample_iid_initial_conditions(
            n_x=solver_n_x,
            n_samples=total,
            std=u0_std,
            zero_mean=zero_mean,
            device=str(solve_device),
            dtype=dtype,
        )
    else:
        raise ValueError("u0_sampler must be one of {'matern','iid'}")

    traj_chunks = []
    chunk_starts = list(range(0, total, solve_batch_size))
    iterator = chunk_starts
    pbar = None
    if show_progress and tqdm is not None:
        iterator = tqdm(chunk_starts, desc="solve KS chunks", dynamic_ncols=True)
        pbar = iterator
    for start in iterator:
        end = min(total, start + solve_batch_size)
        traj_solver = solve_ks_trajectory_etdrk4(
            u0=u0_solver[start:end],
            n_steps=n_steps,
            domain_length=domain_length,
            solver_dt=solver_dt,
            dataset_dt=dataset_dt,
            warmup_time=warmup_time,
            project_zero_mean=zero_mean,
        ).to(dtype=dtype)
        traj_chunks.append(spectral_truncate_periodic_field_1d(traj_solver, target_n_x=n_x).to(dtype=output_dtype))
        if pbar is not None:
            pbar.set_postfix(samples=f"{end}/{total}")
    u_traj = torch.cat(traj_chunks, dim=0)
    u0 = u_traj[:, 0].clone()
    f = torch.zeros(total, n_x, dtype=output_dtype, device=u_traj.device)

    all_data = {"f": f.cpu(), "u0": u0.cpu(), "u_traj": u_traj.cpu()}
    train_end = int(n_train)
    val_end = int(n_train + n_val)
    h = float(domain_length) / float(n_x)

    splits: Dict[str, Dict[str, torch.Tensor]] = {
        "train": _slice_split(all_data, 0, train_end),
        "val": _slice_split(all_data, train_end, val_end),
        "test": _slice_split(all_data, val_end, total),
        "meta": {
            "dataset_version": DATASET_VERSION,
            "solver_note": "ETDRK4 contour coefficients use real Kassam-Trefethen averages.",
            "equation": "kuramoto_sivashinsky_1d_periodic",
            "equation_form": "u_t + u u_x + u_xx + u_xxxx = 0",
            "forcing_mode": "none",
            "boundary_condition": "periodic",
            "periodic": True,
            "domain_length": float(domain_length),
            "n_x": int(n_x),
            "solver_n_x": int(solver_n_x),
            "n_steps": int(n_steps),
            "t_final": float(n_steps) * float(dataset_dt),
            "dataset_dt": float(dataset_dt),
            "solver_dt": float(solver_dt),
            "save_stride": int(record_stride),
            "warmup_time": float(warmup_time),
            "warmup_steps": int(warmup_steps),
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
            "zero_mean": bool(zero_mean),
            "solver_dtype": str(dtype).replace("torch.", ""),
            "storage_dtype": str(output_dtype).replace("torch.", ""),
        },
    }

    torch.random.set_rng_state(rng_state_torch)
    np.random.set_state(rng_state_numpy)
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cached 1D Kuramoto-Sivashinsky dataset splits")
    parser.add_argument("--n-x", type=int, default=256, help="Stored/model spatial grid size")
    parser.add_argument("--solver-n-x", type=int, default=1024, help="Reference solver spatial grid size")
    parser.add_argument("--n-steps", type=int, default=20, help="Number of stored training steps")
    parser.add_argument("--domain-length", type=float, default=32.0 * math.pi)
    parser.add_argument("--dataset-dt", type=float, default=1.0)
    parser.add_argument("--solver-dt", type=float, default=0.01)
    parser.add_argument("--warmup-time", type=float, default=100.0)
    parser.add_argument("--solve-batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--n-train", type=int, default=1500)
    parser.add_argument("--n-val", type=int, default=300)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--u0-sampler", type=str, default="matern", choices=["matern", "iid"])
    parser.add_argument("--u0-std", type=float, default=0.1)
    parser.add_argument("--matern-smoothness", type=float, default=1.5)
    parser.add_argument("--matern-length-scale", type=float, default=4.0)
    parser.add_argument("--disable-zero-mean", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm chunk progress bar")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="grad_flow_l2/ks1d/datasets/ks_periodic_L32pi_snx1024_nx256_dt1_solverdt0p01.pt",
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
            f"u_mean_abs={u_traj.mean(dim=-1).abs().max().item():.4e}, "
            f"u_abs_max={u_traj.abs().max().item():.4e}, finite={bool(torch.isfinite(u_traj).all())}"
        )


def main(args: argparse.Namespace) -> None:
    splits = generate_ks_dataset_splits(
        n_x=args.n_x,
        solver_n_x=args.solver_n_x,
        n_steps=args.n_steps,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        domain_length=args.domain_length,
        solver_dt=args.solver_dt,
        dataset_dt=args.dataset_dt,
        warmup_time=args.warmup_time,
        seed=args.seed,
        u0_sampler=args.u0_sampler,
        u0_std=args.u0_std,
        matern_smoothness=args.matern_smoothness,
        matern_length_scale=args.matern_length_scale,
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
    print(f"Saved KS dataset splits to: {args.dataset_path}")
    _print_split_stats(splits)


if __name__ == "__main__":
    main(parse_args())
