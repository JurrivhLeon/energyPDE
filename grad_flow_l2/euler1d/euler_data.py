"""
Data generation for the 1D compressible Euler equations.

The numerical solver evolves conservative variables
    U = (rho, rho*u, E)
with a first-order finite-volume Rusanov flux. Stored dataset states use
primitive channels
    (rho, u, p)
which are easier for learning and diagnostics.

Dataset format:
    split["f"]      : (n_samples, n_x), zero placeholder forcing
    split["u0"]     : (n_samples, 3, n_x)
    split["u_traj"] : (n_samples, n_steps+1, 3, n_x)
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    from ..heat_data import save_dataset_splits
except ImportError:
    from grad_flow_l2.heat_data import save_dataset_splits


DATASET_VERSION = 1
STATE_CHANNELS = 3
STATE_NAMES_PRIMITIVE = ("rho", "u", "p")
STATE_NAMES_CONSERVED = ("rho", "rho_u", "E")


class Euler1DTrajectoryTensorDataset(Dataset):
    def __init__(self, f_data: torch.Tensor, u0_data: torch.Tensor, u_traj_data: torch.Tensor):
        if f_data.dim() != 2:
            raise ValueError("f_data must have shape (n_samples,n_x)")
        if u0_data.dim() != 3 or int(u0_data.shape[1]) != STATE_CHANNELS:
            raise ValueError("u0_data must have shape (n_samples,3,n_x)")
        if u_traj_data.dim() != 4 or int(u_traj_data.shape[2]) != STATE_CHANNELS:
            raise ValueError("u_traj_data must have shape (n_samples,K+1,3,n_x)")
        n_samples = int(u_traj_data.shape[0])
        n_x = int(u_traj_data.shape[-1])
        if int(f_data.shape[0]) != n_samples or int(f_data.shape[1]) != n_x:
            raise ValueError("f_data and u_traj_data shapes are inconsistent")
        if int(u0_data.shape[0]) != n_samples or int(u0_data.shape[-1]) != n_x:
            raise ValueError("u0_data and u_traj_data shapes are inconsistent")
        self.f_data = f_data
        self.u0_data = u0_data
        self.u_traj_data = u_traj_data

    def __len__(self) -> int:
        return int(self.u0_data.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"f": self.f_data[idx], "u0": self.u0_data[idx], "u_traj": self.u_traj_data[idx]}


class Euler1DStepDataset(Dataset):
    def __init__(self, f_data: torch.Tensor, u_traj_data: torch.Tensor):
        if f_data.dim() != 2:
            raise ValueError("f_data must have shape (n_samples,n_x)")
        if u_traj_data.dim() != 4 or int(u_traj_data.shape[2]) != STATE_CHANNELS:
            raise ValueError("u_traj_data must have shape (n_samples,K+1,3,n_x)")
        if int(f_data.shape[0]) != int(u_traj_data.shape[0]) or int(f_data.shape[1]) != int(u_traj_data.shape[-1]):
            raise ValueError("f_data and u_traj_data shapes are inconsistent")
        self.f_data = f_data
        self.u_traj_data = u_traj_data
        self.n_samples = int(u_traj_data.shape[0])
        self.n_steps = int(u_traj_data.shape[1] - 1)

    def __len__(self) -> int:
        return self.n_samples * self.n_steps

    def __getitem__(self, idx: int):
        i = idx // self.n_steps
        k = idx % self.n_steps
        return self.u_traj_data[i, k], self.u_traj_data[i, k + 1], self.f_data[i]


def build_euler1d_step_dataset(split_or_dataset) -> Euler1DStepDataset:
    if isinstance(split_or_dataset, Euler1DTrajectoryTensorDataset):
        return Euler1DStepDataset(split_or_dataset.f_data, split_or_dataset.u_traj_data)
    if isinstance(split_or_dataset, dict):
        return Euler1DStepDataset(split_or_dataset["f"], split_or_dataset["u_traj"])
    raise TypeError("Expected split dict or Euler1DTrajectoryTensorDataset")


def build_euler1d_trajectory_dataset_from_split(split: Dict[str, torch.Tensor]) -> Euler1DTrajectoryTensorDataset:
    return Euler1DTrajectoryTensorDataset(split["f"], split["u0"], split["u_traj"])


def primitive_to_conserved(q: torch.Tensor, gamma: float = 1.4) -> torch.Tensor:
    if q.dim() != 3 or int(q.shape[1]) != STATE_CHANNELS:
        raise ValueError("primitive state must have shape (batch,3,n_x)")
    rho = q[:, 0].clamp_min(1e-12)
    u = q[:, 1]
    p = q[:, 2].clamp_min(1e-12)
    energy = p / (float(gamma) - 1.0) + 0.5 * rho * u.square()
    return torch.stack([rho, rho * u, energy], dim=1)


def conserved_to_primitive(
    U: torch.Tensor,
    gamma: float = 1.4,
    rho_floor: float = 1e-6,
    p_floor: float = 1e-6,
) -> torch.Tensor:
    if U.dim() != 3 or int(U.shape[1]) != STATE_CHANNELS:
        raise ValueError("conserved state must have shape (batch,3,n_x)")
    rho = U[:, 0].clamp_min(float(rho_floor))
    u = U[:, 1] / rho
    kinetic = 0.5 * rho * u.square()
    p = ((float(gamma) - 1.0) * (U[:, 2] - kinetic)).clamp_min(float(p_floor))
    return torch.stack([rho, u, p], dim=1)


def euler_flux(U: torch.Tensor, gamma: float = 1.4) -> torch.Tensor:
    q = conserved_to_primitive(U, gamma=gamma)
    rho = q[:, 0]
    u = q[:, 1]
    p = q[:, 2]
    return torch.stack([rho * u, rho * u.square() + p, u * (U[:, 2] + p)], dim=1)


def max_wave_speed(U: torch.Tensor, gamma: float = 1.4) -> torch.Tensor:
    q = conserved_to_primitive(U, gamma=gamma)
    c = torch.sqrt(float(gamma) * q[:, 2] / q[:, 0].clamp_min(1e-12))
    return q[:, 1].abs() + c


def _normalize_boundary_condition(boundary_condition: str) -> str:
    bc = str(boundary_condition).strip().lower()
    if bc in {"periodic", "circular", "torus"}:
        return "periodic"
    if bc in {"outflow", "transmissive", "neumann", "replicate"}:
        return "outflow"
    raise ValueError("boundary_condition must be one of {periodic,outflow}")


def _rusanov_interface_flux(U_l: torch.Tensor, U_r: torch.Tensor, gamma: float = 1.4) -> torch.Tensor:
    F_l = euler_flux(U_l, gamma=gamma)
    F_r = euler_flux(U_r, gamma=gamma)
    a = torch.maximum(max_wave_speed(U_l, gamma=gamma), max_wave_speed(U_r, gamma=gamma)).unsqueeze(1)
    return 0.5 * (F_l + F_r) - 0.5 * a * (U_r - U_l)


def _rusanov_step(
    U: torch.Tensor,
    dx: float,
    dt: float,
    gamma: float = 1.4,
    boundary_condition: str = "periodic",
) -> torch.Tensor:
    bc = _normalize_boundary_condition(boundary_condition)
    if bc == "outflow":
        U_ext = torch.cat([U[..., :1], U, U[..., -1:]], dim=-1)
        flux = _rusanov_interface_flux(U_ext[..., :-1], U_ext[..., 1:], gamma=gamma)
        return U - (float(dt) / float(dx)) * (flux[..., 1:] - flux[..., :-1])

    U_r = torch.roll(U, shifts=-1, dims=-1)
    flux_iphalf = _rusanov_interface_flux(U, U_r, gamma=gamma)
    flux_imhalf = torch.roll(flux_iphalf, shifts=1, dims=-1)
    return U - (float(dt) / float(dx)) * (flux_iphalf - flux_imhalf)


def solve_euler1d_trajectory(
    u0_prim: torch.Tensor,
    t_final: float,
    record_dt: float,
    gamma: float = 1.4,
    domain_length: float = 1.0,
    cfl: float = 0.45,
    solver_dt: float | None = None,
    rho_floor: float = 1e-6,
    p_floor: float = 1e-6,
    max_substeps: int = 100000,
    boundary_condition: str = "periodic",
) -> torch.Tensor:
    if u0_prim.dim() != 3 or int(u0_prim.shape[1]) != STATE_CHANNELS:
        raise ValueError("u0_prim must have shape (batch,3,n_x)")
    if t_final <= 0.0 or record_dt <= 0.0:
        raise ValueError("t_final and record_dt must be positive")
    n_steps_float = float(t_final) / float(record_dt)
    n_steps = int(round(n_steps_float))
    if abs(n_steps_float - n_steps) > 1e-8:
        raise ValueError("t_final must be an integer multiple of record_dt")

    n_x = int(u0_prim.shape[-1])
    if domain_length <= 0.0:
        raise ValueError("domain_length must be positive")
    dx = float(domain_length) / float(n_x)
    U = primitive_to_conserved(u0_prim, gamma=gamma)
    records = [conserved_to_primitive(U, gamma=gamma, rho_floor=rho_floor, p_floor=p_floor)]
    t = 0.0
    next_record = float(record_dt)
    substeps = 0

    while len(records) < n_steps + 1:
        speed = max_wave_speed(U, gamma=gamma).amax().item()
        dt_cfl = float(cfl) * dx / max(speed, 1e-12)
        dt = dt_cfl if solver_dt is None else min(float(solver_dt), dt_cfl)
        dt = min(dt, next_record - t)
        U = _rusanov_step(U, dx=dx, dt=dt, gamma=gamma, boundary_condition=boundary_condition)
        q = conserved_to_primitive(U, gamma=gamma, rho_floor=rho_floor, p_floor=p_floor)
        U = primitive_to_conserved(q, gamma=gamma)
        t += dt
        substeps += 1
        if substeps > int(max_substeps):
            raise RuntimeError("Exceeded max_substeps while solving Euler trajectory")
        if t >= next_record - 1e-10:
            records.append(q)
            next_record = min(float(t_final), next_record + float(record_dt))

    return torch.stack(records, dim=1)


def _periodic_grid(n_x: int, device, dtype) -> torch.Tensor:
    return torch.arange(n_x, device=device, dtype=dtype) / float(n_x)


def _random_fourier_field(
    n_x: int,
    n_samples: int,
    amplitude: float,
    max_modes: int,
    decay: float,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    x = _periodic_grid(n_x, device=device, dtype=dtype)
    field = torch.zeros(n_samples, n_x, device=device, dtype=dtype)
    n_terms = max(1, int(max_modes))
    for k in range(1, n_terms + 1):
        phase = 2.0 * np.pi * torch.rand(n_samples, 1, device=device, dtype=dtype)
        coeff_sin = torch.randn(n_samples, 1, device=device, dtype=dtype) / (float(k) ** float(decay))
        coeff_cos = torch.randn(n_samples, 1, device=device, dtype=dtype) / (float(k) ** float(decay))
        angle = 2.0 * np.pi * float(k) * x.view(1, -1) + phase
        field = field + coeff_sin * torch.sin(angle) + coeff_cos * torch.cos(angle)
    field = field - field.mean(dim=-1, keepdim=True)
    field = field / (field.abs().amax(dim=-1, keepdim=True) + 1e-8)
    return float(amplitude) * field


def sample_periodic_grf_1d(
    n_x: int,
    n_samples: int,
    domain_length: float,
    length_scale: float,
    variance: float = 1.0,
    zero_mean: bool = True,
    normalize: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    if n_x <= 0 or n_samples < 0:
        raise ValueError("n_x must be positive and n_samples must be nonnegative")
    if domain_length <= 0.0:
        raise ValueError("domain_length must be positive")
    if length_scale <= 0.0:
        raise ValueError("length_scale must be positive")
    if variance < 0.0:
        raise ValueError("variance must be nonnegative")

    freq = torch.fft.rfftfreq(int(n_x), d=float(domain_length) / float(n_x)).to(device=device, dtype=dtype)
    power = float(variance) * torch.exp(-0.5 * (2.0 * np.pi * float(length_scale) * freq).square())
    if zero_mean:
        power[0] = 0.0

    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    coeff = torch.randn(n_samples, power.numel(), device=device, dtype=complex_dtype)
    coeff = coeff * torch.sqrt(power.clamp_min(0.0)).to(complex_dtype).view(1, -1)
    field = torch.fft.irfft(coeff, n=int(n_x), dim=-1).to(dtype=dtype)
    if zero_mean:
        field = field - field.mean(dim=-1, keepdim=True)
    if normalize:
        field = field / (field.square().mean(dim=-1, keepdim=True).sqrt() + 1e-8)
    return field


def downsample_periodic_primitive(q: torch.Tensor, target_n_x: int) -> torch.Tensor:
    """Downsample periodic primitive states by conservative average pooling."""
    if q.shape[-1] == target_n_x:
        return q.clone()
    source_n_x = int(q.shape[-1])
    if target_n_x <= 0 or source_n_x % int(target_n_x) != 0:
        raise ValueError("target_n_x must divide the source resolution")
    factor = source_n_x // int(target_n_x)
    flat = q.reshape(-1, STATE_CHANNELS, source_n_x)
    pooled = F.avg_pool1d(flat, kernel_size=factor, stride=factor)
    return pooled.reshape(*q.shape[:-1], target_n_x)


def sample_euler1d_initial_conditions(
    n_x: int,
    n_samples: int,
    gamma: float = 1.4,
    rho0: float = 1.0,
    rho_amp: float = 0.15,
    p0: float = 1.0,
    p_amp: float = 0.10,
    mach_min: float = 0.05,
    mach_max: float = 0.35,
    max_modes: int = 5,
    decay: float = 2.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    if n_x <= 0 or n_samples < 0:
        raise ValueError("n_x must be positive and n_samples must be nonnegative")
    rho = float(rho0) * (1.0 + _random_fourier_field(n_x, n_samples, rho_amp, max_modes, decay, device, dtype))
    p = float(p0) * (1.0 + _random_fourier_field(n_x, n_samples, p_amp, max_modes, decay, device, dtype))
    rho = rho.clamp_min(0.1 * float(rho0))
    p = p.clamp_min(0.1 * float(p0))

    base_sound = torch.sqrt(torch.tensor(float(gamma) * float(p0) / float(rho0), device=device, dtype=dtype))
    vel_shape = _random_fourier_field(n_x, n_samples, 1.0, max_modes, decay, device, dtype)
    mach = torch.empty(n_samples, 1, device=device, dtype=dtype).uniform_(float(mach_min), float(mach_max))
    u = mach * base_sound * vel_shape
    return torch.stack([rho, u, p], dim=1)


def sample_euler1d_grf_initial_conditions(
    n_x: int,
    n_samples: int,
    domain_length: float,
    gamma: float = 1.4,
    rho0: float = 1.0,
    rho_amp: float = 0.15,
    p0: float = 1.0,
    p_amp: float = 0.10,
    mach_min: float = 0.05,
    mach_max: float = 0.35,
    length_scale: float = 1.0,
    variance: float = 1.0,
    normalize: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    rho_field = sample_periodic_grf_1d(
        n_x=n_x,
        n_samples=n_samples,
        domain_length=domain_length,
        length_scale=length_scale,
        variance=variance,
        zero_mean=True,
        normalize=normalize,
        device=device,
        dtype=dtype,
    )
    p_field = sample_periodic_grf_1d(
        n_x=n_x,
        n_samples=n_samples,
        domain_length=domain_length,
        length_scale=length_scale,
        variance=variance,
        zero_mean=True,
        normalize=normalize,
        device=device,
        dtype=dtype,
    )
    vel_shape = sample_periodic_grf_1d(
        n_x=n_x,
        n_samples=n_samples,
        domain_length=domain_length,
        length_scale=length_scale,
        variance=variance,
        zero_mean=True,
        normalize=True,
        device=device,
        dtype=dtype,
    )

    rho = float(rho0) * (1.0 + float(rho_amp) * rho_field)
    p = float(p0) * (1.0 + float(p_amp) * p_field)
    rho = rho.clamp_min(0.1 * float(rho0))
    p = p.clamp_min(0.1 * float(p0))

    base_sound = torch.sqrt(torch.tensor(float(gamma) * float(p0) / float(rho0), device=device, dtype=dtype))
    mach = torch.empty(n_samples, 1, device=device, dtype=dtype).uniform_(float(mach_min), float(mach_max))
    u = mach * base_sound * vel_shape
    return torch.stack([rho, u, p], dim=1)


def sample_euler1d_shocktube_initial_conditions(
    n_x: int,
    n_samples: int,
    domain_length: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    x0_middle_fraction: float = 0.2,
) -> torch.Tensor:
    if n_x <= 0 or n_samples < 0:
        raise ValueError("n_x must be positive and n_samples must be nonnegative")
    if domain_length <= 0.0:
        raise ValueError("domain_length must be positive")
    frac = float(x0_middle_fraction)
    if not (0.0 < frac <= 1.0):
        raise ValueError("x0_middle_fraction must be in (0,1]")

    z = torch.rand(n_samples, 6, device=device, dtype=dtype)
    g = 2.0 * z - 1.0
    rho_l = 0.75 + 0.45 * g[:, 0]
    rho_r = 0.40 + 0.30 * g[:, 1]
    u_l = 0.50 + 0.50 * g[:, 2]
    u_r = torch.zeros_like(u_l)
    p_l = 2.50 + 1.60 * g[:, 3]
    p_r = 0.375 + 0.325 * g[:, 4]

    x0_min = 0.5 * float(domain_length) * (1.0 - frac)
    x0_max = 0.5 * float(domain_length) * (1.0 + frac)
    x0 = x0_min + (x0_max - x0_min) * z[:, 5]
    x = (torch.arange(n_x, device=device, dtype=dtype) + 0.5) * (float(domain_length) / float(n_x))
    left = x.view(1, -1) < x0.view(-1, 1)

    rho = torch.where(left, rho_l.view(-1, 1), rho_r.view(-1, 1))
    u = torch.where(left, u_l.view(-1, 1), u_r.view(-1, 1))
    p = torch.where(left, p_l.view(-1, 1), p_r.view(-1, 1))
    return torch.stack([rho, u, p], dim=1)


def _slice_split(data: Dict[str, torch.Tensor], start: int, end: int) -> Dict[str, torch.Tensor]:
    return {
        "f": data["f"][start:end].clone(),
        "u0": data["u0"][start:end].clone(),
        "u_traj": data["u_traj"][start:end].clone(),
    }


def generate_euler1d_dataset_splits(
    n_x: int,
    n_steps: int,
    n_train: int,
    n_val: int,
    n_test: int,
    t_final: float = 0.2,
    domain_length: float = 1.0,
    solver_n_x: int | None = None,
    solver_dt: float | None = None,
    gamma: float = 1.4,
    cfl: float = 0.45,
    rho0: float = 1.0,
    rho_amp: float = 0.15,
    p0: float = 1.0,
    p_amp: float = 0.10,
    mach_min: float = 0.05,
    mach_max: float = 0.35,
    max_modes: int = 5,
    decay: float = 2.0,
    ic_type: str = "smooth",
    boundary_condition: str = "periodic",
    x0_middle_fraction: float = 0.2,
    grf_length_scale: float = 1.0,
    grf_variance: float = 1.0,
    grf_normalize: bool = True,
    solve_batch_size: int = 32,
    max_substeps: int = 100000,
    seed: int = 42,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    output_dtype: torch.dtype = torch.float32,
    show_progress: bool = True,
) -> Dict[str, Dict]:
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    total = int(n_train) + int(n_val) + int(n_test)
    record_dt = float(t_final) / float(n_steps)

    rng_torch = torch.random.get_rng_state()
    rng_np = np.random.get_state()
    torch.manual_seed(seed)
    np.random.seed(seed)

    solve_n_x = int(n_x if solver_n_x is None else solver_n_x)
    if solve_n_x < int(n_x) or solve_n_x % int(n_x) != 0:
        raise ValueError("solver_n_x must be >= n_x and divisible by n_x")
    bc = _normalize_boundary_condition(boundary_condition)
    ic = str(ic_type).strip().lower()

    if ic in {"smooth", "fourier", "random_fourier"}:
        ic = "smooth"
        u0_all = sample_euler1d_initial_conditions(
            n_x=solve_n_x,
            n_samples=total,
            gamma=gamma,
            rho0=rho0,
            rho_amp=rho_amp,
            p0=p0,
            p_amp=p_amp,
            mach_min=mach_min,
            mach_max=mach_max,
            max_modes=max_modes,
            decay=decay,
            device=device,
            dtype=dtype,
        )
    elif ic in {"shocktube", "shock_tube"}:
        ic = "shocktube"
        u0_all = sample_euler1d_shocktube_initial_conditions(
            n_x=solve_n_x,
            n_samples=total,
            domain_length=domain_length,
            device=device,
            dtype=dtype,
            x0_middle_fraction=x0_middle_fraction,
        )
    elif ic in {"grf", "periodic_grf", "gaussian", "periodic_gaussian"}:
        ic = "periodic_grf"
        u0_all = sample_euler1d_grf_initial_conditions(
            n_x=solve_n_x,
            n_samples=total,
            domain_length=domain_length,
            gamma=gamma,
            rho0=rho0,
            rho_amp=rho_amp,
            p0=p0,
            p_amp=p_amp,
            mach_min=mach_min,
            mach_max=mach_max,
            length_scale=grf_length_scale,
            variance=grf_variance,
            normalize=grf_normalize,
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError("ic_type must be one of {smooth,shocktube,grf,periodic_grf}")
    f_all = torch.zeros(total, n_x, dtype=output_dtype)

    starts = range(0, total, int(solve_batch_size))
    if show_progress and tqdm is not None:
        starts = tqdm(starts, total=(total + int(solve_batch_size) - 1) // int(solve_batch_size), desc="solve Euler1D", leave=False)

    chunks = []
    for start in starts:
        end = min(int(start) + int(solve_batch_size), total)
        traj = solve_euler1d_trajectory(
            u0_prim=u0_all[start:end],
            t_final=t_final,
            record_dt=record_dt,
            gamma=gamma,
            domain_length=domain_length,
            cfl=cfl,
            solver_dt=solver_dt,
            max_substeps=max_substeps,
            boundary_condition=bc,
        )
        traj = downsample_periodic_primitive(traj, target_n_x=n_x)
        chunks.append(traj.to(dtype=output_dtype).cpu())

    u_traj = torch.cat(chunks, dim=0) if chunks else torch.empty(0, n_steps + 1, STATE_CHANNELS, n_x, dtype=output_dtype)
    all_data = {"f": f_all.cpu(), "u0": u_traj[:, 0].clone(), "u_traj": u_traj}
    train_end = int(n_train)
    val_end = int(n_train + n_val)

    splits: Dict[str, Dict] = {
        "train": _slice_split(all_data, 0, train_end),
        "val": _slice_split(all_data, train_end, val_end),
        "test": _slice_split(all_data, val_end, total),
        "meta": {
            "dataset_version": DATASET_VERSION,
            "equation": "compressible_euler_1d",
            "domain": f"[0,{float(domain_length)}]",
            "domain_length": float(domain_length),
            "periodic": bc == "periodic",
            "boundary_condition": "periodic" if bc == "periodic" else "neumann",
            "solver_boundary_condition": bc,
            "ic_type": ic,
            "x0_middle_fraction": float(x0_middle_fraction),
            "grf_length_scale": float(grf_length_scale),
            "grf_variance": float(grf_variance),
            "grf_normalize": bool(grf_normalize),
            "n_x": int(n_x),
            "solver_n_x": int(solve_n_x),
            "n_steps": int(n_steps),
            "t_final": float(t_final),
            "record_dt": float(record_dt),
            "dataset_dt": float(record_dt),
            "solver_dt_max": None if solver_dt is None else float(solver_dt),
            "gamma": float(gamma),
            "cfl": float(cfl),
            "rho0": float(rho0),
            "rho_amp": float(rho_amp),
            "p0": float(p0),
            "p_amp": float(p_amp),
            "mach_min": float(mach_min),
            "mach_max": float(mach_max),
            "max_modes": int(max_modes),
            "decay": float(decay),
            "store_primitive": True,
            "state_channels": STATE_CHANNELS,
            "state_names": list(STATE_NAMES_PRIMITIVE),
            "n_train": int(n_train),
            "n_val": int(n_val),
            "n_test": int(n_test),
            "seed": int(seed),
            "solve_batch_size": int(solve_batch_size),
            "max_substeps": int(max_substeps),
            "device": str(device),
        },
    }

    torch.random.set_rng_state(rng_torch)
    np.random.set_state(rng_np)
    return splits


def _print_split_stats(splits: Dict[str, Dict]) -> None:
    meta = splits.get("meta", {})
    print("Euler1D dataset:")
    print(f"  n_x={meta.get('n_x')} steps={meta.get('n_steps')} dt={meta.get('record_dt')} gamma={meta.get('gamma')}")
    for name in ("train", "val", "test"):
        split = splits[name]
        traj = split["u_traj"]
        if int(traj.shape[0]) == 0:
            print(f"  {name}: empty")
            continue
        rho = traj[:, :, 0]
        vel = traj[:, :, 1]
        p = traj[:, :, 2]
        print(
            f"  {name}: u0={tuple(split['u0'].shape)} traj={tuple(traj.shape)} "
            f"rho=[{rho.min().item():.3e},{rho.max().item():.3e}] "
            f"u=[{vel.min().item():.3e},{vel.max().item():.3e}] "
            f"p=[{p.min().item():.3e},{p.max().item():.3e}]"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate 1D compressible Euler dataset splits")
    p.add_argument("--dataset-path", type=str, required=True)
    p.add_argument("--n-x", type=int, default=256)
    p.add_argument("--n-steps", type=int, default=20)
    p.add_argument("--n-train", type=int, default=1500)
    p.add_argument("--n-val", type=int, default=300)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--t-final", type=float, default=0.2)
    p.add_argument("--domain-length", type=float, default=1.0)
    p.add_argument("--solver-n-x", type=int, default=None)
    p.add_argument("--solver-dt", type=float, default=None, help="Maximum solver substep; CFL still caps explicit steps.")
    p.add_argument("--gamma", type=float, default=1.4)
    p.add_argument("--cfl", type=float, default=0.45)
    p.add_argument("--rho0", type=float, default=1.0)
    p.add_argument("--rho-amp", type=float, default=0.15)
    p.add_argument("--p0", type=float, default=1.0)
    p.add_argument("--p-amp", type=float, default=0.10)
    p.add_argument("--mach-min", type=float, default=0.05)
    p.add_argument("--mach-max", type=float, default=0.35)
    p.add_argument("--max-modes", type=int, default=5)
    p.add_argument("--decay", type=float, default=2.0)
    p.add_argument("--ic-type", type=str, default="smooth", choices=["smooth", "shocktube", "grf", "periodic_grf"])
    p.add_argument("--boundary-condition", type=str, default="periodic", choices=["periodic", "outflow", "transmissive", "neumann"])
    p.add_argument("--x0-middle-fraction", type=float, default=0.2)
    p.add_argument("--grf-length-scale", type=float, default=1.0)
    p.add_argument("--grf-variance", type=float, default=1.0)
    p.add_argument("--no-grf-normalize", action="store_true")
    p.add_argument("--solve-batch-size", type=int, default=32)
    p.add_argument("--max-substeps", type=int, default=100000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    splits = generate_euler1d_dataset_splits(
        n_x=args.n_x,
        n_steps=args.n_steps,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        t_final=args.t_final,
        domain_length=args.domain_length,
        solver_n_x=args.solver_n_x,
        solver_dt=args.solver_dt,
        gamma=args.gamma,
        cfl=args.cfl,
        rho0=args.rho0,
        rho_amp=args.rho_amp,
        p0=args.p0,
        p_amp=args.p_amp,
        mach_min=args.mach_min,
        mach_max=args.mach_max,
        max_modes=args.max_modes,
        decay=args.decay,
        ic_type=args.ic_type,
        boundary_condition=args.boundary_condition,
        x0_middle_fraction=args.x0_middle_fraction,
        grf_length_scale=args.grf_length_scale,
        grf_variance=args.grf_variance,
        grf_normalize=not args.no_grf_normalize,
        solve_batch_size=args.solve_batch_size,
        max_substeps=args.max_substeps,
        seed=args.seed,
        device=args.device,
        dtype=torch.float64,
        output_dtype=torch.float32,
        show_progress=not args.no_progress,
    )
    out_dir = os.path.dirname(args.dataset_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_dataset_splits(splits, args.dataset_path)
    print(f"Saved Euler1D dataset splits to: {args.dataset_path}")
    _print_split_stats(splits)


if __name__ == "__main__":
    main(parse_args())
