"""
Data generation for periodic 2D FitzHugh-Nagumo reaction-diffusion:

    u_t = D_u Delta u + u - u^3 - k - v + I(x,y)
    v_t = D_v Delta v + eps * (u + a - b v)

Datasets follow the shared grad_flow_l2 split format:
    split["f"]      : (n_samples, n_x, n_y) scalar current field I
    split["u0"]     : (n_samples, 2, n_x, n_y) initial (u, v)
    split["u_traj"] : (n_samples, n_steps+1, 2, n_x, n_y)
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
STATE_CHANNELS = 2


class FHN2DTrajectoryTensorDataset(Dataset):
    def __init__(self, f_data: torch.Tensor, u0_data: torch.Tensor, u_traj_data: torch.Tensor):
        if f_data.dim() != 3:
            raise ValueError("f_data must have shape (n_samples,n_x,n_y)")
        if u0_data.dim() != 4 or int(u0_data.shape[1]) != STATE_CHANNELS:
            raise ValueError("u0_data must have shape (n_samples,2,n_x,n_y)")
        if u_traj_data.dim() != 5 or int(u_traj_data.shape[2]) != STATE_CHANNELS:
            raise ValueError("u_traj_data must have shape (n_samples,K+1,2,n_x,n_y)")

        n_samples = int(u_traj_data.shape[0])
        n_x = int(u_traj_data.shape[-2])
        n_y = int(u_traj_data.shape[-1])
        if (
            int(f_data.shape[0]) != n_samples
            or tuple(f_data.shape[1:]) != (n_x, n_y)
            or int(u0_data.shape[0]) != n_samples
            or tuple(u0_data.shape[2:]) != (n_x, n_y)
        ):
            raise ValueError("inconsistent FHN dataset tensor shapes")

        self.f_data = f_data
        self.u0_data = u0_data
        self.u_traj_data = u_traj_data

    def __len__(self) -> int:
        return int(self.u0_data.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"f": self.f_data[idx], "u0": self.u0_data[idx], "u_traj": self.u_traj_data[idx]}


class FHN2DStepDataset(Dataset):
    def __init__(self, f_data: torch.Tensor, u_traj_data: torch.Tensor):
        if f_data.dim() != 3:
            raise ValueError("f_data must have shape (n_samples,n_x,n_y)")
        if u_traj_data.dim() != 5 or int(u_traj_data.shape[2]) != STATE_CHANNELS:
            raise ValueError("u_traj_data must have shape (n_samples,K+1,2,n_x,n_y)")
        if int(f_data.shape[0]) != int(u_traj_data.shape[0]) or tuple(f_data.shape[1:]) != tuple(u_traj_data.shape[-2:]):
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


def build_fhn2d_step_dataset(split_or_dataset) -> FHN2DStepDataset:
    if isinstance(split_or_dataset, FHN2DTrajectoryTensorDataset):
        return FHN2DStepDataset(split_or_dataset.f_data, split_or_dataset.u_traj_data)
    if isinstance(split_or_dataset, dict):
        return FHN2DStepDataset(split_or_dataset["f"], split_or_dataset["u_traj"])
    raise TypeError("Expected split dict or FHN2DTrajectoryTensorDataset")


def build_fhn2d_trajectory_dataset_from_split(split: Dict[str, torch.Tensor]) -> FHN2DTrajectoryTensorDataset:
    return FHN2DTrajectoryTensorDataset(split["f"], split["u0"], split["u_traj"])


def _periodic_grid_2d(n_x: int, n_y: int, device: str = "cpu", dtype: torch.dtype = torch.float32):
    x = torch.arange(n_x, device=device, dtype=dtype) / float(n_x)
    y = torch.arange(n_y, device=device, dtype=dtype) / float(n_y)
    return torch.meshgrid(x, y, indexing="ij")


def sample_periodic_grf_2d(
    n_x: int,
    n_y: int,
    n_samples: int,
    length_scale: float = 0.2,
    variance: float = 1.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if length_scale <= 0.0:
        raise ValueError("length_scale must be > 0")
    k_x = torch.fft.fftfreq(n_x, d=1.0 / float(n_x), device=device).to(dtype=dtype)
    k_y = torch.fft.rfftfreq(n_y, d=1.0 / float(n_y), device=device).to(dtype=dtype)
    k2 = k_x[:, None].square() + k_y[None, :].square()
    power = float(variance) * torch.exp(-2.0 * (np.pi * float(length_scale)) ** 2 * k2)
    real = torch.randn(n_samples, n_x, n_y // 2 + 1, device=device, dtype=dtype)
    imag = torch.randn(n_samples, n_x, n_y // 2 + 1, device=device, dtype=dtype)
    spectrum = torch.complex(real, imag) * torch.sqrt(power).unsqueeze(0)
    spectrum[:, 0, 0] = 0.0
    samples = torch.fft.irfft2(spectrum, s=(n_x, n_y), dim=(-2, -1)) * ((n_x * n_y) ** 0.5)
    return samples


def sample_periodic_matern_2d(
    n_x: int,
    n_y: int,
    n_samples: int,
    smoothness: float = 2.5,
    length_scale: float = 0.2,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if smoothness <= 0.0:
        raise ValueError("smoothness must be > 0")
    if length_scale <= 0.0:
        raise ValueError("length_scale must be > 0")
    k_x = torch.fft.fftfreq(n_x, d=1.0 / float(n_x), device=device).to(dtype=dtype)
    k_y = torch.fft.rfftfreq(n_y, d=1.0 / float(n_y), device=device).to(dtype=dtype)
    k2 = k_x[:, None].square() + k_y[None, :].square()
    # Periodic Matérn-like spectral envelope in two spatial dimensions.
    amp = (1.0 + (2.0 * np.pi * float(length_scale)) ** 2 * k2).pow(-0.5 * (float(smoothness) + 1.0))
    amp[0, 0] = 0.0
    real = torch.randn(n_samples, n_x, n_y // 2 + 1, device=device, dtype=dtype)
    imag = torch.randn(n_samples, n_x, n_y // 2 + 1, device=device, dtype=dtype)
    spectrum = torch.complex(real, imag) * amp.unsqueeze(0)
    samples = torch.fft.irfft2(spectrum, s=(n_x, n_y), dim=(-2, -1)) * ((n_x * n_y) ** 0.5)
    return samples


def sample_periodic_sinusoidal_2d(
    n_x: int,
    n_y: int,
    n_samples: int,
    max_modes: int = 6,
    amplitude: float = 1.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    xx, yy = _periodic_grid_2d(n_x, n_y, device=device, dtype=dtype)
    fields = torch.zeros(n_samples, n_x, n_y, device=device, dtype=dtype)
    n_terms = max(3, int(max_modes))
    for _ in range(n_terms):
        k_x = torch.randint(1, int(max_modes) + 1, (n_samples, 1, 1), device=device).to(dtype=dtype)
        k_y = torch.randint(1, int(max_modes) + 1, (n_samples, 1, 1), device=device).to(dtype=dtype)
        ph_x = 2.0 * np.pi * torch.rand(n_samples, 1, 1, device=device, dtype=dtype)
        ph_y = 2.0 * np.pi * torch.rand(n_samples, 1, 1, device=device, dtype=dtype)
        coeff = torch.randn(n_samples, 1, 1, device=device, dtype=dtype) / (k_x + k_y)
        fields = fields + coeff * torch.sin(2.0 * np.pi * k_x * xx + ph_x) * torch.sin(
            2.0 * np.pi * k_y * yy + ph_y
        )
    return float(amplitude) * fields / fields.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)


def sample_forcing_2d(
    n_x: int,
    n_y: int,
    n_samples: int,
    forcing_type: str = "mixed",
    amplitude: float = 0.5,
    grf_prob: float = 0.7,
    length_scale: float = 0.2,
    length_scale_range: tuple[float, float] = (0.08, 0.35),
    max_modes: int = 6,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    forcing_type = forcing_type.lower()
    if forcing_type == "grf":
        field = sample_periodic_grf_2d(n_x, n_y, n_samples, length_scale, device=device, dtype=dtype)
    elif forcing_type == "sinusoidal":
        field = sample_periodic_sinusoidal_2d(n_x, n_y, n_samples, max_modes, device=device, dtype=dtype)
    elif forcing_type == "mixed":
        chunks = []
        for _ in range(n_samples):
            if np.random.rand() < float(grf_prob):
                ls = float(np.random.uniform(*length_scale_range))
                chunks.append(sample_periodic_grf_2d(n_x, n_y, 1, ls, device=device, dtype=dtype))
            else:
                mm = int(np.random.randint(3, int(max_modes) + 1))
                chunks.append(sample_periodic_sinusoidal_2d(n_x, n_y, 1, mm, device=device, dtype=dtype))
        field = torch.cat(chunks, dim=0)
    else:
        raise ValueError("forcing_type must be one of {'grf','sinusoidal','mixed'}")
    field = field - field.mean(dim=(-2, -1), keepdim=True)
    return float(amplitude) * field / field.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)


def sample_initial_conditions(
    n_x: int,
    n_y: int,
    n_samples: int,
    amplitude: float = 0.5,
    smoothness: float = 2.5,
    u_length_scale: float = 0.18,
    v_length_scale: float = 0.25,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    u = sample_periodic_matern_2d(
        n_x,
        n_y,
        n_samples,
        smoothness=smoothness,
        length_scale=u_length_scale,
        device=device,
        dtype=dtype,
    )
    v = sample_periodic_matern_2d(
        n_x,
        n_y,
        n_samples,
        smoothness=smoothness,
        length_scale=v_length_scale,
        device=device,
        dtype=dtype,
    )
    u = float(amplitude) * u / u.std(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    v = 0.5 * float(amplitude) * v / v.std(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return torch.stack([u, v], dim=1)


def _laplacian_periodic(x: torch.Tensor, diffusion: float) -> torch.Tensor:
    n_x = int(x.shape[-2])
    n_y = int(x.shape[-1])
    dtype = x.dtype
    kx = 2.0 * torch.pi * torch.fft.fftfreq(n_x, d=1.0 / float(n_x), device=x.device).to(dtype=dtype)
    ky = 2.0 * torch.pi * torch.fft.rfftfreq(n_y, d=1.0 / float(n_y), device=x.device).to(dtype=dtype)
    k2 = kx[:, None].square() + ky[None, :].square()
    x_hat = torch.fft.rfft2(x, dim=(-2, -1))
    return torch.fft.irfft2(-float(diffusion) * k2 * x_hat, s=(n_x, n_y), dim=(-2, -1))


def _diffuse_periodic(x: torch.Tensor, diffusion: float, dt: float) -> torch.Tensor:
    if float(diffusion) == 0.0 or float(dt) == 0.0:
        return x
    n_x = int(x.shape[-2])
    n_y = int(x.shape[-1])
    dtype = x.dtype
    kx = 2.0 * torch.pi * torch.fft.fftfreq(n_x, d=1.0 / float(n_x), device=x.device).to(dtype=dtype)
    ky = 2.0 * torch.pi * torch.fft.rfftfreq(n_y, d=1.0 / float(n_y), device=x.device).to(dtype=dtype)
    k2 = kx[:, None].square() + ky[None, :].square()
    decay = torch.exp(-float(diffusion) * float(dt) * k2)
    x_hat = torch.fft.rfft2(x, dim=(-2, -1))
    return torch.fft.irfft2(decay * x_hat, s=(n_x, n_y), dim=(-2, -1))


def fhn_rhs(
    state: torch.Tensor,
    forcing: torch.Tensor,
    d_u: float,
    d_v: float,
    eps: float,
    a: float,
    b: float,
    k: float,
) -> torch.Tensor:
    u = state[:, 0]
    v = state[:, 1]
    du = _laplacian_periodic(u, d_u) + u - u.pow(3) - float(k) - v + forcing
    dv = _laplacian_periodic(v, d_v) + float(eps) * (u + float(a) - float(b) * v)
    return torch.stack([du, dv], dim=1)


def fhn_reaction_rhs(
    state: torch.Tensor,
    forcing: torch.Tensor,
    eps: float,
    a: float,
    b: float,
    k: float,
) -> torch.Tensor:
    u = state[:, 0]
    v = state[:, 1]
    du = u - u.pow(3) - float(k) - v + forcing
    dv = float(eps) * (u + float(a) - float(b) * v)
    return torch.stack([du, dv], dim=1)


def rk4_step(
    state: torch.Tensor,
    forcing: torch.Tensor,
    dt: float,
    d_u: float,
    d_v: float,
    eps: float,
    a: float,
    b: float,
    k: float,
):
    k1 = fhn_rhs(state, forcing, d_u, d_v, eps, a, b, k)
    k2 = fhn_rhs(state + 0.5 * float(dt) * k1, forcing, d_u, d_v, eps, a, b, k)
    k3 = fhn_rhs(state + 0.5 * float(dt) * k2, forcing, d_u, d_v, eps, a, b, k)
    k4 = fhn_rhs(state + float(dt) * k3, forcing, d_u, d_v, eps, a, b, k)
    return state + float(dt) * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def reaction_rk4_step(
    state: torch.Tensor,
    forcing: torch.Tensor,
    dt: float,
    eps: float,
    a: float,
    b: float,
    k: float,
) -> torch.Tensor:
    k1 = fhn_reaction_rhs(state, forcing, eps, a, b, k)
    k2 = fhn_reaction_rhs(state + 0.5 * float(dt) * k1, forcing, eps, a, b, k)
    k3 = fhn_reaction_rhs(state + 0.5 * float(dt) * k2, forcing, eps, a, b, k)
    k4 = fhn_reaction_rhs(state + float(dt) * k3, forcing, eps, a, b, k)
    return state + float(dt) * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def strang_step(
    state: torch.Tensor,
    forcing: torch.Tensor,
    dt: float,
    d_u: float,
    d_v: float,
    eps: float,
    a: float,
    b: float,
    k: float,
) -> torch.Tensor:
    half_dt = 0.5 * float(dt)
    state = torch.stack(
        [
            _diffuse_periodic(state[:, 0], d_u, half_dt),
            _diffuse_periodic(state[:, 1], d_v, half_dt),
        ],
        dim=1,
    )
    state = reaction_rk4_step(state, forcing, dt, eps, a, b, k)
    return torch.stack(
        [
            _diffuse_periodic(state[:, 0], d_u, half_dt),
            _diffuse_periodic(state[:, 1], d_v, half_dt),
        ],
        dim=1,
    )


def _integer_ratio(numer: float, denom: float, name: str) -> int:
    ratio = float(numer) / float(denom)
    out = int(round(ratio))
    if out < 1 or abs(ratio - out) > 1e-10:
        raise ValueError(f"{name} must be a positive integer multiple of solver_dt")
    return out


def solve_fhn2d_trajectory(
    u0: torch.Tensor,
    forcing: torch.Tensor,
    n_steps: int,
    solver_dt: float = 1e-3,
    dataset_dt: float = 1e-2,
    d_u: float = 1e-3,
    d_v: float = 5e-3,
    eps: float = 1e-2,
    a: float = 0.7,
    b: float = 0.8,
    k: float = 5e-3,
    solver_method: str = "strang",
) -> torch.Tensor:
    if u0.dim() != 4 or int(u0.shape[1]) != STATE_CHANNELS:
        raise ValueError("u0 must have shape (batch,2,n_x,n_y)")
    if forcing.dim() != 3 or int(forcing.shape[0]) != int(u0.shape[0]) or tuple(forcing.shape[1:]) != tuple(u0.shape[-2:]):
        raise ValueError("forcing must have shape (batch,n_x,n_y) matching u0")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if solver_method not in {"strang", "rk4"}:
        raise ValueError("solver_method must be one of {'strang','rk4'}")
    stride = _integer_ratio(dataset_dt, solver_dt, "dataset_dt")
    state = u0
    states = [state]
    for record_idx in range(1, n_steps + 1):
        for _ in range(stride):
            if solver_method == "strang":
                state = strang_step(state, forcing, solver_dt, d_u, d_v, eps, a, b, k)
            else:
                state = rk4_step(state, forcing, solver_dt, d_u, d_v, eps, a, b, k)
            if not torch.isfinite(state).all():
                raise FloatingPointError(f"Non-finite FHN state before recorded step {record_idx}")
        states.append(state)
    return torch.stack(states, dim=1)


def downsample_field_2d(field: torch.Tensor, n_x: int, n_y: int) -> torch.Tensor:
    if tuple(field.shape[-2:]) == (int(n_x), int(n_y)):
        return field
    flat = field.reshape(-1, 1, int(field.shape[-2]), int(field.shape[-1]))
    out = F.interpolate(flat, size=(int(n_x), int(n_y)), mode="bilinear", align_corners=False)
    return out.reshape(*field.shape[:-2], int(n_x), int(n_y))


def downsample_state_trajectory_2d(traj: torch.Tensor, n_x: int, n_y: int) -> torch.Tensor:
    if tuple(traj.shape[-2:]) == (int(n_x), int(n_y)):
        return traj
    flat = traj.reshape(-1, int(traj.shape[-3]), int(traj.shape[-2]), int(traj.shape[-1]))
    out = F.interpolate(flat, size=(int(n_x), int(n_y)), mode="bilinear", align_corners=False)
    return out.reshape(*traj.shape[:-2], int(n_x), int(n_y))


def _slice_split(data: Dict[str, torch.Tensor], start: int, end: int) -> Dict[str, torch.Tensor]:
    return {"f": data["f"][start:end].clone(), "u0": data["u0"][start:end].clone(), "u_traj": data["u_traj"][start:end].clone()}


def generate_dataset(args: argparse.Namespace) -> Dict[str, Dict[str, torch.Tensor]]:
    dtype = torch.float64 if args.float64 else torch.float32
    total = int(args.n_train + args.n_val + args.n_test)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    solver_n_x = int(args.solver_n_x if args.solver_n_x is not None else args.n_x)
    solver_n_y = int(args.solver_n_y if args.solver_n_y is not None else args.n_y)

    forcing = sample_forcing_2d(
        solver_n_x,
        solver_n_y,
        total,
        forcing_type=args.forcing_type,
        amplitude=args.forcing_amplitude,
        grf_prob=args.grf_prob,
        length_scale=args.forcing_length_scale,
        max_modes=args.max_forcing_modes,
        device=args.device,
        dtype=dtype,
    )
    u0 = sample_initial_conditions(
        solver_n_x,
        solver_n_y,
        total,
        amplitude=args.ic_amplitude,
        smoothness=args.ic_matern_smoothness,
        u_length_scale=args.ic_u_length_scale,
        v_length_scale=args.ic_v_length_scale,
        device=args.device,
        dtype=dtype,
    )

    traj_chunks = []
    iterator = range(0, total, args.batch_generate)
    if tqdm is not None and not args.no_pbar:
        iterator = tqdm(iterator, desc="Generating FHN2D trajectories", leave=False)
    for start in iterator:
        end = min(total, start + args.batch_generate)
        traj_chunks.append(
            solve_fhn2d_trajectory(
                u0[start:end],
                forcing[start:end],
                n_steps=args.n_steps,
                solver_dt=args.solver_dt,
                dataset_dt=args.dataset_dt,
                d_u=args.d_u,
                d_v=args.d_v,
                eps=args.eps,
                a=args.a,
                b=args.b,
                k=args.k,
                solver_method=args.solver_method,
            )
            .to(dtype=torch.float32)
        )

    u_traj_solver = torch.cat(traj_chunks, dim=0)
    u_traj = downsample_state_trajectory_2d(u_traj_solver, args.n_x, args.n_y).to(dtype=torch.float32)
    f_stored = downsample_field_2d(forcing, args.n_x, args.n_y).to(dtype=torch.float32)
    all_data = {"f": f_stored.cpu(), "u0": u_traj[:, 0].cpu(), "u_traj": u_traj.cpu()}
    splits = {
        "train": _slice_split(all_data, 0, args.n_train),
        "val": _slice_split(all_data, args.n_train, args.n_train + args.n_val),
        "test": _slice_split(all_data, args.n_train + args.n_val, total),
        "meta": {
            "dataset_version": DATASET_VERSION,
            "equation": "periodic_2d_fitzhugh_nagumo",
            "state_channels": STATE_CHANNELS,
            "n_x": int(args.n_x),
            "n_y": int(args.n_y),
            "solver_n_x": solver_n_x,
            "solver_n_y": solver_n_y,
            "n_steps": int(args.n_steps),
            "solver_dt": float(args.solver_dt),
            "solver_method": args.solver_method,
            "dataset_dt": float(args.dataset_dt),
            "t_final": float(args.n_steps) * float(args.dataset_dt),
            "d_u": float(args.d_u),
            "d_v": float(args.d_v),
            "eps": float(args.eps),
            "a": float(args.a),
            "b": float(args.b),
            "k": float(args.k),
            "forcing_type": args.forcing_type,
            "forcing_amplitude": float(args.forcing_amplitude),
            "ic_amplitude": float(args.ic_amplitude),
            "ic_sampler": "matern",
            "ic_matern_smoothness": float(args.ic_matern_smoothness),
            "ic_u_length_scale": float(args.ic_u_length_scale),
            "ic_v_length_scale": float(args.ic_v_length_scale),
            "boundary_condition": "periodic",
        },
    }
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate periodic 2D FitzHugh-Nagumo datasets")
    parser.add_argument("--dataset-path", type=str, default="grad_flow_l2/fhn2d/data/fhn2d.pt")
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--n-x", type=int, default=64)
    parser.add_argument("--n-y", type=int, default=64)
    parser.add_argument("--solver-n-x", type=int, default=None)
    parser.add_argument("--solver-n-y", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--solver-dt", type=float, default=1e-3)
    parser.add_argument("--solver-method", type=str, default="strang", choices=["strang", "rk4"])
    parser.add_argument("--dataset-dt", type=float, default=1e-2)
    parser.add_argument("--d-u", type=float, default=1e-3)
    parser.add_argument("--d-v", type=float, default=5e-3)
    parser.add_argument("--eps", type=float, default=1e-2)
    parser.add_argument("--a", type=float, default=0.7)
    parser.add_argument("--b", type=float, default=0.8)
    parser.add_argument("--k", type=float, default=5e-3)
    parser.add_argument("--forcing-type", type=str, default="mixed", choices=["grf", "sinusoidal", "mixed"])
    parser.add_argument("--forcing-amplitude", type=float, default=0.5)
    parser.add_argument("--forcing-length-scale", type=float, default=0.2)
    parser.add_argument("--grf-prob", type=float, default=0.7)
    parser.add_argument("--max-forcing-modes", type=int, default=6)
    parser.add_argument("--ic-amplitude", type=float, default=0.5)
    parser.add_argument("--ic-matern-smoothness", type=float, default=2.5)
    parser.add_argument("--ic-u-length-scale", type=float, default=0.05)
    parser.add_argument("--ic-v-length-scale", type=float, default=0.15)
    parser.add_argument("--batch-generate", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--no-pbar", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    splits = generate_dataset(args)
    os.makedirs(os.path.dirname(args.dataset_path), exist_ok=True)
    save_dataset_splits(splits, args.dataset_path)
    print(f"Saved FHN2D dataset: {args.dataset_path}")
    for name in ["train", "val", "test"]:
        split = splits[name]
        print(
            f"{name}: f={tuple(split['f'].shape)}, u0={tuple(split['u0'].shape)}, "
            f"u_traj={tuple(split['u_traj'].shape)}, finite={bool(torch.isfinite(split['u_traj']).all())}"
        )
    print("meta:", splits["meta"])


if __name__ == "__main__":
    main(parse_args())
