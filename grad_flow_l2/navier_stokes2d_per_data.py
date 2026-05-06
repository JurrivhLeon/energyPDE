"""
Data generation for 2D viscous incompressible Navier-Stokes in vorticity form
on the unit torus.

Trajectories are generated on a periodic solver grid and stored on the same
grid by default (64x64 for the practical data path). The code still supports
high-resolution generation plus Fourier truncation, but the default path is the
lightweight 64x64-only configuration.

Dataset format mirrors the existing grad_flow_l2 generators, but the stored
arrays live on the low-resolution grid:
    split["f"]      : (n_samples, n_x, n_y)
    split["u0"]     : (n_samples, n_x, n_y)
    split["u_traj"] : (n_samples, n_steps+1, n_x, n_y)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .heat_data import save_dataset_splits
    from .navier_stokes2d_per_solver import (
        project_zero_mean_2d,
        sample_periodic_gaussian_field_2d,
        solve_navier_stokes_vorticity_trajectory_pseudospectral,
        spectral_truncate_periodic_field_2d,
    )
except ImportError:
    from grad_flow_l2.heat_data import save_dataset_splits
    from grad_flow_l2.navier_stokes2d_per_solver import (
        project_zero_mean_2d,
        sample_periodic_gaussian_field_2d,
        solve_navier_stokes_vorticity_trajectory_pseudospectral,
        spectral_truncate_periodic_field_2d,
    )


DATASET_VERSION = 2


def _iter_with_progress(
    iterable: Iterable,
    total: int,
    desc: str,
    enabled: bool,
):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm  # type: ignore

        return tqdm(iterable, total=total, desc=desc, leave=False)
    except Exception:
        return iterable


class NavierStokes2DPeriodicTrajectoryTensorDataset(Dataset):
    """
    Dataset wrapper around precomputed tensors:
      f: (n_samples, n_x, n_y)
      u0: (n_samples, n_x, n_y)
      u_traj: (n_samples, K+1, n_x, n_y)
    """

    def __init__(self, f_data: torch.Tensor, u0_data: torch.Tensor, u_traj_data: torch.Tensor):
        if f_data.dim() != 3:
            raise ValueError("f_data must have shape (n_samples,n_x,n_y)")
        if u0_data.dim() != 3:
            raise ValueError("u0_data must have shape (n_samples,n_x,n_y)")
        if u_traj_data.dim() != 4:
            raise ValueError("u_traj_data must have shape (n_samples,K+1,n_x,n_y)")

        n_samples = int(u_traj_data.shape[0])
        n_x = int(u_traj_data.shape[2])
        n_y = int(u_traj_data.shape[3])
        if (
            int(f_data.shape[0]) != n_samples
            or tuple(f_data.shape[1:]) != (n_x, n_y)
            or int(u0_data.shape[0]) != n_samples
            or tuple(u0_data.shape[1:]) != (n_x, n_y)
        ):
            raise ValueError("inconsistent tensor shapes for NavierStokes2D periodic trajectory dataset")

        self.f_data = f_data
        self.u0_data = u0_data
        self.u_traj_data = u_traj_data

    def __len__(self) -> int:
        return int(self.u0_data.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "f": self.f_data[idx],
            "u0": self.u0_data[idx],
            "u_traj": self.u_traj_data[idx],
        }


class NavierStokes2DPeriodicStepDataset(Dataset):
    """
    Flattened one-step dataset from trajectories.
    Returns tuple: (u_k, u_{k+1}, f).
    """

    def __init__(self, f_data: torch.Tensor, u_traj_data: torch.Tensor):
        if f_data.dim() != 3:
            raise ValueError("f_data must have shape (n_samples,n_x,n_y)")
        if u_traj_data.dim() != 4:
            raise ValueError("u_traj_data must have shape (n_samples,K+1,n_x,n_y)")
        if int(f_data.shape[0]) != int(u_traj_data.shape[0]) or tuple(f_data.shape[1:]) != tuple(u_traj_data.shape[2:]):
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


def build_navier_stokes2d_periodic_step_dataset(split_or_dataset) -> NavierStokes2DPeriodicStepDataset:
    if isinstance(split_or_dataset, NavierStokes2DPeriodicTrajectoryTensorDataset):
        f_data = split_or_dataset.f_data
        u_traj_data = split_or_dataset.u_traj_data
    elif isinstance(split_or_dataset, dict):
        f_data = split_or_dataset["f"]
        u_traj_data = split_or_dataset["u_traj"]
    else:
        raise TypeError("Expected split dict or NavierStokes2DPeriodicTrajectoryTensorDataset")
    return NavierStokes2DPeriodicStepDataset(f_data=f_data, u_traj_data=u_traj_data)


def build_navier_stokes2d_periodic_trajectory_dataset_from_split(
    split: Dict[str, torch.Tensor],
) -> NavierStokes2DPeriodicTrajectoryTensorDataset:
    return NavierStokes2DPeriodicTrajectoryTensorDataset(
        f_data=split["f"],
        u0_data=split["u0"],
        u_traj_data=split["u_traj"],
    )


def sample_periodic_grf_2d(
    n_x: int,
    n_y: int,
    n_samples: int = 1,
    length_scale: float = 0.2,
    variance: float = 1.0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Sample smooth periodic 2D Gaussian random fields via spectral synthesis.
    """
    if length_scale <= 0:
        raise ValueError("length_scale must be > 0")

    k_x = torch.fft.fftfreq(n_x, d=1.0 / float(n_x), device=device)
    k_y = torch.fft.rfftfreq(n_y, d=1.0 / float(n_y), device=device)
    k2 = k_x.unsqueeze(1) ** 2 + k_y.unsqueeze(0) ** 2
    power = variance * torch.exp(-2.0 * (np.pi * length_scale) ** 2 * k2)

    real = torch.randn(n_samples, n_x, n_y // 2 + 1, device=device)
    imag = torch.randn(n_samples, n_x, n_y // 2 + 1, device=device)
    spectrum = torch.complex(real, imag) * torch.sqrt(power).unsqueeze(0)
    spectrum[:, 0, 0] = 0.0
    samples = torch.fft.irfft2(spectrum, s=(n_x, n_y)) * ((n_x * n_y) ** 0.5)
    return project_zero_mean_2d(samples)


def sample_periodic_matern15_spectral_2d(
    n_x: int,
    n_y: int,
    n_samples: int = 1,
    length_scale: float = 0.35,
    variance: float = 1.0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Sample 2D periodic fields with an approximate Matérn-1.5 spectrum.
    """
    if length_scale <= 0:
        raise ValueError("length_scale must be > 0")

    nu = 1.5
    dim = 2.0
    kappa2 = (2.0 * nu) / (float(length_scale) * float(length_scale))

    k_x = torch.fft.fftfreq(n_x, d=1.0 / float(n_x), device=device)
    k_y = torch.fft.rfftfreq(n_y, d=1.0 / float(n_y), device=device)
    k2 = k_x.unsqueeze(1) ** 2 + k_y.unsqueeze(0) ** 2
    radial = (2.0 * np.pi) ** 2 * k2
    power = variance * torch.pow(kappa2 + radial, -(nu + dim / 2.0))

    real = torch.randn(n_samples, n_x, n_y // 2 + 1, device=device)
    imag = torch.randn(n_samples, n_x, n_y // 2 + 1, device=device)
    spectrum = torch.complex(real, imag) * torch.sqrt(power).unsqueeze(0)
    spectrum[:, 0, 0] = 0.0
    samples = torch.fft.irfft2(spectrum, s=(n_x, n_y)) * ((n_x * n_y) ** 0.5)
    return project_zero_mean_2d(samples)


def _periodic_grid_2d(
    n_x: int,
    n_y: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(0.0, 1.0, steps=n_x + 1, device=device, dtype=dtype)[:-1]
    y = torch.linspace(0.0, 1.0, steps=n_y + 1, device=device, dtype=dtype)[:-1]
    return torch.meshgrid(x, y, indexing="ij")


def sample_periodic_sinusoidal_2d(
    n_x: int,
    n_y: int,
    n_samples: int = 1,
    max_modes: int = 6,
    amplitude: float = 1.0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Random periodic sinusoidal fields on the torus.
    """
    if max_modes < 1:
        raise ValueError("max_modes must be >= 1")

    xx, yy = _periodic_grid_2d(n_x=n_x, n_y=n_y, device=device)
    fields = torch.zeros(n_samples, n_x, n_y, device=device)
    n_terms = max(3, max_modes)
    for _ in range(n_terms):
        k_x = torch.randint(1, max_modes + 1, (n_samples, 1, 1), device=device).float()
        k_y = torch.randint(1, max_modes + 1, (n_samples, 1, 1), device=device).float()
        ph_x = 2.0 * np.pi * torch.rand(n_samples, 1, 1, device=device)
        ph_y = 2.0 * np.pi * torch.rand(n_samples, 1, 1, device=device)
        coeff = torch.randn(n_samples, 1, 1, device=device) / (k_x + k_y)
        fields = fields + coeff * torch.sin(2.0 * np.pi * k_x * xx.unsqueeze(0) + ph_x) * torch.sin(
            2.0 * np.pi * k_y * yy.unsqueeze(0) + ph_y
        )

    fields = project_zero_mean_2d(fields)
    max_val = fields.abs().amax(dim=(1, 2), keepdim=True) + 1e-8
    fields = float(amplitude) * fields / max_val
    return fields


def sample_periodic_field_mixed_2d(
    n_x: int,
    n_y: int,
    n_samples: int = 1,
    amplitude: float = 1.0,
    sinusoidal_amplitude_range: tuple[float, float] = (0.05, 0.5),
    length_scale_range: tuple[float, float] = (0.05, 0.30),
    max_modes: int = 3,
    grf_prob: float = 0.55,
    matern_prob: float = 0.35,
    matern_length_scale_range: tuple[float, float] = (0.25, 0.80),
    allow_sinusoidal: bool = True,
    show_progress: bool = False,
    progress_desc: str = "sample fields",
    device: str = "cpu",
) -> torch.Tensor:
    """
    Mixed periodic sampler: each sample is GRF, Matérn-1.5, or sinusoidal.
    """
    if grf_prob < 0 or matern_prob < 0:
        raise ValueError("probabilities must be >= 0")
    if allow_sinusoidal:
        if grf_prob + matern_prob > 1.0:
            raise ValueError("grf_prob + matern_prob must be <= 1")
    else:
        if grf_prob + matern_prob <= 0:
            raise ValueError("grf_prob + matern_prob must be > 0 when sinusoidal sampling is disabled")
    if length_scale_range[0] <= 0 or length_scale_range[1] <= 0:
        raise ValueError("length_scale_range values must be > 0")
    if matern_length_scale_range[0] <= 0 or matern_length_scale_range[1] <= 0:
        raise ValueError("matern_length_scale_range values must be > 0")
    if sinusoidal_amplitude_range[0] <= 0 or sinusoidal_amplitude_range[1] <= 0:
        raise ValueError("sinusoidal_amplitude_range values must be > 0")
    if length_scale_range[0] >= length_scale_range[1]:
        raise ValueError("length_scale_range must satisfy min < max")
    if matern_length_scale_range[0] >= matern_length_scale_range[1]:
        raise ValueError("matern_length_scale_range must satisfy min < max")
    if sinusoidal_amplitude_range[0] >= sinusoidal_amplitude_range[1]:
        raise ValueError("sinusoidal_amplitude_range must satisfy min < max")

    out = []
    iterator = _iter_with_progress(
        range(n_samples),
        total=n_samples,
        desc=progress_desc,
        enabled=show_progress,
    )
    for _ in iterator:
        r = float(np.random.rand())
        is_sinusoidal = False
        if allow_sinusoidal:
            if r < grf_prob:
                choose_grf = True
            elif r < grf_prob + matern_prob:
                choose_grf = False
            else:
                mm = int(np.random.randint(3, max_modes + 1))
                sinusoidal_amp = float(np.random.uniform(*sinusoidal_amplitude_range))
                field = sample_periodic_sinusoidal_2d(
                    n_x=n_x,
                    n_y=n_y,
                    n_samples=1,
                    max_modes=mm,
                    amplitude=sinusoidal_amp,
                    device=device,
                )
                choose_grf = None
                is_sinusoidal = True
            if choose_grf is not None:
                if choose_grf:
                    length_scale = float(np.random.uniform(*length_scale_range))
                    field = sample_periodic_grf_2d(
                        n_x=n_x,
                        n_y=n_y,
                        n_samples=1,
                        length_scale=length_scale,
                        variance=1.0,
                        device=device,
                    )
                else:
                    length_scale = float(np.random.uniform(*matern_length_scale_range))
                    field = sample_periodic_matern15_spectral_2d(
                        n_x=n_x,
                        n_y=n_y,
                        n_samples=1,
                        length_scale=length_scale,
                        variance=1.0,
                        device=device,
                    )
        else:
            total_prob = grf_prob + matern_prob
            grf_cut = grf_prob / total_prob
            if r < grf_cut:
                length_scale = float(np.random.uniform(*length_scale_range))
                field = sample_periodic_grf_2d(
                    n_x=n_x,
                    n_y=n_y,
                    n_samples=1,
                    length_scale=length_scale,
                    variance=1.0,
                    device=device,
                )
            else:
                length_scale = float(np.random.uniform(*matern_length_scale_range))
                field = sample_periodic_matern15_spectral_2d(
                    n_x=n_x,
                    n_y=n_y,
                    n_samples=1,
                    length_scale=length_scale,
                    variance=1.0,
                    device=device,
                )

        field = project_zero_mean_2d(field)
        if not is_sinusoidal:
            # Keep the sampled fields mostly within the target value range.
            amp = float(amplitude * (10.0 ** np.random.uniform(-0.15, 0.15)))
            field = amp * field / (field.abs().amax(dim=(1, 2), keepdim=True) + 1e-8)
        out.append(field)
    return torch.cat(out, dim=0)


def _rescale_batch_l2_2d(
    x: torch.Tensor,
    area: float,
    norm_min: float,
    norm_max: float,
) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError("x must have shape (batch, n_x, n_y)")
    norms = torch.sqrt(float(area) * torch.sum(x * x, dim=(1, 2)))
    targets = torch.empty_like(norms).uniform_(norm_min, norm_max)
    scale = targets / (norms + 1e-8)
    return x * scale.view(-1, 1, 1)


def _slice_split(data: Dict[str, torch.Tensor], start: int, end: int) -> Dict[str, torch.Tensor]:
    return {
        "f": data["f"][start:end].clone(),
        "u0": data["u0"][start:end].clone(),
        "u_traj": data["u_traj"][start:end].clone(),
    }


def generate_navier_stokes2d_periodic_dataset_splits(
    n_x: int,
    n_y: int,
    n_steps: int,
    t_final: float,
    n_train: int,
    n_val: int,
    n_test: int,
    nu: float = 0.001,
    seed: int = 42,
    solver_n_x: int = 64,
    solver_n_y: int = 64,
    solver_dt: float = 1e-3,
    record_dt: float = 1.0,
    u0_spectrum_scale: float = 8.0 ** 1.5,
    u0_spectrum_shift: float = 4.0,
    u0_spectrum_power: float = 2.5,
    u0_rescale: float = 10.0,
    forcing_mode: str = "mixed",
    f_amplitude: float = 0.45,
    f_sinusoidal_amplitude_min: float = 0.05,
    f_sinusoidal_amplitude_max: float = 0.5,
    f_grf_prob: float = 0.7,
    f_matern_prob: float = 0.0,
    f_length_scale_min: float = 0.05,
    f_length_scale_max: float = 0.30,
    f_max_modes: int = 3,
    f_allow_sinusoidal: bool = True,
    norm_targeting: bool = False,
    target_u0_norm_range: tuple[float, float] = (0.4, 1.0),
    target_f_norm_range: tuple[float, float] = (0.1, 0.5),
    cfl_adv: float = 0.45,
    chunk_size: int = 256,
    show_progress: bool = False,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, Dict[str, torch.Tensor]]:
    if nu <= 0:
        raise ValueError("nu must be > 0")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if t_final <= 0:
        raise ValueError("t_final must be > 0")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if solver_n_x < n_x or solver_n_y < n_y:
        raise ValueError("solver grid must be at least as large as the output grid")
    if solver_dt <= 0 or record_dt <= 0:
        raise ValueError("solver_dt and record_dt must be > 0")
    if u0_rescale <= 0:
        raise ValueError("u0_rescale must be > 0")
    if f_sinusoidal_amplitude_min <= 0 or f_sinusoidal_amplitude_max <= 0:
        raise ValueError("f_sinusoidal_amplitude_min/max must be > 0")
    if f_sinusoidal_amplitude_min >= f_sinusoidal_amplitude_max:
        raise ValueError("f_sinusoidal_amplitude_min must be < f_sinusoidal_amplitude_max")
    expected_records = int(round(float(t_final) / float(record_dt)))
    if expected_records != int(n_steps):
        raise ValueError(
            "n_steps must match the number of recorded intervals: "
            f"expected {expected_records} from t_final/record_dt, got {n_steps}"
        )

    rng_state_torch = torch.random.get_rng_state()
    rng_state_numpy = np.random.get_state()
    torch.manual_seed(seed)
    np.random.seed(seed)

    total = int(n_train + n_val + n_test)
    h_x = 1.0 / float(n_x)
    h_y = 1.0 / float(n_y)
    area = h_x * h_y
    device = str(device)

    u0_hr = sample_periodic_gaussian_field_2d(
        n_x=solver_n_x,
        n_y=solver_n_y,
        n_samples=total,
        spectrum_scale=u0_spectrum_scale,
        spectrum_shift=u0_spectrum_shift,
        spectrum_power=u0_spectrum_power,
        zero_mean=True,
        device=device,
        dtype=torch.float64,
    )
    u0_hr = float(u0_rescale) * u0_hr
    u0 = spectral_truncate_periodic_field_2d(u0_hr, target_n_x=n_x, target_n_y=n_y).to(dtype=dtype)
    u0 = project_zero_mean_2d(u0)

    if forcing_mode not in ("zero", "mixed"):
        raise ValueError("forcing_mode must be one of {'zero', 'mixed'}")

    if forcing_mode == "zero":
        f = torch.zeros(total, n_x, n_y, dtype=dtype)
        f_hr = None
    else:
        f_hr = sample_periodic_field_mixed_2d(
            n_x=solver_n_x,
            n_y=solver_n_y,
            n_samples=total,
            amplitude=f_amplitude,
            sinusoidal_amplitude_range=(f_sinusoidal_amplitude_min, f_sinusoidal_amplitude_max),
            length_scale_range=(f_length_scale_min, f_length_scale_max),
            max_modes=f_max_modes,
            grf_prob=f_grf_prob,
            matern_prob=f_matern_prob,
            allow_sinusoidal=f_allow_sinusoidal,
            show_progress=show_progress,
            progress_desc="sample forcing",
            device=device,
        ).to(dtype=torch.float64)
        f = spectral_truncate_periodic_field_2d(f_hr, target_n_x=n_x, target_n_y=n_y).to(dtype=dtype)
        f = project_zero_mean_2d(f)

    u_traj_chunks = []
    chunk_starts = range(0, total, int(chunk_size))
    chunk_iter = _iter_with_progress(
        chunk_starts,
        total=(total + int(chunk_size) - 1) // int(chunk_size),
        desc="solve trajectories",
        enabled=show_progress,
    )
    for start in chunk_iter:
        end = min(int(start) + int(chunk_size), total)
        u0_chunk = u0_hr[start:end]
        f_chunk = None if forcing_mode == "zero" else f_hr[start:end]
        u_traj_chunk_hr = solve_navier_stokes_vorticity_trajectory_pseudospectral(
            u0=u0_chunk,
            forcing=f_chunk,
            t_final=t_final,
            dt=float(solver_dt),
            record_dt=float(record_dt),
            nu=nu,
        ).to(dtype=torch.float64)
        u_traj_chunk = spectral_truncate_periodic_field_2d(
            u_traj_chunk_hr.reshape(-1, solver_n_x, solver_n_y),
            target_n_x=n_x,
            target_n_y=n_y,
        ).to(dtype=dtype)
        u_traj_chunk = u_traj_chunk.reshape(end - start, n_steps + 1, n_x, n_y)
        u_traj_chunks.append(u_traj_chunk)
    u_traj = torch.cat(u_traj_chunks, dim=0)

    all_data = {
        "f": f.cpu(),
        "u0": u0.cpu(),
        "u_traj": u_traj.cpu(),
    }
    train_end = int(n_train)
    val_end = int(n_train + n_val)

    splits: Dict[str, Dict[str, torch.Tensor]] = {
        "train": _slice_split(all_data, 0, train_end),
        "val": _slice_split(all_data, train_end, val_end),
        "test": _slice_split(all_data, val_end, total),
        "meta": {
            "dataset_version": DATASET_VERSION,
            "equation": "navier_stokes_2d_vorticity_periodic",
            "domain": "unit_torus",
            "periodic": True,
            "mean_free_vorticity": True,
            "n_x": int(n_x),
            "n_y": int(n_y),
            "output_n_x": int(n_x),
            "output_n_y": int(n_y),
            "solver_n_x": int(solver_n_x),
            "solver_n_y": int(solver_n_y),
            "n_steps": int(n_steps),
            "t_final": float(t_final),
            "solver_dt": float(solver_dt),
            "record_dt": float(record_dt),
            "n_train": int(n_train),
            "n_val": int(n_val),
            "n_test": int(n_test),
            "seed": int(seed),
            "nu": float(nu),
            "forcing_mode": forcing_mode,
            "u0_grid_points": [int(solver_n_x), int(solver_n_y)],
            "f_grid_points": [int(solver_n_x), int(solver_n_y)],
            "u0_spectrum_scale": float(u0_spectrum_scale),
            "u0_spectrum_shift": float(u0_spectrum_shift),
            "u0_spectrum_power": float(u0_spectrum_power),
            "u0_rescale": float(u0_rescale),
            "f_amplitude": float(f_amplitude),
            "f_sinusoidal_amplitude_min": float(f_sinusoidal_amplitude_min),
            "f_sinusoidal_amplitude_max": float(f_sinusoidal_amplitude_max),
            "f_grf_prob": float(f_grf_prob),
            "f_matern_prob": float(f_matern_prob),
            "f_length_scale_min": float(f_length_scale_min),
            "f_length_scale_max": float(f_length_scale_max),
            "f_max_modes": int(f_max_modes),
            "f_allow_sinusoidal": bool(f_allow_sinusoidal),
            "norm_targeting": bool(norm_targeting),
            "target_u0_norm_range": [float(target_u0_norm_range[0]), float(target_u0_norm_range[1])],
            "target_f_norm_range": [float(target_f_norm_range[0]), float(target_f_norm_range[1])],
            "cfl_adv": float(cfl_adv),
            "chunk_size": int(chunk_size),
            "device": device,
        },
    }

    torch.random.set_rng_state(rng_state_torch)
    np.random.set_state(rng_state_numpy)
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cached periodic 2D Navier-Stokes dataset splits")
    parser.add_argument("--n-x", type=int, default=64, help="Number of output grid points in x")
    parser.add_argument("--n-y", type=int, default=64, help="Number of output grid points in y")
    parser.add_argument("--solver-n-x", type=int, default=256, help="Number of solver grid points in x")
    parser.add_argument("--solver-n-y", type=int, default=256, help="Number of solver grid points in y")
    parser.add_argument("--n-steps", type=int, default=10, help="Number of recorded macro steps on [0,t_final]")
    parser.add_argument("--t-final", type=float, default=10.0, help="Final time horizon")
    parser.add_argument("--nu", type=float, default=0.001, help="Viscosity coefficient")
    parser.add_argument("--solver-dt", type=float, default=1e-4, help="Time step used by the pseudospectral solver")
    parser.add_argument("--record-dt", type=float, default=1.0, help="Snapshot spacing used during data generation")

    parser.add_argument("--n-train", type=int, default=1500)
    parser.add_argument("--n-val", type=int, default=300)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--u0-spectrum-scale", type=float, default=8.0 ** 1.5)
    parser.add_argument("--u0-spectrum-shift", type=float, default=4.0)
    parser.add_argument("--u0-spectrum-power", type=float, default=2.5)
    parser.add_argument("--u0-rescale", type=float, default=10.0)
    parser.add_argument("--forcing-mode", type=str, default="mixed", choices=["zero", "mixed"])
    parser.add_argument("--f-amplitude", type=float, default=0.45)
    parser.add_argument("--f-sinusoidal-amplitude-min", type=float, default=0.05)
    parser.add_argument("--f-sinusoidal-amplitude-max", type=float, default=0.5)
    parser.add_argument("--f-grf-prob", type=float, default=0.7)
    parser.add_argument("--f-matern-prob", type=float, default=0.0)
    parser.add_argument("--f-length-scale-min", type=float, default=0.05)
    parser.add_argument("--f-length-scale-max", type=float, default=0.30)
    parser.add_argument("--f-max-modes", type=int, default=3)
    parser.add_argument("--f-allow-sinusoidal", dest="f_allow_sinusoidal", action="store_true", help="Allow sinusoidal forcing samples in the mixed prior")
    parser.add_argument("--f-no-sinusoidal", dest="f_allow_sinusoidal", action="store_false", help="Disable sinusoidal forcing samples in the mixed prior")
    parser.set_defaults(f_allow_sinusoidal=True)
    parser.add_argument("--disable-norm-targeting", action="store_true")
    parser.add_argument("--target-u0-norm-min", type=float, default=0.4)
    parser.add_argument("--target-u0-norm-max", type=float, default=1.0)
    parser.add_argument("--target-f-norm-min", type=float, default=0.1)
    parser.add_argument("--target-f-norm-max", type=float, default=0.5)
    parser.add_argument("--cfl-adv", type=float, default=0.45)
    parser.add_argument("--chunk-size", type=int, default=256, help="Trajectory solver chunk size")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for sampling and solves, e.g. cpu or cuda:0")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")

    parser.add_argument(
        "--dataset-path",
        type=str,
        default="grad_flow_l2/ns2d_per/datasets/navier_stokes2d_vorticity_periodic_lr64_nu0p001_nx64_ny64_steps10.pt",
        help="Path to output dataset file (.pt)",
    )
    parser.add_argument(
        "--settings-path",
        type=str,
        default=None,
        help="Optional path to write a JSON file containing the full generation settings",
    )

    parser.add_argument("--plot-samples", action="store_true", help="Generate sample-grid visualization")
    parser.add_argument("--plot-split", type=str, default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--n-plot-samples", type=int, default=20, help="Number of rows (data instances) to plot")
    parser.add_argument("--snapshot-times", type=str, default="0,1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--plot-dir", type=str, default="grad_flow_l2/ns2d_per/datasets/plots")
    parser.add_argument("--plot-prefix", type=str, default="ns2d_periodic_samples")
    return parser.parse_args()


def _print_split_stats(splits: Dict[str, Dict[str, torch.Tensor]]) -> None:
    print("Dataset meta:", splits.get("meta", {}))
    n_x = int(splits["meta"]["n_x"])
    n_y = int(splits["meta"]["n_y"])
    area = 1.0 / float(n_x * n_y)
    for split_name in ("train", "val", "test"):
        split = splits[split_name]
        f = split["f"]
        u0 = split["u0"]
        u_traj = split["u_traj"]
        if u0.shape[0] == 0:
            print(f"{split_name}: empty split")
            continue
        u0_l2 = torch.sqrt(area * torch.sum(u0 * u0, dim=(1, 2)))
        f_l2 = torch.sqrt(area * torch.sum(f * f, dim=(1, 2)))
        print(
            f"{split_name}: "
            f"f={tuple(f.shape)}, "
            f"u0={tuple(u0.shape)}, "
            f"u_traj={tuple(u_traj.shape)}, "
            f"f_l2_mean={f_l2.mean().item():.4f}, "
            f"u0_l2_mean={u0_l2.mean().item():.4f}, "
            f"f_abs_max={f.abs().max().item():.4f}, "
            f"u0_abs_max={u0.abs().max().item():.4f}"
        )


def _parse_snapshot_times(value: str) -> list[float]:
    out = []
    for token in value.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        out.append(float(stripped))
    if len(out) == 0:
        raise ValueError("snapshot_times must include at least one time value")
    return out


def plot_sample_rows(
    split: Dict[str, torch.Tensor],
    split_name: str,
    out_path: str,
    t_final: float,
    snapshot_times: Sequence[float],
    n_plot_samples: int = 6,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plotting because matplotlib is unavailable: {exc}")
        return

    u_traj = split["u_traj"]  # (n_samples, K+1, n_x, n_y)
    f = split["f"]            # (n_samples, n_x, n_y)
    total = int(u_traj.shape[0])
    if total == 0:
        print(f"Skipping plotting for {split_name}: empty split")
        return

    n_plot = min(max(1, int(n_plot_samples)), total)
    sample_indices = torch.linspace(0, total - 1, n_plot).long().tolist()

    n_steps = int(u_traj.shape[1] - 1)
    n_cols = 1 + len(snapshot_times)  # forcing + snapshots

    fig, axes = plt.subplots(
        n_plot,
        n_cols,
        figsize=(3.1 * n_cols, 2.6 * n_plot),
        squeeze=False,
        constrained_layout=True,
    )

    for row, sample_idx in enumerate(sample_indices):
        f_i = f[sample_idx]
        traj_i = u_traj[sample_idx]
        scale = max(float(torch.max(torch.abs(traj_i)).item()), float(torch.max(torch.abs(f_i)).item()), 1e-8)

        ax_force = axes[row, 0]
        ax_force.imshow(
            f_i.cpu().numpy(),
            origin="lower",
            cmap="coolwarm",
            vmin=-scale,
            vmax=scale,
            extent=[0.0, 1.0, 0.0, 1.0],
            aspect="auto",
        )
        ax_force.set_title("forcing")
        ax_force.set_xticks([])
        ax_force.set_yticks([])
        ax_force.set_ylabel(f"{split_name} #{sample_idx}")

        im_last = ax_force.images[-1]
        for col, t_snap in enumerate(snapshot_times, start=1):
            frac = 0.0 if t_final <= 0 else float(t_snap) / float(t_final)
            k = int(round(frac * n_steps))
            k = max(0, min(n_steps, k))
            ax = axes[row, col]
            ax.imshow(
                traj_i[k].cpu().numpy(),
                origin="lower",
                cmap="coolwarm",
                vmin=-scale,
                vmax=scale,
                extent=[0.0, 1.0, 0.0, 1.0],
                aspect="auto",
            )
            ax.set_title(f"t={t_snap:.1f}")
            ax.set_xticks([])
            ax.set_yticks([])
            im_last = ax.images[-1]

        cbar = fig.colorbar(im_last, ax=axes[row, :], fraction=0.015, pad=0.01)
        cbar.ax.set_ylabel("value", rotation=90)

    folder = os.path.dirname(out_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig.suptitle("2D periodic Navier-Stokes samples: forcing + vorticity snapshots", fontsize=13)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved sample-grid plot: {out_path}")


def main(args: argparse.Namespace) -> None:
    if args.n_x < 2 or args.n_y < 2:
        raise ValueError("--n-x and --n-y must both be >= 2")
    if args.n_steps < 1:
        raise ValueError("--n-steps must be >= 1")
    if args.t_final <= 0:
        raise ValueError("--t-final must be > 0")
    if args.nu <= 0:
        raise ValueError("--nu must be > 0")

    splits = generate_navier_stokes2d_periodic_dataset_splits(
        n_x=args.n_x,
        n_y=args.n_y,
        n_steps=args.n_steps,
        t_final=args.t_final,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        nu=args.nu,
        seed=args.seed,
        solver_n_x=args.solver_n_x,
        solver_n_y=args.solver_n_y,
        solver_dt=args.solver_dt,
        record_dt=args.record_dt,
        u0_spectrum_scale=args.u0_spectrum_scale,
        u0_spectrum_shift=args.u0_spectrum_shift,
        u0_spectrum_power=args.u0_spectrum_power,
        u0_rescale=args.u0_rescale,
        forcing_mode=args.forcing_mode,
        f_amplitude=args.f_amplitude,
        f_sinusoidal_amplitude_min=args.f_sinusoidal_amplitude_min,
        f_sinusoidal_amplitude_max=args.f_sinusoidal_amplitude_max,
        f_grf_prob=args.f_grf_prob,
        f_matern_prob=args.f_matern_prob,
        f_length_scale_min=args.f_length_scale_min,
        f_length_scale_max=args.f_length_scale_max,
        f_max_modes=args.f_max_modes,
        f_allow_sinusoidal=args.f_allow_sinusoidal,
        norm_targeting=False,
        target_u0_norm_range=(args.target_u0_norm_min, args.target_u0_norm_max),
        target_f_norm_range=(args.target_f_norm_min, args.target_f_norm_max),
        cfl_adv=args.cfl_adv,
        chunk_size=args.chunk_size,
        show_progress=(not args.no_progress),
        dtype=torch.float32,
        device=args.device,
    )

    out_dir = os.path.dirname(args.dataset_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_dataset_splits(splits, args.dataset_path)
    print(f"Saved periodic Navier-Stokes dataset splits to: {args.dataset_path}")
    _print_split_stats(splits)

    if args.settings_path:
        settings_dir = os.path.dirname(args.settings_path)
        if settings_dir:
            os.makedirs(settings_dir, exist_ok=True)
        settings_payload = {
            "cli_args": vars(args),
            "meta": splits["meta"],
        }
        with open(args.settings_path, "w", encoding="utf-8") as f:
            json.dump(settings_payload, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"Saved data-generation settings to: {args.settings_path}")

    if args.plot_samples:
        snapshot_times = _parse_snapshot_times(args.snapshot_times)
        split_names = ("train", "val", "test") if args.plot_split == "all" else (args.plot_split,)
        for split_name in split_names:
            out_path = os.path.join(args.plot_dir, f"{args.plot_prefix}_{split_name}.png")
            plot_sample_rows(
                split=splits[split_name],
                split_name=split_name,
                out_path=out_path,
                t_final=float(args.t_final),
                snapshot_times=snapshot_times,
                n_plot_samples=int(args.n_plot_samples),
            )


if __name__ == "__main__":
    main(parse_args())
