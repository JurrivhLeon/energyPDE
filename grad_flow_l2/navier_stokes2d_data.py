"""
Data generation for 2D viscous incompressible Navier-Stokes in vorticity form:

    omega_t + v · grad(omega) = nu * Delta omega + g,
    -Delta psi = omega,
    v = (d_y psi, -d_x psi),

on (0,1)^2 with homogeneous Dirichlet boundary conditions.

Dataset format mirrors existing grad_flow_l2 generators:
    split["f"]      : (n_samples, n_x+2, n_y+2)  (forcing sampled on full grid)
    split["u0"]     : (n_samples, n_x, n_y)
    split["u_traj"] : (n_samples, n_steps+1, n_x, n_y)
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from .heat_data import save_dataset_splits
    from .navier_stokes2d_solver import solve_navier_stokes_vorticity_trajectory, to_interior_field_2d
except ImportError:
    from grad_flow_l2.heat_data import save_dataset_splits
    from grad_flow_l2.navier_stokes2d_solver import solve_navier_stokes_vorticity_trajectory, to_interior_field_2d


DATASET_VERSION = 1


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


class NavierStokes2DTrajectoryTensorDataset(Dataset):
    """
    Dataset wrapper around precomputed tensors:
      f: (n_samples, n_x+2, n_y+2) or (n_samples, n_x, n_y)
      u0: (n_samples, n_x, n_y)
      u_traj: (n_samples, K+1, n_x, n_y)
    """

    def __init__(self, f_data: torch.Tensor, u0_data: torch.Tensor, u_traj_data: torch.Tensor):
        if f_data.dim() != 3:
            raise ValueError("f_data must have shape (n_samples,n_x,n_y) or (n_samples,n_x+2,n_y+2)")
        if u0_data.dim() != 3:
            raise ValueError("u0_data must have shape (n_samples,n_x,n_y)")
        if u_traj_data.dim() != 4:
            raise ValueError("u_traj_data must have shape (n_samples,K+1,n_x,n_y)")

        n_samples = int(u_traj_data.shape[0])
        n_x = int(u_traj_data.shape[2])
        n_y = int(u_traj_data.shape[3])
        valid_f_shape = f_data.shape[1:] in ((n_x, n_y), (n_x + 2, n_y + 2))
        if (
            int(f_data.shape[0]) != n_samples
            or int(u0_data.shape[0]) != n_samples
            or tuple(u0_data.shape[1:]) != (n_x, n_y)
            or (not valid_f_shape)
        ):
            raise ValueError("inconsistent tensor shapes for NavierStokes2D trajectory dataset")

        self.f_data = f_data
        self.u0_data = u0_data
        self.u_traj_data = u_traj_data

    def __len__(self) -> int:
        return int(self.f_data.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "f": self.f_data[idx],
            "u0": self.u0_data[idx],
            "u_traj": self.u_traj_data[idx],
        }


class NavierStokes2DStepDataset(Dataset):
    """
    Flattened one-step dataset from trajectories.
    Returns tuple: (u_k, u_{k+1}, f).
    """

    def __init__(self, f_data: torch.Tensor, u_traj_data: torch.Tensor):
        if f_data.dim() != 3:
            raise ValueError("f_data must have shape (n_samples,n_x,n_y) or (n_samples,n_x+2,n_y+2)")
        if u_traj_data.dim() != 4:
            raise ValueError("u_traj_data must have shape (n_samples,K+1,n_x,n_y)")

        n_samples = int(u_traj_data.shape[0])
        n_x = int(u_traj_data.shape[2])
        n_y = int(u_traj_data.shape[3])
        valid_f_shape = f_data.shape[1:] in ((n_x, n_y), (n_x + 2, n_y + 2))
        if int(f_data.shape[0]) != n_samples or not valid_f_shape:
            raise ValueError("f_data and u_traj_data shapes are inconsistent")

        self.f_data = f_data
        self.u_traj_data = u_traj_data
        self.n_samples = n_samples
        self.n_steps = int(u_traj_data.shape[1] - 1)

    def __len__(self) -> int:
        return self.n_samples * self.n_steps

    def __getitem__(self, idx: int):
        i = idx // self.n_steps
        k = idx % self.n_steps
        return self.u_traj_data[i, k], self.u_traj_data[i, k + 1], self.f_data[i]


def build_navier_stokes2d_step_dataset(split_or_dataset) -> NavierStokes2DStepDataset:
    if isinstance(split_or_dataset, NavierStokes2DTrajectoryTensorDataset):
        f_data = split_or_dataset.f_data
        u_traj_data = split_or_dataset.u_traj_data
    elif isinstance(split_or_dataset, dict):
        f_data = split_or_dataset["f"]
        u_traj_data = split_or_dataset["u_traj"]
    else:
        raise TypeError("Expected split dict or NavierStokes2DTrajectoryTensorDataset")
    return NavierStokes2DStepDataset(f_data=f_data, u_traj_data=u_traj_data)


def build_navier_stokes2d_trajectory_dataset_from_split(
    split: Dict[str, torch.Tensor],
) -> NavierStokes2DTrajectoryTensorDataset:
    return NavierStokes2DTrajectoryTensorDataset(
        f_data=split["f"],
        u0_data=split["u0"],
        u_traj_data=split["u_traj"],
    )


def sample_grf_2d(
    n_x: int,
    n_y: int,
    n_samples: int = 1,
    length_scale: float = 0.2,
    variance: float = 1.0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Sample periodic-like 2D Gaussian random fields via spectral synthesis.
    """
    k_x = torch.fft.fftfreq(n_x, d=1.0 / n_x, device=device)
    k_y = torch.fft.fftfreq(n_y, d=1.0 / n_y, device=device)
    k2 = k_x.unsqueeze(1) ** 2 + k_y.unsqueeze(0) ** 2
    power = variance * (2.0 * np.pi * length_scale * length_scale)
    power = power * torch.exp(-2.0 * (np.pi * length_scale) ** 2 * k2)

    real = torch.randn(n_samples, n_x, n_y, device=device)
    imag = torch.randn(n_samples, n_x, n_y, device=device)
    spectrum = (real + 1j * imag) * torch.sqrt(power).unsqueeze(0)
    samples = torch.fft.ifft2(spectrum).real * ((n_x * n_y) ** 0.5)
    return samples


def sample_matern15_spectral_2d(
    n_x: int,
    n_y: int,
    n_samples: int = 1,
    length_scale: float = 0.35,
    variance: float = 1.0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Sample 2D GP fields with approximate Matérn kernel (nu=1.5) in spectral form.
    """
    if length_scale <= 0:
        raise ValueError("length_scale must be > 0")

    nu = 1.5
    dim = 2.0
    kappa2 = (2.0 * nu) / (float(length_scale) * float(length_scale))

    k_x = torch.fft.fftfreq(n_x, d=1.0 / n_x, device=device)
    k_y = torch.fft.fftfreq(n_y, d=1.0 / n_y, device=device)
    k2 = k_x.unsqueeze(1) ** 2 + k_y.unsqueeze(0) ** 2
    radial = (2.0 * np.pi) ** 2 * k2
    power = variance * torch.pow(kappa2 + radial, -(nu + dim / 2.0))

    real = torch.randn(n_samples, n_x, n_y, device=device)
    imag = torch.randn(n_samples, n_x, n_y, device=device)
    spectrum = (real + 1j * imag) * torch.sqrt(power).unsqueeze(0)
    samples = torch.fft.ifft2(spectrum).real * ((n_x * n_y) ** 0.5)
    return samples


def sample_coarse_matern15_2d(
    n_x: int,
    n_y: int,
    n_samples: int = 1,
    coarse_factor: int = 4,
    length_scale_range: tuple[float, float] = (0.25, 0.80),
    amplitude: float = 1.0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Coarse Matérn(1.5) samples: draw on coarse grid, then upsample to target size.
    """
    if coarse_factor < 2:
        raise ValueError("coarse_factor must be >= 2")
    if length_scale_range[0] <= 0 or length_scale_range[1] <= 0:
        raise ValueError("length_scale_range values must be > 0")
    if length_scale_range[0] >= length_scale_range[1]:
        raise ValueError("length_scale_range must satisfy min < max")

    n_x_coarse = max(4, int(np.ceil(n_x / coarse_factor)))
    n_y_coarse = max(4, int(np.ceil(n_y / coarse_factor)))
    out = []
    for _ in range(n_samples):
        ls = float(np.random.uniform(length_scale_range[0], length_scale_range[1]))
        coarse = sample_matern15_spectral_2d(
            n_x=n_x_coarse,
            n_y=n_y_coarse,
            n_samples=1,
            length_scale=ls,
            variance=1.0,
            device=device,
        )  # (1, n_x_coarse, n_y_coarse)
        up = F.interpolate(
            coarse.unsqueeze(1),
            size=(n_x, n_y),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)
        max_val = up.abs().amax(dim=(1, 2), keepdim=True) + 1e-8
        up = float(amplitude) * up / max_val
        out.append(up)
    return torch.cat(out, dim=0)


def sample_sinusoidal_2d(
    n_x: int,
    n_y: int,
    n_samples: int = 1,
    max_modes: int = 6,
    amplitude: float = 1.0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Random sinusoidal fields on interior grid.
    """
    x = torch.linspace(0.0, 1.0, n_x + 2, device=device)[1:-1]
    y = torch.linspace(0.0, 1.0, n_y + 2, device=device)[1:-1]
    xx, yy = torch.meshgrid(x, y, indexing="ij")

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

    max_val = fields.abs().amax(dim=(1, 2), keepdim=True) + 1e-8
    fields = float(amplitude) * fields / max_val
    return fields


def sample_field_mixed_2d(
    n_x: int,
    n_y: int,
    n_samples: int = 1,
    amplitude: float = 1.0,
    length_scale_range: tuple[float, float] = (0.05, 0.30),
    max_modes: int = 6,
    grf_prob: float = 0.7,
    matern_coarse_prob: float = 0.0,
    matern_coarse_factor: int = 4,
    matern_length_scale_range: tuple[float, float] = (0.25, 0.80),
    show_progress: bool = False,
    progress_desc: str = "sample fields",
    device: str = "cpu",
) -> torch.Tensor:
    """
    Mixed 2D sampler: each sample is GRF, coarse Matérn(1.5), or sinusoidal.
    """
    if grf_prob < 0 or matern_coarse_prob < 0:
        raise ValueError("probabilities must be >= 0")
    if grf_prob + matern_coarse_prob > 1.0:
        raise ValueError("grf_prob + matern_coarse_prob must be <= 1")

    out = []
    iterator = _iter_with_progress(
        range(n_samples),
        total=n_samples,
        desc=progress_desc,
        enabled=show_progress,
    )
    for _ in iterator:
        r = float(np.random.rand())
        if r < grf_prob:
            length_scale = float(np.random.uniform(*length_scale_range))
            field = sample_grf_2d(
                n_x=n_x,
                n_y=n_y,
                n_samples=1,
                length_scale=length_scale,
                variance=1.0,
                device=device,
            )
        elif r < grf_prob + matern_coarse_prob:
            field = sample_coarse_matern15_2d(
                n_x=n_x,
                n_y=n_y,
                n_samples=1,
                coarse_factor=matern_coarse_factor,
                length_scale_range=matern_length_scale_range,
                amplitude=1.0,
                device=device,
            )
        else:
            mm = int(np.random.randint(3, max_modes + 1))
            field = sample_sinusoidal_2d(
                n_x=n_x,
                n_y=n_y,
                n_samples=1,
                max_modes=mm,
                amplitude=1.0,
                device=device,
            )

        amp = float(amplitude * (10.0 ** np.random.uniform(-0.35, 0.35)))
        field = amp * field / (field.abs().amax(dim=(1, 2), keepdim=True) + 1e-8)
        out.append(field)
    return torch.cat(out, dim=0)


def _dirichlet_taper_2d(
    n_x: int,
    n_y: int,
    power: float = 1.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if power <= 0:
        raise ValueError("power must be > 0")
    x = torch.linspace(0.0, 1.0, n_x + 2, device=device, dtype=dtype)[1:-1]
    y = torch.linspace(0.0, 1.0, n_y + 2, device=device, dtype=dtype)[1:-1]
    tx = (4.0 * x * (1.0 - x)) ** power
    ty = (4.0 * y * (1.0 - y)) ** power
    taper = torch.outer(tx, ty)
    return taper / (torch.max(taper) + 1e-12)


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


def generate_navier_stokes2d_dataset_splits(
    n_x: int,
    n_y: int,
    n_steps: int,
    t_final: float,
    n_train: int,
    n_val: int,
    n_test: int,
    nu: float = 0.01,
    seed: int = 42,
    u0_amplitude: float = 2.5,
    u0_grf_prob: float = 0.55,
    u0_matern_coarse_prob: float = 0.35,
    u0_matern_coarse_factor: int = 4,
    u0_matern_length_scale_min: float = 0.25,
    u0_matern_length_scale_max: float = 0.80,
    forcing_mode: str = "zero",
    f_amplitude: float = 0.0,
    f_grf_prob: float = 0.7,
    f_length_scale_min: float = 0.05,
    f_length_scale_max: float = 0.30,
    f_max_modes: int = 6,
    taper_power: float = 1.0,
    norm_targeting: bool = True,
    target_u0_norm_range: tuple[float, float] = (0.5, 1.5),
    target_f_norm_range: tuple[float, float] = (0.1, 0.8),
    cfl_adv: float = 0.45,
    chunk_size: int = 256,
    show_progress: bool = False,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Dict[str, torch.Tensor]]:
    if nu <= 0:
        raise ValueError("nu must be > 0")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if t_final <= 0:
        raise ValueError("t_final must be > 0")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    rng_state_torch = torch.random.get_rng_state()
    rng_state_numpy = np.random.get_state()
    torch.manual_seed(seed)
    np.random.seed(seed)

    total = int(n_train + n_val + n_test)
    h_x = 1.0 / float(n_x + 1)
    h_y = 1.0 / float(n_y + 1)
    area = h_x * h_y

    u0 = sample_field_mixed_2d(
        n_x=n_x,
        n_y=n_y,
        n_samples=total,
        amplitude=u0_amplitude,
        grf_prob=u0_grf_prob,
        matern_coarse_prob=u0_matern_coarse_prob,
        matern_coarse_factor=u0_matern_coarse_factor,
        matern_length_scale_range=(u0_matern_length_scale_min, u0_matern_length_scale_max),
        show_progress=show_progress,
        progress_desc="sample u0",
        device="cpu",
    ).to(dtype=dtype)
    taper = _dirichlet_taper_2d(n_x=n_x, n_y=n_y, power=taper_power, device="cpu", dtype=dtype)
    u0 = u0 * taper.unsqueeze(0)
    if norm_targeting:
        u0 = _rescale_batch_l2_2d(
            x=u0,
            area=area,
            norm_min=float(target_u0_norm_range[0]),
            norm_max=float(target_u0_norm_range[1]),
        )

    if forcing_mode not in ("zero", "mixed"):
        raise ValueError("forcing_mode must be one of {'zero', 'mixed'}")

    if forcing_mode == "zero":
        f = torch.zeros(total, n_x + 2, n_y + 2, dtype=dtype)
    else:
        f_int = sample_field_mixed_2d(
            n_x=n_x,
            n_y=n_y,
            n_samples=total,
            amplitude=f_amplitude,
            length_scale_range=(f_length_scale_min, f_length_scale_max),
            max_modes=f_max_modes,
            grf_prob=f_grf_prob,
            matern_coarse_prob=0.0,
            show_progress=show_progress,
            progress_desc="sample forcing",
            device="cpu",
        ).to(dtype=dtype)
        if norm_targeting:
            f_int = _rescale_batch_l2_2d(
                x=f_int,
                area=area,
                norm_min=float(target_f_norm_range[0]),
                norm_max=float(target_f_norm_range[1]),
            )
        f = torch.zeros(total, n_x + 2, n_y + 2, dtype=dtype)
        f[:, 1:-1, 1:-1] = f_int

    # Solve in chunks so large datasets provide visible progress and use memory more smoothly.
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
        u_traj_chunk = solve_navier_stokes_vorticity_trajectory(
            u0=u0[start:end],
            forcing=f[start:end],
            n_steps=n_steps,
            nu=nu,
            t_final=t_final,
            cfl_adv=cfl_adv,
        ).to(dtype=dtype)
        u_traj_chunks.append(u_traj_chunk)
    u_traj = torch.cat(u_traj_chunks, dim=0)

    all_data = {"f": f, "u0": u0, "u_traj": u_traj}
    train_end = int(n_train)
    val_end = int(n_train + n_val)

    splits: Dict[str, Dict[str, torch.Tensor]] = {
        "train": _slice_split(all_data, 0, train_end),
        "val": _slice_split(all_data, train_end, val_end),
        "test": _slice_split(all_data, val_end, total),
        "meta": {
            "dataset_version": DATASET_VERSION,
            "equation": "navier_stokes_2d_vorticity_dirichlet",
            "n_x": int(n_x),
            "n_y": int(n_y),
            "n_steps": int(n_steps),
            "t_final": float(t_final),
            "n_train": int(n_train),
            "n_val": int(n_val),
            "n_test": int(n_test),
            "seed": int(seed),
            "nu": float(nu),
            "forcing_mode": forcing_mode,
            "u0_grid_points": [int(n_x), int(n_y)],
            "f_grid_points": [int(f.shape[-2]), int(f.shape[-1])],
            "u0_amplitude": float(u0_amplitude),
            "u0_grf_prob": float(u0_grf_prob),
            "u0_matern_coarse_prob": float(u0_matern_coarse_prob),
            "u0_matern_coarse_factor": int(u0_matern_coarse_factor),
            "u0_matern_length_scale_min": float(u0_matern_length_scale_min),
            "u0_matern_length_scale_max": float(u0_matern_length_scale_max),
            "f_amplitude": float(f_amplitude),
            "f_grf_prob": float(f_grf_prob),
            "f_length_scale_min": float(f_length_scale_min),
            "f_length_scale_max": float(f_length_scale_max),
            "f_max_modes": int(f_max_modes),
            "taper_power": float(taper_power),
            "norm_targeting": bool(norm_targeting),
            "target_u0_norm_range": [float(target_u0_norm_range[0]), float(target_u0_norm_range[1])],
            "target_f_norm_range": [float(target_f_norm_range[0]), float(target_f_norm_range[1])],
            "cfl_adv": float(cfl_adv),
            "chunk_size": int(chunk_size),
        },
    }

    torch.random.set_rng_state(rng_state_torch)
    np.random.set_state(rng_state_numpy)
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cached 2D Navier-Stokes vorticity dataset splits")
    parser.add_argument("--n-x", type=int, default=32, help="Number of interior x-grid points")
    parser.add_argument("--n-y", type=int, default=32, help="Number of interior y-grid points")
    parser.add_argument("--n-steps", type=int, default=10, help="Number of macro time steps on [0,t_final]")
    parser.add_argument("--t-final", type=float, default=1.0, help="Final time horizon")
    parser.add_argument("--nu", type=float, default=0.01, help="Viscosity coefficient")

    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--u0-amplitude", type=float, default=2.5)
    parser.add_argument("--u0-grf-prob", type=float, default=0.55)
    parser.add_argument("--u0-matern-coarse-prob", type=float, default=0.35)
    parser.add_argument("--u0-matern-coarse-factor", type=int, default=4)
    parser.add_argument("--u0-matern-ls-min", type=float, default=0.25)
    parser.add_argument("--u0-matern-ls-max", type=float, default=0.80)
    parser.add_argument("--forcing-mode", type=str, default="zero", choices=["zero", "mixed"])
    parser.add_argument("--f-amplitude", type=float, default=0.5)
    parser.add_argument("--f-grf-prob", type=float, default=0.7)
    parser.add_argument("--f-length-scale-min", type=float, default=0.05)
    parser.add_argument("--f-length-scale-max", type=float, default=0.30)
    parser.add_argument("--f-max-modes", type=int, default=6)
    parser.add_argument("--taper-power", type=float, default=1.0)
    parser.add_argument("--disable-norm-targeting", action="store_true")
    parser.add_argument("--target-u0-norm-min", type=float, default=0.5)
    parser.add_argument("--target-u0-norm-max", type=float, default=1.5)
    parser.add_argument("--target-f-norm-min", type=float, default=0.1)
    parser.add_argument("--target-f-norm-max", type=float, default=0.8)
    parser.add_argument("--cfl-adv", type=float, default=0.45)
    parser.add_argument("--chunk-size", type=int, default=256, help="Trajectory solver chunk size")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")

    parser.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/navier_stokes2d_vorticity_l2_nu0p01_nx32_ny32_steps10.pt",
        help="Path to output dataset file (.pt)",
    )

    parser.add_argument("--plot-samples", action="store_true", help="Generate sample-grid visualization")
    parser.add_argument("--plot-split", type=str, default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--n-plot-samples", type=int, default=10, help="Number of rows (data instances) to plot")
    parser.add_argument("--snapshot-times", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--plot-dir", type=str, default="grad_flow_l2/outputs/data_samples")
    parser.add_argument("--plot-prefix", type=str, default="ns2d_samples")
    return parser.parse_args()


def _print_split_stats(splits: Dict[str, Dict[str, torch.Tensor]]) -> None:
    print("Dataset meta:", splits.get("meta", {}))
    n_x = int(splits["meta"]["n_x"])
    n_y = int(splits["meta"]["n_y"])
    area = 1.0 / float((n_x + 1) * (n_y + 1))
    for split_name in ("train", "val", "test"):
        split = splits[split_name]
        f = split["f"]
        u0 = split["u0"]
        u_traj = split["u_traj"]
        if u0.shape[0] == 0:
            print(f"{split_name}: empty split")
            continue
        u0_l2 = torch.sqrt(area * torch.sum(u0 * u0, dim=(1, 2)))
        f_int = to_interior_field_2d(f, n_x=n_x, n_y=n_y)
        f_l2 = torch.sqrt(area * torch.sum(f_int * f_int, dim=(1, 2)))
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
    f = split["f"]            # (n_samples, n_x+2, n_y+2) or (n_samples, n_x, n_y)
    total = int(u_traj.shape[0])
    if total == 0:
        print(f"Skipping plotting for {split_name}: empty split")
        return

    n_plot = min(max(1, int(n_plot_samples)), total)
    sample_indices = torch.linspace(0, total - 1, n_plot).long().tolist()

    n_steps = int(u_traj.shape[1] - 1)
    n_x = int(u_traj.shape[-2])
    n_y = int(u_traj.shape[-1])
    n_cols = 1 + len(snapshot_times)  # forcing + snapshots

    fig, axes = plt.subplots(
        n_plot,
        n_cols,
        figsize=(3.1 * n_cols, 2.6 * n_plot),
        squeeze=False,
        constrained_layout=True,
    )

    for row, sample_idx in enumerate(sample_indices):
        f_int = to_interior_field_2d(f[sample_idx], n_x=n_x, n_y=n_y).squeeze(0)
        traj_i = u_traj[sample_idx]
        scale = max(float(torch.max(torch.abs(traj_i)).item()), float(torch.max(torch.abs(f_int)).item()), 1e-8)

        ax_force = axes[row, 0]
        ax_force.imshow(
            f_int.cpu().numpy(),
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
    fig.suptitle("2D Navier-Stokes Samples: forcing + vorticity snapshots", fontsize=13)
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

    splits = generate_navier_stokes2d_dataset_splits(
        n_x=args.n_x,
        n_y=args.n_y,
        n_steps=args.n_steps,
        t_final=args.t_final,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        nu=args.nu,
        seed=args.seed,
        u0_amplitude=args.u0_amplitude,
        u0_grf_prob=args.u0_grf_prob,
        u0_matern_coarse_prob=args.u0_matern_coarse_prob,
        u0_matern_coarse_factor=args.u0_matern_coarse_factor,
        u0_matern_length_scale_min=args.u0_matern_ls_min,
        u0_matern_length_scale_max=args.u0_matern_ls_max,
        forcing_mode=args.forcing_mode,
        f_amplitude=args.f_amplitude,
        f_grf_prob=args.f_grf_prob,
        f_length_scale_min=args.f_length_scale_min,
        f_length_scale_max=args.f_length_scale_max,
        f_max_modes=args.f_max_modes,
        taper_power=args.taper_power,
        norm_targeting=(not args.disable_norm_targeting),
        target_u0_norm_range=(args.target_u0_norm_min, args.target_u0_norm_max),
        target_f_norm_range=(args.target_f_norm_min, args.target_f_norm_max),
        cfl_adv=args.cfl_adv,
        chunk_size=args.chunk_size,
        show_progress=(not args.no_progress),
        dtype=torch.float32,
    )

    out_dir = os.path.dirname(args.dataset_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_dataset_splits(splits, args.dataset_path)
    print(f"Saved Navier-Stokes dataset splits to: {args.dataset_path}")
    _print_split_stats(splits)

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
