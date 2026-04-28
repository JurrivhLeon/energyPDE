"""
Data generation for 1D non-homogeneous heat equation:
    u_t = kappa * u_xx + f(x),  x in [0,1], t in [0,1]
with homogeneous Dirichlet boundary conditions.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Optional

import torch
import numpy as np
from torch.utils.data import Dataset

try:
    from .utils import (
        build_laplacian_1d_dirichlet,
        check_dirichlet_1d,
        pad_dirichlet_1d,
        prepare_implicit_matrix,
        solve_heat_trajectory,
    )
except ImportError:
    from utils import (
        build_laplacian_1d_dirichlet,
        check_dirichlet_1d,
        pad_dirichlet_1d,
        prepare_implicit_matrix,
        solve_heat_trajectory,
    )


DATASET_VERSION = 5


def sample_grf_1d(
    n_points: int,
    n_samples: int = 1,
    length_scale: float = 0.2,
    variance: float = 1.0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Sample 1D Gaussian random fields with a spectral construction.

    Returns shape (n_samples, n_points).
    """
    freqs = torch.fft.fftfreq(n_points, d=1.0 / n_points, device=device)
    power = variance * (2.0 * np.pi * length_scale * length_scale) ** 0.5
    power = power * torch.exp(-2.0 * (np.pi * length_scale * freqs) ** 2)

    real = torch.randn(n_samples, n_points, device=device)
    imag = torch.randn(n_samples, n_points, device=device)
    spectrum = (real + 1j * imag) * power.sqrt().unsqueeze(0)

    samples = torch.fft.ifft(spectrum).real * (n_points ** 0.5)
    return samples


def sample_sinusoidal_1d(
    n_points: int,
    n_samples: int = 1,
    max_modes: int = 6,
    amplitude: float = 1.0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Random Fourier-like sinusoidal fields on interior grid.
    """
    x = torch.linspace(0.0, 1.0, n_points + 2, device=device)[1:-1]
    fields = torch.zeros(n_samples, n_points, device=device)

    n_terms = max(2, max_modes)
    for _ in range(n_terms):
        k = torch.randint(1, max_modes + 1, (n_samples, 1), device=device).float()
        phase = 2 * np.pi * torch.rand(n_samples, 1, device=device)
        coeff = torch.randn(n_samples, 1, device=device) / k
        fields = fields + coeff * torch.sin(2.0 * np.pi * k * x.unsqueeze(0) + phase)

    max_val = fields.abs().amax(dim=1, keepdim=True) + 1e-8
    fields = amplitude * fields / max_val
    return fields


def sample_field_mixed(
    n_points: int,
    n_samples: int = 1,
    amplitude: float = 1.0,
    length_scale_range: tuple[float, float] = (0.06, 0.35),
    max_modes: int = 6,
    grf_prob: float = 0.7,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Mixed sampler: each sample is either GRF or sinusoidal.
    """
    out = []
    for _ in range(n_samples):
        if np.random.rand() < grf_prob:
            ls = float(np.random.uniform(*length_scale_range))
            field = sample_grf_1d(
                n_points=n_points,
                n_samples=1,
                length_scale=ls,
                variance=1.0,
                device=device,
            )
        else:
            mm = int(np.random.randint(3, max_modes + 1))
            field = sample_sinusoidal_1d(
                n_points=n_points,
                n_samples=1,
                max_modes=mm,
                amplitude=1.0,
                device=device,
            )

        # Per-sample amplitude variation for diversity.
        amp = float(amplitude * 10 ** np.random.uniform(-0.35, 0.35))
        field = amp * field / (field.abs().amax(dim=1, keepdim=True) + 1e-8)
        out.append(field)

    return torch.cat(out, dim=0)


def generate_heat_trajectory_batch(
    u0: torch.Tensor,
    f: torch.Tensor,
    dt: float,
    n_steps: int,
    kappa: float = 1.0,
    solver_cache: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Generate trajectory batch with implicit Euler solver.

    Args:
        u0: (batch, n_x)
        f: (batch, n_x)

    Returns:
        traj: (batch, n_steps+1, n_x)
    """
    if u0.dim() == 1:
        u0 = u0.unsqueeze(0)
    if f.dim() == 1:
        f = f.unsqueeze(0)

    n_x = u0.shape[-1]
    if solver_cache is None:
        h = 1.0 / (n_x + 1)
        solver_cache = prepare_implicit_matrix(
            n_x=n_x,
            dt=dt,
            h=h,
            kappa=kappa,
            device=u0.device,
            dtype=u0.dtype,
        )

    return solve_heat_trajectory(
        u0=u0,
        f=f[:, 1:-1] if f.shape[-1] == n_x + 2 else f,
        dt=dt,
        n_steps=n_steps,
        kappa=kappa,
        matrix_cache=solver_cache,
    )


def _rescale_batch_to_l2_norm_range(
    x: torch.Tensor,
    h: float,
    norm_min: float,
    norm_max: float,
) -> torch.Tensor:
    """
    Rescale each sample in a batch to a random L2 norm in [norm_min, norm_max].
    """
    if x.dim() != 2:
        raise ValueError("x must have shape (batch, n_x)")
    norms = torch.sqrt(h * torch.sum(x * x, dim=-1))  # (batch,)
    targets = torch.empty_like(norms).uniform_(norm_min, norm_max)
    scale = targets / (norms + 1e-8)
    return x * scale.unsqueeze(-1)


def _rescale_forcing_by_target_steady_state_norm(
    f: torch.Tensor,
    n_x: int,
    h: float,
    kappa: float,
    target_uss_norm_min: float,
    target_uss_norm_max: float,
) -> torch.Tensor:
    """
    Rescale forcing so the steady-state solve (-D2)u_ss=f has ||u_ss||_L2
    in a target range. This avoids nearly-zero trajectories.
    """
    if f.dim() != 2:
        raise ValueError("f must have shape (batch, n_x) or (batch, n_x+2)")
    if f.shape[1] == n_x:
        f_interior = f
    elif f.shape[1] == n_x + 2:
        f_interior = f[:, 1:-1]
    else:
        raise ValueError("f width must be n_x or n_x+2")

    d2 = build_laplacian_1d_dirichlet(n_x=n_x, h=h, device=f.device, dtype=f.dtype)
    m = -float(kappa) * d2  # SPD for Dirichlet Laplacian
    u_ss = torch.linalg.solve(m, f_interior.unsqueeze(-1)).squeeze(-1)
    uss_norm = torch.sqrt(h * torch.sum(u_ss * u_ss, dim=-1))
    targets = torch.empty_like(uss_norm).uniform_(target_uss_norm_min, target_uss_norm_max)
    scale = targets / (uss_norm + 1e-8)
    return f * scale.unsqueeze(-1)


def _dirichlet_smooth_taper(
    n_x: int,
    power: float = 1.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Smooth taper on interior points that vanishes at boundaries in the full domain.
    """
    if power <= 0:
        raise ValueError("power must be > 0")
    x = torch.linspace(0.0, 1.0, n_x + 2, device=device, dtype=dtype)[1:-1]
    taper = (4 * x * (1.0 - x)) ** power
    # Normalize to keep center scale near 1.
    taper = taper / (torch.max(taper) + 1e-12)
    return taper


def _apply_u0_boundary_taper(
    u0: torch.Tensor,
    power: float = 1.0,
) -> torch.Tensor:
    """
    Apply smooth Dirichlet-compatible taper to interior u0 samples.
    """
    if u0.dim() != 2:
        raise ValueError("u0 must have shape (batch, n_x)")
    taper = _dirichlet_smooth_taper(
        n_x=u0.shape[-1],
        power=power,
        device=u0.device,
        dtype=u0.dtype,
    )
    return u0 * taper.unsqueeze(0)


class HeatTrajectoryDataset(Dataset):
    """
    Dataset returning samples:
        {"f": (n_x+2,), "u0": (n_x,), "u_traj": (K+1, n_x)}

    Note:
        States are represented on interior grid points only.
        Dirichlet boundary values are fixed as zero and imposed implicitly.
    """

    def __init__(
        self,
        n_x: int,
        n_steps: int,
        n_samples: int,
        kappa: float = 0.10,
        f_amplitude: float = 1.0,
        u0_amplitude: float = 1.0,
        enforce_u0_boundary_taper: bool = True,
        u0_taper_power: float = 1.0,
        norm_targeting: bool = True,
        target_u0_norm_range: tuple[float, float] = (0.6, 1.6),
        target_uss_norm_range: tuple[float, float] = (0.25, 1.0),
        pregenerate: bool = True,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.n_x = n_x
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.kappa = float(kappa)
        self.f_amplitude = f_amplitude
        self.u0_amplitude = u0_amplitude
        self.enforce_u0_boundary_taper = enforce_u0_boundary_taper
        self.u0_taper_power = u0_taper_power
        self.norm_targeting = norm_targeting
        self.target_u0_norm_range = target_u0_norm_range
        self.target_uss_norm_range = target_uss_norm_range
        self.pregenerate = pregenerate
        self.device = device
        self.dtype = dtype

        self.dt = 1.0 / float(n_steps)
        self.h = 1.0 / float(n_x + 1)

        self._solver_cache_cpu = prepare_implicit_matrix(
            n_x=n_x,
            dt=self.dt,
            h=self.h,
            kappa=self.kappa,
            device="cpu",
            dtype=dtype,
        )

        self.f_data: Optional[torch.Tensor] = None
        self.u0_data: Optional[torch.Tensor] = None
        self.u_traj_data: Optional[torch.Tensor] = None

        if pregenerate:
            self._pregenerate()

    def _pregenerate(self) -> None:
        f = sample_field_mixed(
            n_points=self.n_x + 2,
            n_samples=self.n_samples,
            amplitude=self.f_amplitude,
            device="cpu",
        ).to(self.dtype)
        u0 = sample_field_mixed(
            n_points=self.n_x,
            n_samples=self.n_samples,
            amplitude=self.u0_amplitude,
            device="cpu",
        ).to(self.dtype)
        if self.enforce_u0_boundary_taper:
            u0 = _apply_u0_boundary_taper(u0, power=self.u0_taper_power)

        if self.norm_targeting:
            f = _rescale_forcing_by_target_steady_state_norm(
                f=f,
                n_x=self.n_x,
                h=self.h,
                kappa=self.kappa,
                target_uss_norm_min=self.target_uss_norm_range[0],
                target_uss_norm_max=self.target_uss_norm_range[1],
            )
            u0 = _rescale_batch_to_l2_norm_range(
                x=u0,
                h=self.h,
                norm_min=self.target_u0_norm_range[0],
                norm_max=self.target_u0_norm_range[1],
            )

        u_traj = generate_heat_trajectory_batch(
            u0=u0,
            f=f,
            dt=self.dt,
            n_steps=self.n_steps,
            kappa=self.kappa,
            solver_cache=self._solver_cache_cpu,
        )

        self.f_data = f
        self.u0_data = u0
        self.u_traj_data = u_traj

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.pregenerate and self.f_data is not None:
            return {
                "f": self.f_data[idx],
                "u0": self.u0_data[idx],
                "u_traj": self.u_traj_data[idx],
            }

        # On-the-fly generation for a single sample.
        f = sample_field_mixed(
            n_points=self.n_x + 2,
            n_samples=1,
            amplitude=self.f_amplitude,
            device="cpu",
        )[0].to(self.dtype)
        u0 = sample_field_mixed(
            n_points=self.n_x,
            n_samples=1,
            amplitude=self.u0_amplitude,
            device="cpu",
        )[0].to(self.dtype)
        if self.enforce_u0_boundary_taper:
            u0 = _apply_u0_boundary_taper(u0.unsqueeze(0), power=self.u0_taper_power)[0]

        if self.norm_targeting:
            f = _rescale_forcing_by_target_steady_state_norm(
                f=f.unsqueeze(0),
                n_x=self.n_x,
                h=self.h,
                kappa=self.kappa,
                target_uss_norm_min=self.target_uss_norm_range[0],
                target_uss_norm_max=self.target_uss_norm_range[1],
            )[0]
            u0 = _rescale_batch_to_l2_norm_range(
                x=u0.unsqueeze(0),
                h=self.h,
                norm_min=self.target_u0_norm_range[0],
                norm_max=self.target_u0_norm_range[1],
            )[0]

        u_traj = generate_heat_trajectory_batch(
            u0=u0,
            f=f,
            dt=self.dt,
            n_steps=self.n_steps,
            kappa=self.kappa,
            solver_cache=self._solver_cache_cpu,
        )[0]

        return {"f": f, "u0": u0, "u_traj": u_traj}


class HeatTrajectoryTensorDataset(Dataset):
    """
    Dataset wrapper around precomputed tensors:
        f: (n_samples, n_x+2)
        u0: (n_samples, n_x)
        u_traj: (n_samples, K+1, n_x)
    """

    def __init__(self, f_data: torch.Tensor, u0_data: torch.Tensor, u_traj_data: torch.Tensor):
        if f_data.dim() != 2:
            raise ValueError("f_data must have shape (n_samples, n_x)")
        if u0_data.dim() != 2:
            raise ValueError("u0_data must have shape (n_samples, n_x)")
        if u_traj_data.dim() != 3:
            raise ValueError("u_traj_data must have shape (n_samples, K+1, n_x)")
        n_x = u_traj_data.shape[2]
        if (
            f_data.shape[0] != u0_data.shape[0]
            or f_data.shape[0] != u_traj_data.shape[0]
            or u0_data.shape[1] != n_x
            or f_data.shape[1] not in (n_x, n_x + 2)
        ):
            raise ValueError("inconsistent tensor shapes for precomputed trajectory data")

        self.f_data = f_data
        self.u0_data = u0_data
        self.u_traj_data = u_traj_data

    def __len__(self) -> int:
        return self.f_data.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "f": self.f_data[idx],
            "u0": self.u0_data[idx],
            "u_traj": self.u_traj_data[idx],
        }


class HeatStepDataset(Dataset):
    """
    Flattened one-step dataset from full trajectories.

    Returns tuple:
        (u_k, u_{k+1}, f)
    each with shape (n_x,).
    """

    def __init__(self, f_data: torch.Tensor, u_traj_data: torch.Tensor):
        if f_data.dim() != 2:
            raise ValueError("f_data must have shape (n_samples, n_x)")
        if u_traj_data.dim() != 3:
            raise ValueError("u_traj_data must have shape (n_samples, K+1, n_x)")
        n_x = u_traj_data.shape[2]
        if f_data.shape[0] != u_traj_data.shape[0] or f_data.shape[1] not in (n_x, n_x + 2):
            raise ValueError("f_data and u_traj_data shapes are inconsistent")

        self.f_data = f_data
        self.u_traj_data = u_traj_data
        self.n_samples = u_traj_data.shape[0]
        self.n_steps = u_traj_data.shape[1] - 1

    def __len__(self) -> int:
        return self.n_samples * self.n_steps

    def __getitem__(self, idx: int):
        traj_idx = idx // self.n_steps
        k = idx % self.n_steps

        u_k = self.u_traj_data[traj_idx, k]
        u_k1 = self.u_traj_data[traj_idx, k + 1]
        f = self.f_data[traj_idx]

        return u_k, u_k1, f


def build_step_dataset(traj_dataset) -> HeatStepDataset:
    """
    Build one-step dataset from a HeatTrajectoryDataset.
    """
    if isinstance(traj_dataset, HeatTrajectoryDataset):
        if traj_dataset.f_data is None or traj_dataset.u_traj_data is None:
            traj_dataset._pregenerate()
        f_data = traj_dataset.f_data
        u_traj_data = traj_dataset.u_traj_data
    elif isinstance(traj_dataset, HeatTrajectoryTensorDataset):
        f_data = traj_dataset.f_data
        u_traj_data = traj_dataset.u_traj_data
    elif isinstance(traj_dataset, dict):
        f_data = traj_dataset["f"]
        u_traj_data = traj_dataset["u_traj"]
    else:
        raise TypeError(
            "build_step_dataset expects HeatTrajectoryDataset, HeatTrajectoryTensorDataset, or split dict"
        )

    if f_data is None or u_traj_data is None:
        raise ValueError("Could not resolve precomputed tensors for step dataset construction")

    return HeatStepDataset(
        f_data=f_data,
        u_traj_data=u_traj_data,
    )


def build_trajectory_dataset_from_split(split: Dict[str, torch.Tensor]) -> HeatTrajectoryTensorDataset:
    """
    Build a trajectory dataset from a split dictionary.
    Expected keys: f, u0, u_traj.
    """
    return HeatTrajectoryTensorDataset(
        f_data=split["f"],
        u0_data=split["u0"],
        u_traj_data=split["u_traj"],
    )


def _slice_split(data: Dict[str, torch.Tensor], start: int, end: int) -> Dict[str, torch.Tensor]:
    return {
        "f": data["f"][start:end].clone(),
        "u0": data["u0"][start:end].clone(),
        "u_traj": data["u_traj"][start:end].clone(),
    }


def generate_dataset_splits(
    n_x: int,
    n_steps: int,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = 42,
    kappa: float = 0.10,
    f_amplitude: float = 1.0,
    u0_amplitude: float = 1.0,
    enforce_u0_boundary_taper: bool = True,
    u0_taper_power: float = 1.0,
    norm_targeting: bool = True,
    target_u0_norm_range: tuple[float, float] = (0.6, 1.6),
    target_uss_norm_range: tuple[float, float] = (0.25, 1.0),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Generate one precomputed dataset and split it into train/val/test.
    """
    rng_state_torch = torch.random.get_rng_state()
    rng_state_numpy = np.random.get_state()
    torch.manual_seed(seed)
    np.random.seed(seed)

    total = n_train + n_val + n_test
    base_ds = HeatTrajectoryDataset(
        n_x=n_x,
        n_steps=n_steps,
        n_samples=total,
        kappa=kappa,
        f_amplitude=f_amplitude,
        u0_amplitude=u0_amplitude,
        enforce_u0_boundary_taper=enforce_u0_boundary_taper,
        u0_taper_power=u0_taper_power,
        norm_targeting=norm_targeting,
        target_u0_norm_range=target_u0_norm_range,
        target_uss_norm_range=target_uss_norm_range,
        pregenerate=True,
        dtype=dtype,
    )

    data_all = {
        "f": base_ds.f_data,
        "u0": base_ds.u0_data,
        "u_traj": base_ds.u_traj_data,
    }

    train_end = n_train
    val_end = n_train + n_val

    splits = {
        "train": _slice_split(data_all, 0, train_end),
        "val": _slice_split(data_all, train_end, val_end),
        "test": _slice_split(data_all, val_end, total),
        "meta": {
            "dataset_version": DATASET_VERSION,
            "n_x": n_x,
            "n_steps": n_steps,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "seed": seed,
            "kappa": float(kappa),
            "f_grid_points": n_x + 2,
            "u0_grid_points": n_x,
            "f_amplitude": f_amplitude,
            "u0_amplitude": u0_amplitude,
            "enforce_u0_boundary_taper": enforce_u0_boundary_taper,
            "u0_taper_power": u0_taper_power,
            "norm_targeting": norm_targeting,
            "target_u0_norm_range": list(target_u0_norm_range),
            "target_uss_norm_range": list(target_uss_norm_range),
        },
    }

    torch.random.set_rng_state(rng_state_torch)
    np.random.set_state(rng_state_numpy)
    return splits


def build_dataset_meta(
    n_x: int,
    n_steps: int,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    kappa: float,
    f_amplitude: float,
    u0_amplitude: float,
    enforce_u0_boundary_taper: bool,
    u0_taper_power: float,
    norm_targeting: bool,
    target_u0_norm_range: tuple[float, float],
    target_uss_norm_range: tuple[float, float],
) -> Dict[str, object]:
    """
    Build expected metadata for cached dataset split consistency checks.
    """
    return {
        "dataset_version": DATASET_VERSION,
        "n_x": n_x,
        "n_steps": n_steps,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "seed": seed,
        "kappa": float(kappa),
        "f_grid_points": n_x + 2,
        "u0_grid_points": n_x,
        "f_amplitude": float(f_amplitude),
        "u0_amplitude": float(u0_amplitude),
        "enforce_u0_boundary_taper": bool(enforce_u0_boundary_taper),
        "u0_taper_power": float(u0_taper_power),
        "norm_targeting": bool(norm_targeting),
        "target_u0_norm_range": list(target_u0_norm_range),
        "target_uss_norm_range": list(target_uss_norm_range),
    }


def prepare_or_load_dataset_splits(
    dataset_path: str,
    n_x: int,
    n_steps: int,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = 42,
    kappa: float = 0.10,
    f_amplitude: float = 1.0,
    u0_amplitude: float = 1.0,
    enforce_u0_boundary_taper: bool = True,
    u0_taper_power: float = 1.0,
    norm_targeting: bool = True,
    target_u0_norm_range: tuple[float, float] = (0.6, 1.6),
    target_uss_norm_range: tuple[float, float] = (0.25, 1.0),
    force_regenerate: bool = False,
    map_location: str = "cpu",
    dtype: torch.dtype = torch.float32,
    verbose: bool = True,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Load cached dataset if metadata matches; otherwise regenerate and cache.
    """
    expected_meta = build_dataset_meta(
        n_x=n_x,
        n_steps=n_steps,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        seed=seed,
        kappa=kappa,
        f_amplitude=f_amplitude,
        u0_amplitude=u0_amplitude,
        enforce_u0_boundary_taper=enforce_u0_boundary_taper,
        u0_taper_power=u0_taper_power,
        norm_targeting=norm_targeting,
        target_u0_norm_range=target_u0_norm_range,
        target_uss_norm_range=target_uss_norm_range,
    )

    should_regenerate = force_regenerate or (not os.path.exists(dataset_path))
    reason = "forced" if force_regenerate else "missing"

    splits = None
    if not should_regenerate:
        splits = load_dataset_splits(dataset_path, map_location=map_location)
        if splits.get("meta", {}) != expected_meta:
            should_regenerate = True
            reason = "meta_mismatch"

    if should_regenerate:
        if verbose:
            print(f"Preparing dataset splits ({reason}) at: {dataset_path}")
        splits = generate_dataset_splits(
            n_x=n_x,
            n_steps=n_steps,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            seed=seed,
            kappa=kappa,
            f_amplitude=f_amplitude,
            u0_amplitude=u0_amplitude,
            enforce_u0_boundary_taper=enforce_u0_boundary_taper,
            u0_taper_power=u0_taper_power,
            norm_targeting=norm_targeting,
            target_u0_norm_range=target_u0_norm_range,
            target_uss_norm_range=target_uss_norm_range,
            dtype=dtype,
        )
        save_dataset_splits(splits, dataset_path)
    elif verbose:
        print(f"Loaded cached dataset splits: {dataset_path}")

    return splits


def validate_split_dirichlet_u0(split: Dict[str, torch.Tensor], split_name: str = "train") -> None:
    """
    Validate interior representation is Dirichlet-compatible after zero-padding.
    """
    if "u0" not in split:
        raise KeyError(f"Missing key '{split_name}.u0'")
    if not check_dirichlet_1d(split["u0"]):
        raise RuntimeError(f"Dirichlet boundary check failed for {split_name} initial conditions")


def plot_trajectory_samples(
    split: Dict[str, torch.Tensor],
    split_name: str,
    out_path: str,
    n_plot_samples: int = 4,
) -> None:
    """
    Plot forcing, trajectory heatmap, and snapshot curves for selected samples.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plotting because matplotlib is unavailable: {exc}")
        return

    u_traj = split["u_traj"]  # (n_samples, K+1, n_x), interior representation
    f = split["f"]            # (n_samples, n_x) or (n_samples, n_x+2)

    total = u_traj.shape[0]
    n_plot = min(max(1, n_plot_samples), total)
    idx = torch.linspace(0, total - 1, n_plot).long()

    n_x = u_traj.shape[-1]
    h = 1.0 / (n_x + 1)
    n_steps = u_traj.shape[1] - 1
    x_full = torch.linspace(0.0, 1.0, n_x + 2)
    x_interior = torch.linspace(h, 1.0 - h, n_x)
    snapshot_times = [0.0, 0.25, 0.50, 0.75, 1.0]
    fig, axes = plt.subplots(n_plot, 3, figsize=(16, 3.4 * n_plot), squeeze=False)

    for row, i_t in enumerate(idx.tolist()):
        ax_f = axes[row, 0]
        ax_traj = axes[row, 1]
        ax_snap = axes[row, 2]

        u_traj_full = pad_dirichlet_1d(u_traj[i_t])  # (K+1, n_x+2)

        # Panel 1: forcing field f(x)
        if f.shape[1] == n_x + 2:
            x_f = x_full
            f_plot = f[i_t]
            f_title = "forcing f(x) (full grid)"
        else:
            x_f = x_interior
            f_plot = f[i_t]
            f_title = "forcing f(x) (interior grid)"
        ax_f.plot(x_f.cpu().numpy(), f_plot.cpu().numpy(), linewidth=2, color="tab:purple")
        ax_f.set_title(f"{split_name} sample {i_t}: {f_title}")
        ax_f.set_xlabel("x")
        ax_f.set_ylabel("f")
        ax_f.grid(alpha=0.3)

        # Panel 2: trajectory heatmap u(x,t)
        im = ax_traj.imshow(
            u_traj_full.cpu().numpy(),
            aspect="auto",
            origin="lower",
            extent=[0.0, 1.0, 0.0, 1.0],
            cmap="viridis",
        )
        ax_traj.set_title(f"{split_name} sample {i_t}: trajectory u(x,t) with Dirichlet BC")
        ax_traj.set_xlabel("x")
        ax_traj.set_ylabel("t")
        fig.colorbar(im, ax=ax_traj, fraction=0.046, pad=0.04)

        # Panel 3: snapshots u(·,t) at requested times
        color_seq = ["k", "tab:blue", "tab:orange", "tab:green", "tab:red"]
        for t_snap, color in zip(snapshot_times, color_seq):
            idx_t = int(round(t_snap * n_steps))
            idx_t = max(0, min(n_steps, idx_t))
            ax_snap.plot(
                x_full.cpu().numpy(),
                u_traj_full[idx_t].cpu().numpy(),
                linewidth=2,
                color=color,
                label=f"t={t_snap:.2f}",
            )
        ax_snap.set_title(f"{split_name} sample {i_t}: u(·,t) snapshots")
        ax_snap.set_xlabel("x")
        ax_snap.set_ylabel("u")
        ax_snap.grid(alpha=0.3)
        ax_snap.legend(loc="best")

    folder = os.path.dirname(out_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved trajectory plot: {out_path}")


def save_dataset_splits(splits: Dict[str, Dict[str, torch.Tensor]], path: str) -> None:
    """
    Save precomputed train/val/test splits to disk as a single .pt file.
    """
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    torch.save(splits, path)


def load_dataset_splits(path: str, map_location: str = "cpu") -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Load precomputed train/val/test splits from disk.
    """
    splits = torch.load(path, map_location=map_location)
    for split_name in ("train", "val", "test"):
        if split_name not in splits:
            raise ValueError(f"Missing split '{split_name}' in {path}")
        for key in ("f", "u0", "u_traj"):
            if key not in splits[split_name]:
                raise ValueError(f"Missing key '{split_name}.{key}' in {path}")
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare cached heat-equation dataset splits and plots")

    parser.add_argument("--n-x", type=int, default=100, help="Number of interior spatial points")
    parser.add_argument("--n-steps", type=int, default=10, help="Number of time steps on [0,1]")
    parser.add_argument("--kappa", type=float, default=0.05, help="Diffusion coefficient in u_t = kappa*u_xx + f")

    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/heat_l2_nx100_steps10.pt",
        help="Path to cached train/val/test dataset file (.pt)",
    )
    parser.add_argument("--force-regenerate-data", action="store_true")

    parser.add_argument("--f-amplitude", type=float, default=10.0)
    parser.add_argument("--u0-amplitude", type=float, default=2.5)
    parser.add_argument("--disable-u0-boundary-taper", action="store_true")
    parser.add_argument("--u0-taper-power", type=float, default=1.0)
    parser.add_argument("--disable-norm-targeting", type=bool, default=True)
    parser.add_argument("--target-u0-norm-min", type=float, default=0.6)
    parser.add_argument("--target-u0-norm-max", type=float, default=1.6)
    parser.add_argument("--target-uss-norm-min", type=float, default=0.25)
    parser.add_argument("--target-uss-norm-max", type=float, default=1.0)

    parser.add_argument("--plot-trajectories", type=bool, default=True)
    parser.add_argument("--plot-split", type=str, default="all", choices=["all", "train", "val", "test"])
    parser.add_argument("--n-plot-samples", type=int, default=4)
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="outputs/data_samples",
        help="Directory where trajectory sample plots are saved",
    )
    parser.add_argument("--plot-prefix", type=str, default="trajectory_samples")

    return parser.parse_args()


def _print_split_stats(splits: Dict[str, Dict[str, torch.Tensor]]) -> None:
    meta = splits.get("meta", {})
    print("Dataset meta:", meta)
    for split_name in ("train", "val", "test"):
        split = splits[split_name]
        f = split["f"]
        u0 = split["u0"]
        u_traj = split["u_traj"]
        maxabs = f.abs().amax(dim=1)
        print(
            f"{split_name}: "
            f"f{tuple(f.shape)} u0{tuple(u0.shape)} u_traj{tuple(u_traj.shape)} "
            f"| max|f| median={float(torch.quantile(maxabs, 0.5)):.4f} "
            f"q95={float(torch.quantile(maxabs, 0.95)):.4f}"
        )


def main(args: argparse.Namespace) -> None:
    splits = prepare_or_load_dataset_splits(
        dataset_path=args.dataset_path,
        n_x=args.n_x,
        n_steps=args.n_steps,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seed=args.seed,
        kappa=args.kappa,
        f_amplitude=args.f_amplitude,
        u0_amplitude=args.u0_amplitude,
        enforce_u0_boundary_taper=(not args.disable_u0_boundary_taper),
        u0_taper_power=args.u0_taper_power,
        norm_targeting=(not args.disable_norm_targeting),
        target_u0_norm_range=(args.target_u0_norm_min, args.target_u0_norm_max),
        target_uss_norm_range=(args.target_uss_norm_min, args.target_uss_norm_max),
        force_regenerate=args.force_regenerate_data,
        map_location="cpu",
        verbose=True,
    )

    for split_name in ("train", "val", "test"):
        validate_split_dirichlet_u0(splits[split_name], split_name=split_name)

    _print_split_stats(splits)

    if args.plot_trajectories:
        if args.plot_split == "all":
            split_names = ("train", "val", "test")
        else:
            split_names = (args.plot_split,)
        os.makedirs(args.plot_dir, exist_ok=True)
        for split_name in split_names:
            out_path = os.path.join(args.plot_dir, f"{args.plot_prefix}_{split_name}.png")
            plot_trajectory_samples(
                split=splits[split_name],
                split_name=split_name,
                out_path=out_path,
                n_plot_samples=args.n_plot_samples,
            )


if __name__ == "__main__":
    main(parse_args())
