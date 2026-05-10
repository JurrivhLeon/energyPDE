"""
Data generation for 1D viscous Burgers equation:
    u_t + (u^2 / 2)_x = nu * u_xx + g(x),  x in [0, 1], t in [0, 1]
with periodic boundary conditions by default.

The dataset format matches grad_flow_l2/data.py:
    split["f"]      : (n_samples, n_x)     (periodic forcing grid)
    split["u0"]     : (n_samples, n_x)
    split["u_traj"] : (n_samples, n_steps+1, n_x)
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import torch

try:
    from .heat_data import sample_field_mixed, save_dataset_splits
except ImportError:
    from grad_flow_l2.heat_data import sample_field_mixed, save_dataset_splits


DATASET_VERSION = 3


def _periodic_grid_1d(
    n_x: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.arange(n_x, device=device, dtype=dtype) / float(n_x)


def _remove_mean_1d(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=-1, keepdim=True)


def sample_periodic_grf_1d(
    n_points: int,
    n_samples: int = 1,
    length_scale: float = 0.2,
    variance: float = 1.0,
    zero_mean: bool = True,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Sample smooth periodic 1D Gaussian random fields by spectral synthesis.
    """
    k = torch.fft.rfftfreq(n_points, d=1.0 / float(n_points), device=device)
    power = variance * torch.exp(-0.5 * (2.0 * np.pi * float(length_scale) * k).square())
    if zero_mean:
        power[0] = 0.0

    real = torch.randn(n_samples, power.shape[0], device=device)
    imag = torch.randn(n_samples, power.shape[0], device=device)
    imag[:, 0] = 0.0
    if n_points % 2 == 0:
        imag[:, -1] = 0.0
    spectrum = (real + 1j * imag) * torch.sqrt(power).unsqueeze(0)
    samples = torch.fft.irfft(spectrum, n=n_points, dim=-1) * (n_points ** 0.5)
    if zero_mean:
        samples = _remove_mean_1d(samples)
    return samples


def sample_periodic_sinusoidal_1d(
    n_points: int,
    n_samples: int = 1,
    max_modes: int = 6,
    amplitude: float = 1.0,
    zero_mean: bool = True,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Random periodic sinusoidal fields on the torus.
    """
    x = _periodic_grid_1d(n_x=n_points, device=device)
    fields = torch.zeros(n_samples, n_points, device=device)
    n_terms = max(2, int(max_modes))
    for _ in range(n_terms):
        k = torch.randint(1, max_modes + 1, (n_samples, 1), device=device).float()
        phase = 2.0 * np.pi * torch.rand(n_samples, 1, device=device)
        coeff = torch.randn(n_samples, 1, device=device) / k
        fields = fields + coeff * torch.sin(2.0 * np.pi * k * x.unsqueeze(0) + phase)

    if zero_mean:
        fields = _remove_mean_1d(fields)
    max_val = fields.abs().amax(dim=1, keepdim=True) + 1e-8
    return amplitude * fields / max_val


def sample_periodic_field_mixed_1d(
    n_points: int,
    n_samples: int = 1,
    amplitude: float = 1.0,
    length_scale_range: tuple[float, float] = (0.06, 0.35),
    max_modes: int = 6,
    grf_prob: float = 0.7,
    zero_mean: bool = True,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Mixed periodic sampler: each sample is either spectral GRF or sinusoidal.
    """
    out = []
    for _ in range(n_samples):
        if np.random.rand() < grf_prob:
            ls = float(np.random.uniform(*length_scale_range))
            field = sample_periodic_grf_1d(
                n_points=n_points,
                n_samples=1,
                length_scale=ls,
                variance=1.0,
                zero_mean=zero_mean,
                device=device,
            )
        else:
            mm = int(np.random.randint(3, max_modes + 1))
            field = sample_periodic_sinusoidal_1d(
                n_points=n_points,
                n_samples=1,
                max_modes=mm,
                amplitude=1.0,
                zero_mean=zero_mean,
                device=device,
            )

        amp = float(amplitude * 10 ** np.random.uniform(-0.35, 0.35))
        field = amp * field / (field.abs().amax(dim=1, keepdim=True) + 1e-8)
        out.append(field)

    return torch.cat(out, dim=0)


def sample_elastic_forcing_1d(
    n_points: int,
    n_samples: int = 1,
    stiffness: float = 1.0,
    length_scale_range: tuple[float, float] = (0.06, 0.35),
    max_modes: int = 6,
    grf_prob: float = 0.7,
    zero_mean: bool = True,
    boundary_condition: str = "periodic",
    taper_power: float = 1.0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Sample static elastic forcing fields as Hookean restoring profiles.

    The sampled smooth displacement profile q(x) is converted to a static
    additive force f(x) = -k q(x). This keeps the existing Burgers dataset
    schema while giving the forcing a simple elastic interpretation.
    """
    if stiffness <= 0:
        raise ValueError("stiffness must be > 0")
    bc = boundary_condition.strip().lower()
    if bc == "periodic":
        displacement = sample_periodic_field_mixed_1d(
            n_points=n_points,
            n_samples=n_samples,
            amplitude=1.0,
            length_scale_range=length_scale_range,
            max_modes=max_modes,
            grf_prob=grf_prob,
            zero_mean=zero_mean,
            device=device,
        )
        return -float(stiffness) * displacement
    if bc == "dirichlet":
        interior = sample_field_mixed(
            n_points=n_points - 2,
            n_samples=n_samples,
            amplitude=1.0,
            length_scale_range=length_scale_range,
            max_modes=max_modes,
            grf_prob=grf_prob,
            device=device,
        )
        taper = _dirichlet_taper(n_x=n_points - 2, power=taper_power, device=device)
        interior = -float(stiffness) * interior * taper.unsqueeze(0)
        zeros = torch.zeros(n_samples, 1, device=device)
        return torch.cat([zeros, interior, zeros], dim=-1)
    raise ValueError("boundary_condition must be one of {'periodic','dirichlet'}")


def sample_burgers_paper_u0_1d(
    n_points: int,
    n_samples: int = 1,
    covariance_scale: float = 625.0,
    mass: float = 25.0,
    zero_mean: bool = False,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Sample u0 ~ N(0, covariance_scale * (-Delta + mass I)^-2) on the unit torus.

    The periodic Laplacian eigenvalues are (2*pi*k)^2, so each real Fourier
    coefficient is scaled by sqrt(covariance_scale) / ((2*pi*k)^2 + mass).
    """
    k = torch.fft.rfftfreq(n_points, d=1.0 / float(n_points), device=device)
    eig = (2.0 * np.pi * k).square() + float(mass)
    amp = (float(covariance_scale) ** 0.5) / eig
    if zero_mean:
        amp[0] = 0.0

    real = torch.randn(n_samples, amp.shape[0], device=device)
    imag = torch.randn(n_samples, amp.shape[0], device=device)
    imag[:, 0] = 0.0
    if n_points % 2 == 0:
        imag[:, -1] = 0.0
    spectrum = (real + 1j * imag) * amp.unsqueeze(0)
    samples = torch.fft.irfft(spectrum, n=n_points, dim=-1) * (n_points ** 0.5)
    if zero_mean:
        samples = _remove_mean_1d(samples)
    return samples


def spectral_truncate_periodic_field_1d(field: torch.Tensor, target_n_x: int) -> torch.Tensor:
    """
    Downsample a periodic 1D field by truncating high Fourier modes.
    """
    if field.shape[-1] == target_n_x:
        return field.clone()
    if target_n_x <= 0:
        raise ValueError("target_n_x must be positive")
    orig_n_x = int(field.shape[-1])
    if target_n_x > orig_n_x:
        raise ValueError("target_n_x cannot exceed the source resolution")

    field_hat = torch.fft.rfft(field, dim=-1, norm="ortho")
    target_freq = target_n_x // 2 + 1
    out_hat = field_hat[..., :target_freq].clone()
    if target_n_x % 2 == 0:
        out_hat[..., -1] = out_hat[..., -1].real + 0j
    out = torch.fft.irfft(out_hat, n=target_n_x, dim=-1, norm="ortho")
    return out * (float(target_n_x) / float(orig_n_x)) ** 0.5


def _dirichlet_taper(
    n_x: int,
    power: float = 1.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if power <= 0:
        raise ValueError("power must be > 0")
    x = torch.linspace(0.0, 1.0, n_x + 2, device=device, dtype=dtype)[1:-1]
    taper = (4.0 * x * (1.0 - x)) ** power
    return taper / (torch.max(taper) + 1e-12)


def _rescale_batch_l2(
    x: torch.Tensor,
    h: float,
    norm_min: float,
    norm_max: float,
) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError("x must have shape (batch, n_x)")
    norms = torch.sqrt(h * torch.sum(x * x, dim=-1))
    targets = torch.empty_like(norms).uniform_(norm_min, norm_max)
    scale = targets / (norms + 1e-8)
    return x * scale.unsqueeze(-1)


def _cap_batch_abs(x: torch.Tensor, max_abs: float | None) -> torch.Tensor:
    if max_abs is None or float(max_abs) <= 0.0:
        return x
    current = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(float(max_abs) / (current + 1e-8), max=1.0)
    return x * scale


def _to_interior_forcing(f: torch.Tensor, n_x: int) -> torch.Tensor:
    if f.dim() == 1:
        if f.shape[0] == n_x:
            return f.unsqueeze(0)
        if f.shape[0] == n_x + 2:
            return f[1:-1].unsqueeze(0)
        raise ValueError(f"forcing width must be {n_x} or {n_x+2}, got {f.shape[0]}")
    if f.dim() == 2:
        if f.shape[1] == n_x:
            return f
        if f.shape[1] == n_x + 2:
            return f[:, 1:-1]
        raise ValueError(f"forcing width must be {n_x} or {n_x+2}, got {f.shape[1]}")
    raise ValueError("forcing must have shape (n_x,), (n_x+2,), (batch,n_x), or (batch,n_x+2)")


def _rescale_forcing_l2(
    f: torch.Tensor,
    h: float,
    norm_min: float,
    norm_max: float,
    n_x: int,
) -> torch.Tensor:
    f_int = _to_interior_forcing(f, n_x=n_x)
    norms = torch.sqrt(h * torch.sum(f_int * f_int, dim=-1))
    targets = torch.empty_like(norms).uniform_(norm_min, norm_max)
    scale = targets / (norms + 1e-8)
    if f.dim() == 1:
        return f * scale[0]
    return f * scale.unsqueeze(-1)


def burgers_step_rusanov_dirichlet(
    u: torch.Tensor,
    dt: float,
    h: float,
    nu: float,
    forcing: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Single explicit step with:
      - Rusanov flux for nonlinear convection
      - central second-order finite-difference diffusion
      - zero Dirichlet boundary values through ghost cells
    """
    if u.dim() == 1:
        u = u.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    zeros = torch.zeros(u.shape[0], 1, device=u.device, dtype=u.dtype)
    u_full = torch.cat([zeros, u, zeros], dim=-1)  # (batch, n_x+2)

    # Interface fluxes on (n_x+1) faces.
    u_l = u_full[:, :-1]
    u_r = u_full[:, 1:]
    flux_l = 0.5 * (u_l * u_l)
    flux_r = 0.5 * (u_r * u_r)
    a = torch.maximum(torch.abs(u_l), torch.abs(u_r))
    flux_half = 0.5 * (flux_l + flux_r) - 0.5 * a * (u_r - u_l)

    convection = (flux_half[:, 1:] - flux_half[:, :-1]) / h
    diffusion = float(nu) * (u_full[:, :-2] - 2.0 * u_full[:, 1:-1] + u_full[:, 2:]) / (h * h)

    if forcing is None:
        forcing_int = torch.zeros_like(u)
    else:
        forcing_int = _to_interior_forcing(forcing, n_x=u.shape[-1]).to(device=u.device, dtype=u.dtype)
        if forcing_int.shape[0] == 1 and u.shape[0] > 1:
            forcing_int = forcing_int.expand(u.shape[0], -1)
        if forcing_int.shape[0] != u.shape[0]:
            raise ValueError("forcing batch size must match u batch size or be 1")

    u_next = u - dt * convection + dt * diffusion + dt * forcing_int
    if squeeze:
        return u_next.squeeze(0)
    return u_next


def burgers_step_rusanov_periodic(
    u: torch.Tensor,
    dt: float,
    h: float,
    nu: float,
    forcing: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Single periodic IMEX step with explicit Rusanov convection and forcing,
    plus implicit spectral diffusion.
    """
    if u.dim() == 1:
        u = u.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    u_l = u
    u_r = torch.roll(u, shifts=-1, dims=-1)
    flux_l = 0.5 * (u_l * u_l)
    flux_r = 0.5 * (u_r * u_r)
    a = torch.maximum(torch.abs(u_l), torch.abs(u_r))
    flux_half = 0.5 * (flux_l + flux_r) - 0.5 * a * (u_r - u_l)

    convection = (flux_half - torch.roll(flux_half, shifts=1, dims=-1)) / h

    if forcing is None:
        forcing_int = torch.zeros_like(u)
    else:
        forcing_int = _to_interior_forcing(forcing, n_x=u.shape[-1]).to(device=u.device, dtype=u.dtype)
        if forcing_int.shape[0] == 1 and u.shape[0] > 1:
            forcing_int = forcing_int.expand(u.shape[0], -1)
        if forcing_int.shape[0] != u.shape[0]:
            raise ValueError("forcing batch size must match u batch size or be 1")

    rhs = u - dt * convection + dt * forcing_int
    rhs_hat = torch.fft.rfft(rhs, dim=-1)
    k = 2.0 * torch.pi * torch.fft.rfftfreq(u.shape[-1], d=1.0 / float(u.shape[-1]), device=u.device).to(dtype=u.dtype)
    u_next = torch.fft.irfft(rhs_hat / (1.0 + float(dt) * float(nu) * k.square()).unsqueeze(0), n=u.shape[-1], dim=-1)
    if squeeze:
        return u_next.squeeze(0)
    return u_next


def burgers_step_spectral_periodic(
    u: torch.Tensor,
    dt: float,
    nu: float,
    forcing: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Periodic split-step update:
      1. exact heat solve in Fourier space,
      2. forward Euler spectral update for -d_x(u^2/2) + forcing.
    """
    if u.dim() == 1:
        u = u.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    n_x = int(u.shape[-1])
    k = 2.0 * torch.pi * torch.fft.rfftfreq(n_x, d=1.0 / float(n_x), device=u.device).to(dtype=u.dtype)

    u_hat = torch.fft.rfft(u, dim=-1)
    u_heat = torch.fft.irfft(torch.exp(-float(nu) * k.square() * float(dt)).unsqueeze(0) * u_hat, n=n_x, dim=-1)

    flux = 0.5 * u_heat.square()
    flux_hat = torch.fft.rfft(flux, dim=-1)
    convection = torch.fft.irfft((1j * k).unsqueeze(0) * flux_hat, n=n_x, dim=-1)

    if forcing is None:
        forcing_int = torch.zeros_like(u_heat)
    else:
        forcing_int = _to_interior_forcing(forcing, n_x=n_x).to(device=u.device, dtype=u.dtype)
        if forcing_int.shape[0] == 1 and u.shape[0] > 1:
            forcing_int = forcing_int.expand(u.shape[0], -1)
        if forcing_int.shape[0] != u.shape[0]:
            raise ValueError("forcing batch size must match u batch size or be 1")

    u_next = u_heat + float(dt) * (-convection + forcing_int)
    if squeeze:
        return u_next.squeeze(0)
    return u_next


def solve_burgers_trajectory(
    u0: torch.Tensor,
    forcing: torch.Tensor | None,
    n_steps: int,
    nu: float,
    t_final: float = 1.0,
    boundary_condition: str = "periodic",
    cfl_adv: float = 0.45,
    cfl_diff: float = 0.45,
    max_substeps_per_step: int = 2000,
) -> torch.Tensor:
    """
    Integrate Burgers equation and return [u_0, ..., u_K].

    Uses adaptive substepping inside each macro-step to satisfy explicit
    advection+diffusion stability constraints.
    """
    if nu <= 0:
        raise ValueError("nu must be > 0 for this viscous Burgers generator")
    if u0.dim() == 1:
        u0 = u0.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    n_x = int(u0.shape[-1])
    bc = boundary_condition.strip().lower()
    if bc not in {"periodic", "dirichlet"}:
        raise ValueError("boundary_condition must be one of {'periodic','dirichlet'}")
    h = 1.0 / float(n_x) if bc == "periodic" else 1.0 / float(n_x + 1)
    dt_macro = float(t_final) / float(n_steps)
    dt_diff = float(cfl_diff) * (h * h) / float(nu)
    forcing_int = None if forcing is None else _to_interior_forcing(forcing, n_x=n_x).to(device=u0.device, dtype=u0.dtype)
    if forcing_int is not None and forcing_int.shape[0] == 1 and u0.shape[0] > 1:
        forcing_int = forcing_int.expand(u0.shape[0], -1)

    u = u0.clone()
    states = [u.clone()]
    for _ in range(n_steps):
        remaining = dt_macro
        substeps = 0
        while remaining > 1e-15:
            umax = float(torch.max(torch.abs(u)).item())
            dt_adv = float(cfl_adv) * h / max(umax, 1e-8)
            dt_stable = max(1e-10, dt_adv if bc == "periodic" else min(dt_adv, dt_diff))
            dt_sub = min(remaining, dt_stable)
            if bc == "periodic":
                u = burgers_step_rusanov_periodic(u, dt=dt_sub, h=h, nu=nu, forcing=forcing_int)
            else:
                u = burgers_step_rusanov_dirichlet(u, dt=dt_sub, h=h, nu=nu, forcing=forcing_int)
            remaining -= dt_sub
            substeps += 1
            if substeps > max_substeps_per_step:
                raise RuntimeError(
                    "Exceeded max_substeps_per_step; reduce amplitudes, increase n_steps, or relax CFL."
                )
        states.append(u.clone())

    traj = torch.stack(states, dim=1)  # (batch, n_steps+1, n_x)
    if squeeze:
        return traj.squeeze(0)
    return traj


def solve_burgers_trajectory_periodic_spectral(
    u0: torch.Tensor,
    forcing: torch.Tensor | None,
    dataset_dt: float,
    solver_dt: float,
    nu: float,
    t_final: float = 1.0,
) -> torch.Tensor:
    """
    Reference periodic solver with fine solver_dt and stored frames every dataset_dt.
    """
    if nu <= 0:
        raise ValueError("nu must be > 0")
    if dataset_dt <= 0 or solver_dt <= 0:
        raise ValueError("dataset_dt and solver_dt must be > 0")
    if dataset_dt < solver_dt:
        raise ValueError("dataset_dt must be >= solver_dt")
    ratio = float(dataset_dt) / float(solver_dt)
    steps_per_frame = int(round(ratio))
    if abs(ratio - steps_per_frame) > 1e-10:
        raise ValueError("dataset_dt must be an integer multiple of solver_dt")
    n_frames_float = float(t_final) / float(dataset_dt)
    n_steps = int(round(n_frames_float))
    if abs(n_frames_float - n_steps) > 1e-10:
        raise ValueError("t_final must be an integer multiple of dataset_dt")

    if u0.dim() == 1:
        u0 = u0.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    n_x = int(u0.shape[-1])
    forcing_int = None if forcing is None else _to_interior_forcing(forcing, n_x=n_x).to(device=u0.device, dtype=u0.dtype)
    if forcing_int is not None and forcing_int.shape[0] == 1 and u0.shape[0] > 1:
        forcing_int = forcing_int.expand(u0.shape[0], -1)

    u = u0.clone()
    states = [u.clone()]
    for _ in range(n_steps):
        for _ in range(steps_per_frame):
            u = burgers_step_spectral_periodic(u, dt=solver_dt, nu=nu, forcing=forcing_int)
        states.append(u.clone())

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


def generate_burgers_dataset_splits(
    n_x: int,
    n_steps: int,
    t_final: float,
    n_train: int,
    n_val: int,
    n_test: int,
    nu: float = 0.01,
    seed: int = 42,
    u0_amplitude: float = 2.5,
    u0_sampler: str = "sinusoidal",
    u0_covariance_scale: float = 625.0,
    u0_mass: float = 25.0,
    forcing_mode: str = "zero",
    forcing_sampler: str = "grf",
    elastic_stiffness: float = 1.0,
    f_amplitude: float = 1.5,
    f_grf_prob: float = 0.7,
    f_length_scale_min: float = 0.06,
    f_length_scale_max: float = 0.35,
    f_max_modes: int = 6,
    taper_power: float = 1.0,
    boundary_condition: str = "periodic",
    zero_mean: bool = True,
    norm_targeting: bool = True,
    target_u0_norm_range: tuple[float, float] = (0.6, 1.6),
    target_f_norm_range: tuple[float, float] = (0.2, 1.2),
    target_u0_abs_max: float | None = 1.0,
    target_f_abs_max: float | None = 0.5,
    solver_n_x: int | None = None,
    solver_dt: float | None = None,
    dataset_dt: float | None = None,
    solve_batch_size: int = 64,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Dict[str, torch.Tensor]]:
    rng_state_torch = torch.random.get_rng_state()
    rng_state_numpy = np.random.get_state()
    torch.manual_seed(seed)
    np.random.seed(seed)

    total = int(n_train + n_val + n_test)
    bc = boundary_condition.strip().lower()
    if bc not in {"periodic", "dirichlet"}:
        raise ValueError("boundary_condition must be one of {'periodic','dirichlet'}")
    if solver_n_x is None:
        solver_n_x = n_x
    solver_n_x = int(solver_n_x)
    if solver_n_x < n_x:
        raise ValueError("solver_n_x must be >= n_x")
    solve_batch_size = max(1, int(solve_batch_size))
    if bc != "periodic" and solver_n_x != n_x:
        raise ValueError("high-resolution solver downsampling is only supported for periodic Burgers")
    if dataset_dt is None:
        dataset_dt = float(t_final) / float(n_steps)
    if solver_dt is None:
        solver_dt = dataset_dt
    solve_device = torch.device(device)
    if solve_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but torch.cuda.is_available() is False")
    expected_steps = int(round(float(t_final) / float(dataset_dt)))
    if expected_steps != int(n_steps) or abs(float(t_final) / float(dataset_dt) - expected_steps) > 1e-10:
        raise ValueError("n_steps must equal t_final / dataset_dt")
    h = 1.0 / float(n_x) if bc == "periodic" else 1.0 / float(n_x + 1)
    h_solver = 1.0 / float(solver_n_x) if bc == "periodic" else h

    if bc == "periodic":
        if u0_sampler == "paper":
            u0_solver = sample_burgers_paper_u0_1d(
                n_points=solver_n_x,
                n_samples=total,
                covariance_scale=u0_covariance_scale,
                mass=u0_mass,
                zero_mean=zero_mean,
                device=str(solve_device),
            ).to(dtype=dtype)
            if norm_targeting:
                u0_solver = _rescale_batch_l2(
                    u0_solver,
                    h=h_solver,
                    norm_min=float(target_u0_norm_range[0]),
                    norm_max=float(target_u0_norm_range[1]),
                )
        elif u0_sampler == "sinusoidal":
            u0_solver = sample_periodic_sinusoidal_1d(
                n_points=solver_n_x,
                n_samples=total,
                max_modes=f_max_modes,
                amplitude=u0_amplitude,
                zero_mean=zero_mean,
                device=str(solve_device),
            ).to(dtype=dtype)
            if norm_targeting:
                u0_solver = _rescale_batch_l2(
                    u0_solver,
                    h=h_solver,
                    norm_min=float(target_u0_norm_range[0]),
                    norm_max=float(target_u0_norm_range[1]),
                )
        elif u0_sampler == "mixed":
            u0_solver = sample_periodic_field_mixed_1d(
                n_points=solver_n_x,
                n_samples=total,
                amplitude=u0_amplitude,
                length_scale_range=(f_length_scale_min, f_length_scale_max),
                max_modes=f_max_modes,
                grf_prob=f_grf_prob,
                zero_mean=zero_mean,
                device=str(solve_device),
            ).to(dtype=dtype)
            if norm_targeting:
                u0_solver = _rescale_batch_l2(
                    u0_solver,
                    h=h_solver,
                    norm_min=float(target_u0_norm_range[0]),
                    norm_max=float(target_u0_norm_range[1]),
                )
        else:
            raise ValueError("u0_sampler must be one of {'paper','sinusoidal','mixed'}")
        u0_solver = _cap_batch_abs(u0_solver, target_u0_abs_max)
        u0 = spectral_truncate_periodic_field_1d(u0_solver, target_n_x=n_x).to(dtype=dtype)
    else:
        if u0_sampler not in {"mixed", "paper", "sinusoidal"}:
            raise ValueError("u0_sampler must be one of {'paper','sinusoidal','mixed'}")
        u0 = sample_field_mixed(
            n_points=n_x,
            n_samples=total,
            amplitude=u0_amplitude,
            device=str(solve_device),
        ).to(dtype=dtype)
        taper = _dirichlet_taper(n_x=n_x, power=taper_power, device=str(solve_device), dtype=dtype)
        u0 = u0 * taper.unsqueeze(0)
        if norm_targeting:
            u0 = _rescale_batch_l2(
                u0,
                h=h,
                norm_min=float(target_u0_norm_range[0]),
                norm_max=float(target_u0_norm_range[1]),
            )
        u0 = _cap_batch_abs(u0, target_u0_abs_max)

    if forcing_mode not in ("zero", "mixed"):
        raise ValueError("forcing_mode must be one of {'zero', 'mixed'}")
    if forcing_sampler not in {"grf", "mixed", "elastic"}:
        raise ValueError("forcing_sampler must be one of {'grf','mixed','elastic'}")

    if forcing_mode == "zero":
        f_width = solver_n_x if bc == "periodic" else n_x + 2
        f_solver = torch.zeros(total, f_width, dtype=dtype, device=solve_device)
    else:
        if bc == "periodic":
            if forcing_sampler == "elastic":
                f_solver = sample_elastic_forcing_1d(
                    n_points=solver_n_x,
                    n_samples=total,
                    stiffness=elastic_stiffness,
                    length_scale_range=(f_length_scale_min, f_length_scale_max),
                    max_modes=f_max_modes,
                    grf_prob=f_grf_prob,
                    zero_mean=zero_mean,
                    boundary_condition="periodic",
                    device=str(solve_device),
                ).to(dtype=dtype)
                f_solver = f_amplitude * f_solver / (f_solver.abs().amax(dim=1, keepdim=True) + 1e-8)
            elif forcing_sampler == "grf":
                ls = float(0.5 * (f_length_scale_min + f_length_scale_max))
                f_solver = sample_periodic_grf_1d(
                    n_points=solver_n_x,
                    n_samples=total,
                    length_scale=ls,
                    variance=1.0,
                    zero_mean=zero_mean,
                    device=str(solve_device),
                ).to(dtype=dtype)
                f_solver = f_amplitude * f_solver / (f_solver.abs().amax(dim=1, keepdim=True) + 1e-8)
            else:
                f_solver = sample_periodic_field_mixed_1d(
                    n_points=solver_n_x,
                    n_samples=total,
                    amplitude=f_amplitude,
                    length_scale_range=(f_length_scale_min, f_length_scale_max),
                    max_modes=f_max_modes,
                    grf_prob=f_grf_prob,
                    zero_mean=zero_mean,
                    device=str(solve_device),
                ).to(dtype=dtype)
        else:
            if forcing_sampler == "elastic":
                f_solver = sample_elastic_forcing_1d(
                    n_points=n_x + 2,
                    n_samples=total,
                    stiffness=elastic_stiffness,
                    length_scale_range=(f_length_scale_min, f_length_scale_max),
                    max_modes=f_max_modes,
                    grf_prob=f_grf_prob,
                    zero_mean=zero_mean,
                    boundary_condition="dirichlet",
                    taper_power=taper_power,
                    device=str(solve_device),
                ).to(dtype=dtype)
                f_solver = f_amplitude * f_solver / (f_solver.abs().amax(dim=1, keepdim=True) + 1e-8)
            else:
                f_solver = sample_field_mixed(
                    n_points=n_x + 2,
                    n_samples=total,
                    amplitude=f_amplitude,
                    length_scale_range=(f_length_scale_min, f_length_scale_max),
                    max_modes=f_max_modes,
                    grf_prob=f_grf_prob,
                    device=str(solve_device),
                ).to(dtype=dtype)
        if norm_targeting:
            f_solver = _rescale_forcing_l2(
                f=f_solver,
                h=h_solver,
                norm_min=float(target_f_norm_range[0]),
                norm_max=float(target_f_norm_range[1]),
                n_x=solver_n_x if bc == "periodic" else n_x,
            )
        f_solver = _cap_batch_abs(f_solver, target_f_abs_max)
    f = spectral_truncate_periodic_field_1d(f_solver, target_n_x=n_x).to(dtype=dtype) if bc == "periodic" else f_solver

    if bc == "periodic":
        traj_chunks = []
        for start in range(0, total, solve_batch_size):
            end = min(total, start + solve_batch_size)
            u_traj_solver = solve_burgers_trajectory_periodic_spectral(
                u0=u0_solver[start:end],
                forcing=f_solver[start:end],
                dataset_dt=float(dataset_dt),
                solver_dt=float(solver_dt),
                nu=nu,
                t_final=t_final,
            ).to(dtype=dtype)
            traj_chunks.append(spectral_truncate_periodic_field_1d(u_traj_solver, target_n_x=n_x).to(dtype=dtype))
        u_traj = torch.cat(traj_chunks, dim=0)
    else:
        u_traj = solve_burgers_trajectory(
            u0=u0,
            forcing=f,
            n_steps=n_steps,
            nu=nu,
            t_final=t_final,
            boundary_condition=bc,
        ).to(dtype=dtype)

    all_data = {"f": f.cpu(), "u0": u0.cpu(), "u_traj": u_traj.cpu()}
    train_end = n_train
    val_end = n_train + n_val

    splits: Dict[str, Dict[str, torch.Tensor]] = {
        "train": _slice_split(all_data, 0, train_end),
        "val": _slice_split(all_data, train_end, val_end),
        "test": _slice_split(all_data, val_end, total),
        "meta": {
            "dataset_version": DATASET_VERSION,
            "equation": f"burgers_1d_{bc}",
            "boundary_condition": bc,
            "periodic": bool(bc == "periodic"),
            "n_x": int(n_x),
            "solver_n_x": int(solver_n_x),
            "n_steps": int(n_steps),
            "t_final": float(t_final),
            "dataset_dt": float(dataset_dt),
            "solver_dt": float(solver_dt),
            "solve_batch_size": int(solve_batch_size),
            "generation_device": str(solve_device),
            "h": float(h),
            "h_solver": float(h_solver),
            "n_train": int(n_train),
            "n_val": int(n_val),
            "n_test": int(n_test),
            "seed": int(seed),
            "nu": float(nu),
            "u0_sampler": u0_sampler,
            "u0_covariance_scale": float(u0_covariance_scale),
            "u0_mass": float(u0_mass),
            "forcing_mode": forcing_mode,
            "forcing_sampler": forcing_sampler,
            "elastic_stiffness": float(elastic_stiffness),
            "f_grid_points": int(f.shape[-1]),
            "f_solver_grid_points": int(f_solver.shape[-1]),
            "u0_grid_points": int(n_x),
            "u0_amplitude": float(u0_amplitude),
            "f_amplitude": float(f_amplitude),
            "f_grf_prob": float(f_grf_prob),
            "f_length_scale_min": float(f_length_scale_min),
            "f_length_scale_max": float(f_length_scale_max),
            "f_max_modes": int(f_max_modes),
            "taper_power": float(taper_power),
            "zero_mean": bool(zero_mean),
            "norm_targeting": bool(norm_targeting),
            "target_u0_norm_range": [float(target_u0_norm_range[0]), float(target_u0_norm_range[1])],
            "target_f_norm_range": [float(target_f_norm_range[0]), float(target_f_norm_range[1])],
            "target_u0_abs_max": None if target_u0_abs_max is None else float(target_u0_abs_max),
            "target_f_abs_max": None if target_f_abs_max is None else float(target_f_abs_max),
        },
    }

    torch.random.set_rng_state(rng_state_torch)
    np.random.set_state(rng_state_numpy)
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cached 1D Burgers trajectory dataset splits")
    parser.add_argument("--n-x", type=int, default=256, help="Number of spatial grid points")
    parser.add_argument("--n-steps", type=int, default=10, help="Number of macro time steps on [0,t_final]")
    parser.add_argument("--t-final", type=float, default=1.0, help="Final time horizon")
    parser.add_argument("--dataset-dt", type=float, default=0.1, help="Stored trajectory time step")
    parser.add_argument("--solver-dt", type=float, default=1e-3, help="Fine reference solver time step")
    parser.add_argument("--solver-n-x", type=int, default=8192, help="Fine reference solver spatial resolution")
    parser.add_argument("--solve-batch-size", type=int, default=64, help="Number of samples per high-res solve chunk")
    parser.add_argument("--device", type=str, default="cpu", help="Device for reference generation, e.g. cpu or cuda:0")
    parser.add_argument("--nu", type=float, default=0.1, help="Viscosity coefficient")

    parser.add_argument("--n-train", type=int, default=1500)
    parser.add_argument("--n-val", type=int, default=300)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--u0-amplitude", type=float, default=1.0)
    parser.add_argument("--u0-sampler", type=str, default="sinusoidal", choices=["paper", "sinusoidal", "mixed"])
    parser.add_argument("--u0-covariance-scale", type=float, default=625.0)
    parser.add_argument("--u0-mass", type=float, default=25.0)
    parser.add_argument("--boundary-condition", type=str, default="periodic", choices=["periodic", "dirichlet"])
    parser.add_argument("--forcing-mode", type=str, default="mixed", choices=["zero", "mixed"])
    parser.add_argument("--forcing-sampler", type=str, default="grf", choices=["grf", "mixed", "elastic"])
    parser.add_argument("--elastic-stiffness", type=float, default=1.0)
    parser.add_argument("--f-amplitude", type=float, default=0.5)
    parser.add_argument("--f-grf-prob", type=float, default=0.7)
    parser.add_argument("--f-length-scale-min", type=float, default=0.06)
    parser.add_argument("--f-length-scale-max", type=float, default=0.35)
    parser.add_argument("--f-max-modes", type=int, default=6)
    parser.add_argument("--taper-power", type=float, default=1.0)
    parser.add_argument("--disable-zero-mean", action="store_true")
    parser.add_argument("--disable-norm-targeting", action="store_true")
    parser.add_argument("--target-u0-norm-min", type=float, default=0.6)
    parser.add_argument("--target-u0-norm-max", type=float, default=1.0)
    parser.add_argument("--target-f-norm-min", type=float, default=0.2)
    parser.add_argument("--target-f-norm-max", type=float, default=0.5)
    parser.add_argument("--target-u0-abs-max", type=float, default=1.0)
    parser.add_argument("--target-f-abs-max", type=float, default=0.5)

    parser.add_argument(
        "--dataset-path",
        type=str,
        default="grad_flow_l2/datasets/burgers_periodic_forced_l2_nu0p1_snx8192_nx256_dt0p1.pt",
        help="Path to output dataset file (.pt)",
    )
    return parser.parse_args()


def _print_split_stats(splits: Dict[str, Dict[str, torch.Tensor]]) -> None:
    print("Dataset meta:", splits.get("meta", {}))
    for split_name in ("train", "val", "test"):
        split = splits[split_name]
        f = split["f"]
        u0 = split["u0"]
        u_traj = split["u_traj"]
        if u0.shape[0] == 0:
            print(f"{split_name}: empty split")
            continue
        meta = splits.get("meta", {})
        n_x = int(u0.shape[-1])
        h = float(meta.get("h", 1.0 / float(n_x) if meta.get("periodic", False) else 1.0 / float(n_x + 1)))
        l2_u0 = torch.sqrt(h * torch.sum(u0 * u0, dim=-1))
        f_int = _to_interior_forcing(f, n_x=n_x)
        l2_f = torch.sqrt(h * torch.sum(f_int * f_int, dim=-1))
        print(
            f"{split_name}: "
            f"f={tuple(f.shape)}, "
            f"u0={tuple(u0.shape)}, "
            f"u_traj={tuple(u_traj.shape)}, "
            f"f_l2_mean={l2_f.mean().item():.4f}, "
            f"u0_l2_mean={l2_u0.mean().item():.4f}, "
            f"f_abs_max={f.abs().max().item():.4f}, "
            f"u0_abs_max={u0.abs().max().item():.4f}"
        )


def main(args: argparse.Namespace) -> None:
    if args.nu <= 0:
        raise ValueError("--nu must be > 0 for this generator")
    if args.n_steps < 1:
        raise ValueError("--n-steps must be >= 1")
    if args.t_final <= 0:
        raise ValueError("--t-final must be > 0")

    splits = generate_burgers_dataset_splits(
        n_x=args.n_x,
        n_steps=args.n_steps,
        t_final=args.t_final,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        nu=args.nu,
        seed=args.seed,
        u0_amplitude=args.u0_amplitude,
        u0_sampler=args.u0_sampler,
        u0_covariance_scale=args.u0_covariance_scale,
        u0_mass=args.u0_mass,
        forcing_mode=args.forcing_mode,
        forcing_sampler=args.forcing_sampler,
        elastic_stiffness=args.elastic_stiffness,
        f_amplitude=args.f_amplitude,
        f_grf_prob=args.f_grf_prob,
        f_length_scale_min=args.f_length_scale_min,
        f_length_scale_max=args.f_length_scale_max,
        f_max_modes=args.f_max_modes,
        taper_power=args.taper_power,
        boundary_condition=args.boundary_condition,
        zero_mean=not args.disable_zero_mean,
        norm_targeting=not args.disable_norm_targeting,
        target_u0_norm_range=(args.target_u0_norm_min, args.target_u0_norm_max),
        target_f_norm_range=(args.target_f_norm_min, args.target_f_norm_max),
        target_u0_abs_max=args.target_u0_abs_max,
        target_f_abs_max=args.target_f_abs_max,
        solver_n_x=args.solver_n_x,
        solver_dt=args.solver_dt,
        dataset_dt=args.dataset_dt,
        solve_batch_size=args.solve_batch_size,
        device=args.device,
        dtype=torch.float32,
    )

    out_dir = os.path.dirname(args.dataset_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_dataset_splits(splits, args.dataset_path)
    print(f"Saved Burgers dataset splits to: {args.dataset_path}")
    _print_split_stats(splits)


if __name__ == "__main__":
    main(parse_args())
