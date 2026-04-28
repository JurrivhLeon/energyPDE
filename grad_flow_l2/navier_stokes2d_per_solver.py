"""
Reference solver for 2D viscous incompressible Navier-Stokes in vorticity form
on the unit torus:

    omega_t + u · grad(omega) = nu * Delta omega + f,
    div u = 0,

with periodic boundary conditions in both directions.

We represent the state on a full periodic grid with shape (n_x, n_y),
corresponding to x_i = i / n_x and y_j = j / n_y.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Optional

import torch


def _ensure_batch_2d(x: torch.Tensor, name: str = "tensor") -> tuple[torch.Tensor, bool]:
    if x.dim() == 2:
        return x.unsqueeze(0), True
    if x.dim() == 3:
        return x, False
    raise ValueError(f"{name} must have shape (n_x,n_y) or (batch,n_x,n_y), got {tuple(x.shape)}")


def _match_batch_2d(x: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
    x_b, _ = _ensure_batch_2d(x, name=name)
    if int(x_b.shape[0]) == batch_size:
        return x_b
    if int(x_b.shape[0]) == 1 and batch_size > 1:
        return x_b.expand(batch_size, -1, -1)
    raise ValueError(f"{name} batch size must match omega batch size or be 1")


def _fft2(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fft2(x)


def _ifft2(x_hat: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifft2(x_hat)


def to_periodic_field_2d(field: torch.Tensor, n_x: int, n_y: int) -> torch.Tensor:
    """
    Accept a periodic field stored either on the full grid (n_x, n_y)
    or in legacy zero-padded form (n_x+2, n_y+2), and return the periodic
    representation on the full grid.
    """
    if field.dim() == 2:
        if field.shape == (n_x, n_y):
            return field.unsqueeze(0)
        if field.shape == (n_x + 2, n_y + 2):
            return field[1:-1, 1:-1].unsqueeze(0)
        raise ValueError(f"2D field shape must be ({n_x},{n_y}) or ({n_x+2},{n_y+2}), got {tuple(field.shape)}")
    if field.dim() == 3:
        if field.shape[1:] == (n_x, n_y):
            return field
        if field.shape[1:] == (n_x + 2, n_y + 2):
            return field[:, 1:-1, 1:-1]
        raise ValueError(
            f"3D field shape must be (batch,{n_x},{n_y}) or (batch,{n_x+2},{n_y+2}), got {tuple(field.shape)}"
        )
    raise ValueError("field must have 2 or 3 dimensions")


def project_zero_mean_2d(field: torch.Tensor) -> torch.Tensor:
    """
    Remove the spatial mean from a periodic field.
    """
    field_b, squeeze = _ensure_batch_2d(field, name="field")
    field_b = field_b - torch.mean(field_b, dim=(-2, -1), keepdim=True)
    if squeeze:
        return field_b.squeeze(0)
    return field_b


def spectral_truncate_periodic_field_2d(
    field: torch.Tensor,
    target_n_x: int,
    target_n_y: int,
) -> torch.Tensor:
    """
    Downsample a periodic field by truncating high Fourier modes.

    The crop is centered in Fourier space and the coefficients are rescaled so
    that the retained low-frequency modes represent the same continuous field
    on the smaller grid.
    """
    field_b, squeeze = _ensure_batch_2d(field, name="field")
    n_x, n_y = int(field_b.shape[-2]), int(field_b.shape[-1])
    if target_n_x < 1 or target_n_y < 1:
        raise ValueError("target_n_x and target_n_y must be >= 1")
    if target_n_x > n_x or target_n_y > n_y:
        raise ValueError("target grid must not be larger than the source grid")
    if target_n_x == n_x and target_n_y == n_y:
        return field_b.squeeze(0) if squeeze else field_b.clone()

    field_hat = torch.fft.fftshift(_fft2(field_b), dim=(-2, -1))
    start_x = (n_x - target_n_x) // 2
    start_y = (n_y - target_n_y) // 2
    field_hat_small = field_hat[:, start_x : start_x + target_n_x, start_y : start_y + target_n_y]
    scale = (float(target_n_x) * float(target_n_y)) / (float(n_x) * float(n_y))
    field_small = torch.fft.ifft2(
        torch.fft.ifftshift(field_hat_small * scale, dim=(-2, -1)),
        s=(target_n_x, target_n_y),
    ).real
    if squeeze:
        return field_small.squeeze(0)
    return field_small


def sample_periodic_gaussian_field_2d(
    n_x: int,
    n_y: int,
    n_samples: int = 1,
    spectrum_scale: float = 5.0 ** 1.5,
    spectrum_shift: float = 25.0,
    spectrum_power: float = 2.5,
    zero_mean: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample a periodic Gaussian random field with spectral covariance

        spectrum_scale * (|k|^2 + spectrum_shift)^(-spectrum_power).

    The zero mode is removed when zero_mean=True to match the periodic
    incompressible Navier-Stokes setting.
    """
    if n_x < 1 or n_y < 1:
        raise ValueError("n_x and n_y must be >= 1")
    if spectrum_scale <= 0 or spectrum_shift <= 0 or spectrum_power <= 0:
        raise ValueError("spectrum parameters must be > 0")

    k_x = torch.fft.fftfreq(n_x, d=1.0 / float(n_x), device=device, dtype=dtype)
    k_y = torch.fft.rfftfreq(n_y, d=1.0 / float(n_y), device=device, dtype=dtype)
    k2 = k_x.unsqueeze(1) ** 2 + k_y.unsqueeze(0) ** 2
    power = float(spectrum_scale) * torch.pow(k2 + float(spectrum_shift), -float(spectrum_power))

    real = torch.randn(n_samples, n_x, n_y // 2 + 1, device=device, dtype=dtype)
    imag = torch.randn(n_samples, n_x, n_y // 2 + 1, device=device, dtype=dtype)
    noise = torch.complex(real, imag) / (2.0 ** 0.5)
    spectrum = noise * torch.sqrt(power).unsqueeze(0)
    if zero_mean:
        spectrum[:, 0, 0] = 0.0
    samples = torch.fft.irfft2(spectrum, s=(n_x, n_y)) * ((n_x * n_y) ** 0.5)
    samples = project_zero_mean_2d(samples) if zero_mean else samples
    return samples


def prepare_ns2d_periodic_spectral_cache(
    n_x: int,
    n_y: int,
    h_x: float,
    h_y: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """
    Prepare Fourier wave numbers and Laplacian eigenvalues for the unit torus.
    """
    if n_x < 1 or n_y < 1:
        raise ValueError("n_x and n_y must be >= 1")
    if h_x <= 0 or h_y <= 0:
        raise ValueError("h_x and h_y must be > 0")

    k_x = 2.0 * torch.pi * torch.fft.fftfreq(n_x, d=1.0 / float(n_x), device=device).to(dtype=dtype)
    k_y = 2.0 * torch.pi * torch.fft.fftfreq(n_y, d=1.0 / float(n_y), device=device).to(dtype=dtype)
    kx_grid = k_x.unsqueeze(1).expand(n_x, n_y)
    ky_grid = k_y.unsqueeze(0).expand(n_x, n_y)
    laplace_eigs = kx_grid * kx_grid + ky_grid * ky_grid
    inv_laplace_eigs = torch.where(laplace_eigs > 0.0, 1.0 / laplace_eigs, torch.zeros_like(laplace_eigs))
    dealias_cutoff_x = float(n_x) / 3.0
    dealias_cutoff_y = float(n_y) / 3.0
    dealias_mask = (
        (torch.abs(torch.fft.fftfreq(n_x, d=1.0 / float(n_x), device=device).to(dtype=dtype)).unsqueeze(1).expand(n_x, n_y)
         <= dealias_cutoff_x)
        & (torch.abs(torch.fft.fftfreq(n_y, d=1.0 / float(n_y), device=device).to(dtype=dtype)).unsqueeze(0).expand(n_x, n_y)
           <= dealias_cutoff_y)
    )

    return {
        "kx_grid": kx_grid,
        "ky_grid": ky_grid,
        "laplace_eigs": laplace_eigs,
        "inv_laplace_eigs": inv_laplace_eigs,
        "dealias_mask": dealias_mask,
        "h_x": torch.tensor(float(h_x), device=device, dtype=dtype),
        "h_y": torch.tensor(float(h_y), device=device, dtype=dtype),
    }


def solve_poisson_periodic_2d(
    rhs: torch.Tensor,
    cache: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Solve -Delta psi = rhs on the periodic grid with zero-mean gauge.
    """
    rhs_b, squeeze = _ensure_batch_2d(rhs, name="rhs")
    rhs_hat = _fft2(rhs_b)
    psi_hat = rhs_hat * cache["inv_laplace_eigs"].unsqueeze(0)
    psi_hat[:, 0, 0] = 0.0
    psi = _ifft2(psi_hat).real
    if squeeze:
        return psi.squeeze(0)
    return psi


def solve_implicit_diffusion_periodic_2d(
    rhs: torch.Tensor,
    dt: float,
    nu: float,
    cache: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Solve (I - dt*nu*Delta) omega_next = rhs with periodic boundaries.
    """
    rhs_b, squeeze = _ensure_batch_2d(rhs, name="rhs")
    rhs_hat = _fft2(rhs_b)
    denom = 1.0 + float(dt) * float(nu) * cache["laplace_eigs"].unsqueeze(0)
    omega_hat = rhs_hat / denom
    omega_next = _ifft2(omega_hat).real
    if squeeze:
        return omega_next.squeeze(0)
    return omega_next


def streamfunction_to_velocity_periodic(
    psi: torch.Tensor,
    cache: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the incompressible velocity u = (d_y psi, -d_x psi).
    """
    psi_b, squeeze = _ensure_batch_2d(psi, name="psi")
    psi_hat = _fft2(psi_b)
    v_x_hat = 1j * cache["ky_grid"].unsqueeze(0) * psi_hat
    v_y_hat = -1j * cache["kx_grid"].unsqueeze(0) * psi_hat
    v_x = _ifft2(v_x_hat).real
    v_y = _ifft2(v_y_hat).real
    if squeeze:
        return v_x.squeeze(0), v_y.squeeze(0)
    return v_x, v_y


def spectral_gradient_periodic_2d(
    field: torch.Tensor,
    cache: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute spectral gradients of a periodic field.
    """
    field_b, squeeze = _ensure_batch_2d(field, name="field")
    field_hat = _fft2(field_b)
    dx_hat = 1j * cache["kx_grid"].unsqueeze(0) * field_hat
    dy_hat = 1j * cache["ky_grid"].unsqueeze(0) * field_hat
    d_x = _ifft2(dx_hat).real
    d_y = _ifft2(dy_hat).real
    if squeeze:
        return d_x.squeeze(0), d_y.squeeze(0)
    return d_x, d_y


def pseudospectral_advection_term_hat_periodic(
    omega: torch.Tensor,
    cache: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Compute the Fourier coefficients of u · grad(omega) using a pseudospectral
    step and de-alias the nonlinear term in Fourier space.
    """
    omega_b, squeeze = _ensure_batch_2d(omega, name="omega")
    omega_hat = _fft2(omega_b)

    psi_hat = omega_hat * cache["inv_laplace_eigs"].unsqueeze(0)
    psi_hat[:, 0, 0] = 0.0
    v_x_hat = 1j * cache["ky_grid"].unsqueeze(0) * psi_hat
    v_y_hat = -1j * cache["kx_grid"].unsqueeze(0) * psi_hat
    v_x = _ifft2(v_x_hat).real
    v_y = _ifft2(v_y_hat).real

    omega_x = _ifft2(1j * cache["kx_grid"].unsqueeze(0) * omega_hat).real
    omega_y = _ifft2(1j * cache["ky_grid"].unsqueeze(0) * omega_hat).real
    adv = v_x * omega_x + v_y * omega_y
    adv_hat = _fft2(adv)
    adv_hat = adv_hat * cache["dealias_mask"].unsqueeze(0)
    if squeeze:
        return adv_hat.squeeze(0)
    return adv_hat


def pseudospectral_crank_nicolson_step_periodic(
    omega: torch.Tensor,
    dt: float,
    nu: float,
    cache: Dict[str, torch.Tensor],
    forcing_hat: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    One pseudospectral Crank-Nicolson step for periodic vorticity dynamics.

    The nonlinear advection term is explicit, while diffusion is advanced with
    Crank-Nicolson:

        (I - 0.5 dt nu Delta) omega^{n+1}
            = (I + 0.5 dt nu Delta) omega^n - dt N(omega^n) + dt f.
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if nu < 0:
        raise ValueError("nu must be >= 0")

    omega_b, squeeze = _ensure_batch_2d(omega, name="omega")
    omega_hat = _fft2(omega_b)
    adv_hat = pseudospectral_advection_term_hat_periodic(omega_b, cache=cache)

    rhs_hat = (1.0 - 0.5 * float(dt) * float(nu) * cache["laplace_eigs"].unsqueeze(0)) * omega_hat
    rhs_hat = rhs_hat - float(dt) * adv_hat
    if forcing_hat is not None:
        rhs_hat = rhs_hat + float(dt) * forcing_hat

    denom = 1.0 + 0.5 * float(dt) * float(nu) * cache["laplace_eigs"].unsqueeze(0)
    omega_next_hat = rhs_hat / denom
    omega_next_hat[:, 0, 0] = 0.0
    omega_next = _ifft2(omega_next_hat).real
    omega_next = project_zero_mean_2d(omega_next)
    if squeeze:
        return omega_next.squeeze(0)
    return omega_next


def solve_navier_stokes_vorticity_trajectory_pseudospectral(
    u0: torch.Tensor,
    forcing: Optional[torch.Tensor],
    t_final: float,
    dt: float = 1e-4,
    record_dt: float = 1.0,
    nu: float = 1e-3,
    solver_cache: Optional[Dict[str, torch.Tensor]] = None,
    max_step_per_record: Optional[int] = None,
) -> torch.Tensor:
    """
    Integrate periodic Navier-Stokes in vorticity form on a full periodic grid.

    The returned trajectory contains the state at t=0 and then every
    `record_dt` time units.
    """
    if t_final <= 0:
        raise ValueError("t_final must be > 0")
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if record_dt <= 0:
        raise ValueError("record_dt must be > 0")
    if nu < 0:
        raise ValueError("nu must be >= 0")

    u0_b, squeeze = _ensure_batch_2d(u0, name="u0")
    n_x, n_y = int(u0_b.shape[-2]), int(u0_b.shape[-1])
    h_x = 1.0 / float(n_x)
    h_y = 1.0 / float(n_y)

    if solver_cache is None:
        solver_cache = prepare_ns2d_periodic_spectral_cache(
            n_x=n_x,
            n_y=n_y,
            h_x=h_x,
            h_y=h_y,
            device=str(u0_b.device),
            dtype=u0_b.dtype,
        )

    forcing_hat = None
    if forcing is not None:
        forcing_b = to_periodic_field_2d(forcing, n_x=n_x, n_y=n_y).to(device=u0_b.device, dtype=u0_b.dtype)
        forcing_b = project_zero_mean_2d(forcing_b)
        if forcing_b.shape[0] == 1 and u0_b.shape[0] > 1:
            forcing_b = forcing_b.expand(u0_b.shape[0], -1, -1)
        if forcing_b.shape[0] != u0_b.shape[0]:
            raise ValueError("forcing batch size must match u0 batch size or be 1")
        forcing_hat = _fft2(forcing_b)
        forcing_hat[:, 0, 0] = 0.0

    steps_per_record = int(round(float(record_dt) / float(dt)))
    if steps_per_record < 1:
        raise ValueError("record_dt must be at least one dt")
    if abs(float(record_dt) - steps_per_record * float(dt)) > 1e-10:
        raise ValueError("record_dt must be an integer multiple of dt")

    n_records = int(round(float(t_final) / float(record_dt)))
    if abs(float(t_final) - n_records * float(record_dt)) > 1e-10:
        raise ValueError("t_final must be an integer multiple of record_dt")

    omega = project_zero_mean_2d(u0_b.to(dtype=u0_b.dtype).clone())
    states = [omega.clone()]

    for _ in range(n_records):
        for _ in range(steps_per_record):
            omega = pseudospectral_crank_nicolson_step_periodic(
                omega=omega,
                dt=dt,
                nu=nu,
                cache=solver_cache,
                forcing_hat=forcing_hat,
            )
        states.append(omega.clone())

    traj = torch.stack(states, dim=1)
    if squeeze:
        return traj.squeeze(0)
    return traj


def vorticity_to_velocity(
    omega: torch.Tensor,
    cache: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute (psi, v_x, v_y) from periodic vorticity.
    """
    psi = solve_poisson_periodic_2d(omega, cache)
    v_x, v_y = streamfunction_to_velocity_periodic(psi, cache)
    return psi, v_x, v_y


def upwind_advection_term_periodic(
    omega: torch.Tensor,
    v_x: torch.Tensor,
    v_y: torch.Tensor,
    h_x: float,
    h_y: float,
) -> torch.Tensor:
    """
    First-order upwind approximation of u · grad(omega) with periodic wrap-around.
    """
    omega_b, squeeze = _ensure_batch_2d(omega, name="omega")
    vx_b = _match_batch_2d(v_x, int(omega_b.shape[0]), name="v_x")
    vy_b = _match_batch_2d(v_y, int(omega_b.shape[0]), name="v_y")

    w_left = torch.roll(omega_b, shifts=1, dims=-2)
    w_right = torch.roll(omega_b, shifts=-1, dims=-2)
    w_down = torch.roll(omega_b, shifts=1, dims=-1)
    w_up = torch.roll(omega_b, shifts=-1, dims=-1)

    dw_dx_backward = (omega_b - w_left) / float(h_x)
    dw_dx_forward = (w_right - omega_b) / float(h_x)
    dw_dy_backward = (omega_b - w_down) / float(h_y)
    dw_dy_forward = (w_up - omega_b) / float(h_y)

    dw_dx = torch.where(vx_b >= 0.0, dw_dx_backward, dw_dx_forward)
    dw_dy = torch.where(vy_b >= 0.0, dw_dy_backward, dw_dy_forward)
    adv = vx_b * dw_dx + vy_b * dw_dy

    if squeeze:
        return adv.squeeze(0)
    return adv


def navier_stokes_vorticity_step(
    omega: torch.Tensor,
    dt: float,
    nu: float,
    cache: Dict[str, torch.Tensor],
    forcing: Optional[torch.Tensor] = None,
    velocity: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    One split step:
      1) explicit upwind advection + forcing
      2) implicit diffusion
    """
    omega_b, squeeze = _ensure_batch_2d(omega, name="omega")
    n_x, n_y = int(omega_b.shape[-2]), int(omega_b.shape[-1])

    if velocity is None:
        _, v_x, v_y = vorticity_to_velocity(omega_b, cache)
    else:
        v_x, v_y = velocity
        v_x = _match_batch_2d(v_x, int(omega_b.shape[0]), name="v_x")
        v_y = _match_batch_2d(v_y, int(omega_b.shape[0]), name="v_y")

    adv = upwind_advection_term_periodic(
        omega=omega_b,
        v_x=v_x,
        v_y=v_y,
        h_x=float(cache["h_x"].item()),
        h_y=float(cache["h_y"].item()),
    )

    if forcing is None:
        forcing_int = 0.0
    else:
        forcing_int = to_periodic_field_2d(forcing, n_x=n_x, n_y=n_y).to(device=omega_b.device, dtype=omega_b.dtype)
        forcing_int = project_zero_mean_2d(forcing_int)
        if forcing_int.shape[0] == 1 and omega_b.shape[0] > 1:
            forcing_int = forcing_int.expand(omega_b.shape[0], -1, -1)
        if forcing_int.shape[0] != omega_b.shape[0]:
            raise ValueError("forcing batch size must match omega batch size or be 1")

    rhs = omega_b - float(dt) * adv + float(dt) * forcing_int
    omega_next = solve_implicit_diffusion_periodic_2d(rhs=rhs, dt=dt, nu=nu, cache=cache)
    omega_next = project_zero_mean_2d(omega_next)
    if squeeze:
        return omega_next.squeeze(0)
    return omega_next


def solve_navier_stokes_vorticity_trajectory(
    u0: torch.Tensor,
    forcing: Optional[torch.Tensor],
    n_steps: int,
    nu: float,
    t_final: float = 1.0,
    cfl_adv: float = 0.45,
    max_substeps_per_step: int = 4000,
    max_dt_substep: Optional[float] = None,
    solver_cache: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Integrate and return [omega_0, ..., omega_K] on the periodic grid.
    """
    if nu <= 0:
        raise ValueError("nu must be > 0")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if t_final <= 0:
        raise ValueError("t_final must be > 0")
    if cfl_adv <= 0:
        raise ValueError("cfl_adv must be > 0")

    u0_b, squeeze = _ensure_batch_2d(u0, name="u0")
    n_x, n_y = int(u0_b.shape[-2]), int(u0_b.shape[-1])
    h_x = 1.0 / float(n_x)
    h_y = 1.0 / float(n_y)

    if solver_cache is None:
        solver_cache = prepare_ns2d_periodic_spectral_cache(
            n_x=n_x,
            n_y=n_y,
            h_x=h_x,
            h_y=h_y,
            device=str(u0_b.device),
            dtype=u0_b.dtype,
        )

    forcing_int = None
    if forcing is not None:
        forcing_int = to_periodic_field_2d(forcing, n_x=n_x, n_y=n_y).to(device=u0_b.device, dtype=u0_b.dtype)
        forcing_int = project_zero_mean_2d(forcing_int)
        if forcing_int.shape[0] == 1 and u0_b.shape[0] > 1:
            forcing_int = forcing_int.expand(u0_b.shape[0], -1, -1)
        if forcing_int.shape[0] != u0_b.shape[0]:
            raise ValueError("forcing batch size must match u0 batch size or be 1")

    dt_macro = float(t_final) / float(n_steps)
    omega = project_zero_mean_2d(u0_b.clone())
    states = [omega.clone()]

    for _ in range(n_steps):
        remaining = dt_macro
        substeps = 0
        while remaining > 1e-14:
            _, v_x, v_y = vorticity_to_velocity(omega, solver_cache)
            speed = torch.abs(v_x) / h_x + torch.abs(v_y) / h_y
            max_speed = float(torch.max(speed).item())
            dt_adv = float(cfl_adv) / max(max_speed, 1e-8)
            dt_stable = dt_adv if max_dt_substep is None else min(dt_adv, float(max_dt_substep))
            dt_sub = min(remaining, max(1e-10, dt_stable))

            omega = navier_stokes_vorticity_step(
                omega=omega,
                dt=dt_sub,
                nu=nu,
                cache=solver_cache,
                forcing=forcing_int,
                velocity=(v_x, v_y),
            )
            omega = project_zero_mean_2d(omega)
            remaining -= dt_sub
            substeps += 1
            if substeps > max_substeps_per_step:
                raise RuntimeError(
                    "Exceeded max_substeps_per_step; reduce amplitudes, increase n_steps, or relax CFL settings."
                )
        states.append(omega.clone())

    traj = torch.stack(states, dim=1)
    if squeeze:
        return traj.squeeze(0)
    return traj


def _sample_reference_initial_vorticity(
    n_x: int,
    n_y: int,
    amplitude: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    x = torch.arange(n_x, device=device, dtype=dtype) / float(n_x)
    y = torch.arange(n_y, device=device, dtype=dtype) / float(n_y)
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    field = torch.zeros(n_x, n_y, device=device, dtype=dtype)
    for k_x in range(0, 4):
        for k_y in range(0, 4):
            if k_x == 0 and k_y == 0:
                continue
            coeff = torch.randn((), device=device, dtype=dtype) / float(1 + k_x + k_y)
            phase = 2.0 * torch.pi * torch.rand((), device=device, dtype=dtype)
            field = field + coeff * torch.sin(2.0 * torch.pi * (k_x * xx + k_y * yy) + phase)

    field = project_zero_mean_2d(field)
    max_abs = torch.max(torch.abs(field)) + 1e-8
    return float(amplitude) * field / max_abs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reference 2D periodic Navier-Stokes vorticity solver")
    parser.add_argument("--n-x", type=int, default=32, help="Number of grid points in x")
    parser.add_argument("--n-y", type=int, default=32, help="Number of grid points in y")
    parser.add_argument("--n-steps", type=int, default=10, help="Number of macro time steps on [0,t_final]")
    parser.add_argument("--t-final", type=float, default=1.0, help="Final simulation time")
    parser.add_argument("--nu", type=float, default=0.001, help="Viscosity coefficient")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampled initial condition")
    parser.add_argument("--u0-amplitude", type=float, default=2.0, help="Amplitude of sampled initial vorticity")
    parser.add_argument("--cfl-adv", type=float, default=0.45, help="Advection CFL safety factor")
    parser.add_argument("--max-substeps-per-step", type=int, default=4000)
    parser.add_argument(
        "--out-path",
        type=str,
        default="grad_flow_l2/outputs/navier_stokes2d_periodic_reference_sample.pt",
        help="Where to save {u0, u_traj, meta}",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.n_x < 2 or args.n_y < 2:
        raise ValueError("--n-x and --n-y must both be >= 2")
    if args.n_steps < 1:
        raise ValueError("--n-steps must be >= 1")
    if args.nu <= 0:
        raise ValueError("--nu must be > 0")
    if args.t_final <= 0:
        raise ValueError("--t-final must be > 0")

    torch.manual_seed(int(args.seed))
    u0 = _sample_reference_initial_vorticity(
        n_x=args.n_x,
        n_y=args.n_y,
        amplitude=float(args.u0_amplitude),
        device="cpu",
        dtype=torch.float32,
    )
    u_traj = solve_navier_stokes_vorticity_trajectory(
        u0=u0,
        forcing=None,
        n_steps=args.n_steps,
        nu=args.nu,
        t_final=args.t_final,
        cfl_adv=args.cfl_adv,
        max_substeps_per_step=args.max_substeps_per_step,
    )

    area = 1.0 / float(args.n_x * args.n_y)
    u0_l2 = torch.sqrt(area * torch.sum(u0 * u0)).item()
    u_final_l2 = torch.sqrt(area * torch.sum(u_traj[-1] * u_traj[-1])).item()
    print(
        "Reference solve complete: "
        f"u0_shape={tuple(u0.shape)}, "
        f"traj_shape={tuple(u_traj.shape)}, "
        f"u0_l2={u0_l2:.4f}, "
        f"uT_l2={u_final_l2:.4f}"
    )

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = {
        "u0": u0,
        "u_traj": u_traj,
        "meta": {
            "equation": "navier_stokes_2d_vorticity_periodic",
            "domain": "unit_torus",
            "n_x": int(args.n_x),
            "n_y": int(args.n_y),
            "n_steps": int(args.n_steps),
            "t_final": float(args.t_final),
            "nu": float(args.nu),
            "seed": int(args.seed),
            "cfl_adv": float(args.cfl_adv),
            "periodic": True,
        },
    }
    torch.save(payload, args.out_path)
    print(f"Saved reference trajectory: {args.out_path}")


if __name__ == "__main__":
    main(parse_args())
