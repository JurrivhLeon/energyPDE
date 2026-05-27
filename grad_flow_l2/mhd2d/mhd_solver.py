"""
Reference solver for periodic 2D incompressible resistive MHD on the unit torus.

The stored state is ``(omega, a)`` where
    -Delta psi = omega,
    u = (d_y psi, -d_x psi),
    b = (d_y a, -d_x a),
    j = -Delta a.

The evolution equations are
    omega_t + u . grad(omega) = b . grad(j) + nu * Delta(omega) + f,
    a_t     + u . grad(a)     = eta * Delta(a).

The default time integration is a fully explicit RK4 Fourier pseudospectral
scheme. Nonlinear products are evaluated on a 3/2 padded grid and truncated
back to the solver grid, which is the standard zero-padding dealiasing used
for quadratic nonlinearities on periodic domains.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _ensure_batch_field_2d(x: torch.Tensor, name: str) -> tuple[torch.Tensor, bool]:
    if x.dim() == 2:
        return x.unsqueeze(0), True
    if x.dim() == 3:
        return x, False
    raise ValueError(f"{name} must have shape (n_x,n_y) or (batch,n_x,n_y), got {tuple(x.shape)}")


def _ensure_batch_state_2d(state: torch.Tensor, name: str = "state") -> tuple[torch.Tensor, bool]:
    if state.dim() == 3 and int(state.shape[0]) == 2:
        return state.unsqueeze(0), True
    if state.dim() == 4 and int(state.shape[1]) == 2:
        return state, False
    raise ValueError(
        f"{name} must have shape (2,n_x,n_y) or (batch,2,n_x,n_y), got {tuple(state.shape)}"
    )


def project_zero_mean_2d(field: torch.Tensor) -> torch.Tensor:
    field_b, squeeze = _ensure_batch_field_2d(field, name="field")
    out = field_b - field_b.mean(dim=(-2, -1), keepdim=True)
    return out.squeeze(0) if squeeze else out


def prepare_mhd2d_periodic_spectral_cache(
    n_x: int,
    n_y: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    dealias_factor: float = 1.5,
    include_dealias_cache: bool = True,
) -> Dict[str, torch.Tensor]:
    if n_x < 2 or n_y < 2:
        raise ValueError("n_x and n_y must both be >= 2")
    kx_1d = 2.0 * torch.pi * torch.fft.fftfreq(n_x, d=1.0 / float(n_x), device=device).to(dtype=dtype)
    ky_1d = 2.0 * torch.pi * torch.fft.fftfreq(n_y, d=1.0 / float(n_y), device=device).to(dtype=dtype)
    kx = kx_1d.unsqueeze(1).expand(n_x, n_y)
    ky = ky_1d.unsqueeze(0).expand(n_x, n_y)
    laplace_eigs = kx.square() + ky.square()
    inv_laplace_eigs = torch.where(laplace_eigs > 0.0, 1.0 / laplace_eigs, torch.zeros_like(laplace_eigs))

    cache: Dict[str, torch.Tensor] = {
        "kx": kx,
        "ky": ky,
        "laplace_eigs": laplace_eigs,
        "inv_laplace_eigs": inv_laplace_eigs,
        "n_x": torch.tensor(n_x, device=device),
        "n_y": torch.tensor(n_y, device=device),
    }
    if include_dealias_cache and dealias_factor > 1.0:
        n_x_pad = int(round(float(dealias_factor) * float(n_x)))
        n_y_pad = int(round(float(dealias_factor) * float(n_y)))
        if n_x_pad <= n_x or n_y_pad <= n_y:
            raise ValueError("dealias_factor must produce a larger padded grid")
        cache["dealias_factor"] = torch.tensor(float(dealias_factor), device=device, dtype=dtype)
        cache["dealias_n_x"] = torch.tensor(n_x_pad, device=device)
        cache["dealias_n_y"] = torch.tensor(n_y_pad, device=device)
        cache["dealias_cache"] = prepare_mhd2d_periodic_spectral_cache(
            n_x=n_x_pad,
            n_y=n_y_pad,
            device=device,
            dtype=dtype,
            dealias_factor=1.0,
            include_dealias_cache=False,
        )
    return cache


def solve_poisson_from_vorticity(omega: torch.Tensor, cache: Dict[str, torch.Tensor]) -> torch.Tensor:
    omega_b, squeeze = _ensure_batch_field_2d(omega, name="omega")
    psi_hat = torch.fft.fft2(omega_b) * cache["inv_laplace_eigs"].unsqueeze(0)
    psi_hat[:, 0, 0] = 0.0
    psi = torch.fft.ifft2(psi_hat).real
    return psi.squeeze(0) if squeeze else psi


def streamfunction_to_velocity(psi: torch.Tensor, cache: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    psi_b, squeeze = _ensure_batch_field_2d(psi, name="psi")
    psi_hat = torch.fft.fft2(psi_b)
    u_x = torch.fft.ifft2(1j * cache["ky"].unsqueeze(0) * psi_hat).real
    u_y = torch.fft.ifft2(-1j * cache["kx"].unsqueeze(0) * psi_hat).real
    if squeeze:
        return u_x.squeeze(0), u_y.squeeze(0)
    return u_x, u_y


def magnetic_potential_to_field_and_current(
    a: torch.Tensor,
    cache: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a_b, squeeze = _ensure_batch_field_2d(a, name="a")
    a_hat = torch.fft.fft2(a_b)
    b_x = torch.fft.ifft2(1j * cache["ky"].unsqueeze(0) * a_hat).real
    b_y = torch.fft.ifft2(-1j * cache["kx"].unsqueeze(0) * a_hat).real
    j = torch.fft.ifft2(cache["laplace_eigs"].unsqueeze(0) * a_hat).real
    if squeeze:
        return b_x.squeeze(0), b_y.squeeze(0), j.squeeze(0)
    return b_x, b_y, j


def spectral_gradient(field: torch.Tensor, cache: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    field_b, squeeze = _ensure_batch_field_2d(field, name="field")
    field_hat = torch.fft.fft2(field_b)
    d_x = torch.fft.ifft2(1j * cache["kx"].unsqueeze(0) * field_hat).real
    d_y = torch.fft.ifft2(1j * cache["ky"].unsqueeze(0) * field_hat).real
    if squeeze:
        return d_x.squeeze(0), d_y.squeeze(0)
    return d_x, d_y


def velocity_and_magnetic_field(
    state: torch.Tensor,
    cache: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    state_b, squeeze = _ensure_batch_state_2d(state)
    omega = state_b[:, 0]
    a = state_b[:, 1]
    psi = solve_poisson_from_vorticity(omega, cache)
    u_x, u_y = streamfunction_to_velocity(psi, cache)
    b_x, b_y, j = magnetic_potential_to_field_and_current(a, cache)
    if squeeze:
        return u_x.squeeze(0), u_y.squeeze(0), b_x.squeeze(0), b_y.squeeze(0), j.squeeze(0)
    return u_x, u_y, b_x, b_y, j


def _pad_spectral_hat_2d(field_hat: torch.Tensor, target_n_x: int, target_n_y: int) -> torch.Tensor:
    """Zero-pad unnormalised FFT coefficients and preserve physical amplitudes."""
    n_x = int(field_hat.shape[-2])
    n_y = int(field_hat.shape[-1])
    if target_n_x < n_x or target_n_y < n_y:
        raise ValueError("target grid must be at least as large as input grid")
    if target_n_x == n_x and target_n_y == n_y:
        return field_hat
    shifted = torch.fft.fftshift(field_hat, dim=(-2, -1))
    pad_x_left = (target_n_x - n_x) // 2
    pad_x_right = target_n_x - n_x - pad_x_left
    pad_y_left = (target_n_y - n_y) // 2
    pad_y_right = target_n_y - n_y - pad_y_left
    padded = F.pad(shifted, (pad_y_left, pad_y_right, pad_x_left, pad_x_right))
    padded = torch.fft.ifftshift(padded, dim=(-2, -1))
    return padded * (float(target_n_x * target_n_y) / float(n_x * n_y))


def _truncate_spectral_hat_2d(field_hat: torch.Tensor, target_n_x: int, target_n_y: int) -> torch.Tensor:
    """Crop unnormalised FFT coefficients and preserve physical amplitudes."""
    n_x = int(field_hat.shape[-2])
    n_y = int(field_hat.shape[-1])
    if target_n_x > n_x or target_n_y > n_y:
        raise ValueError("target grid must be no larger than input grid")
    if target_n_x == n_x and target_n_y == n_y:
        return field_hat
    shifted = torch.fft.fftshift(field_hat, dim=(-2, -1))
    start_x = (n_x - target_n_x) // 2
    start_y = (n_y - target_n_y) // 2
    cropped = shifted[..., start_x : start_x + target_n_x, start_y : start_y + target_n_y]
    cropped = torch.fft.ifftshift(cropped, dim=(-2, -1))
    return cropped * (float(target_n_x * target_n_y) / float(n_x * n_y))


def _state_to_padded_grid(
    state_b: torch.Tensor,
    cache: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    pad_cache = cache.get("dealias_cache", cache)
    n_x_pad = int(pad_cache["kx"].shape[-2])
    n_y_pad = int(pad_cache["kx"].shape[-1])
    omega_hat_pad = _pad_spectral_hat_2d(torch.fft.fft2(state_b[:, 0]), n_x_pad, n_y_pad)
    a_hat_pad = _pad_spectral_hat_2d(torch.fft.fft2(state_b[:, 1]), n_x_pad, n_y_pad)
    state_pad = torch.stack(
        [torch.fft.ifft2(omega_hat_pad).real, torch.fft.ifft2(a_hat_pad).real],
        dim=1,
    )
    return state_pad, pad_cache


def nonlinear_terms_hat(
    state: torch.Tensor,
    cache: Dict[str, torch.Tensor],
    dealiased: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_b, squeeze = _ensure_batch_state_2d(state)
    target_n_x = int(state_b.shape[-2])
    target_n_y = int(state_b.shape[-1])
    if dealiased:
        state_eval, eval_cache = _state_to_padded_grid(state_b, cache)
    else:
        state_eval, eval_cache = state_b, cache

    omega = state_eval[:, 0]
    a = state_eval[:, 1]
    u_x, u_y, b_x, b_y, j = velocity_and_magnetic_field(state_eval, eval_cache)
    omega_x, omega_y = spectral_gradient(omega, eval_cache)
    a_x, a_y = spectral_gradient(a, eval_cache)
    j_x, j_y = spectral_gradient(j, eval_cache)

    omega_rhs = -(u_x * omega_x + u_y * omega_y) + (b_x * j_x + b_y * j_y)
    a_rhs = -(u_x * a_x + u_y * a_y)
    omega_hat = torch.fft.fft2(omega_rhs)
    a_hat = torch.fft.fft2(a_rhs)
    if tuple(omega_hat.shape[-2:]) != (target_n_x, target_n_y):
        omega_hat = _truncate_spectral_hat_2d(omega_hat, target_n_x, target_n_y)
        a_hat = _truncate_spectral_hat_2d(a_hat, target_n_x, target_n_y)
    omega_hat[:, 0, 0] = 0.0
    a_hat[:, 0, 0] = 0.0
    if squeeze:
        return omega_hat.squeeze(0), a_hat.squeeze(0)
    return omega_hat, a_hat


def crank_nicolson_step(
    state: torch.Tensor,
    dt: float,
    nu: float,
    eta: float,
    cache: Dict[str, torch.Tensor],
    forcing_hat: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if dt <= 0.0:
        raise ValueError("dt must be > 0")
    if nu < 0.0 or eta < 0.0:
        raise ValueError("nu and eta must be >= 0")
    state_b, squeeze = _ensure_batch_state_2d(state)
    omega_hat = torch.fft.fft2(state_b[:, 0])
    a_hat = torch.fft.fft2(state_b[:, 1])
    n_omega_hat, n_a_hat = nonlinear_terms_hat(state_b, cache, dealiased=True)
    lam = cache["laplace_eigs"].unsqueeze(0)

    rhs_omega = (1.0 - 0.5 * float(dt) * float(nu) * lam) * omega_hat + float(dt) * n_omega_hat
    if forcing_hat is not None:
        rhs_omega = rhs_omega + float(dt) * forcing_hat
    rhs_a = (1.0 - 0.5 * float(dt) * float(eta) * lam) * a_hat + float(dt) * n_a_hat

    next_omega_hat = rhs_omega / (1.0 + 0.5 * float(dt) * float(nu) * lam)
    next_a_hat = rhs_a / (1.0 + 0.5 * float(dt) * float(eta) * lam)
    next_omega_hat[:, 0, 0] = 0.0
    next_a_hat[:, 0, 0] = 0.0
    out = torch.stack(
        [torch.fft.ifft2(next_omega_hat).real, torch.fft.ifft2(next_a_hat).real],
        dim=1,
    )
    return out.squeeze(0) if squeeze else out


def rhs_hat_mhd2d(
    state: torch.Tensor,
    nu: float,
    eta: float,
    cache: Dict[str, torch.Tensor],
    forcing_hat: Optional[torch.Tensor] = None,
    dealiased: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_b, squeeze = _ensure_batch_state_2d(state)
    omega_hat = torch.fft.fft2(state_b[:, 0])
    a_hat = torch.fft.fft2(state_b[:, 1])
    n_omega_hat, n_a_hat = nonlinear_terms_hat(state_b, cache, dealiased=dealiased)
    lam = cache["laplace_eigs"].unsqueeze(0)
    rhs_omega_hat = n_omega_hat - float(nu) * lam * omega_hat
    rhs_a_hat = n_a_hat - float(eta) * lam * a_hat
    if forcing_hat is not None:
        rhs_omega_hat = rhs_omega_hat + forcing_hat
    rhs_omega_hat[:, 0, 0] = 0.0
    rhs_a_hat[:, 0, 0] = 0.0
    if squeeze:
        return rhs_omega_hat.squeeze(0), rhs_a_hat.squeeze(0)
    return rhs_omega_hat, rhs_a_hat


def rk4_step(
    state: torch.Tensor,
    dt: float,
    nu: float,
    eta: float,
    cache: Dict[str, torch.Tensor],
    forcing_hat: Optional[torch.Tensor] = None,
    dealiased: bool = True,
) -> torch.Tensor:
    if dt <= 0.0:
        raise ValueError("dt must be > 0")
    if nu < 0.0 or eta < 0.0:
        raise ValueError("nu and eta must be >= 0")
    state_b, squeeze = _ensure_batch_state_2d(state)

    def add_hat(base: torch.Tensor, k_omega: torch.Tensor, k_a: torch.Tensor, alpha: float) -> torch.Tensor:
        omega_hat = torch.fft.fft2(base[:, 0]) + float(alpha) * float(dt) * k_omega
        a_hat = torch.fft.fft2(base[:, 1]) + float(alpha) * float(dt) * k_a
        omega_hat[:, 0, 0] = 0.0
        a_hat[:, 0, 0] = 0.0
        return torch.stack([torch.fft.ifft2(omega_hat).real, torch.fft.ifft2(a_hat).real], dim=1)

    k1_omega, k1_a = rhs_hat_mhd2d(state_b, nu, eta, cache, forcing_hat=forcing_hat, dealiased=dealiased)
    s2 = add_hat(state_b, k1_omega, k1_a, 0.5)
    k2_omega, k2_a = rhs_hat_mhd2d(s2, nu, eta, cache, forcing_hat=forcing_hat, dealiased=dealiased)
    s3 = add_hat(state_b, k2_omega, k2_a, 0.5)
    k3_omega, k3_a = rhs_hat_mhd2d(s3, nu, eta, cache, forcing_hat=forcing_hat, dealiased=dealiased)
    s4 = add_hat(state_b, k3_omega, k3_a, 1.0)
    k4_omega, k4_a = rhs_hat_mhd2d(s4, nu, eta, cache, forcing_hat=forcing_hat, dealiased=dealiased)

    omega_hat = torch.fft.fft2(state_b[:, 0]) + (float(dt) / 6.0) * (
        k1_omega + 2.0 * k2_omega + 2.0 * k3_omega + k4_omega
    )
    a_hat = torch.fft.fft2(state_b[:, 1]) + (float(dt) / 6.0) * (
        k1_a + 2.0 * k2_a + 2.0 * k3_a + k4_a
    )
    omega_hat[:, 0, 0] = 0.0
    a_hat[:, 0, 0] = 0.0
    out = torch.stack([torch.fft.ifft2(omega_hat).real, torch.fft.ifft2(a_hat).real], dim=1)
    return out.squeeze(0) if squeeze else out


def solve_mhd2d_trajectory_pseudospectral(
    u0: torch.Tensor,
    forcing: Optional[torch.Tensor],
    t_final: float,
    dt: float,
    record_dt: float,
    nu: float,
    eta: float,
    solver_cache: Optional[Dict[str, torch.Tensor]] = None,
    progress_callback=None,
    time_integrator: str = "rk4",
    dealias_factor: float = 1.5,
) -> torch.Tensor:
    if t_final <= 0.0:
        raise ValueError("t_final must be > 0")
    if dt <= 0.0 or record_dt <= 0.0:
        raise ValueError("dt and record_dt must be > 0")
    state, squeeze = _ensure_batch_state_2d(u0, name="u0")
    batch_size, _, n_x, n_y = state.shape
    if solver_cache is None:
        solver_cache = prepare_mhd2d_periodic_spectral_cache(
            n_x=n_x,
            n_y=n_y,
            device=str(state.device),
            dtype=state.dtype,
            dealias_factor=dealias_factor,
        )

    forcing_hat = None
    if forcing is not None:
        f_b, _ = _ensure_batch_field_2d(forcing, name="forcing")
        f_b = project_zero_mean_2d(f_b).to(device=state.device, dtype=state.dtype)
        if int(f_b.shape[0]) == 1 and batch_size > 1:
            f_b = f_b.expand(batch_size, -1, -1)
        if int(f_b.shape[0]) != batch_size or tuple(f_b.shape[1:]) != (n_x, n_y):
            raise ValueError("forcing must match the state batch/spatial shape or have batch size 1")
        forcing_hat = torch.fft.fft2(f_b)
        forcing_hat[:, 0, 0] = 0.0

    steps_per_record = int(round(float(record_dt) / float(dt)))
    if steps_per_record < 1 or abs(float(record_dt) - steps_per_record * float(dt)) > 1e-10:
        raise ValueError("record_dt must be an integer multiple of dt")
    n_records = int(round(float(t_final) / float(record_dt)))
    if abs(float(t_final) - n_records * float(record_dt)) > 1e-10:
        raise ValueError("t_final must be an integer multiple of record_dt")
    if time_integrator not in {"rk4", "cn", "crank_nicolson"}:
        raise ValueError("time_integrator must be one of {'rk4', 'cn', 'crank_nicolson'}")

    state = state.clone()
    state[:, 0] = project_zero_mean_2d(state[:, 0])
    state[:, 1] = project_zero_mean_2d(state[:, 1])
    states = [state.clone()]
    for _ in range(n_records):
        for _ in range(steps_per_record):
            if time_integrator == "rk4":
                state = rk4_step(
                    state=state,
                    dt=dt,
                    nu=nu,
                    eta=eta,
                    cache=solver_cache,
                    forcing_hat=forcing_hat,
                    dealiased=dealias_factor > 1.0,
                )
            else:
                state = crank_nicolson_step(
                    state=state,
                    dt=dt,
                    nu=nu,
                    eta=eta,
                    cache=solver_cache,
                    forcing_hat=forcing_hat,
                )
        if not torch.isfinite(state).all():
            raise FloatingPointError("Non-finite MHD state encountered during trajectory solve")
        states.append(state.clone())
        if progress_callback is not None:
            progress_callback(1)
    traj = torch.stack(states, dim=1)
    return traj.squeeze(0) if squeeze else traj
