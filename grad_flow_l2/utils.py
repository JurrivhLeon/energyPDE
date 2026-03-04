"""
Utility functions for 1D heat-equation gradient-flow experiments.

PDE:
    u_t = u_xx + f(x),   x in [0, 1], t in [0, 1]
with homogeneous Dirichlet boundary conditions.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


def pad_dirichlet_1d(u_interior: torch.Tensor) -> torch.Tensor:
    """
    Pad 1D interior state(s) with zero Dirichlet boundaries.

    Args:
        u_interior: shape (..., n_x)

    Returns:
        u_full: shape (..., n_x + 2) with u_full[..., 0] = u_full[..., -1] = 0
    """
    if u_interior.dim() < 1:
        raise ValueError("u_interior must have at least one dimension")
    zeros_shape = (*u_interior.shape[:-1], 1)
    zeros = torch.zeros(zeros_shape, device=u_interior.device, dtype=u_interior.dtype)
    return torch.cat([zeros, u_interior, zeros], dim=-1)


def check_dirichlet_1d(u_interior: torch.Tensor, atol: float = 0.0) -> bool:
    """
    Check Dirichlet BC after padding interior values.
    """
    u_full = pad_dirichlet_1d(u_interior)
    left_ok = torch.all(torch.abs(u_full[..., 0]) <= atol)
    right_ok = torch.all(torch.abs(u_full[..., -1]) <= atol)
    return bool(left_ok and right_ok)


def build_laplacian_1d_dirichlet(
    n_x: int,
    h: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build the interior finite-difference Laplacian D2 for Dirichlet BCs.

    Returns matrix D2 with shape (n_x, n_x):
        (D2 u)_i = (u_{i-1} - 2u_i + u_{i+1}) / h^2
    where boundary values are zero.
    """
    diag = -2.0 * torch.ones(n_x, device=device, dtype=dtype)
    off = torch.ones(n_x - 1, device=device, dtype=dtype)
    d2 = torch.diag(diag)
    if n_x > 1:
        d2 = d2 + torch.diag(off, diagonal=1) + torch.diag(off, diagonal=-1)
    d2 = d2 / (h * h)
    return d2


def prepare_implicit_matrix(
    n_x: int,
    dt: float,
    h: float,
    kappa: float = 1.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """
    Prepare cache for implicit Euler update:
        A u_{k+1} = u_k + dt f
    with A = I - dt * kappa * D2.
    """
    d2 = build_laplacian_1d_dirichlet(n_x=n_x, h=h, device=device, dtype=dtype)
    eye = torch.eye(n_x, device=device, dtype=dtype)
    a = eye - dt * float(kappa) * d2

    cache: Dict[str, torch.Tensor] = {
        "A": a,
        "dt": torch.tensor(dt, device=device, dtype=dtype),
        "h": torch.tensor(h, device=device, dtype=dtype),
        "kappa": torch.tensor(float(kappa), device=device, dtype=dtype),
    }

    try:
        chol = torch.linalg.cholesky(a)
        cache["chol"] = chol
    except RuntimeError:
        # Fallback to direct solve if Cholesky is not available.
        pass

    return cache


def _ensure_batch(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if x.dim() == 1:
        return x.unsqueeze(0), True
    return x, False


def implicit_euler_heat_step(
    u_k: torch.Tensor,
    f: torch.Tensor,
    matrix_cache: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Run one implicit Euler step for the heat equation.

    Args:
        u_k: Current state, shape (batch, n_x) or (n_x,)
        f: Forcing field, shape (batch, n_x) or (n_x,)
        matrix_cache: Output of prepare_implicit_matrix

    Returns:
        u_{k+1} with same shape as u_k.
    """
    u_k_b, squeeze = _ensure_batch(u_k)
    f_b, _ = _ensure_batch(f)

    dt = matrix_cache["dt"]
    rhs = u_k_b + dt * f_b

    a = matrix_cache["A"]
    rhs_col = rhs.unsqueeze(-1)

    if "chol" in matrix_cache:
        u_next = torch.cholesky_solve(rhs_col, matrix_cache["chol"]).squeeze(-1)
    else:
        u_next = torch.linalg.solve(a, rhs_col).squeeze(-1)

    if squeeze:
        return u_next.squeeze(0)
    return u_next


def solve_heat_trajectory(
    u0: torch.Tensor,
    f: torch.Tensor,
    dt: float,
    n_steps: int,
    kappa: float = 1.0,
    matrix_cache: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Solve and return full trajectory [u_0, ..., u_K].

    Args:
        u0: Initial state, shape (n_x,) or (batch, n_x)
        f: Forcing, shape matching u0
        dt: Time step
        n_steps: Number of implicit Euler steps
        matrix_cache: Optional matrix cache

    Returns:
        trajectory: shape (K+1, n_x) or (batch, K+1, n_x)
    """
    u0_b, squeeze = _ensure_batch(u0)
    f_b, _ = _ensure_batch(f)

    n_x = u0_b.shape[-1]
    if matrix_cache is None:
        h = 1.0 / (n_x + 1)
        matrix_cache = prepare_implicit_matrix(
            n_x=n_x,
            dt=dt,
            h=h,
            kappa=kappa,
            device=u0_b.device,
            dtype=u0_b.dtype,
        )

    states = [u0_b]
    u = u0_b
    for _ in range(n_steps):
        u = implicit_euler_heat_step(u, f_b, matrix_cache)
        states.append(u)

    traj = torch.stack(states, dim=1)  # (batch, K+1, n_x)
    if squeeze:
        return traj.squeeze(0)
    return traj


def compute_relative_l2_error(
    u_pred: torch.Tensor,
    u_ref: torch.Tensor,
    h: Optional[float] = None,
) -> torch.Tensor:
    """
    Relative L2 error over last dimension.

    Returns tensor of shape u_pred.shape[:-1].
    """
    diff = u_pred - u_ref
    if h is None:
        num = torch.sqrt(torch.sum(diff * diff, dim=-1))
        den = torch.sqrt(torch.sum(u_ref * u_ref, dim=-1))
    else:
        num = torch.sqrt(h * torch.sum(diff * diff, dim=-1))
        den = torch.sqrt(h * torch.sum(u_ref * u_ref, dim=-1))
    return num / (den + 1e-8)


def rollout_model(
    model,
    u0: torch.Tensor,
    f: torch.Tensor,
    n_steps: int,
    dt: Optional[float] = None,
) -> torch.Tensor:
    """
    Roll out learned model for n_steps.

    Args:
        model: Expects model.predict_step(u, f, dt=...) or model(u, f, dt=...)
        u0: (batch, n_x) or (n_x,)
        f: (batch, n_x) or (n_x,)
    """
    u0_b, squeeze = _ensure_batch(u0)
    f_b, _ = _ensure_batch(f)

    states = [u0_b]
    u = u0_b

    for _ in range(n_steps):
        if hasattr(model, "predict_step"):
            u = model.predict_step(u, f_b, dt=dt)
        else:
            u = model(u, f_b, dt=dt)
        states.append(u)

    traj = torch.stack(states, dim=1)
    if squeeze:
        return traj.squeeze(0)
    return traj
