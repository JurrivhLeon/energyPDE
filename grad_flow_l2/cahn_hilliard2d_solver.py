"""
Reference solver utilities for the 2D Cahn-Hilliard equation:

    u_t = M * Delta(mu),
    mu = -epsilon^2 * Delta(u) + (u^3 - u),

on (0,1)^2 with homogeneous Neumann boundary conditions:

    d_n u = 0,   d_n mu = 0.

State representation:
    We store only interior grid values with shape (n_x, n_y), where
    h_x = 1 / (n_x + 1), h_y = 1 / (n_y + 1).

Neumann BC enforcement:
    We use a finite-difference operator derived from mirrored ghost points:
        u_0 = u_1, u_{n+1} = u_n  (and same in y).
    This yields a 1D negative Laplacian whose first/last rows are
    [1, -1, 0, ...] / h^2 and [..., 0, -1, 1] / h^2.
    The same operator is used consistently in both physical-space diagnostics
    and spectral semi-implicit time stepping.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


def _ensure_batch_2d(x: torch.Tensor, name: str = "tensor") -> tuple[torch.Tensor, bool]:
    if x.dim() == 2:
        return x.unsqueeze(0), True
    if x.dim() == 3:
        return x, False
    raise ValueError(f"{name} must have shape (n_x,n_y) or (batch,n_x,n_y), got {tuple(x.shape)}")


def _as_batch_param(
    value,
    batch_size: int,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if torch.is_tensor(value):
        v = value.to(device=device, dtype=dtype)
        if v.dim() == 0:
            if float(v.item()) <= 0.0:
                raise ValueError(f"{name} must be > 0")
            return v.view(1, 1, 1).expand(batch_size, 1, 1)
        if v.dim() == 1 and int(v.shape[0]) == batch_size:
            if torch.any(v <= 0):
                raise ValueError(f"all {name} values must be > 0")
            return v.view(batch_size, 1, 1)
        raise ValueError(f"{name} tensor must be scalar or shape (batch,), got {tuple(v.shape)}")

    scalar = float(value)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return torch.full((batch_size, 1, 1), scalar, device=device, dtype=dtype)


def build_negative_laplacian_1d_neumann(
    n: int,
    h: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build 1D matrix A approximating -d_xx with homogeneous Neumann BC.

    For interior unknowns (u_1,...,u_n), mirrored ghosts enforce d_n u = 0:
      u_0 = u_1 and u_{n+1} = u_n.
    Substituting into centered second differences gives boundary rows with
    stencil [1, -1] / h^2 for -d_xx.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    if h <= 0:
        raise ValueError("h must be > 0")

    a = torch.zeros(n, n, device=device, dtype=dtype)
    a[0, 0] = 1.0
    a[0, 1] = -1.0
    a[-1, -1] = 1.0
    a[-1, -2] = -1.0
    if n > 2:
        idx = torch.arange(1, n - 1, device=device)
        a[idx, idx] = 2.0
        a[idx, idx - 1] = -1.0
        a[idx, idx + 1] = -1.0
    return a / (h * h)


def prepare_ch2d_spectral_cache(
    n_x: int,
    n_y: int,
    h_x: float,
    h_y: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """
    Prepare separable eigendecomposition cache for Neumann Laplacian operators.
    """
    a_x = build_negative_laplacian_1d_neumann(n=n_x, h=h_x, device=device, dtype=dtype)
    a_y = build_negative_laplacian_1d_neumann(n=n_y, h=h_y, device=device, dtype=dtype)

    eig_x, q_x = torch.linalg.eigh(a_x)
    eig_y, q_y = torch.linalg.eigh(a_y)
    laplace_eigs = eig_x.unsqueeze(1) + eig_y.unsqueeze(0)

    return {
        "a_x": a_x,
        "a_y": a_y,
        "q_x": q_x,
        "q_y": q_y,
        "q_x_t": q_x.transpose(0, 1),
        "q_y_t": q_y.transpose(0, 1),
        "laplace_eigs": laplace_eigs,
        "h_x": torch.tensor(float(h_x), device=device, dtype=dtype),
        "h_y": torch.tensor(float(h_y), device=device, dtype=dtype),
    }


def _apply_separable_transform(rhs: torch.Tensor, cache: Dict[str, torch.Tensor]) -> torch.Tensor:
    # rhs_hat = Qx^T rhs Qy
    return torch.matmul(torch.matmul(cache["q_x_t"], rhs), cache["q_y"])


def _apply_inverse_separable_transform(rhs_hat: torch.Tensor, cache: Dict[str, torch.Tensor]) -> torch.Tensor:
    # rhs = Qx rhs_hat Qy^T
    return torch.matmul(torch.matmul(cache["q_x"], rhs_hat), cache["q_y_t"])


def apply_negative_laplacian_2d(u: torch.Tensor, cache: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Apply discrete negative Laplacian A = A_x + A_y with Neumann BC.
    """
    u_b, squeeze = _ensure_batch_2d(u, name="u")
    ax_u = torch.matmul(cache["a_x"], u_b)
    ay_u = torch.matmul(u_b, cache["a_y"].transpose(0, 1))
    out = ax_u + ay_u
    if squeeze:
        return out.squeeze(0)
    return out


def compute_total_mass_2d(u: torch.Tensor, h_x: float, h_y: float) -> torch.Tensor:
    """
    Compute approximate total mass int_Omega u dx dy on interior grid.
    """
    u_b, squeeze = _ensure_batch_2d(u, name="u")
    mass = float(h_x) * float(h_y) * torch.sum(u_b, dim=(-2, -1))
    if squeeze:
        return mass.squeeze(0)
    return mass


def compute_ch_free_energy_2d(
    u: torch.Tensor,
    epsilon,
    h_x: float,
    h_y: float,
    cache: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Discrete Cahn-Hilliard energy:

      E(u) = int [ epsilon^2/2 * |grad u|^2 + (u^2 - 1)^2 / 4 ] dx.

    The gradient contribution is evaluated via the symmetric Neumann
    operator using the identity int |grad u|^2 ~ <u, -Delta_h u>.
    """
    u_b, squeeze = _ensure_batch_2d(u, name="u")
    batch_size = int(u_b.shape[0])
    eps_b = _as_batch_param(
        epsilon,
        batch_size=batch_size,
        name="epsilon",
        device=u_b.device,
        dtype=u_b.dtype,
    )

    if cache is None:
        n_x = int(u_b.shape[-2])
        n_y = int(u_b.shape[-1])
        cache = prepare_ch2d_spectral_cache(
            n_x=n_x,
            n_y=n_y,
            h_x=float(h_x),
            h_y=float(h_y),
            device=str(u_b.device),
            dtype=u_b.dtype,
        )

    area = float(h_x) * float(h_y)
    au = apply_negative_laplacian_2d(u_b, cache)
    grad_part = 0.5 * (eps_b.squeeze(-1).squeeze(-1) ** 2) * area * torch.sum(u_b * au, dim=(-2, -1))
    pot_part = area * torch.sum(0.25 * (u_b * u_b - 1.0) ** 2, dim=(-2, -1))
    energy = grad_part + pot_part
    if squeeze:
        return energy.squeeze(0)
    return energy


def cahn_hilliard_semi_implicit_step(
    u: torch.Tensor,
    dt: float,
    epsilon,
    mobility,
    cache: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    One semi-implicit step:

      (u^{n+1} - u^n) / dt = M * Delta( (u^n)^3 - u^n - epsilon^2 * Delta u^{n+1} ).

    In terms of A = -Delta_h (Neumann),

      (I + dt * M * epsilon^2 * A^2) u^{n+1}
        = u^n - dt * M * A( (u^n)^3 - u^n ).
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")

    u_b, squeeze = _ensure_batch_2d(u, name="u")
    batch_size = int(u_b.shape[0])

    eps_b = _as_batch_param(
        epsilon,
        batch_size=batch_size,
        name="epsilon",
        device=u_b.device,
        dtype=u_b.dtype,
    )
    mob_b = _as_batch_param(
        mobility,
        batch_size=batch_size,
        name="mobility",
        device=u_b.device,
        dtype=u_b.dtype,
    )

    nonlin = u_b * u_b * u_b - u_b
    u_hat = _apply_separable_transform(u_b, cache)
    nonlin_hat = _apply_separable_transform(nonlin, cache)

    lam = cache["laplace_eigs"].unsqueeze(0)  # (1, n_x, n_y)
    coeff = mob_b * (eps_b * eps_b)  # (batch, 1, 1)
    denom = 1.0 + float(dt) * coeff * (lam * lam)
    rhs = u_hat - float(dt) * mob_b * lam * nonlin_hat
    u_next_hat = rhs / denom

    # Preserve the zero-frequency mode exactly to maintain total mass.
    u_next_hat[:, 0, 0] = u_hat[:, 0, 0]

    u_next = _apply_inverse_separable_transform(u_next_hat, cache)
    if squeeze:
        return u_next.squeeze(0)
    return u_next


def solve_cahn_hilliard_trajectory(
    u0: torch.Tensor,
    n_steps: int,
    t_final: float = 1.0,
    epsilon: float | torch.Tensor = 0.04,
    mobility: float | torch.Tensor = 1.0,
    cfl_nonlinear: float = 0.20,
    max_substeps_per_step: int = 4000,
    max_dt_substep: Optional[float] = None,
    enforce_mass_correction: bool = True,
    solver_cache: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Integrate and return [u_0, ..., u_K] on interior grid.

    The explicit nonlinear term can become stiff. We therefore substep each
    macro interval with a conservative nonlinear CFL heuristic.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if t_final <= 0:
        raise ValueError("t_final must be > 0")
    if cfl_nonlinear <= 0:
        raise ValueError("cfl_nonlinear must be > 0")
    if max_substeps_per_step < 1:
        raise ValueError("max_substeps_per_step must be >= 1")

    u0_b, squeeze = _ensure_batch_2d(u0, name="u0")
    batch_size = int(u0_b.shape[0])
    n_x = int(u0_b.shape[-2])
    n_y = int(u0_b.shape[-1])
    h_x = 1.0 / float(n_x + 1)
    h_y = 1.0 / float(n_y + 1)

    if solver_cache is None:
        solver_cache = prepare_ch2d_spectral_cache(
            n_x=n_x,
            n_y=n_y,
            h_x=h_x,
            h_y=h_y,
            device=str(u0_b.device),
            dtype=u0_b.dtype,
        )

    eps_b = _as_batch_param(
        epsilon,
        batch_size=batch_size,
        name="epsilon",
        device=u0_b.device,
        dtype=u0_b.dtype,
    ).squeeze(-1).squeeze(-1)
    mob_b = _as_batch_param(
        mobility,
        batch_size=batch_size,
        name="mobility",
        device=u0_b.device,
        dtype=u0_b.dtype,
    ).squeeze(-1).squeeze(-1)

    dt_macro = float(t_final) / float(n_steps)
    min_h2 = min(h_x * h_x, h_y * h_y)

    u = u0_b.clone()
    states = [u.clone()]
    for _ in range(n_steps):
        remaining = dt_macro
        substeps = 0
        while remaining > 1e-14:
            # Explicit treatment of Delta(u^3-u):
            # local Lipschitz ~ |3u^2 - 1|, so we shrink dt when fields sharpen.
            lipschitz = torch.max(torch.abs(3.0 * u * u - 1.0), dim=-1)[0]
            lipschitz = torch.max(lipschitz, dim=-1)[0]
            max_lipschitz = float(torch.max(lipschitz).item())
            max_mob = float(torch.max(mob_b).item())
            dt_nonlinear = float(cfl_nonlinear) * min_h2 / max(max_mob * max_lipschitz, 1e-8)
            if max_dt_substep is not None:
                dt_nonlinear = min(dt_nonlinear, float(max_dt_substep))
            dt_sub = min(remaining, max(1e-10, dt_nonlinear))

            if enforce_mass_correction:
                mass_before = compute_total_mass_2d(u, h_x=h_x, h_y=h_y)
            u = cahn_hilliard_semi_implicit_step(
                u=u,
                dt=dt_sub,
                epsilon=eps_b,
                mobility=mob_b,
                cache=solver_cache,
            )
            if enforce_mass_correction:
                mass_after = compute_total_mass_2d(u, h_x=h_x, h_y=h_y)
                delta = (mass_before - mass_after) / (float(h_x) * float(h_y) * float(n_x * n_y))
                u = u + delta.view(-1, 1, 1)

            if not torch.isfinite(u).all():
                raise RuntimeError("Encountered non-finite values during Cahn-Hilliard integration.")

            remaining -= dt_sub
            substeps += 1
            if substeps > max_substeps_per_step:
                raise RuntimeError(
                    "Exceeded max_substeps_per_step. Increase n_steps, decrease t_final, "
                    "reduce cfl_nonlinear, or use a smaller max_dt_substep."
                )
        states.append(u.clone())

    traj = torch.stack(states, dim=1)
    if squeeze:
        return traj.squeeze(0)
    return traj

