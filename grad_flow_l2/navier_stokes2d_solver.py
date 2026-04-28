"""
Reference solver for 2D viscous incompressible Navier-Stokes in vorticity form:

    omega_t + v · grad(omega) = nu * Delta omega + g,
    -Delta psi = omega,
    v = (d_y psi, -d_x psi),

on (0,1)^2 with homogeneous Dirichlet boundaries on the represented scalar
fields (interior-grid formulation with zero boundary padding).
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


def pad_dirichlet_2d(u_interior: torch.Tensor) -> torch.Tensor:
    """
    Zero-pad interior field(s) to enforce homogeneous Dirichlet boundaries.

    Args:
        u_interior: (..., n_x, n_y)
    Returns:
        (..., n_x+2, n_y+2)
    """
    if u_interior.dim() < 2:
        raise ValueError("u_interior must have at least 2 dimensions")
    left_right = torch.zeros(*u_interior.shape[:-2], u_interior.shape[-2], 1, device=u_interior.device, dtype=u_interior.dtype)
    with_lr = torch.cat([left_right, u_interior, left_right], dim=-1)
    top_bottom = torch.zeros(*with_lr.shape[:-2], 1, with_lr.shape[-1], device=with_lr.device, dtype=with_lr.dtype)
    return torch.cat([top_bottom, with_lr, top_bottom], dim=-2)


def build_negative_laplacian_1d_dirichlet(
    n: int,
    h: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build 1D matrix L = -D2 with homogeneous Dirichlet BCs.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if h <= 0:
        raise ValueError("h must be > 0")
    diag = 2.0 * torch.ones(n, device=device, dtype=dtype)
    off = -1.0 * torch.ones(n - 1, device=device, dtype=dtype)
    mat = torch.diag(diag)
    if n > 1:
        mat = mat + torch.diag(off, diagonal=1) + torch.diag(off, diagonal=-1)
    return mat / (h * h)


def prepare_ns2d_spectral_cache(
    n_x: int,
    n_y: int,
    h_x: float,
    h_y: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """
    Prepare separable eigendecomposition cache for fast Poisson/Helmholtz solves.
    """
    l_x = build_negative_laplacian_1d_dirichlet(n=n_x, h=h_x, device=device, dtype=dtype)
    l_y = build_negative_laplacian_1d_dirichlet(n=n_y, h=h_y, device=device, dtype=dtype)

    eig_x, q_x = torch.linalg.eigh(l_x)  # l_x = q_x diag(eig_x) q_x^T
    eig_y, q_y = torch.linalg.eigh(l_y)  # l_y = q_y diag(eig_y) q_y^T
    laplace_eigs = eig_x.unsqueeze(1) + eig_y.unsqueeze(0)  # eigenvalues of (-Delta)

    return {
        "q_x": q_x,
        "q_y": q_y,
        "q_x_t": q_x.transpose(0, 1),
        "q_y_t": q_y.transpose(0, 1),
        "laplace_eigs": laplace_eigs,
        "h_x": torch.tensor(float(h_x), device=device, dtype=dtype),
        "h_y": torch.tensor(float(h_y), device=device, dtype=dtype),
    }


def to_interior_field_2d(field: torch.Tensor, n_x: int, n_y: int) -> torch.Tensor:
    """
    Accept interior/full-grid field and return interior representation.
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


def _apply_separable_transform(rhs: torch.Tensor, cache: Dict[str, torch.Tensor]) -> torch.Tensor:
    # rhs_hat = Qx^T rhs Qy
    return torch.matmul(torch.matmul(cache["q_x_t"], rhs), cache["q_y"])


def _apply_inverse_separable_transform(rhs_hat: torch.Tensor, cache: Dict[str, torch.Tensor]) -> torch.Tensor:
    # rhs = Qx rhs_hat Qy^T
    return torch.matmul(torch.matmul(cache["q_x"], rhs_hat), cache["q_y_t"])


def solve_poisson_dirichlet_2d(
    rhs: torch.Tensor,
    cache: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Solve -Delta psi = rhs on interior grid with homogeneous Dirichlet BC.
    """
    rhs_b, squeeze = _ensure_batch_2d(rhs, name="rhs")
    rhs_hat = _apply_separable_transform(rhs_b, cache)
    denom = cache["laplace_eigs"].unsqueeze(0)  # strictly positive for Dirichlet
    psi_hat = rhs_hat / (denom + 1e-12)
    psi = _apply_inverse_separable_transform(psi_hat, cache)
    if squeeze:
        return psi.squeeze(0)
    return psi


def solve_implicit_diffusion_2d(
    rhs: torch.Tensor,
    dt: float,
    nu: float,
    cache: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Solve (I - dt*nu*Delta) omega_next = rhs with homogeneous Dirichlet BC.
    """
    rhs_b, squeeze = _ensure_batch_2d(rhs, name="rhs")
    rhs_hat = _apply_separable_transform(rhs_b, cache)
    denom = 1.0 + float(dt) * float(nu) * cache["laplace_eigs"].unsqueeze(0)
    omega_hat = rhs_hat / denom
    omega_next = _apply_inverse_separable_transform(omega_hat, cache)
    if squeeze:
        return omega_next.squeeze(0)
    return omega_next


def streamfunction_to_velocity(
    psi: torch.Tensor,
    h_x: float,
    h_y: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert streamfunction to incompressible velocity:
        v_x = d_y psi, v_y = -d_x psi
    """
    psi_b, squeeze = _ensure_batch_2d(psi, name="psi")
    psi_full = pad_dirichlet_2d(psi_b)

    dpsi_dx = (psi_full[:, 2:, 1:-1] - psi_full[:, :-2, 1:-1]) / (2.0 * float(h_x))
    dpsi_dy = (psi_full[:, 1:-1, 2:] - psi_full[:, 1:-1, :-2]) / (2.0 * float(h_y))

    v_x = dpsi_dy
    v_y = -dpsi_dx
    if squeeze:
        return v_x.squeeze(0), v_y.squeeze(0)
    return v_x, v_y


def vorticity_to_velocity(
    omega: torch.Tensor,
    cache: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute (psi, v_x, v_y) from vorticity by Poisson solve + centered differences.
    """
    psi = solve_poisson_dirichlet_2d(omega, cache)
    h_x = float(cache["h_x"].item())
    h_y = float(cache["h_y"].item())
    v_x, v_y = streamfunction_to_velocity(psi, h_x=h_x, h_y=h_y)
    return psi, v_x, v_y


def upwind_advection_term(
    omega: torch.Tensor,
    v_x: torch.Tensor,
    v_y: torch.Tensor,
    h_x: float,
    h_y: float,
) -> torch.Tensor:
    """
    First-order upwind approximation of v · grad(omega).
    """
    omega_b, squeeze = _ensure_batch_2d(omega, name="omega")
    vx_b, _ = _ensure_batch_2d(v_x, name="v_x")
    vy_b, _ = _ensure_batch_2d(v_y, name="v_y")

    if vx_b.shape != omega_b.shape or vy_b.shape != omega_b.shape:
        raise ValueError("omega, v_x, v_y must have matching shapes")

    w_full = pad_dirichlet_2d(omega_b)
    w_c = omega_b
    w_im1 = w_full[:, :-2, 1:-1]
    w_ip1 = w_full[:, 2:, 1:-1]
    w_jm1 = w_full[:, 1:-1, :-2]
    w_jp1 = w_full[:, 1:-1, 2:]

    dw_dx_backward = (w_c - w_im1) / float(h_x)
    dw_dx_forward = (w_ip1 - w_c) / float(h_x)
    dw_dy_backward = (w_c - w_jm1) / float(h_y)
    dw_dy_forward = (w_jp1 - w_c) / float(h_y)

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
      1) explicit upwind advection (and forcing)
      2) implicit diffusion
    """
    omega_b, squeeze = _ensure_batch_2d(omega, name="omega")
    n_x, n_y = omega_b.shape[-2], omega_b.shape[-1]

    if velocity is None:
        _, v_x, v_y = vorticity_to_velocity(omega_b, cache)
    else:
        v_x, v_y = velocity
        v_x, _ = _ensure_batch_2d(v_x, name="v_x")
        v_y, _ = _ensure_batch_2d(v_y, name="v_y")

    adv = upwind_advection_term(
        omega=omega_b,
        v_x=v_x,
        v_y=v_y,
        h_x=float(cache["h_x"].item()),
        h_y=float(cache["h_y"].item()),
    )

    if forcing is None:
        forcing_int = 0.0
    else:
        forcing_int = to_interior_field_2d(forcing, n_x=n_x, n_y=n_y).to(device=omega_b.device, dtype=omega_b.dtype)
        if forcing_int.shape[0] == 1 and omega_b.shape[0] > 1:
            forcing_int = forcing_int.expand(omega_b.shape[0], -1, -1)
        if forcing_int.shape[0] != omega_b.shape[0]:
            raise ValueError("forcing batch size must match omega batch size or be 1")

    rhs = omega_b - float(dt) * adv + float(dt) * forcing_int
    omega_next = solve_implicit_diffusion_2d(rhs=rhs, dt=dt, nu=nu, cache=cache)
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
    Integrate and return [omega_0, ..., omega_K] on interior grid.
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
    h_x = 1.0 / float(n_x + 1)
    h_y = 1.0 / float(n_y + 1)
    if solver_cache is None:
        solver_cache = prepare_ns2d_spectral_cache(
            n_x=n_x,
            n_y=n_y,
            h_x=h_x,
            h_y=h_y,
            device=str(u0_b.device),
            dtype=u0_b.dtype,
        )

    forcing_int = None
    if forcing is not None:
        forcing_int = to_interior_field_2d(forcing, n_x=n_x, n_y=n_y).to(device=u0_b.device, dtype=u0_b.dtype)
        if forcing_int.shape[0] == 1 and u0_b.shape[0] > 1:
            forcing_int = forcing_int.expand(u0_b.shape[0], -1, -1)
        if forcing_int.shape[0] != u0_b.shape[0]:
            raise ValueError("forcing batch size must match u0 batch size or be 1")

    dt_macro = float(t_final) / float(n_steps)
    omega = u0_b.clone()
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
            remaining -= dt_sub
            substeps += 1
            if substeps > max_substeps_per_step:
                raise RuntimeError(
                    "Exceeded max_substeps_per_step; reduce amplitudes, increase n_steps, or relax CFL settings."
                )
        states.append(omega.clone())

    traj = torch.stack(states, dim=1)  # (batch, n_steps+1, n_x, n_y)
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
    x = torch.linspace(0.0, 1.0, n_x + 2, device=device, dtype=dtype)[1:-1]
    y = torch.linspace(0.0, 1.0, n_y + 2, device=device, dtype=dtype)[1:-1]
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    field = torch.zeros(n_x, n_y, device=device, dtype=dtype)
    for k_x in range(1, 4):
        for k_y in range(1, 4):
            coeff = torch.randn((), device=device, dtype=dtype) / float(k_x + k_y)
            phase = 2.0 * torch.pi * torch.rand((), device=device, dtype=dtype)
            field = field + coeff * torch.sin(torch.pi * k_x * xx + phase) * torch.sin(torch.pi * k_y * yy + phase)
    max_abs = torch.max(torch.abs(field)) + 1e-8
    return float(amplitude) * field / max_abs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reference 2D Navier-Stokes vorticity solver on one sample")
    parser.add_argument("--n-x", type=int, default=32, help="Number of interior x-grid points")
    parser.add_argument("--n-y", type=int, default=32, help="Number of interior y-grid points")
    parser.add_argument("--n-steps", type=int, default=10, help="Number of macro time steps on [0,t_final]")
    parser.add_argument("--t-final", type=float, default=1.0, help="Final simulation time")
    parser.add_argument("--nu", type=float, default=0.01, help="Viscosity coefficient")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampled initial condition")
    parser.add_argument("--u0-amplitude", type=float, default=2.0, help="Amplitude of sampled initial vorticity")
    parser.add_argument("--cfl-adv", type=float, default=0.45, help="Advection CFL safety factor")
    parser.add_argument("--max-substeps-per-step", type=int, default=4000)
    parser.add_argument(
        "--out-path",
        type=str,
        default="grad_flow_l2/outputs/navier_stokes2d_reference_sample.pt",
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

    area = 1.0 / float((args.n_x + 1) * (args.n_y + 1))
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
            "equation": "navier_stokes_2d_vorticity_dirichlet",
            "n_x": int(args.n_x),
            "n_y": int(args.n_y),
            "n_steps": int(args.n_steps),
            "t_final": float(args.t_final),
            "nu": float(args.nu),
            "seed": int(args.seed),
            "cfl_adv": float(args.cfl_adv),
        },
    }
    torch.save(payload, args.out_path)
    print(f"Saved reference trajectory: {args.out_path}")


if __name__ == "__main__":
    main(parse_args())
