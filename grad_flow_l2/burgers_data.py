"""
Data generation for 1D viscous Burgers equation:
    u_t + (u^2 / 2)_x = nu * u_xx + g(x),  x in [0, 1], t in [0, 1]
with homogeneous Dirichlet boundary conditions.

The dataset format matches grad_flow_l2/data.py:
    split["f"]      : (n_samples, n_x+2)   (forcing sampled on full grid)
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


DATASET_VERSION = 2


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


def solve_burgers_trajectory(
    u0: torch.Tensor,
    forcing: torch.Tensor | None,
    n_steps: int,
    nu: float,
    t_final: float = 1.0,
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
    h = 1.0 / float(n_x + 1)
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
            dt_stable = max(1e-10, min(dt_adv, dt_diff))
            dt_sub = min(remaining, dt_stable)
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
    forcing_mode: str = "zero",
    f_amplitude: float = 1.5,
    f_grf_prob: float = 0.7,
    f_length_scale_min: float = 0.06,
    f_length_scale_max: float = 0.35,
    f_max_modes: int = 6,
    taper_power: float = 1.0,
    norm_targeting: bool = True,
    target_u0_norm_range: tuple[float, float] = (0.6, 1.6),
    target_f_norm_range: tuple[float, float] = (0.2, 1.2),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Dict[str, torch.Tensor]]:
    rng_state_torch = torch.random.get_rng_state()
    rng_state_numpy = np.random.get_state()
    torch.manual_seed(seed)
    np.random.seed(seed)

    total = int(n_train + n_val + n_test)
    h = 1.0 / float(n_x + 1)

    u0 = sample_field_mixed(
        n_points=n_x,
        n_samples=total,
        amplitude=u0_amplitude,
        device="cpu",
    ).to(dtype=dtype)

    taper = _dirichlet_taper(n_x=n_x, power=taper_power, device="cpu", dtype=dtype)
    u0 = u0 * taper.unsqueeze(0)

    if norm_targeting:
        u0 = _rescale_batch_l2(
            u0,
            h=h,
            norm_min=float(target_u0_norm_range[0]),
            norm_max=float(target_u0_norm_range[1]),
        )

    if forcing_mode not in ("zero", "mixed"):
        raise ValueError("forcing_mode must be one of {'zero', 'mixed'}")

    if forcing_mode == "zero":
        f = torch.zeros(total, n_x + 2, dtype=dtype)
    else:
        f = sample_field_mixed(
            n_points=n_x + 2,
            n_samples=total,
            amplitude=f_amplitude,
            length_scale_range=(f_length_scale_min, f_length_scale_max),
            max_modes=f_max_modes,
            grf_prob=f_grf_prob,
            device="cpu",
        ).to(dtype=dtype)
        if norm_targeting:
            f = _rescale_forcing_l2(
                f=f,
                h=h,
                norm_min=float(target_f_norm_range[0]),
                norm_max=float(target_f_norm_range[1]),
                n_x=n_x,
            )

    u_traj = solve_burgers_trajectory(
        u0=u0,
        forcing=f,
        n_steps=n_steps,
        nu=nu,
        t_final=t_final,
    ).to(dtype=dtype)

    all_data = {"f": f, "u0": u0, "u_traj": u_traj}
    train_end = n_train
    val_end = n_train + n_val

    splits: Dict[str, Dict[str, torch.Tensor]] = {
        "train": _slice_split(all_data, 0, train_end),
        "val": _slice_split(all_data, train_end, val_end),
        "test": _slice_split(all_data, val_end, total),
        "meta": {
            "dataset_version": DATASET_VERSION,
            "equation": "burgers_1d_dirichlet",
            "n_x": int(n_x),
            "n_steps": int(n_steps),
            "t_final": float(t_final),
            "n_train": int(n_train),
            "n_val": int(n_val),
            "n_test": int(n_test),
            "seed": int(seed),
            "nu": float(nu),
            "forcing_mode": forcing_mode,
            "f_grid_points": int(n_x + 2),
            "u0_grid_points": int(n_x),
            "u0_amplitude": float(u0_amplitude),
            "f_amplitude": float(f_amplitude),
            "f_grf_prob": float(f_grf_prob),
            "f_length_scale_min": float(f_length_scale_min),
            "f_length_scale_max": float(f_length_scale_max),
            "f_max_modes": int(f_max_modes),
            "taper_power": float(taper_power),
            "norm_targeting": bool(norm_targeting),
            "target_u0_norm_range": [float(target_u0_norm_range[0]), float(target_u0_norm_range[1])],
            "target_f_norm_range": [float(target_f_norm_range[0]), float(target_f_norm_range[1])],
        },
    }

    torch.random.set_rng_state(rng_state_torch)
    np.random.set_state(rng_state_numpy)
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cached 1D Burgers trajectory dataset splits")
    parser.add_argument("--n-x", type=int, default=100, help="Number of interior spatial points")
    parser.add_argument("--n-steps", type=int, default=10, help="Number of macro time steps on [0,t_final]")
    parser.add_argument("--t-final", type=float, default=1.0, help="Final time horizon")
    parser.add_argument("--nu", type=float, default=0.01, help="Viscosity coefficient")

    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--u0-amplitude", type=float, default=2.5)
    parser.add_argument("--forcing-mode", type=str, default="zero", choices=["zero", "mixed"])
    parser.add_argument("--f-amplitude", type=float, default=1.5)
    parser.add_argument("--f-grf-prob", type=float, default=0.7)
    parser.add_argument("--f-length-scale-min", type=float, default=0.06)
    parser.add_argument("--f-length-scale-max", type=float, default=0.35)
    parser.add_argument("--f-max-modes", type=int, default=6)
    parser.add_argument("--taper-power", type=float, default=1.0)
    parser.add_argument("--disable-norm-targeting", action="store_true")
    parser.add_argument("--target-u0-norm-min", type=float, default=0.6)
    parser.add_argument("--target-u0-norm-max", type=float, default=1.6)
    parser.add_argument("--target-f-norm-min", type=float, default=0.2)
    parser.add_argument("--target-f-norm-max", type=float, default=1.2)

    parser.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/burgers_forced_l2_nu0p01_nx100_steps10.pt",
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
        l2_u0 = torch.sqrt((1.0 / (u0.shape[-1] + 1)) * torch.sum(u0 * u0, dim=-1))
        n_x = int(u0.shape[-1])
        f_int = _to_interior_forcing(f, n_x=n_x)
        l2_f = torch.sqrt((1.0 / (n_x + 1)) * torch.sum(f_int * f_int, dim=-1))
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
        forcing_mode=args.forcing_mode,
        f_amplitude=args.f_amplitude,
        f_grf_prob=args.f_grf_prob,
        f_length_scale_min=args.f_length_scale_min,
        f_length_scale_max=args.f_length_scale_max,
        f_max_modes=args.f_max_modes,
        taper_power=args.taper_power,
        norm_targeting=not args.disable_norm_targeting,
        target_u0_norm_range=(args.target_u0_norm_min, args.target_u0_norm_max),
        target_f_norm_range=(args.target_f_norm_min, args.target_f_norm_max),
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
