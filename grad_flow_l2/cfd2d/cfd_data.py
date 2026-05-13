"""
Data generation for PDEBench-style 2D compressible Navier-Stokes.

Default setup follows Appendix D.5 of PDEBench (arXiv:2210.07182v7):
periodic 2D random-field initial conditions, ideal-gas EOS with Gamma=5/3,
Mach-number-controlled velocity scale, HLLC+MUSCL inviscid fluxes, and
central-difference viscous terms.

Dataset format:
    split["f"]      : (n_samples, n_x, n_y), zero placeholder forcing
    split["u0"]     : (n_samples, 4, n_x, n_y)
    split["u_traj"] : (n_samples, n_steps+1, 4, n_x, n_y)

By default stored channels are PDEBench-style primitive variables
    (rho, vx, vy, p).
Use ``--store-conserved`` to store conservative channels
    (rho, rho_vx, rho_vy, E).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    from ..heat_data import save_dataset_splits
except ImportError:
    from grad_flow_l2.heat_data import save_dataset_splits


DATASET_VERSION = 1
STATE_CHANNELS = 4
STATE_NAMES_PRIMITIVE = ("rho", "vx", "vy", "p")
STATE_NAMES_CONSERVED = ("rho", "rho_vx", "rho_vy", "E")


# ─── Dataset classes ──────────────────────────────────────────────────────────

class CFD2DTrajectoryTensorDataset(Dataset):
    def __init__(self, f_data: torch.Tensor, u0_data: torch.Tensor, u_traj_data: torch.Tensor):
        if f_data.dim() != 3:
            raise ValueError("f_data must have shape (n_samples,n_x,n_y)")
        if u0_data.dim() != 4 or int(u0_data.shape[1]) != STATE_CHANNELS:
            raise ValueError("u0_data must have shape (n_samples,4,n_x,n_y)")
        if u_traj_data.dim() != 5 or int(u_traj_data.shape[2]) != STATE_CHANNELS:
            raise ValueError("u_traj_data must have shape (n_samples,K+1,4,n_x,n_y)")
        n_samples = int(u_traj_data.shape[0])
        n_x = int(u_traj_data.shape[-2])
        n_y = int(u_traj_data.shape[-1])
        if (
            int(f_data.shape[0]) != n_samples
            or tuple(f_data.shape[1:]) != (n_x, n_y)
            or int(u0_data.shape[0]) != n_samples
            or tuple(u0_data.shape[2:]) != (n_x, n_y)
        ):
            raise ValueError("inconsistent CFD2D dataset tensor shapes")
        self.f_data = f_data
        self.u0_data = u0_data
        self.u_traj_data = u_traj_data

    def __len__(self) -> int:
        return int(self.u0_data.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"f": self.f_data[idx], "u0": self.u0_data[idx], "u_traj": self.u_traj_data[idx]}


class CFD2DStepDataset(Dataset):
    def __init__(self, f_data: torch.Tensor, u_traj_data: torch.Tensor):
        if f_data.dim() != 3:
            raise ValueError("f_data must have shape (n_samples,n_x,n_y)")
        if u_traj_data.dim() != 5 or int(u_traj_data.shape[2]) != STATE_CHANNELS:
            raise ValueError("u_traj_data must have shape (n_samples,K+1,4,n_x,n_y)")
        if int(f_data.shape[0]) != int(u_traj_data.shape[0]) or tuple(f_data.shape[1:]) != tuple(u_traj_data.shape[-2:]):
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


def build_cfd2d_step_dataset(split_or_dataset) -> CFD2DStepDataset:
    if isinstance(split_or_dataset, CFD2DTrajectoryTensorDataset):
        return CFD2DStepDataset(split_or_dataset.f_data, split_or_dataset.u_traj_data)
    if isinstance(split_or_dataset, dict):
        return CFD2DStepDataset(split_or_dataset["f"], split_or_dataset["u_traj"])
    raise TypeError("Expected split dict or CFD2DTrajectoryTensorDataset")


def build_cfd2d_trajectory_dataset_from_split(split: Dict[str, torch.Tensor]) -> CFD2DTrajectoryTensorDataset:
    return CFD2DTrajectoryTensorDataset(split["f"], split["u0"], split["u_traj"])


# ─── EOS & state conversion ───────────────────────────────────────────────────

def primitive_to_conserved(q: torch.Tensor, gamma: float = 5.0 / 3.0) -> torch.Tensor:
    """(batch,4,nx,ny) primitive (rho,vx,vy,p) → conserved (rho, rho*vx, rho*vy, E)."""
    if q.dim() != 4 or int(q.shape[1]) != STATE_CHANNELS:
        raise ValueError("primitive state must have shape (batch,4,n_x,n_y)")
    rho = q[:, 0].clamp_min(1e-12)
    vx = q[:, 1]
    vy = q[:, 2]
    p = q[:, 3].clamp_min(1e-12)
    energy = p / (float(gamma) - 1.0) + 0.5 * rho * (vx.square() + vy.square())
    return torch.stack([rho, rho * vx, rho * vy, energy], dim=1)


def conserved_to_primitive(
    u: torch.Tensor,
    gamma: float = 5.0 / 3.0,
    rho_floor: float = 1e-6,
    p_floor: float = 1e-6,
) -> torch.Tensor:
    """(batch,4,nx,ny) conserved → primitive (rho,vx,vy,p)."""
    if u.dim() != 4 or int(u.shape[1]) != STATE_CHANNELS:
        raise ValueError("conserved state must have shape (batch,4,n_x,n_y)")
    rho = u[:, 0].clamp_min(float(rho_floor))
    vx = u[:, 1] / rho
    vy = u[:, 2] / rho
    kinetic = 0.5 * rho * (vx.square() + vy.square())
    p = (float(gamma) - 1.0) * (u[:, 3] - kinetic)
    p = p.clamp_min(float(p_floor))
    return torch.stack([rho, vx, vy, p], dim=1)


# ─── Initial-condition sampling ───────────────────────────────────────────────

def _sample_sinusoidal_ic_2d(
    n_x: int,
    n_y: int,
    n_samples: int,
    n_modes_min: int = 4,
    n_modes_max: int = 16,
    k_max: int = 4,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    PDEBench Eq. 8 extended to 2D:
      u(x,y) = Σᵢ [Aᵢ/(mᵢ²+nᵢ²)] · sin(2π(mᵢx + nᵢy) + φᵢ)
    where
      Aᵢ ~ U[0,1],  φᵢ ~ U[0,2π),
      mᵢ, nᵢ ∈ {-k_max,...,-1,1,...,k_max}  (no zeros, all four quadrant directions),
      number of modes ~ U{n_modes_min,...,n_modes_max}  per sample.
    Returns (n_samples, n_x, n_y), zero-mean, unit-std per sample.
    """
    x = torch.linspace(0.0, 1.0, n_x + 1, device=device, dtype=dtype)[:-1]
    y = torch.linspace(0.0, 1.0, n_y + 1, device=device, dtype=dtype)[:-1]
    xx, yy = torch.meshgrid(x, y, indexing="ij")   # (n_x, n_y)

    M = int(n_modes_max)  # allocate for max; mask out unused modes per sample

    m_mag = torch.randint(1, k_max + 1, (n_samples, M), device=device)
    n_mag = torch.randint(1, k_max + 1, (n_samples, M), device=device)
    sm    = 2 * torch.randint(0, 2, (n_samples, M), device=device) - 1  # ±1
    sn    = 2 * torch.randint(0, 2, (n_samples, M), device=device) - 1
    m_int = (m_mag * sm).to(dtype)   # ∈ {-k_max,...,-1,1,...,k_max}
    n_int = (n_mag * sn).to(dtype)

    kx  = (2.0 * torch.pi * m_int).view(n_samples, M, 1, 1)
    ky  = (2.0 * torch.pi * n_int).view(n_samples, M, 1, 1)
    A   = torch.rand(n_samples, M, 1, 1, device=device, dtype=dtype)
    phi = torch.rand(n_samples, M, 1, 1, device=device, dtype=dtype) * (2.0 * torch.pi)
    damp = 1.0 / (m_int.pow(2) + n_int.pow(2)).view(n_samples, M, 1, 1)

    # Per-sample random mode count; mask out modes beyond that count
    n_modes_per = torch.randint(int(n_modes_min), M + 1, (n_samples,), device=device)
    mask = (torch.arange(M, device=device).unsqueeze(0) < n_modes_per.unsqueeze(1)
            ).to(dtype).view(n_samples, M, 1, 1)

    fields = (mask * A * damp * torch.sin(kx * xx + ky * yy + phi)).sum(dim=1)
    std = fields.std(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    return fields / std


def _sample_grf_periodic_2d(
    n_x: int,
    n_y: int,
    n_samples: int,
    length_scale: "float | torch.Tensor" = 0.2,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Periodic 2D GRF via Gaussian spectral filter (matching ns2d_per convention).
    Power spectrum ∝ exp(-2(π·ℓ)²·|k|²).
    length_scale may be a scalar or a (n_samples,) tensor for per-sample ℓ.
    Returns (n_samples, n_x, n_y), zero-mean, unit-std per sample.
    """
    kx = torch.fft.fftfreq(n_x, d=1.0 / float(n_x), device=device, dtype=dtype)
    ky = torch.fft.rfftfreq(n_y, d=1.0 / float(n_y), device=device, dtype=dtype)
    k2 = kx.unsqueeze(1) ** 2 + ky.unsqueeze(0) ** 2            # (n_x, n_y//2+1)
    if isinstance(length_scale, torch.Tensor):
        ls = length_scale.to(device=device, dtype=dtype).view(-1, 1, 1)
        power = torch.exp(-2.0 * (torch.pi * ls) ** 2 * k2.unsqueeze(0))
    else:
        power = torch.exp(-2.0 * (torch.pi * float(length_scale)) ** 2 * k2).unsqueeze(0)
    real = torch.randn(n_samples, n_x, n_y // 2 + 1, device=device, dtype=dtype)
    imag = torch.randn(n_samples, n_x, n_y // 2 + 1, device=device, dtype=dtype)
    spec = torch.complex(real, imag) * power.sqrt()
    spec[:, 0, 0] = 0.0
    samples = torch.fft.irfft2(spec, s=(n_x, n_y)) * float(n_x * n_y) ** 0.5
    std = samples.std(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    return samples / std


def sample_cfd2d_initial_conditions(
    n_x: int,
    n_y: int,
    n_samples: int,
    gamma: float = 5.0 / 3.0,
    rho0: float = 1.0,
    p0: float = -1.0,          # <0 → auto: 1/gamma (so sound speed = 1)
    rho_amp: float = 0.1,
    mach_min: float = 0.1,
    mach_max: float = 0.5,
    p_amp: float = 0.05,
    ic_type: str = "grf",          # "grf" (Gaussian spectral) or "sinusoidal" (PDEBench Eq.8)
    n_modes_min: int = 4,          # sinusoidal only: min modes per field
    n_modes_max: int = 16,         # sinusoidal only: max modes per field
    k_max: int = 4,                # sinusoidal only: max integer wavenumber |m|,|n|
    grf_ls_min: float = 0.05,      # grf only: lower bound of per-sample length scale
    grf_ls_max: float = 0.15,      # grf only: upper bound of per-sample length scale
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample random primitive initial conditions (n_samples, 4, n_x, n_y).

    Background state: rho=rho0, p=p0 (default 1/gamma → c₀=1),
    sinusoidal (PDEBench Eq. 8) or GRF perturbations on all fields,
    velocity amplitude scaled to Mach M ~ Uniform(mach_min, mach_max).
    """
    g = float(gamma)
    p0_val = 1.0 / g if float(p0) < 0.0 else float(p0)
    c0 = (g * p0_val / float(rho0)) ** 0.5

    # One length-scale per sample, shared across all four fields so that
    # rho, vx, vy, p have the same spatial correlation length within a sample.
    if ic_type == "sinusoidal":
        def _sample_field() -> torch.Tensor:
            return _sample_sinusoidal_ic_2d(n_x, n_y, n_samples,
                                            n_modes_min, n_modes_max, k_max,
                                            device=device, dtype=dtype)
    else:
        ls = torch.empty(n_samples, device=device, dtype=dtype).uniform_(
            float(grf_ls_min), float(grf_ls_max))
        def _sample_field() -> torch.Tensor:
            return _sample_grf_periodic_2d(n_x, n_y, n_samples, ls,
                                           device=device, dtype=dtype)

    f_rho = _sample_field()
    f_vx  = _sample_field()
    f_vy  = _sample_field()
    f_p   = _sample_field()

    mach = torch.empty(n_samples, device=device, dtype=dtype).uniform_(float(mach_min), float(mach_max))
    rho = (float(rho0) + float(rho_amp) * f_rho).clamp(min=1e-6)
    vx  = c0 * mach.view(-1, 1, 1) * f_vx
    vy  = c0 * mach.view(-1, 1, 1) * f_vy
    p   = (p0_val * (1.0 + float(p_amp) * f_p)).clamp(min=1e-6)
    return torch.stack([rho, vx, vy, p], dim=1)


# ─── MUSCL-minmod reconstruction ─────────────────────────────────────────────

def _minmod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.sign(a) + torch.sign(b)) * torch.minimum(a.abs(), b.abs())


def _muscl_x(q_prim: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MUSCL-minmod in x (dims=2) on primitive variables.
    Returns (qL, qR) at each i+1/2 interface.
    """
    q_im1 = torch.roll(q_prim, 1, dims=2)
    q_ip1 = torch.roll(q_prim, -1, dims=2)
    slope     = _minmod(q_prim - q_im1, q_ip1 - q_prim)
    slope_ip1 = torch.roll(slope, -1, dims=2)
    qL = q_prim + 0.5 * slope
    qR = q_ip1  - 0.5 * slope_ip1
    qL = torch.cat([qL[:, :1].clamp(min=1e-12), qL[:, 1:3], qL[:, 3:].clamp(min=1e-12)], dim=1)
    qR = torch.cat([qR[:, :1].clamp(min=1e-12), qR[:, 1:3], qR[:, 3:].clamp(min=1e-12)], dim=1)
    return qL, qR


def _muscl_y(q_prim: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """MUSCL-minmod in y (dims=3) on primitive variables."""
    q_jm1 = torch.roll(q_prim, 1, dims=3)
    q_jp1 = torch.roll(q_prim, -1, dims=3)
    slope     = _minmod(q_prim - q_jm1, q_jp1 - q_prim)
    slope_jp1 = torch.roll(slope, -1, dims=3)
    qL = q_prim + 0.5 * slope
    qR = q_jp1  - 0.5 * slope_jp1
    qL = torch.cat([qL[:, :1].clamp(min=1e-12), qL[:, 1:3], qL[:, 3:].clamp(min=1e-12)], dim=1)
    qR = torch.cat([qR[:, :1].clamp(min=1e-12), qR[:, 1:3], qR[:, 3:].clamp(min=1e-12)], dim=1)
    return qL, qR


# ─── HLLC Riemann solver ─────────────────────────────────────────────────────

def _hllc_flux_x(qL: torch.Tensor, qR: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    HLLC flux in x-direction (normal velocity = vx).
    qL, qR: (batch,4,nx,ny) primitive at i+1/2 interfaces.
    Returns (batch,4,nx,ny) conservative flux F_x.
    """
    g = float(gamma)
    eps = 1e-12
    rL, uL, vL, pL = qL[:, 0], qL[:, 1], qL[:, 2], qL[:, 3]
    rR, uR, vR, pR = qR[:, 0], qR[:, 1], qR[:, 2], qR[:, 3]

    aL = (g * pL / rL).clamp(min=eps).sqrt()
    aR = (g * pR / rR).clamp(min=eps).sqrt()

    SL = torch.minimum(uL - aL, uR - aR)
    SR = torch.maximum(uL + aL, uR + aR)

    num_S = pR - pL + rL * uL * (SL - uL) - rR * uR * (SR - uR)
    den_S = rL * (SL - uL) - rR * (SR - uR)
    Sstar = torch.where(den_S.abs() > eps, num_S / den_S, 0.5 * (uL + uR))

    EL = pL / (g - 1.0) + 0.5 * rL * (uL.square() + vL.square())
    ER = pR / (g - 1.0) + 0.5 * rR * (uR.square() + vR.square())

    FL = torch.stack([rL * uL, rL * uL.square() + pL, rL * uL * vL, uL * (EL + pL)], dim=1)
    FR = torch.stack([rR * uR, rR * uR.square() + pR, rR * uR * vR, uR * (ER + pR)], dim=1)

    # Star-state scale factors: rhoK*(SK-uK)/(SK-S*)
    dL = torch.where((SL - Sstar).abs() > eps, SL - Sstar, SL.new_full((), -eps).expand_as(SL))
    dR = torch.where((SR - Sstar).abs() > eps, SR - Sstar, SR.new_full((),  eps).expand_as(SR))
    fL = rL * (SL - uL) / dL
    fR = rR * (SR - uR) / dR

    # Energy formula: E*K = fK * [EK/rhoK + (S*-uK)*(S* + pK/(rhoK*(SK-uK)))]
    dLuL = torch.where((SL - uL).abs() > eps, SL - uL, SL.new_full((), -eps).expand_as(SL))
    dRuR = torch.where((SR - uR).abs() > eps, SR - uR, SR.new_full((),  eps).expand_as(SR))
    EsL = fL * (EL / rL + (Sstar - uL) * (Sstar + pL / (rL * dLuL)))
    EsR = fR * (ER / rR + (Sstar - uR) * (Sstar + pR / (rR * dRuR)))

    UL_star = torch.stack([fL, fL * Sstar, fL * vL, EsL], dim=1)
    UR_star = torch.stack([fR, fR * Sstar, fR * vR, EsR], dim=1)
    UL_cons = torch.stack([rL, rL * uL, rL * vL, EL], dim=1)
    UR_cons = torch.stack([rR, rR * uR, rR * vR, ER], dim=1)

    FL_star = FL + SL.unsqueeze(1) * (UL_star - UL_cons)
    FR_star = FR + SR.unsqueeze(1) * (UR_star - UR_cons)

    return torch.where(
        SL.unsqueeze(1) >= 0.0, FL,
        torch.where(Sstar.unsqueeze(1) >= 0.0, FL_star,
        torch.where(SR.unsqueeze(1) >= 0.0, FR_star, FR)),
    )


def _hllc_flux_y(qL: torch.Tensor, qR: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    HLLC flux in y-direction (normal velocity = vy).
    qL, qR: (batch,4,nx,ny) primitive at j+1/2 interfaces.
    Returns (batch,4,nx,ny) conservative flux G_y.
    """
    g = float(gamma)
    eps = 1e-12
    rL, uL, vL, pL = qL[:, 0], qL[:, 1], qL[:, 2], qL[:, 3]
    rR, uR, vR, pR = qR[:, 0], qR[:, 1], qR[:, 2], qR[:, 3]

    aL = (g * pL / rL).clamp(min=eps).sqrt()
    aR = (g * pR / rR).clamp(min=eps).sqrt()

    # Normal wave speeds based on vy
    SL = torch.minimum(vL - aL, vR - aR)
    SR = torch.maximum(vL + aL, vR + aR)

    num_S = pR - pL + rL * vL * (SL - vL) - rR * vR * (SR - vR)
    den_S = rL * (SL - vL) - rR * (SR - vR)
    Sstar = torch.where(den_S.abs() > eps, num_S / den_S, 0.5 * (vL + vR))

    EL = pL / (g - 1.0) + 0.5 * rL * (uL.square() + vL.square())
    ER = pR / (g - 1.0) + 0.5 * rR * (uR.square() + vR.square())

    GL = torch.stack([rL * vL, rL * uL * vL, rL * vL.square() + pL, vL * (EL + pL)], dim=1)
    GR = torch.stack([rR * vR, rR * uR * vR, rR * vR.square() + pR, vR * (ER + pR)], dim=1)

    dL = torch.where((SL - Sstar).abs() > eps, SL - Sstar, SL.new_full((), -eps).expand_as(SL))
    dR = torch.where((SR - Sstar).abs() > eps, SR - Sstar, SR.new_full((),  eps).expand_as(SR))
    fL = rL * (SL - vL) / dL
    fR = rR * (SR - vR) / dR

    dLvL = torch.where((SL - vL).abs() > eps, SL - vL, SL.new_full((), -eps).expand_as(SL))
    dRvR = torch.where((SR - vR).abs() > eps, SR - vR, SR.new_full((),  eps).expand_as(SR))
    EsL = fL * (EL / rL + (Sstar - vL) * (Sstar + pL / (rL * dLvL)))
    EsR = fR * (ER / rR + (Sstar - vR) * (Sstar + pR / (rR * dRvR)))

    # y-direction: tangential = u, normal = S*
    UL_star = torch.stack([fL, fL * uL, fL * Sstar, EsL], dim=1)
    UR_star = torch.stack([fR, fR * uR, fR * Sstar, EsR], dim=1)
    UL_cons = torch.stack([rL, rL * uL, rL * vL, EL], dim=1)
    UR_cons = torch.stack([rR, rR * uR, rR * vR, ER], dim=1)

    GL_star = GL + SL.unsqueeze(1) * (UL_star - UL_cons)
    GR_star = GR + SR.unsqueeze(1) * (UR_star - UR_cons)

    return torch.where(
        SL.unsqueeze(1) >= 0.0, GL,
        torch.where(Sstar.unsqueeze(1) >= 0.0, GL_star,
        torch.where(SR.unsqueeze(1) >= 0.0, GR_star, GR)),
    )


# ─── Viscous terms ────────────────────────────────────────────────────────────

def _viscous_rhs(
    q_prim: torch.Tensor,
    mu: float,
    zeta: float,
    kappa: float,
    dx: float,
    dy: float,
) -> torch.Tensor:
    """
    Viscous + heat-conduction RHS (cell-centred 2nd-order central differences).
    Stress tensor: Newtonian with shear viscosity mu (η) and bulk viscosity zeta (ζ).
    τ_xx = 2η ∂u/∂x + (ζ - 2η/3) ∇·v,  τ_yy = 2η ∂v/∂y + (ζ - 2η/3) ∇·v,
    τ_xy = η(∂u/∂y + ∂v/∂x).
    Temperature: T = p/rho  (ideal gas with R=1).
    Returns (batch,4,nx,ny) contribution to ∂_t(conserved).
    """
    rho, u, v, p = q_prim[:, 0], q_prim[:, 1], q_prim[:, 2], q_prim[:, 3]
    T = p / rho.clamp(min=1e-12)

    def cd_x(f: torch.Tensor) -> torch.Tensor:
        return (torch.roll(f, -1, -2) - torch.roll(f, 1, -2)) / (2.0 * float(dx))

    def cd_y(f: torch.Tensor) -> torch.Tensor:
        return (torch.roll(f, -1, -1) - torch.roll(f, 1, -1)) / (2.0 * float(dy))

    du_dx, du_dy = cd_x(u), cd_y(u)
    dv_dx, dv_dy = cd_x(v), cd_y(v)
    dT_dx, dT_dy = cd_x(T), cd_y(T)

    div_v = du_dx + dv_dy
    bulk_coef = float(zeta) - (2.0 / 3.0) * float(mu)
    tau_xx = 2.0 * float(mu) * du_dx + bulk_coef * div_v
    tau_yy = 2.0 * float(mu) * dv_dy + bulk_coef * div_v
    tau_xy = float(mu) * (du_dy + dv_dx)

    # Momentum viscous flux divergences
    vis_mu = cd_x(tau_xx) + cd_y(tau_xy)
    vis_mv = cd_x(tau_xy) + cd_y(tau_yy)

    # Energy: ∂_x(τxx·u + τxy·v + κ∂_xT) + ∂_y(τxy·u + τyy·v + κ∂_yT)
    Fvx_E = tau_xx * u + tau_xy * v + float(kappa) * dT_dx
    Fvy_E = tau_xy * u + tau_yy * v + float(kappa) * dT_dy
    vis_E = cd_x(Fvx_E) + cd_y(Fvy_E)

    return torch.stack([torch.zeros_like(rho), vis_mu, vis_mv, vis_E], dim=1)


# ─── Full RHS and adaptive time-step ─────────────────────────────────────────

def _inviscid_rhs(u_cons: torch.Tensor, gamma: float, dx: float, dy: float) -> torch.Tensor:
    q = conserved_to_primitive(u_cons, gamma)
    qL_x, qR_x = _muscl_x(q)
    Fx = _hllc_flux_x(qL_x, qR_x, gamma)
    qL_y, qR_y = _muscl_y(q)
    Fy = _hllc_flux_y(qL_y, qR_y, gamma)
    # Flux divergence: (F_{i+1/2} - F_{i-1/2}) / h
    return (
        -(Fx - torch.roll(Fx, 1, dims=2)) / float(dx)
        - (Fy - torch.roll(Fy, 1, dims=3)) / float(dy)
    )


def _compute_rhs(
    u_cons: torch.Tensor,
    gamma: float,
    mu: float,
    zeta: float,
    kappa: float,
    dx: float,
    dy: float,
) -> torch.Tensor:
    rhs = _inviscid_rhs(u_cons, gamma, dx, dy)
    if mu > 0.0 or zeta > 0.0 or kappa > 0.0:
        q = conserved_to_primitive(u_cons, gamma)
        rhs = rhs + _viscous_rhs(q, mu, zeta, kappa, dx, dy)
    return rhs


def _max_wavespeed(u_cons: torch.Tensor, gamma: float) -> float:
    """Max |v| + a across the entire batch and all grid points."""
    q = conserved_to_primitive(u_cons, gamma)
    a = (float(gamma) * q[:, 3] / q[:, 0].clamp(min=1e-12)).clamp(min=0.0).sqrt()
    sx = (q[:, 1].abs() + a).amax().item()
    sy = (q[:, 2].abs() + a).amax().item()
    return max(sx, sy, 1e-8)


def _adaptive_dt(u_cons: torch.Tensor, gamma: float, mu: float, zeta: float,
                 kappa: float, dx: float, dy: float, cfl: float) -> float:
    """CFL-based adaptive timestep (acoustic + viscous)."""
    dt = float(cfl) * min(dx, dy) / _max_wavespeed(u_cons, gamma)
    if mu > 0.0 or zeta > 0.0 or kappa > 0.0:
        g = float(gamma)
        # 4/3*mu + zeta is the diagonal viscous coefficient (bulk+shear)
        max_diff = max(4.0 / 3.0 * mu + zeta, kappa * (g - 1.0))
        dt_vis = 0.5 * min(dx, dy) ** 2 / (max_diff + 1e-30)
        dt = min(dt, dt_vis)
    return dt


def _ssprk2_step(
    u_cons: torch.Tensor,
    dt: float,
    gamma: float,
    mu: float,
    zeta: float,
    kappa: float,
    dx: float,
    dy: float,
) -> torch.Tensor:
    """
    One SSP-RK2 (Heun) step.  Positivity is enforced by projecting back
    through primitive→conserved after each stage.
    """
    L1 = _compute_rhs(u_cons, gamma, mu, zeta, kappa, dx, dy)
    u1 = primitive_to_conserved(conserved_to_primitive(u_cons + float(dt) * L1, gamma), gamma)

    L2 = _compute_rhs(u1, gamma, mu, zeta, kappa, dx, dy)
    u2 = 0.5 * (u_cons + u1 + float(dt) * L2)
    return primitive_to_conserved(conserved_to_primitive(u2, gamma), gamma)


# ─── Trajectory solver ────────────────────────────────────────────────────────

def solve_cfd2d_trajectory(
    u0_prim: torch.Tensor,
    t_final: float,
    record_dt: float,
    gamma: float = 5.0 / 3.0,
    mu: float = 1e-3,
    zeta: float = -1.0,        # <0 → auto: zeta = mu  (PDEBench: η = ζ)
    kappa: float = -1.0,       # <0 → auto from prandtl
    prandtl: float = 0.72,
    cfl: float = 0.45,
    max_substeps: int = 200_000,
    store_primitive: bool = True,
) -> torch.Tensor:
    """
    Integrate 2D compressible NS from u0_prim to t_final, recording snapshots
    every record_dt time units.

    Parameters
    ----------
    u0_prim : (batch,4,nx,ny) or (4,nx,ny) primitive initial state
    Returns
    -------
    trajectory : (batch, n_records+1, 4, nx, ny)  in primitive or conserved form
    """
    squeeze = u0_prim.dim() == 3
    if squeeze:
        u0_prim = u0_prim.unsqueeze(0)
    if u0_prim.dim() != 4 or u0_prim.shape[1] != 4:
        raise ValueError("u0_prim must have shape (batch,4,nx,ny)")

    t_final   = float(t_final)
    record_dt = float(record_dt)
    n_records = int(round(t_final / record_dt))
    if abs(n_records * record_dt - t_final) > 1e-10 * t_final:
        raise ValueError("t_final must be an integer multiple of record_dt")

    device = u0_prim.device
    _, _, n_x, n_y = u0_prim.shape
    dx = 1.0 / float(n_x)
    dy = 1.0 / float(n_y)

    g = float(gamma)
    zeta_val  = float(mu) if float(zeta) < 0.0 else float(zeta)
    kappa_val = (float(mu) * g / ((g - 1.0) * float(prandtl))) if float(kappa) < 0.0 else float(kappa)

    # Integrate in float64 for numerical stability
    u_cons = primitive_to_conserved(u0_prim, gamma).to(dtype=torch.float64)

    def _snap() -> torch.Tensor:
        q = conserved_to_primitive(u_cons, gamma)
        if store_primitive:
            return q.to(dtype=torch.float32)
        return u_cons.to(dtype=torch.float32)

    states = [_snap()]
    t_now = 0.0

    for _ in range(n_records):
        t_target = t_now + record_dt
        substep = 0
        while t_now < t_target - 1e-14:
            dt = _adaptive_dt(u_cons, gamma, float(mu), zeta_val, kappa_val, dx, dy, float(cfl))
            dt = min(dt, t_target - t_now)
            dt = max(dt, 1e-16)
            u_cons = _ssprk2_step(u_cons, dt, gamma, float(mu), zeta_val, kappa_val, dx, dy)
            t_now += dt
            substep += 1
            if substep > int(max_substeps):
                raise RuntimeError(
                    f"Exceeded max_substeps={max_substeps} within one record interval. "
                    "Try a lower Mach number, larger viscosity, or stricter CFL."
                )
        states.append(_snap())

    traj = torch.stack(states, dim=1)  # (batch, n_records+1, 4, nx, ny)
    return traj.squeeze(0) if squeeze else traj


# ─── Dataset generation ───────────────────────────────────────────────────────

def _slice_split(data: Dict[str, torch.Tensor], start: int, end: int) -> Dict[str, torch.Tensor]:
    return {k: v[start:end].clone() for k, v in data.items()}


def generate_cfd2d_dataset_splits(
    n_x: int,                   # dataset (output) spatial resolution
    n_y: int,
    n_steps: int,               # number of stored time intervals  (PDEBench: 20)
    t_final: float,             # end time                          (PDEBench: 2.0)
    n_train: int,
    n_val: int,
    n_test: int,
    gamma: float = 5.0 / 3.0,
    mu: float = 1e-6,
    zeta: float = -1.0,        # <0 → auto: zeta = mu  (PDEBench: η = ζ)
    prandtl: float = 0.72,
    seed: int = 42,
    record_dt: float = -1.0,   # <0 → t_final/n_steps
    cfl: float = 0.45,
    solver_factor: int = 1,    # solve at (n_x*solver_factor)×(n_y*solver_factor),
                                # then avg-pool down to n_x×n_y
    rho0: float = 1.0,
    rho_amp: float = 0.1,
    mach_min: float = 0.1,
    mach_max: float = 0.1,
    p_amp: float = 0.05,
    ic_type: str = "grf",
    n_modes_min: int = 4,
    n_modes_max: int = 16,
    k_max: int = 4,
    grf_ls_min: float = 0.05,
    grf_ls_max: float = 0.15,
    store_primitive: bool = True,
    chunk_size: int = 8,
    show_progress: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Generate train / val / test splits for 2D compressible NS.

    With solver_factor > 1, ICs and trajectories are computed at
    (n_x * solver_factor) × (n_y * solver_factor) and spatially
    avg-pooled to n_x × n_y before storage.

    Returns a dict with keys 'train', 'val', 'test', 'meta'.
    Each split contains tensors 'f', 'u0', 'u_traj'.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if t_final <= 0.0:
        raise ValueError("t_final must be > 0")

    sf  = int(solver_factor)
    nxs = n_x * sf          # solver spatial resolution
    nys = n_y * sf

    rdt = float(t_final) / float(n_steps) if float(record_dt) < 0.0 else float(record_dt)
    expected = int(round(float(t_final) / rdt))
    if expected != n_steps:
        raise ValueError("n_steps must equal round(t_final / record_dt)")

    total = int(n_train + n_val + n_test)
    rng_torch = torch.random.get_rng_state()
    rng_np    = np.random.get_state()
    torch.manual_seed(seed)
    np.random.seed(seed)

    u0_all = sample_cfd2d_initial_conditions(
        n_x=nxs, n_y=nys, n_samples=total,
        gamma=gamma, rho0=rho0, rho_amp=rho_amp,
        mach_min=mach_min, mach_max=mach_max, p_amp=p_amp,
        ic_type=ic_type, n_modes_min=n_modes_min, n_modes_max=n_modes_max, k_max=k_max,
        grf_ls_min=grf_ls_min, grf_ls_max=grf_ls_max,
        device=device, dtype=dtype,
    )
    f_all = torch.zeros(total, n_x, n_y, dtype=dtype)

    chunk_starts = range(0, total, int(chunk_size))
    if show_progress and tqdm is not None:
        chunk_starts = tqdm(
            chunk_starts,
            total=(total + int(chunk_size) - 1) // int(chunk_size),
            desc="solve trajectories",
            leave=False,
        )

    traj_chunks = []
    for start in chunk_starts:
        end = min(int(start) + int(chunk_size), total)
        u0_ch = u0_all[start:end].to(device=device, dtype=dtype)
        traj_ch = solve_cfd2d_trajectory(
            u0_prim=u0_ch,
            t_final=float(t_final),
            record_dt=rdt,
            gamma=gamma,
            mu=mu,
            zeta=zeta,
            prandtl=prandtl,
            cfl=cfl,
            store_primitive=store_primitive,
        )  # (chunk, n_steps+1, 4, nxs, nys)
        if sf > 1:
            B, T, C, H, W = traj_ch.shape
            traj_ch = F.avg_pool2d(
                traj_ch.reshape(B * T, C, H, W), kernel_size=sf
            ).reshape(B, T, C, n_x, n_y)
        traj_chunks.append(traj_ch.cpu())

    u_traj = torch.cat(traj_chunks, dim=0)
    u0_stored = u_traj[:, 0].clone()

    all_data = {"f": f_all.cpu(), "u0": u0_stored, "u_traj": u_traj}
    train_end = int(n_train)
    val_end   = int(n_train + n_val)

    g = float(gamma)
    zeta_val  = float(mu) if float(zeta) < 0.0 else float(zeta)
    kappa_val = float(mu) * g / ((g - 1.0) * float(prandtl))
    splits: Dict[str, Dict] = {
        "train": _slice_split(all_data, 0, train_end),
        "val":   _slice_split(all_data, train_end, val_end),
        "test":  _slice_split(all_data, val_end, total),
        "meta": {
            "dataset_version":  DATASET_VERSION,
            "equation":         "compressible_navier_stokes_2d",
            "domain":           "unit_torus",
            "periodic":         True,
            "n_x":              int(n_x),
            "n_y":              int(n_y),
            "solver_factor":    sf,
            "n_x_solver":       nxs,
            "n_y_solver":       nys,
            "n_steps":          int(n_steps),
            "t_final":          float(t_final),
            "record_dt":        float(rdt),
            "gamma":            float(gamma),
            "mu":               float(mu),
            "zeta":             float(zeta_val),
            "kappa":            float(kappa_val),
            "prandtl":          float(prandtl),
            "cfl":              float(cfl),
            "rho0":             float(rho0),
            "rho_amp":          float(rho_amp),
            "mach_min":         float(mach_min),
            "mach_max":         float(mach_max),
            "p_amp":            float(p_amp),
            "ic_type":          str(ic_type),
            "n_modes_min":      int(n_modes_min),
            "n_modes_max":      int(n_modes_max),
            "k_max":            int(k_max),
            "grf_ls_min":       float(grf_ls_min),
            "grf_ls_max":       float(grf_ls_max),
            "store_primitive":  bool(store_primitive),
            "state_channels":   STATE_CHANNELS,
            "state_names":      list(STATE_NAMES_PRIMITIVE if store_primitive else STATE_NAMES_CONSERVED),
            "n_train":          int(n_train),
            "n_val":            int(n_val),
            "n_test":           int(n_test),
            "seed":             int(seed),
            "chunk_size":       int(chunk_size),
            "device":           str(device),
        },
    }

    torch.random.set_rng_state(rng_torch)
    np.random.set_state(rng_np)
    return splits


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate 2D compressible NS dataset splits")
    # PDEBench-aligned defaults: 64×64 output, 256×256 solver, 20 steps over t=2.0
    p.add_argument("--n-x",           type=int,   default=64)
    p.add_argument("--n-y",           type=int,   default=64)
    p.add_argument("--solver-factor", type=int,   default=4,
                   help="Solve at (n_x*solver_factor)×(n_y*solver_factor), avg-pool to n_x×n_y")
    p.add_argument("--n-steps",       type=int,   default=20)
    p.add_argument("--t-final",       type=float, default=2.0)
    p.add_argument("--record-dt",     type=float, default=-1.0,
                   help="Snapshot spacing; -1 → t_final/n_steps")
    p.add_argument("--gamma",         type=float, default=5.0 / 3.0)
    p.add_argument("--mu",            type=float, default=1e-6)
    p.add_argument("--zeta",          type=float, default=-1.0,
                   help="Bulk viscosity; -1 → auto: zeta = mu (PDEBench: η = ζ)")
    p.add_argument("--prandtl",       type=float, default=0.72)
    p.add_argument("--cfl",           type=float, default=0.45)
    p.add_argument("--rho0",          type=float, default=1.0)
    p.add_argument("--rho-amp",       type=float, default=0.1)
    p.add_argument("--mach-min",      type=float, default=0.1)
    p.add_argument("--mach-max",      type=float, default=0.1)
    p.add_argument("--p-amp",         type=float, default=0.05)
    p.add_argument("--ic-type",       type=str,   default="grf",
                   choices=["sinusoidal", "grf"])
    p.add_argument("--n-modes-min",   type=int,   default=4)
    p.add_argument("--n-modes-max",   type=int,   default=16)
    p.add_argument("--k-max",         type=int,   default=4)
    p.add_argument("--grf-ls-min",    type=float, default=0.05)
    p.add_argument("--grf-ls-max",    type=float, default=0.15)
    p.add_argument("--n-train",       type=int,   default=800)
    p.add_argument("--n-val",         type=int,   default=100)
    p.add_argument("--n-test",        type=int,   default=100)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--chunk-size",    type=int,   default=8)
    p.add_argument("--device",        type=str,   default="cuda:0")
    p.add_argument("--no-progress",   action="store_true")
    p.add_argument("--store-conserved", action="store_true",
                   help="Store conserved (rho,rho*vx,rho*vy,E) instead of primitive")
    p.add_argument(
        "--dataset-path", type=str,
        default="grad_flow_l2/cfd2d/datasets/cfd2d_n1000_t2.pt",
    )
    p.add_argument("--settings-path", type=str, default=None)
    return p.parse_args()


def _print_stats(splits: Dict[str, Dict[str, torch.Tensor]]) -> None:
    meta = splits.get("meta", {})
    print("Meta:", {k: v for k, v in meta.items() if not isinstance(v, list)})
    for name in ("train", "val", "test"):
        sp = splits[name]
        u_traj = sp["u_traj"]
        if u_traj.shape[0] == 0:
            print(f"  {name}: empty"); continue
        u0 = sp["u0"]
        rho_mean = u0[:, 0].mean().item()
        vx_rms   = u0[:, 1].pow(2).mean().sqrt().item()
        p_mean   = u0[:, 3].mean().item()
        u_max    = u_traj.abs().amax().item()
        print(
            f"  {name}: shape={tuple(u_traj.shape)}, "
            f"rho0_mean={rho_mean:.4f}, vx0_rms={vx_rms:.4f}, "
            f"p0_mean={p_mean:.4f}, u_traj_max={u_max:.4f}"
        )


def main(args: argparse.Namespace) -> None:
    splits = generate_cfd2d_dataset_splits(
        n_x=args.n_x,
        n_y=args.n_y,
        n_steps=args.n_steps,
        t_final=args.t_final,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        gamma=args.gamma,
        mu=args.mu,
        zeta=args.zeta,
        prandtl=args.prandtl,
        solver_factor=args.solver_factor,
        seed=args.seed,
        record_dt=args.record_dt,
        cfl=args.cfl,
        rho0=args.rho0,
        rho_amp=args.rho_amp,
        mach_min=args.mach_min,
        mach_max=args.mach_max,
        p_amp=args.p_amp,
        ic_type=args.ic_type,
        n_modes_min=args.n_modes_min,
        n_modes_max=args.n_modes_max,
        k_max=args.k_max,
        grf_ls_min=args.grf_ls_min,
        grf_ls_max=args.grf_ls_max,
        store_primitive=not args.store_conserved,
        chunk_size=args.chunk_size,
        show_progress=not args.no_progress,
        device=args.device,
        dtype=torch.float32,
    )

    out_dir = os.path.dirname(args.dataset_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_dataset_splits(splits, args.dataset_path)
    print(f"Saved dataset to: {args.dataset_path}")
    _print_stats(splits)

    if args.settings_path:
        sd = os.path.dirname(args.settings_path)
        if sd:
            os.makedirs(sd, exist_ok=True)
        with open(args.settings_path, "w", encoding="utf-8") as fh:
            json.dump({"cli_args": vars(args), "meta": splits["meta"]}, fh, indent=2)
            fh.write("\n")
        print(f"Saved settings to: {args.settings_path}")


if __name__ == "__main__":
    main(parse_args())
