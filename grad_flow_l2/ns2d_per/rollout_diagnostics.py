"""Rollout diagnostics for periodic 2D vorticity fields."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def _scalar_vorticity(u: torch.Tensor) -> torch.Tensor:
    if u.dim() == 4:
        return u
    if u.dim() == 5 and u.shape[2] == 1:
        return u[:, :, 0]
    raise ValueError(f"expected vorticity trajectory shape (batch,time,n_x,n_y), got {tuple(u.shape)}")


def _wavenumber_grids(n_x: int, n_y: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    kx = 2.0 * torch.pi * torch.fft.fftfreq(n_x, d=1.0 / float(n_x), device=device).to(dtype=dtype)
    ky = 2.0 * torch.pi * torch.fft.fftfreq(n_y, d=1.0 / float(n_y), device=device).to(dtype=dtype)
    return torch.meshgrid(kx, ky, indexing="ij")


def _enstrophy(u: torch.Tensor, area: float) -> torch.Tensor:
    u = _scalar_vorticity(u)
    return 0.5 * float(area) * torch.sum(u.square(), dim=(-2, -1))


def _palinstrophy(u: torch.Tensor, area: float) -> torch.Tensor:
    u = _scalar_vorticity(u)
    n_x, n_y = int(u.shape[-2]), int(u.shape[-1])
    u_hat = torch.fft.fft2(u, dim=(-2, -1), norm="ortho")
    kx_grid, ky_grid = _wavenumber_grids(n_x, n_y, u.device, u.real.dtype)
    grad_weight = kx_grid.square() + ky_grid.square()
    power = u_hat.real.square() + u_hat.imag.square()
    return 0.5 * float(area) * torch.sum(power * grad_weight, dim=(-2, -1))


def _radial_energy_spectrum(u: torch.Tensor, area: float) -> torch.Tensor:
    u = _scalar_vorticity(u)
    n_x, n_y = int(u.shape[-2]), int(u.shape[-1])
    u_hat = torch.fft.fft2(u, dim=(-2, -1), norm="ortho")
    kx = torch.fft.fftfreq(n_x, d=1.0 / float(n_x), device=u.device)
    ky = torch.fft.fftfreq(n_y, d=1.0 / float(n_y), device=u.device)
    kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing="ij")
    shell = torch.round(torch.sqrt(kx_grid.square() + ky_grid.square())).long()
    n_shells = int(shell.max().item()) + 1

    angular_k_sq = (2.0 * torch.pi) ** 2 * (kx_grid.square() + ky_grid.square())
    vorticity_power = u_hat.real.square() + u_hat.imag.square()
    velocity_power = torch.where(angular_k_sq > 0, vorticity_power / angular_k_sq, torch.zeros_like(vorticity_power))
    power = 0.5 * float(area) * velocity_power
    flat_power = power.reshape(-1, n_x * n_y)
    flat_shell = shell.reshape(1, n_x * n_y).expand(flat_power.shape[0], -1)
    spectrum = torch.zeros(flat_power.shape[0], n_shells, device=u.device, dtype=u.real.dtype)
    spectrum.scatter_add_(dim=1, index=flat_shell, src=flat_power)
    return spectrum.reshape(*u.shape[:-2], n_shells)


def _scalar_relative_diagnostics(pred: torch.Tensor, ref: torch.Tensor, eps: float) -> tuple[np.ndarray, float, np.ndarray]:
    abs_ref = ref.abs()
    curve = torch.mean((pred - ref).abs() / (abs_ref + eps), dim=0).detach().cpu().numpy().astype(np.float64)
    samples = torch.sqrt(torch.sum((pred - ref).square(), dim=1) / (torch.sum(ref.square(), dim=1) + eps))
    samples_np = samples.detach().cpu().numpy().astype(np.float64)
    aggregate = float(np.nanmean(samples_np))
    return curve, aggregate, samples_np


def compute_rollout_diagnostics(u_pred: torch.Tensor, u_ref: torch.Tensor, area: float) -> Dict[str, object]:
    """Compute invariant and spectrum rollout diagnostics for predicted snapshots only."""
    u_pred = u_pred[:, 1:]
    u_ref = u_ref[:, 1:]
    ens_curve, ens_agg, ens_samples = _scalar_relative_diagnostics(
        _enstrophy(u_pred, area), _enstrophy(u_ref, area), eps=1e-24
    )
    pal_curve, pal_agg, pal_samples = _scalar_relative_diagnostics(
        _palinstrophy(u_pred, area), _palinstrophy(u_ref, area), eps=1e-24
    )

    spec_pred = _radial_energy_spectrum(u_pred, area)
    spec_ref = _radial_energy_spectrum(u_ref, area)
    spec_rel = torch.sqrt(
        torch.sum((spec_pred - spec_ref).square(), dim=-1) / (torch.sum(spec_ref.square(), dim=-1) + 1e-24)
    )
    spec_curve = torch.mean(spec_rel, dim=0).detach().cpu().numpy().astype(np.float64)
    spec_samples_t = torch.sqrt(
        torch.sum((spec_pred - spec_ref).square(), dim=(1, 2)) / (torch.sum(spec_ref.square(), dim=(1, 2)) + 1e-24)
    )
    spec_samples = spec_samples_t.detach().cpu().numpy().astype(np.float64)
    spec_agg = float(np.nanmean(spec_samples))

    return {
        "enstrophy_rel_curve_mean": ens_curve,
        "enstrophy_rel_error": ens_agg,
        "enstrophy_rel_samples": ens_samples,
        "palinstrophy_rel_curve_mean": pal_curve,
        "palinstrophy_rel_error": pal_agg,
        "palinstrophy_rel_samples": pal_samples,
        "spectrum_rel_curve_mean": spec_curve,
        "spectrum_rel_error": spec_agg,
        "spectrum_rel_samples": spec_samples,
    }
