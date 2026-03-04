"""
Utility functions for the energy-based PDE solver.
"""

import torch
import numpy as np


def solve_poisson_exact(
    a: torch.Tensor,
    f: torch.Tensor,
    h: float
) -> torch.Tensor:
    """
    Solve the 1D Poisson equation exactly using direct linear solve.
    
    The discretized system is: K(a) u = f
    
    Args:
        a: Coefficient on full grid, shape (batch, n+2) or (n+2,)
        f: Forcing on interior grid, shape (batch, n) or (n,)
        h: Grid spacing
        
    Returns:
        u: Exact solution on interior grid, same shape as f
    """
    from energy import build_stiffness_matrix
    
    squeeze_output = False
    if a.dim() == 1:
        a = a.unsqueeze(0)
        f = f.unsqueeze(0)
        squeeze_output = True
    
    batch_size = a.shape[0]
    n = f.shape[1]
    
    K = build_stiffness_matrix(a, h)  # (batch, n, n)
    
    # Solve K u = f for each batch
    # Note: actual PDE solve would be K u = h * f, but we store
    # the energy as (h/2) u^T K u - h f^T u, so minimizer satisfies h K u = h f
    # i.e., K u = f
    u = torch.linalg.solve(K, f.unsqueeze(-1)).squeeze(-1)
    
    if squeeze_output:
        u = u.squeeze(0)
    
    return u


def compute_relative_l2_error(
    u_pred: torch.Tensor,
    u_exact: torch.Tensor,
    h: float = None
) -> torch.Tensor:
    """
    Compute relative L2 error.
    
    Args:
        u_pred: Predicted solution, shape (batch, n) or (n,)
        u_exact: Exact solution, same shape
        h: Grid spacing (optional, for proper L2 norm. If None, uses discrete norm)
        
    Returns:
        error: Relative error, scalar or (batch,)
    """
    diff = u_pred - u_exact
    
    if h is not None:
        # Proper L2 norm with quadrature
        error_norm = (h * (diff ** 2).sum(dim=-1)) ** 0.5
        exact_norm = (h * (u_exact ** 2).sum(dim=-1)) ** 0.5
    else:
        # Discrete norm
        error_norm = (diff ** 2).sum(dim=-1) ** 0.5
        exact_norm = (u_exact ** 2).sum(dim=-1) ** 0.5
    
    return error_norm / (exact_norm + 1e-8)


def pad_solution(u_interior: torch.Tensor) -> torch.Tensor:
    """
    Pad interior solution with zero boundary values.
    
    Args:
        u_interior: Interior values, shape (..., n)
        
    Returns:
        u_full: Full solution with boundaries, shape (..., n+2)
    """
    shape = u_interior.shape[:-1]
    zeros = torch.zeros(*shape, 1, device=u_interior.device, dtype=u_interior.dtype)
    return torch.cat([zeros, u_interior, zeros], dim=-1)


def get_grid_points(n_interior: int, include_boundary: bool = True) -> torch.Tensor:
    """
    Get grid point locations.
    
    Args:
        n_interior: Number of interior points
        include_boundary: If True, include x=0 and x=1
        
    Returns:
        x: Grid points
    """
    n_full = n_interior + 2
    h = 1.0 / (n_interior + 1)
    
    if include_boundary:
        return torch.linspace(0, 1, n_full)
    else:
        return torch.linspace(h, 1 - h, n_interior)


class AverageMeter:
    """Track running average of a metric."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from energy import build_stiffness_matrix, compute_energy
    from data import sample_coefficient_a, sample_forcing_f
    
    torch.manual_seed(42)
    
    n = 20
    h = 1.0 / (n + 1)
    
    # Generate data
    a = sample_coefficient_a(n, 5)
    f = sample_forcing_f(n, 5)
    
    # Solve exactly
    u_exact = solve_poisson_exact(a, f, h)
    
    print(f"Exact solution shape: {u_exact.shape}")
    
    # Verify solution minimizes energy
    energies_at_exact = compute_energy(u_exact, a, f, h)
    print(f"Energy at exact solution: {energies_at_exact}")
    
    # Compare with perturbed solution
    u_perturbed = u_exact + 0.1 * torch.randn_like(u_exact)
    energies_perturbed = compute_energy(u_perturbed, a, f, h)
    print(f"Energy at perturbed: {energies_perturbed}")
    print(f"Energy increased: {(energies_perturbed > energies_at_exact).all()}")
