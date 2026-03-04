"""
Utility functions for the 1D Burgers equation solver.
"""

import torch
import numpy as np

from reference_solver import (
    solve_burgers_reference,
    solve_burgers_trajectory,
    solve_burgers_pseudospectral
)


def get_grid_points(n_grid: int, L: float = 1.0, device: str = 'cpu') -> torch.Tensor:
    """
    Get periodic grid point locations [0, L) (excluding endpoint).
    
    Args:
        n_grid: Number of grid points
        L: Domain length
        device: Torch device
        
    Returns:
        x: Grid points, shape (n_grid,)
    """
    return torch.linspace(0, L, n_grid + 1, device=device)[:-1]


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
        h: Grid spacing (optional, for proper L2 norm)
        
    Returns:
        error: Relative error, scalar or (batch,)
    """
    diff = u_pred - u_exact
    
    if h is not None:
        error_norm = (h * (diff ** 2).sum(dim=-1)) ** 0.5
        exact_norm = (h * (u_exact ** 2).sum(dim=-1)) ** 0.5
    else:
        error_norm = (diff ** 2).sum(dim=-1) ** 0.5
        exact_norm = (u_exact ** 2).sum(dim=-1) ** 0.5
    
    return error_norm / (exact_norm + 1e-8)


def _spectral_derivatives_numpy(u: np.ndarray, L: float) -> tuple:
    """
    Compute spectral derivatives in NumPy (for reference solver).
    
    Args:
        u: Function values on periodic grid, shape (n,)
        L: Domain length
        
    Returns:
        u_x, u_xx: First and second derivatives
    """
    n = len(u)
    k = np.fft.fftfreq(n, d=L / n) * (2 * np.pi)
    
    u_hat = np.fft.fft(u)
    u_x = np.real(np.fft.ifft(1j * k * u_hat))
    u_xx = np.real(np.fft.ifft(-(k ** 2) * u_hat))
    
    return u_x, u_xx


def solve_burgers_multi_step(
    u0: torch.Tensor,
    nu: float,
    dt: float,
    n_steps: int,
    L: float = 1.0,
    solver_dt: float = 1e-4
) -> torch.Tensor:
    """
    Solve multiple time steps of the Burgers equation.
    
    Each step advances by dt using the pseudo-spectral solver internally.
    
    Args:
        u0: Initial condition, shape (n,) or (batch, n)
        nu: Viscosity
        dt: Time step between snapshots
        n_steps: Number of time steps
        L: Domain length
        solver_dt: Internal solver time step
        
    Returns:
        trajectory: Solution at each time step, shape (n_steps+1, ..., n)
    """
    trajectory = [u0.clone()]
    u = u0.clone()
    
    for step in range(n_steps):
        u = solve_burgers_reference(u, nu, dt, L, solver_dt=solver_dt)
        trajectory.append(u.clone())
    
    return torch.stack(trajectory, dim=0)


class AverageMeter:
    """Computes and stores the average and current value."""
    
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


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from energy import compute_residual
    
    torch.manual_seed(42)
    
    n = 64
    L = 1.0
    h = L / n
    nu = 0.01
    dt = 0.01
    
    x = get_grid_points(n, L)
    
    # Smooth initial condition (periodic on [0, L])
    u_curr = torch.sin(2 * np.pi * x / L)
    
    print("=== Reference Solver Test ===")
    
    # Solve one step
    u_next = solve_burgers_reference(u_curr, nu, dt, L)
    
    # Check residual
    R = compute_residual(u_next, u_curr, nu, dt, L)
    res_norm = (h * (R ** 2).sum()) ** 0.5
    print(f"  Residual L2 norm: {res_norm.item():.2e}")
    print(f"  Max residual: {R.abs().max().item():.2e}")
    
    # Multi-step solve
    print("\n=== Multi-step Solve ===")
    n_steps = 10
    traj = solve_burgers_multi_step(u_curr, nu, dt, n_steps, L)
    print(f"  Trajectory shape: {traj.shape}")
    
    # Check that energy of true solution is near zero
    from energy import compute_energy
    for step in range(1, n_steps + 1):
        H = compute_energy(traj[step], traj[step - 1], nu, dt, L)
        print(f"  Step {step}: energy = {H.item():.2e}")
    
    # Batch test
    print("\n=== Batch Test ===")
    u_batch = torch.stack([
        torch.sin(2 * np.pi * x / L),
        torch.sin(4 * np.pi * x / L),
        torch.cos(2 * np.pi * x / L)
    ], dim=0)
    u_next_batch = solve_burgers_reference(u_batch, nu, dt, L)
    print(f"  Input shape: {u_batch.shape}, Output shape: {u_next_batch.shape}")
    
    for i in range(3):
        R = compute_residual(u_next_batch[i], u_batch[i], nu, dt, L)
        print(f"  Sample {i}: residual = {(h * (R ** 2).sum()) ** 0.5:.2e}")
    
    # L2 error test
    print("\n=== L2 Error Test ===")
    u_perturbed = u_next + 0.01 * torch.randn_like(u_next)
    err = compute_relative_l2_error(u_perturbed, u_next, h)
    print(f"  Relative L2 error (small perturbation): {err.item():.4f}")
