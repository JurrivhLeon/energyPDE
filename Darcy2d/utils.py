"""
Utility functions for 2D Darcy flow solver.
"""

import torch
import numpy as np


def solve_darcy_exact(
    a: torch.Tensor,
    f: torch.Tensor,
    h: float
) -> torch.Tensor:
    """
    Solve the 2D Darcy equation exactly using direct linear solve.
    
    For constant a=1 (Poisson), uses spectral method with 2D DST.
    For variable a, uses iterative conjugate gradient.
    
    Args:
        a: Permeability on full grid, shape (batch, N+2, N+2) or (N+2, N+2)
        f: Forcing on interior grid, shape (batch, N, N) or (N, N)
        h: Grid spacing
        
    Returns:
        u: Exact solution on interior grid, same shape as f
    """
    squeeze_output = False
    if a.dim() == 2:
        a = a.unsqueeze(0)
        f = f.unsqueeze(0)
        squeeze_output = True
    
    batch_size, N_full, _ = a.shape
    N = N_full - 2
    
    # Check if a is constant (standard Poisson)
    a_interior = a[:, 1:-1, 1:-1]
    is_constant_a = (a_interior.max() - a_interior.min() < 1e-6).all()
    
    if is_constant_a:
        # Use spectral method for constant a
        u = _solve_poisson_spectral(f, h, a_interior[0, 0, 0].item())
    else:
        # Use conjugate gradient for variable a
        u = _solve_darcy_cg(a, f, h)
    
    if squeeze_output:
        u = u.squeeze(0)
    
    return u


def _solve_poisson_spectral(
    f: torch.Tensor,
    h: float,
    a_val: float = 1.0
) -> torch.Tensor:
    """
    Solve -a*Δu = f using 2D DST (spectral method).
    
    Eigenvalues of -Δ in sine basis: λ_{m,n} = (2/h²)(2 - cos(mπh) - cos(nπh))
    """
    from preconditioned_langevin import dst2_type1, idst2_type1
    
    batch_size, N, _ = f.shape
    device = f.device
    dtype = f.dtype
    
    # Mode indices
    m = torch.arange(1, N + 1, device=device, dtype=dtype)
    n = torch.arange(1, N + 1, device=device, dtype=dtype)
    M, N_grid = torch.meshgrid(m, n, indexing='ij')
    
    # Eigenvalues of discrete Laplacian: λ = (4/h²)(sin²(mπh/2) + sin²(nπh/2))
    # For continuous approx: λ ≈ (mπ)² + (nπ)²
    lambda_mn = (2.0 / (h * h)) * (
        2 - torch.cos(M * np.pi * h) - torch.cos(N_grid * np.pi * h)
    )
    
    # Transform f to spectral domain
    f_hat = dst2_type1(f)
    
    # Solve: a * λ * u_hat = f_hat  =>  u_hat = f_hat / (a * λ)
    u_hat = f_hat / (a_val * lambda_mn + 1e-10)
    
    # Transform back
    u = idst2_type1(u_hat)
    
    return u


def _solve_darcy_cg(
    a: torch.Tensor,
    f: torch.Tensor,
    h: float,
    max_iter: int = 1000,
    tol: float = 1e-6
) -> torch.Tensor:
    """
    Solve -∇·(a∇u) = f using conjugate gradient.
    
    This applies the discrete operator via finite differences.
    """
    from energy import compute_energy_gradient
    
    batch_size, N, _ = f.shape
    device = f.device
    
    # Initial guess
    u = torch.zeros_like(f)
    
    # Residual: r = f - Au (where Au corresponds to -∇·(a∇u))
    # We use the gradient of the quadratic part of energy
    beta_fake = 1.0  # We want Au, which is grad of (1/2)u^T A u
    
    def apply_operator(v):
        """Apply the discrete Darcy operator to v."""
        # grad_u [(h²/2) sum a|∇u|²] = h² * (discrete divergence of a*∇u)
        grad = compute_energy_gradient(v, a, torch.zeros_like(f), h, beta=1.0)
        return grad / (h * h)  # Undo the h factor from energy gradient
    
    # CG iteration
    Au = apply_operator(u)
    r = f - Au
    p = r.clone()
    rsold = (r * r).sum(dim=(-2, -1))
    
    for i in range(max_iter):
        Ap = apply_operator(p)
        pAp = (p * Ap).sum(dim=(-2, -1))
        alpha = rsold / (pAp + 1e-10)
        
        u = u + alpha.unsqueeze(-1).unsqueeze(-1) * p
        r = r - alpha.unsqueeze(-1).unsqueeze(-1) * Ap
        
        rsnew = (r * r).sum(dim=(-2, -1))
        
        if (rsnew.sqrt() < tol).all():
            break
        
        beta = rsnew / (rsold + 1e-10)
        p = r + beta.unsqueeze(-1).unsqueeze(-1) * p
        rsold = rsnew
    
    return u


def compute_relative_l2_error_2d(
    u_pred: torch.Tensor,
    u_exact: torch.Tensor,
    h: float = None
) -> torch.Tensor:
    """
    Compute relative L2 error for 2D fields.
    
    Args:
        u_pred: Predicted solution, shape (batch, N, N) or (N, N)
        u_exact: Exact solution, same shape
        h: Grid spacing (optional)
        
    Returns:
        error: Relative error, scalar or (batch,)
    """
    diff = u_pred - u_exact
    
    if h is not None:
        # Proper L2 norm with quadrature: ||u||² = h² Σ u²
        error_norm = (h * h * (diff ** 2).sum(dim=(-2, -1))) ** 0.5
        exact_norm = (h * h * (u_exact ** 2).sum(dim=(-2, -1))) ** 0.5
    else:
        # Discrete norm
        error_norm = (diff ** 2).sum(dim=(-2, -1)) ** 0.5
        exact_norm = (u_exact ** 2).sum(dim=(-2, -1)) ** 0.5
    
    return error_norm / (exact_norm + 1e-8)


def pad_solution_2d(u_interior: torch.Tensor) -> torch.Tensor:
    """
    Pad interior solution with zero boundary values.
    
    Args:
        u_interior: Interior values, shape (..., N, N)
        
    Returns:
        u_full: Full solution with boundaries, shape (..., N+2, N+2)
    """
    return torch.nn.functional.pad(u_interior, (1, 1, 1, 1), mode='constant', value=0.0)


def get_grid_points_2d(N: int, include_boundary: bool = True) -> tuple:
    """
    Get 2D grid point locations.
    
    Args:
        N: Number of interior points per dimension
        include_boundary: If True, include boundary points
        
    Returns:
        X, Y: Meshgrid tensors
    """
    h = 1.0 / (N + 1)
    
    if include_boundary:
        x = torch.linspace(0, 1, N + 2)
        y = torch.linspace(0, 1, N + 2)
    else:
        x = torch.linspace(h, 1 - h, N)
        y = torch.linspace(h, 1 - h, N)
    
    X, Y = torch.meshgrid(x, y, indexing='ij')
    return X, Y


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
    from energy import compute_energy
    from data import sample_coefficient_a, sample_forcing_f
    
    torch.manual_seed(42)
    
    N = 32
    h = 1.0 / (N + 1)
    
    # Test with constant a (uses spectral method)
    a_const = torch.ones(4, N + 2, N + 2)
    f = sample_forcing_f(N, 4, method='sinusoidal', amplitude=1.0)
    
    u_exact = solve_darcy_exact(a_const, f, h)
    print(f"Exact solution shape: {u_exact.shape}")
    
    # Verify solution minimizes energy
    energies_at_exact = compute_energy(u_exact, a_const, f, h, beta=1.0)
    print(f"Energy at exact solution: {energies_at_exact}")
    
    # Compare with perturbed solution
    u_perturbed = u_exact + 0.1 * torch.randn_like(u_exact)
    energies_perturbed = compute_energy(u_perturbed, a_const, f, h, beta=1.0)
    print(f"Energy at perturbed: {energies_perturbed}")
    print(f"Energy increased: {(energies_perturbed > energies_at_exact).all()}")
    
    # Test relative L2 error
    err = compute_relative_l2_error_2d(u_perturbed, u_exact, h)
    print(f"\nRelative L2 error of perturbed: {err}")
    
    # Test grid points
    X, Y = get_grid_points_2d(N, include_boundary=False)
    print(f"\nGrid X shape: {X.shape}, range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"Grid Y shape: {Y.shape}, range: [{Y.min():.3f}, {Y.max():.3f}]")
