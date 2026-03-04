"""
Utility functions for the nonlinear Poisson (Ginzburg-Landau) solver.
"""

import torch
import numpy as np
from energy import compute_energy_gradient

def get_grid_points(n_points, include_boundary=False, device='cpu'):
    """
    Get grid point locations.
    """
    if include_boundary:
        return torch.linspace(0, 1, n_points + 2, device=device)
    else:
        return torch.linspace(0, 1, n_points + 2, device=device)[1:-1]

def compute_relative_l2_error(u, u_ref, h):
    """Compute relative L2 error: ||u - u_ref|| / ||u_ref||."""
    diff = u - u_ref
    diff_norm = torch.sqrt(h * torch.sum(diff**2, dim=-1))
    ref_norm = torch.sqrt(h * torch.sum(u_ref**2, dim=-1))
    return diff_norm / (ref_norm + 1e-10)

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

def solve_newton_method(u0, s, h, K=None, max_iter=200, tol=1e-6):
    """
    Solve the nonlinear system F(u) = K*u + u^3 - u - s = 0 using Newton's method.
    Jacobian J = K + diag(3u^2 - 1).
    """
    u = u0.clone()
    batch_size, n = u.shape
    
    # Ensure K is available
    if K is None:
        raise ValueError("Stiffness matrix K must be provided for Newton solver")
        
    print(f"Computing reference solution (Newton's method)...")
    
    # Warm start with Gradient Descent
    # This helps get into the basin of attraction if u0=0 is poor or unstable 
    lr_warm = 1e-3
    print("  Warm starting with GD...", end="", flush=True)
    for i in range(1000):
        # Use proper energy gradient (includes h scaling)
        grad = compute_energy_gradient(u, s, h, K=K)
        u = u - lr_warm * grad
    print(" Done.")
    
    for k in range(max_iter):
        # 1. Evaluate Residual F(u)
        # F(u) = K u + u^3 - u - s
        
        # Linear part: K u
        # Handle batch dimension if needed, currently assuming batch=1 or shared K
        # K is (n, n), u is (batch, n) -> (batch, n)
        Ku = torch.matmul(u, K) 
        
        # Nonlinear part
        f_u = u**3 - u
        
        # Residual
        F = Ku + f_u - s # (batch, n)
        
        # Check convergence
        res_norm = torch.norm(F).item()
        if res_norm < tol:
            print(f"  Newton converged in {k} iterations (residual={res_norm:.4e})")
            return u
            
        # 2. Compute Jacobian and update
        # J = K + diag(3u^2 - 1)
        # We can vectorize this using broadcasting
        
        # K is (n, n), u is (batch, n)
        # J is (batch, n, n)
        
        # Diagonal term: (batch, n)
        diag_term = 3 * u**2 - 1
        
        # Expand K to batch size: (1, n, n) -> (batch, n, n)
        K_batched = K.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Add diagonal: torch.diag_embed creates (batch, n, n)
        J = K_batched + torch.diag_embed(diag_term)
        
        # Solve linear system J * delta = F
        # delta = J^{-1} F
        try:
            delta = torch.linalg.solve(J, F) # F is (batch, n), returns (batch, n)
        except RuntimeError as e:
            print(f"  Newton failed: Linear solve error at iter {k}")
            break
            
        u = u - delta
                
    print(f"  Newton did not converge in {max_iter} iterations (residual={res_norm:.4e})")
    return u

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
