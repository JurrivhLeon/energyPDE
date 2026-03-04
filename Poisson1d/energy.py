"""
Energy function for 1D Poisson equation.

The energy functional for the boundary value problem:
    -d/dx(a(x) du/dx) = f(x),  u(0) = u(1) = 0

is given by:
    J(u; a, f) = (1/2) * integral(a(x)|u'(x)|^2 dx) - integral(f(x)u(x) dx)

In discretized form on n interior grid points with spacing h = 1/(n+1):
    J(u; a, f) = (h/2) * u^T K(a) u - h * f^T u

where K(a) is the stiffness matrix.
"""

import torch
import torch.nn.functional as F


def build_stiffness_matrix(a: torch.Tensor, h: float) -> torch.Tensor:
    """
    Build the stiffness matrix K(a) for the 1D Poisson equation.
    
    Uses harmonic mean for mid-point coefficient values:
        a_{i+1/2} = 2 / (1/a_i + 1/a_{i+1})
    
    Args:
        a: Coefficient function values on full grid including boundaries,
           shape (batch, n+2) or (n+2,) where n is number of interior points
        h: Grid spacing, h = 1/(n+1)
    
    Returns:
        K: Stiffness matrix, shape (batch, n, n) or (n, n)
    """
    squeeze_output = False
    if a.dim() == 1:
        a = a.unsqueeze(0)
        squeeze_output = True
    
    batch_size = a.shape[0]
    n = a.shape[1] - 2  # number of interior points
    
    # Compute mid-point coefficients using harmonic mean
    # a_{i+1/2} for i = 0, 1, ..., n (n+1 values)
    a_left = a[:, :-1]   # a_0, a_1, ..., a_n
    a_right = a[:, 1:]   # a_1, a_2, ..., a_{n+1}
    
    # Harmonic mean: 2 / (1/a_left + 1/a_right)
    a_mid = 2.0 / (1.0 / a_left + 1.0 / a_right + 1e-8)  # shape (batch, n+1)
    
    # Build tridiagonal matrix
    # K_{ii} = (a_{i-1/2} + a_{i+1/2}) / h^2
    # K_{i,i-1} = -a_{i-1/2} / h^2
    # K_{i,i+1} = -a_{i+1/2} / h^2
    
    # For interior points i = 1, ..., n (0-indexed: 0, ..., n-1)
    # a_{i-1/2} corresponds to a_mid[:, i-1] = a_mid[:, 0:n]
    # a_{i+1/2} corresponds to a_mid[:, i] = a_mid[:, 1:n+1]
    
    a_minus = a_mid[:, :n]     # a_{1/2}, a_{3/2}, ..., a_{n-1/2}
    a_plus = a_mid[:, 1:n+1]   # a_{3/2}, a_{5/2}, ..., a_{n+1/2}
    
    # Diagonal elements
    diag = (a_minus + a_plus) / (h * h)  # shape (batch, n)
    
    # Off-diagonal elements
    off_diag_lower = -a_minus[:, 1:] / (h * h)   # K_{i,i-1} for i=1,...,n-1, shape (batch, n-1)
    off_diag_upper = -a_plus[:, :-1] / (h * h)   # K_{i,i+1} for i=0,...,n-2, shape (batch, n-1)
    
    # Construct the matrix
    K = torch.zeros(batch_size, n, n, device=a.device, dtype=a.dtype)
    
    # Fill diagonal
    K[:, range(n), range(n)] = diag
    
    # Fill off-diagonals
    if n > 1:
        K[:, range(1, n), range(n-1)] = off_diag_lower
        K[:, range(n-1), range(1, n)] = off_diag_upper
    
    if squeeze_output:
        K = K.squeeze(0)
    
    return K


def compute_energy(
    u: torch.Tensor,
    a: torch.Tensor,
    f: torch.Tensor,
    h: float
) -> torch.Tensor:
    """
    Compute the energy J(u; a, f).
    
    Discretized energy:
        J = (h/2) * u^T K(a) u - h * f^T u
    
    Args:
        u: Interior solution values, shape (batch, n) or (n,)
        a: Coefficient function on full grid, shape (batch, n+2) or (n+2,)
        f: Forcing term on interior grid, shape (batch, n) or (n,)
        h: Grid spacing
        
    Returns:
        J: Energy value(s), shape (batch,) or scalar
    """
    squeeze_output = False
    if u.dim() == 1:
        u = u.unsqueeze(0)
        a = a.unsqueeze(0)
        f = f.unsqueeze(0)
        squeeze_output = True
    
    K = build_stiffness_matrix(a, h)  # (batch, n, n)
    
    # Compute u^T K u for each batch element
    Ku = torch.bmm(K, u.unsqueeze(-1)).squeeze(-1)  # (batch, n)
    quadratic_term = 0.5 * h * torch.sum(u * Ku, dim=-1)  # (batch,)
    
    # Linear term
    linear_term = h * torch.sum(f * u, dim=-1)  # (batch,)
    
    # Total energy
    J = quadratic_term - linear_term
    
    if squeeze_output:
        J = J.squeeze(0)
    
    return J


def compute_energy_gradient(
    u: torch.Tensor,
    a: torch.Tensor,
    f: torch.Tensor,
    h: float
) -> torch.Tensor:
    """
    Compute the gradient of the energy with respect to u.
    
    grad_u J = h * (K(a) u - f)
    
    Args:
        u: Interior solution values, shape (batch, n) or (n,)
        a: Coefficient function on full grid, shape (batch, n+2) or (n+2,)
        f: Forcing term on interior grid, shape (batch, n) or (n,)
        h: Grid spacing
        
    Returns:
        grad: Gradient of J w.r.t. u, same shape as u
    """
    squeeze_output = False
    if u.dim() == 1:
        u = u.unsqueeze(0)
        a = a.unsqueeze(0)
        f = f.unsqueeze(0)
        squeeze_output = True
    
    K = build_stiffness_matrix(a, h)  # (batch, n, n)
    
    # Compute K u
    Ku = torch.bmm(K, u.unsqueeze(-1)).squeeze(-1)  # (batch, n)
    
    # Gradient
    grad = h * (Ku - f)
    
    if squeeze_output:
        grad = grad.squeeze(0)
    
    return grad


if __name__ == "__main__":
    # Quick test
    torch.manual_seed(42)
    
    n = 10
    h = 1.0 / (n + 1)
    
    # Constant coefficient a = 1 (standard Laplacian)
    a = torch.ones(n + 2)
    f = torch.sin(torch.linspace(0, 3.14159, n))
    u = torch.randn(n, requires_grad=True)
    
    # Compute energy and gradient
    H = compute_energy(u, a, f, h)
    H.backward()
    grad_auto = u.grad.clone()
    
    grad_manual = compute_energy_gradient(u.detach(), a, f, h)
    
    print(f"Energy: {H.item():.6f}")
    print(f"Max gradient error: {(grad_auto - grad_manual).abs().max().item():.2e}")
    
    # Check stiffness matrix for a=1 case
    K = build_stiffness_matrix(a, h)
    print(f"\nStiffness matrix (a=1):\n{K}")
    
    # Expected: K_ii = 2/h^2, K_{i,i±1} = -1/h^2
    expected_diag = 2.0 / (h * h)
    expected_off = -1.0 / (h * h)
    print(f"\nExpected diagonal: {expected_diag:.2f}, Got: {K[0, 0].item():.2f}")
    print(f"Expected off-diag: {expected_off:.2f}, Got: {K[0, 1].item():.2f}")
