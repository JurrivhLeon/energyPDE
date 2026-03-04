"""
Energy function for 1D Nonlinear Poisson (Ginzburg-Landau / Double-Well) equation.

The nonlinear Poisson equation with Dirichlet boundary conditions:
    -u''(x) + (u(x)^3 - u(x)) = s(x),  x in (0,1),  u(0) = u(1) = 0

The energy functional:
    J(u,s) = (1/2) * integral(|u'(x)|^2 dx) + integral(V(u(x)) dx) - integral(s(x)u(x) dx)
    
where V(u) = (1/4)u^4 - (1/2)u^2 is the double-well potential.

In discretized form on n interior grid points with spacing h = 1/(n+1):
    J(u,s) = (h/2) * u^T K u + h * sum(V(u_i)) - h * s^T u
    
where K is the standard Laplacian stiffness matrix.

The energy gradient:
    grad_u J = h * (K u + f(u) - s)
    
where f(u) = u^3 - u is the derivative of V(u).
"""

import torch
import torch.nn.functional as F


def double_well_potential(u: torch.Tensor) -> torch.Tensor:
    """
    Double-well potential V(u) = (1/4)u^4 - (1/2)u^2.
    
    Args:
        u: Input values, any shape
        
    Returns:
        V(u): Same shape as u
    """
    return 0.25 * u**4 - 0.5 * u**2


def double_well_derivative(u: torch.Tensor) -> torch.Tensor:
    """
    Derivative of double-well potential: f(u) = V'(u) = u^3 - u.
    
    This is the reaction term in the PDE.
    
    Args:
        u: Input values, any shape
        
    Returns:
        f(u) = u^3 - u: Same shape as u
    """
    return u**3 - u


def build_laplacian_matrix(n: int, h: float, device: str = 'cpu', dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Build the standard Laplacian stiffness matrix K for Dirichlet BCs.
    
    For -u'' = f with u(0) = u(1) = 0:
        K_{ii} = 2/h^2
        K_{i,i+1} = K_{i,i-1} = -1/h^2
    
    Args:
        n: Number of interior grid points
        h: Grid spacing, h = 1/(n+1)
        device: Torch device
        dtype: Torch dtype
        
    Returns:
        K: Stiffness matrix, shape (n, n)
    """
    diag = 2.0 / (h * h) * torch.ones(n, device=device, dtype=dtype)
    off_diag = -1.0 / (h * h) * torch.ones(n - 1, device=device, dtype=dtype)
    
    K = torch.diag(diag) + torch.diag(off_diag, 1) + torch.diag(off_diag, -1)
    
    return K


def compute_energy(
    u: torch.Tensor,
    s: torch.Tensor,
    h: float,
    K: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute the energy J(u; s).
    
    Discretized energy:
        J = (h/2) * u^T K u + h * sum(V(u_i)) - h * s^T u
    
    Args:
        u: Interior solution values, shape (batch, n) or (n,)
        s: Source term on interior grid, shape (batch, n) or (n,)
        h: Grid spacing
        K: Pre-computed stiffness matrix (n, n). If None, will be built.
        
    Returns:
        J: Energy value(s), shape (batch,) or scalar
    """
    squeeze_output = False
    if u.dim() == 1:
        u = u.unsqueeze(0)
        s = s.unsqueeze(0)
        squeeze_output = True
    
    batch_size, n = u.shape
    
    if K is None:
        K = build_laplacian_matrix(n, h, device=u.device, dtype=u.dtype)
    
    # Quadratic term: (h/2) * u^T K u
    Ku = torch.matmul(u, K)  # (batch, n)
    quadratic_term = 0.5 * h * torch.sum(u * Ku, dim=-1)  # (batch,)
    
    # Double-well potential: h * sum(V(u_i))
    V_u = double_well_potential(u)  # (batch, n)
    potential_term = h * torch.sum(V_u, dim=-1)  # (batch,)
    
    # Linear term: h * s^T u
    linear_term = h * torch.sum(s * u, dim=-1)  # (batch,)
    
    # Total energy
    J = quadratic_term + potential_term - linear_term
    
    if squeeze_output:
        J = J.squeeze(0)
    
    return J


def compute_energy_gradient(
    u: torch.Tensor,
    s: torch.Tensor,
    h: float,
    K: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute the gradient of the energy with respect to u.
    
    grad_u J = h * (K u + f(u) - s)
    
    where f(u) = u^3 - u is the derivative of double-well potential.
    
    Args:
        u: Interior solution values, shape (batch, n) or (n,)
        s: Source term on interior grid, shape (batch, n) or (n,)
        h: Grid spacing
        K: Pre-computed stiffness matrix (n, n). If None, will be built.
        
    Returns:
        grad: Gradient of J w.r.t. u, same shape as u
    """
    squeeze_output = False
    if u.dim() == 1:
        u = u.unsqueeze(0)
        s = s.unsqueeze(0)
        squeeze_output = True
    
    batch_size, n = u.shape
    
    if K is None:
        K = build_laplacian_matrix(n, h, device=u.device, dtype=u.dtype)
    
    # Compute K u
    Ku = torch.matmul(u, K)  # (batch, n)
    
    # Compute reaction term f(u) = u^3 - u
    f_u = double_well_derivative(u)  # (batch, n)
    
    # Gradient: h * (K u + f(u) - s)
    grad = h * (Ku + f_u - s)
    
    if squeeze_output:
        grad = grad.squeeze(0)
    
    return grad


if __name__ == "__main__":
    # Quick test with gradient verification
    torch.manual_seed(42)
    
    n = 20
    h = 1.0 / (n + 1)
    
    # Source term (sinusoidal)
    x = torch.linspace(h, 1 - h, n)
    s = torch.sin(2 * 3.14159 * x)
    
    # Random initial solution
    u = torch.randn(n, requires_grad=True)
    
    # Pre-build stiffness matrix
    K = build_laplacian_matrix(n, h)
    
    # Compute energy and gradient via autograd
    H = compute_energy(u, s, h, K=K)
    H.backward()
    grad_auto = u.grad.clone()
    
    # Compute gradient via analytical formula
    grad_manual = compute_energy_gradient(u.detach(), s, h, K=K)
    
    print(f"Energy: {H.item():.6f}")
    print(f"Max gradient error: {(grad_auto - grad_manual).abs().max().item():.2e}")
    
    # Check stiffness matrix
    print(f"\nStiffness matrix (first 5x5):\n{K[:5, :5]}")
    
    # Expected: K_ii = 2/h^2, K_{i,i±1} = -1/h^2
    expected_diag = 2.0 / (h * h)
    expected_off = -1.0 / (h * h)
    print(f"\nExpected diagonal: {expected_diag:.2f}, Got: {K[0, 0].item():.2f}")
    print(f"Expected off-diag: {expected_off:.2f}, Got: {K[0, 1].item():.2f}")
    
    # Test double-well potential
    u_test = torch.tensor([-1.0, 0.0, 1.0])
    V_test = double_well_potential(u_test)
    print(f"\nDouble-well potential at u=[-1, 0, 1]: {V_test.tolist()}")
    print("Expected: [-0.25, 0.0, -0.25] (minima at ±1)")
    
    f_test = double_well_derivative(u_test)
    print(f"Derivative at u=[-1, 0, 1]: {f_test.tolist()}")
    print("Expected: [0, 0, 0] (zeros at ±1 and 0)")
    
    # Batch test
    u_batch = torch.randn(4, n)
    s_batch = torch.randn(4, n)
    H_batch = compute_energy(u_batch, s_batch, h, K=K)
    grad_batch = compute_energy_gradient(u_batch, s_batch, h, K=K)
    print(f"\nBatch test - Energy shape: {H_batch.shape}, Gradient shape: {grad_batch.shape}")
