"""
Energy function for 2D Darcy flow equation.

The boundary value problem (Dirichlet BC):
    ∇·(a(x)∇u(x)) = f(x)  in Ω = (0,1)²
    u = 0  on ∂Ω

Energy functional:
    J(u; a, f) = (1/2) ∫_Ω a(x)|∇u(x)|² dx - ∫_Ω f(x)u(x) dx

Discretization on N×N interior grid with h = 1/(N+1):
    J(u; a, f) = (h²/2) Σ_{i,j} [a_{i+1/2,j}(u_x)²_{i+1/2,j} + a_{i,j+1/2}(u_y)²_{i,j+1/2}]
                 - h² Σ_{i,j} u_{i,j} f_{i,j}

The gradient ∇_u H is computed using autograd (recommended in proposal).
"""

import torch
import torch.nn.functional as F


def pad_with_bc(u: torch.Tensor) -> torch.Tensor:
    """
    Pad interior solution with zero Dirichlet boundary.
    
    Args:
        u: Interior values, shape (batch, N, N)
        
    Returns:
        u_padded: With boundaries, shape (batch, N+2, N+2)
    """
    return F.pad(u, (1, 1, 1, 1), mode='constant', value=0.0)


def compute_energy(
    u: torch.Tensor,
    a: torch.Tensor,
    f: torch.Tensor,
    h: float,
    beta: float = 1.0
) -> torch.Tensor:
    """
    Compute the Hamiltonian H(u; a, f) = beta * J(u; a, f).
    
    Uses arithmetic mean for permeability at cell edges.
    Properly includes ALL edges including boundary contributions.
    
    Args:
        u: Interior solution values, shape (batch, N, N) or (N, N)
        a: Permeability on full grid including boundaries, shape (batch, N+2, N+2) or (N+2, N+2)
        f: Forcing term on interior grid, shape (batch, N, N) or (N, N)
        h: Grid spacing, h = 1/(N+1)
        beta: Inverse temperature
        
    Returns:
        H: Energy value(s), shape (batch,) or scalar
    """
    squeeze_output = False
    if u.dim() == 2:
        u = u.unsqueeze(0)
        a = a.unsqueeze(0)
        f = f.unsqueeze(0)
        squeeze_output = True
    
    batch_size, N, _ = u.shape
    
    # Pad u with zero boundary conditions
    u_padded = pad_with_bc(u)  # (batch, N+2, N+2)
    
    # =========================================================
    # Compute gradient energy: (1/2) ∫ a|∇u|² dx
    # 
    # We need to sum over ALL edges in the grid:
    # - x-edges: (N+1) × N (including west and east boundary edges)
    # - y-edges: N × (N+1) (including south and north boundary edges)
    # =========================================================
    
    # x-derivatives at ALL vertical edges (i+1/2, j) for i=0..N, j=1..N
    # u_padded indices: rows 0..N+1, cols 1..N (in padded coords)
    # (u_{i+1,j} - u_{i,j})/h for all edges
    ux_all = (u_padded[:, 1:, 1:-1] - u_padded[:, :-1, 1:-1]) / h  # (batch, N+1, N)
    
    # y-derivatives at ALL horizontal edges (i, j+1/2) for i=1..N, j=0..N
    uy_all = (u_padded[:, 1:-1, 1:] - u_padded[:, 1:-1, :-1]) / h  # (batch, N, N+1)
    
    # Permeability at edges using arithmetic mean
    # For x-edges at (i+1/2, j): average of a[i,j] and a[i+1,j]
    a_x = 0.5 * (a[:, :-1, 1:-1] + a[:, 1:, 1:-1])  # (batch, N+1, N)
    
    # For y-edges at (i, j+1/2): average of a[i,j] and a[i,j+1]
    a_y = 0.5 * (a[:, 1:-1, :-1] + a[:, 1:-1, 1:])  # (batch, N, N+1)
    
    # Gradient energy: (h²/2) Σ [a * (∂u)²] for all edges
    # Note: h² is the area element, each edge contributes h² * 0.5 * a * (du/dx)²
    grad_energy_x = 0.5 * h * h * torch.sum(a_x * ux_all**2, dim=(-2, -1))
    grad_energy_y = 0.5 * h * h * torch.sum(a_y * uy_all**2, dim=(-2, -1))
    grad_energy = grad_energy_x + grad_energy_y  # (batch,)
    
    # Potential term (forcing): ∫ f·u dx ≈ h² Σ f_{i,j} u_{i,j}
    forcing_term = h * h * torch.sum(f * u, dim=(-2, -1))  # (batch,)
    
    # Total energy
    J = grad_energy - forcing_term
    H = beta * J
    
    if squeeze_output:
        H = H.squeeze(0)
    
    return H


def compute_energy_gradient(
    u: torch.Tensor,
    a: torch.Tensor,
    f: torch.Tensor,
    h: float,
    beta: float = 1.0
) -> torch.Tensor:
    """
    Compute the gradient of the Hamiltonian with respect to u using autograd.
    
    Args:
        u: Interior solution values, shape (batch, N, N) or (N, N)
        a: Permeability on full grid, shape (batch, N+2, N+2) or (N+2, N+2)
        f: Forcing term on interior grid, shape (batch, N, N) or (N, N)
        h: Grid spacing
        beta: Inverse temperature
        
    Returns:
        grad: Gradient of H w.r.t. u, same shape as u
    """
    squeeze_output = False
    if u.dim() == 2:
        u = u.unsqueeze(0)
        a = a.unsqueeze(0)
        f = f.unsqueeze(0)
        squeeze_output = True
    
    # Create a fresh tensor for autograd
    u_var = u.detach().clone().requires_grad_(True)
    
    with torch.enable_grad():
        # Compute energy
        H = compute_energy(u_var, a, f, h, beta)
        
        # Compute gradient
        grad = torch.autograd.grad(H.sum(), u_var)[0]
    
    if squeeze_output:
        grad = grad.squeeze(0)
    
    return grad


def build_laplacian_matrix_2d(
    N: int, 
    h: float,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Build the 2D Laplacian stiffness matrix for constant a=1.
    
    For -Δu = f with u=0 on boundary, the matrix is N²×N² with
    standard 5-point stencil structure.
    
    Args:
        N: Grid size (N×N interior points)
        h: Grid spacing
        
    Returns:
        K: Stiffness matrix, shape (N², N²)
    """
    n_total = N * N
    
    # Main diagonal: 4/h²
    diag_main = (4.0 / (h * h)) * torch.ones(n_total, device=device, dtype=dtype)
    
    # Off-diagonals for neighbors
    off_diag = (-1.0 / (h * h)) * torch.ones(n_total - 1, device=device, dtype=dtype)
    off_diag_N = (-1.0 / (h * h)) * torch.ones(n_total - N, device=device, dtype=dtype)
    
    # Remove connections across row boundaries
    for i in range(1, N):
        off_diag[i * N - 1] = 0.0
    
    K = torch.diag(diag_main)
    K += torch.diag(off_diag, 1) + torch.diag(off_diag, -1)
    K += torch.diag(off_diag_N, N) + torch.diag(off_diag_N, -N)
    
    return K


if __name__ == "__main__":
    """Test energy computation and gradient."""
    import numpy as np
    
    torch.manual_seed(42)
    
    N = 16  # Grid size
    h = 1.0 / (N + 1)
    
    # Constant permeability a = 1
    a = torch.ones(N + 2, N + 2)
    
    # Simple forcing
    x = torch.linspace(h, 1 - h, N)
    y = torch.linspace(h, 1 - h, N)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    f = torch.sin(np.pi * X) * torch.sin(np.pi * Y)
    
    # Random solution
    u = torch.randn(N, N, requires_grad=True)
    
    # Compute energy
    H = compute_energy(u, a, f, h, beta=1.0)
    print(f"Energy: {H.item():.6f}")
    
    # Compute gradient via autograd (direct)
    H.backward()
    grad_direct = u.grad.clone()
    
    # Compute gradient via our function
    grad_func = compute_energy_gradient(u.detach(), a, f, h, beta=1.0)
    
    # Compare
    max_err = (grad_direct - grad_func).abs().max().item()
    print(f"Max gradient error (autograd vs function): {max_err:.2e}")
    
    # Numerical gradient check
    eps = 1e-4
    u_test = u.detach().clone()
    i, j = N // 2, N // 2
    
    u_plus = u_test.clone()
    u_plus[i, j] += eps
    H_plus = compute_energy(u_plus, a, f, h, beta=1.0)
    
    u_minus = u_test.clone()
    u_minus[i, j] -= eps
    H_minus = compute_energy(u_minus, a, f, h, beta=1.0)
    
    grad_numerical = (H_plus - H_minus) / (2 * eps)
    print(f"Numerical gradient at ({i},{j}): {grad_numerical.item():.6f}")
    print(f"Autograd gradient at ({i},{j}): {grad_func[i, j].item():.6f}")
    print(f"Difference: {abs(grad_numerical.item() - grad_func[i, j].item()):.2e}")
    
    # Batch test
    batch_size = 4
    u_batch = torch.randn(batch_size, N, N)
    a_batch = torch.ones(batch_size, N + 2, N + 2)
    f_batch = f.unsqueeze(0).expand(batch_size, -1, -1).clone()
    
    H_batch = compute_energy(u_batch, a_batch, f_batch, h, beta=1.0)
    grad_batch = compute_energy_gradient(u_batch, a_batch, f_batch, h, beta=1.0)
    
    print(f"\nBatch test:")
    print(f"  Energy shape: {H_batch.shape}")
    print(f"  Gradient shape: {grad_batch.shape}")
    print(f"  Energy values: {H_batch.tolist()}")
