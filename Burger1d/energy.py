"""
Energy function for 1D viscous Burgers equation (time-marching formulation).

The viscous Burgers equation:
    u_t + u u_x = nu u_xx,    x in [0, L),  periodic BCs

Using backward Euler time-discretization, the residual is:
    R(u^{n+1}; u^n) = (u^{n+1} - u^n)/dt + u^{n+1} * d_x(u^{n+1}) - nu * d_xx(u^{n+1})

The residual energy (Hamiltonian):
    H(u^{n+1}; u^n) = (1/2) * h * ||R||^2

where h = L/n is the grid spacing.

Spatial derivatives are computed spectrally (FFT) on a periodic domain [0, L).
"""

import torch
import numpy as np


def compute_spectral_derivatives(
    u: torch.Tensor,
    L: float = 1.0
) -> tuple:
    """
    Compute first and second spatial derivatives using FFT (periodic domain).
    
    Args:
        u: Function values on periodic grid, shape (..., n)
        L: Domain length
        
    Returns:
        u_x: First derivative, shape (..., n)
        u_xx: Second derivative, shape (..., n)
    """
    n = u.shape[-1]
    
    # Wavenumbers for a periodic domain of length L
    # k = 2*pi/L * [0, 1, 2, ..., n/2-1, -n/2, -n/2+1, ..., -1]
    k = torch.fft.fftfreq(n, d=L / n, device=u.device, dtype=u.dtype) * (2 * np.pi)
    
    # FFT
    u_hat = torch.fft.fft(u)
    
    # First derivative: multiply by ik
    u_x_hat = 1j * k * u_hat
    u_x = torch.fft.ifft(u_x_hat).real
    
    # Second derivative: multiply by -k^2
    u_xx_hat = -(k ** 2) * u_hat
    u_xx = torch.fft.ifft(u_xx_hat).real
    
    return u_x, u_xx


def compute_residual(
    u_next: torch.Tensor,
    u_curr: torch.Tensor,
    nu: float,
    dt: float,
    L: float = 1.0
) -> torch.Tensor:
    """
    Compute the backward Euler residual for the Burgers equation.
    
    R = (u^{n+1} - u^n)/dt + u^{n+1} * d_x(u^{n+1}) - nu * d_xx(u^{n+1})
    
    Args:
        u_next: Proposed next-step solution, shape (batch, n) or (n,)
        u_curr: Current solution, shape (batch, n) or (n,)
        nu: Viscosity coefficient
        dt: Time step size
        L: Domain length
        
    Returns:
        R: Residual, same shape as u_next
    """
    # Spectral derivatives of u^{n+1}
    u_x, u_xx = compute_spectral_derivatives(u_next, L)
    
    # Backward Euler residual
    R = (u_next - u_curr) / dt + u_next * u_x - nu * u_xx
    
    return R


def compute_energy(
    u_next: torch.Tensor,
    u_curr: torch.Tensor,
    nu: float,
    dt: float,
    L: float = 1.0
) -> torch.Tensor:
    """
    Compute the residual energy H = (1/2) * h * ||R||^2.
    
    Args:
        u_next: Proposed next-step solution, shape (batch, n) or (n,)
        u_curr: Current solution, shape (batch, n) or (n,)
        nu: Viscosity coefficient
        dt: Time step size
        L: Domain length
        
    Returns:
        H: Energy value(s), shape (batch,) or scalar
    """
    squeeze_output = False
    if u_next.dim() == 1:
        u_next = u_next.unsqueeze(0)
        u_curr = u_curr.unsqueeze(0)
        squeeze_output = True
    
    n = u_next.shape[-1]
    h = L / n
    
    R = compute_residual(u_next, u_curr, nu, dt, L)
    
    # L2 norm squared: h * sum(R^2)
    H = 0.5 * h * torch.sum(R ** 2, dim=-1)
    
    if squeeze_output:
        H = H.squeeze(0)
    
    return H


def compute_energy_gradient(
    u_next: torch.Tensor,
    u_curr: torch.Tensor,
    nu: float,
    dt: float,
    L: float = 1.0
) -> torch.Tensor:
    """
    Compute the gradient of H with respect to u^{n+1} using autograd.
    
    Args:
        u_next: Proposed next-step solution, shape (batch, n) or (n,)
        u_curr: Current solution, shape (batch, n) or (n,)
        nu: Viscosity coefficient
        dt: Time step size
        L: Domain length
        
    Returns:
        grad: Gradient of H w.r.t. u_next, same shape as u_next
    """
    squeeze_output = False
    if u_next.dim() == 1:
        u_next = u_next.unsqueeze(0)
        u_curr = u_curr.unsqueeze(0)
        squeeze_output = True
    
    # Create fresh tensor for autograd
    u_var = u_next.detach().clone().requires_grad_(True)
    
    with torch.enable_grad():
        H = compute_energy(u_var, u_curr, nu, dt, L)
        grad = torch.autograd.grad(H.sum(), u_var)[0]
    
    if squeeze_output:
        grad = grad.squeeze(0)
    
    return grad


if __name__ == "__main__":
    """Test energy computation and gradient verification."""
    torch.manual_seed(42)
    
    n = 64
    L = 1.0
    h = L / n
    nu = 0.01
    dt = 0.01
    
    x = torch.linspace(0, L, n + 1)[:-1]  # Periodic grid, exclude endpoint
    
    # Create smooth initial condition (2π*x/L so it's periodic on [0,L])
    u_curr = torch.sin(2 * np.pi * x / L)
    u_next = torch.sin(2 * np.pi * x / L) + 0.1 * torch.randn(n)
    
    # ---- Test spectral derivatives ----
    u_x, u_xx = compute_spectral_derivatives(u_curr, L)
    u_x_exact = (2 * np.pi / L) * torch.cos(2 * np.pi * x / L)
    u_xx_exact = -(2 * np.pi / L)**2 * torch.sin(2 * np.pi * x / L)
    
    print("=== Spectral Derivative Test ===")
    print(f"  d/dx sin(x): max error = {(u_x - u_x_exact).abs().max().item():.2e}")
    print(f"  d²/dx² sin(x): max error = {(u_xx - u_xx_exact).abs().max().item():.2e}")
    
    # ---- Test energy ----
    H = compute_energy(u_next, u_curr, nu, dt, L)
    print(f"\nEnergy: {H.item():.6f}")
    
    # ---- Gradient verification: autograd vs finite differences ----
    u_next_var = u_next.clone().requires_grad_(True)
    H = compute_energy(u_next_var, u_curr, nu, dt, L)
    H.backward()
    grad_auto = u_next_var.grad.clone()
    
    grad_func = compute_energy_gradient(u_next.detach(), u_curr, nu, dt, L)
    
    print(f"\n=== Gradient Verification ===")
    print(f"  Max error (autograd vs function): {(grad_auto - grad_func).abs().max().item():.2e}")
    
    # Numerical gradient check at a single point
    eps = 1e-5
    idx = n // 3
    u_plus = u_next.detach().clone()
    u_plus[idx] += eps
    u_minus = u_next.detach().clone()
    u_minus[idx] -= eps
    H_plus = compute_energy(u_plus, u_curr, nu, dt, L)
    H_minus = compute_energy(u_minus, u_curr, nu, dt, L)
    grad_numerical = (H_plus - H_minus) / (2 * eps)
    
    print(f"  Numerical gradient at idx={idx}: {grad_numerical.item():.6f}")
    print(f"  Autograd gradient at idx={idx}: {grad_func[idx].item():.6f}")
    print(f"  Finite diff error: {abs(grad_numerical.item() - grad_func[idx].item()):.2e}")
    
    # ---- Batch test ----
    u_curr_batch = torch.randn(4, n)
    u_next_batch = torch.randn(4, n)
    H_batch = compute_energy(u_next_batch, u_curr_batch, nu, dt, L)
    grad_batch = compute_energy_gradient(u_next_batch, u_curr_batch, nu, dt, L)
    print(f"\n=== Batch Test ===")
    print(f"  Energy shape: {H_batch.shape}")
    print(f"  Gradient shape: {grad_batch.shape}")
