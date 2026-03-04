"""
Preconditioned Langevin dynamics for 2D Darcy flow.

Simplified parameterization:
    u_{k+1} = u_k - step_size * M ∇J(u_k) + noise_scale * M^{1/2} ξ_k,   ξ_k ~ N(0, I)

where:
    - step_size: controls the gradient descent magnitude
    - noise_scale: controls the stochastic noise magnitude (0 = deterministic)
    - M = (κ² I - Δ)^{-α} is a Matérn-type covariance operator
    - J(u) is the energy functional (no beta scaling in the gradient)

In the 2D sine basis (for Dirichlet BCs), M acts diagonally with eigenvalues:
    λ_{m,n} = (κ² + (mπ)² + (nπ)²)^{-α}

This uses separable 2D DST computed via 1D DST along each axis.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List

from energy import compute_energy_gradient, compute_energy


def dst_type1(x: torch.Tensor) -> torch.Tensor:
    """
    Discrete Sine Transform (Type-I) using FFT.
    
    DST-I: X_k = sum_{n=0}^{N-1} x_n * sin(π(n+1)(k+1)/(N+1))
    
    Args:
        x: Input tensor, shape (..., n)
        
    Returns:
        X: DST coefficients, shape (..., n)
    """
    n = x.shape[-1]
    
    # Create extended array with odd symmetry
    zeros = torch.zeros_like(x[..., :1])
    x_extended = torch.cat([zeros, x, zeros, -x.flip(-1)], dim=-1)
    
    # FFT
    X_fft = torch.fft.fft(x_extended, dim=-1)
    
    # Extract DST coefficients
    X = -0.5 * X_fft[..., 1:n+1].imag
    
    return X


def idst_type1(X: torch.Tensor) -> torch.Tensor:
    """
    Inverse Discrete Sine Transform (Type-I).
    """
    n = X.shape[-1]
    return dst_type1(X) * (2.0 / (n + 1))


def dst2_type1(x: torch.Tensor) -> torch.Tensor:
    """
    2D Discrete Sine Transform (Type-I) using separable 1D DST.
    
    Args:
        x: Input tensor, shape (..., N, N)
        
    Returns:
        X: 2D DST coefficients, shape (..., N, N)
    """
    # Apply 1D DST along last axis
    X_temp = dst_type1(x)
    # Apply 1D DST along second-to-last axis
    X = dst_type1(X_temp.transpose(-2, -1)).transpose(-2, -1)
    return X


def idst2_type1(X: torch.Tensor) -> torch.Tensor:
    """
    Inverse 2D Discrete Sine Transform (Type-I).
    """
    # Apply 1D iDST along last axis
    x_temp = idst_type1(X)
    # Apply 1D iDST along second-to-last axis
    x = idst_type1(x_temp.transpose(-2, -1)).transpose(-2, -1)
    return x


class MaternPreconditioner2d:
    """
    Matérn-type preconditioner M = (κ² I - Δ)^{-α} for 2D Dirichlet BCs.
    
    In the 2D sine basis, this operator is diagonal with eigenvalues:
        λ_{m,n} = ((κπ)² + (mπ)² + (nπ)²)^{-α}   for m,n = 1, 2, ..., N
        
    Args:
        N: Grid size (N×N interior points)
        kappa: Correlation length parameter κ
        alpha: Smoothing exponent α
        normalize: If True, scale eigenvalues so max(λ) = 1.0
        mode: 'matern' or 'inverse_laplacian'
        device: Torch device
        dtype: Torch dtype
    """
    
    def __init__(
        self,
        N: int,
        kappa: float = 1.0,
        alpha: float = 1.5,
        normalize: bool = True,
        mode: str = 'matern',
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32
    ):
        self.N = N
        self.kappa = kappa
        self.alpha = alpha
        self.mode = mode
        self.device = device
        self.dtype = dtype
        
        # 2D mode indices
        m = torch.arange(1, N + 1, device=device, dtype=dtype)
        n = torch.arange(1, N + 1, device=device, dtype=dtype)
        M, N_grid = torch.meshgrid(m, n, indexing='ij')
        
        omega_m_sq = (M * np.pi) ** 2
        omega_n_sq = (N_grid * np.pi) ** 2
        
        if mode == 'inverse_laplacian':
            # Eigenvalues of (-Δ)^{-α}: λ_{m,n} = 1/((mπ)² + (nπ)²)^α
            eigenvalues = 1.0 / (omega_m_sq + omega_n_sq) ** alpha
        else:
            # Matérn: λ_{m,n} = ((κπ)² + (mπ)² + (nπ)²)^{-α}
            eigenvalues = (kappa**2 * np.pi**2 + omega_m_sq + omega_n_sq) ** (-alpha)
        
        # Normalize
        if normalize:
            eigenvalues = eigenvalues / eigenvalues.max()
        
        self.eigenvalues = eigenvalues
        self.sqrt_eigenvalues = self.eigenvalues ** 0.5
    
    def to(self, device: str):
        """Move preconditioner to specified device."""
        self.device = device
        self.eigenvalues = self.eigenvalues.to(device)
        self.sqrt_eigenvalues = self.sqrt_eigenvalues.to(device)
        return self
    
    def apply(self, v: torch.Tensor) -> torch.Tensor:
        """
        Apply M to tensor v: M v = iDST2( λ · DST2(v) )
        
        Args:
            v: Input tensor, shape (..., N, N)
        Returns:
            Mv: Result of applying M, shape (..., N, N)
        """
        v_hat = dst2_type1(v)
        Mv_hat = v_hat * self.eigenvalues
        Mv = idst2_type1(Mv_hat)
        return Mv
    
    def apply_sqrt(self, v: torch.Tensor) -> torch.Tensor:
        """
        Apply M^{1/2} to tensor v.
        """
        v_hat = dst2_type1(v)
        Mv_hat = v_hat * self.sqrt_eigenvalues
        M_sqrt_v = idst2_type1(Mv_hat)
        return M_sqrt_v
    
    def sample_noise(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Sample colored noise M^{1/2} ξ where ξ ~ N(0, I).
        
        Args:
            shape: Shape of the output, last two dimensions should be (N, N)
            
        Returns:
            noise: Colored noise sample, shape = shape
        """
        assert shape[-2:] == (self.N, self.N), f"Last two dims must be ({self.N}, {self.N})"
        
        xi = torch.randn(shape, device=self.device, dtype=self.dtype)
        return self.apply_sqrt(xi)


def preconditioned_langevin_step(
    u: torch.Tensor,
    a: torch.Tensor,
    f: torch.Tensor,
    h: float,
    step_size: float,
    noise_scale: float,
    preconditioner: MaternPreconditioner2d,
    grad_clip: float = None
) -> torch.Tensor:
    """
    Perform one step of Preconditioned Langevin Algorithm for 2D.
    
    Update rule:
        u' = u - step_size * M ∇J(u) + noise_scale * M^{1/2} ξ
    
    where:
        - step_size controls gradient descent magnitude
        - noise_scale controls stochastic noise (0 = deterministic GD)
        - M is the preconditioner
        - J(u) is the energy functional
    
    Args:
        u: Current solution, shape (batch, N, N) or (N, N)
        a: Permeability, shape (batch, N+2, N+2) or (N+2, N+2)
        f: Forcing term, shape (batch, N, N) or (N, N)
        h: Grid spacing
        step_size: Gradient descent step size
        noise_scale: Noise magnitude (0.0 for deterministic)
        preconditioner: MaternPreconditioner2d instance
        grad_clip: Maximum absolute value for gradient clipping
        
    Returns:
        u_next: Updated solution, same shape as u
    """
    # Compute gradient of J (energy functional, no beta scaling)
    grad_J = compute_energy_gradient(u, a, f, h, beta=1.0)
    
    # Apply preconditioner to gradient
    M_grad = preconditioner.apply(grad_J)
    
    # Clip gradient if requested
    if grad_clip is not None:
        M_grad = torch.clamp(M_grad, -grad_clip, grad_clip)
    
    # Sample colored noise
    if noise_scale > 0:
        colored_noise = preconditioner.sample_noise(u.shape)
        noise_term = noise_scale * colored_noise
    else:
        noise_term = 0.0
    
    # Update
    u_next = u - step_size * M_grad + noise_term
    
    return u_next


def preconditioned_langevin_refine(
    u0: torch.Tensor,
    a: torch.Tensor,
    f: torch.Tensor,
    h: float,
    step_size: float,
    noise_scale: float,
    K: int,
    preconditioner: MaternPreconditioner2d,
    return_trajectory: bool = False,
    grad_clip: float = None
) -> torch.Tensor:
    """
    Refine initial sample using K steps of preconditioned Langevin dynamics.
    
    Args:
        u0: Initial solution, shape (batch, N, N) or (N, N)
        a: Permeability, shape (batch, N+2, N+2) or (N+2, N+2)
        f: Forcing term, shape (batch, N, N) or (N, N)
        h: Grid spacing
        step_size: Gradient descent step size
        noise_scale: Noise magnitude (0 = deterministic)
        K: Number of Langevin steps
        preconditioner: MaternPreconditioner2d instance
        return_trajectory: If True, return all intermediate samples
        grad_clip: Gradient clipping value
        
    Returns:
        u_K: Refined solution after K steps
        trajectory: (optional) List of all samples if return_trajectory=True
    """
    u = u0.clone()
    
    if return_trajectory:
        trajectory = [u.clone()]
    
    for _ in range(K):
        u = preconditioned_langevin_step(u, a, f, h, step_size, noise_scale, preconditioner, grad_clip=grad_clip)
        if return_trajectory:
            trajectory.append(u.clone())
    
    if return_trajectory:
        return u, trajectory
    return u


def preconditioned_langevin_refine_with_energy(
    u0: torch.Tensor,
    a: torch.Tensor,
    f: torch.Tensor,
    h: float,
    step_size: float,
    noise_scale: float,
    K: int,
    preconditioner: MaternPreconditioner2d,
    grad_clip: float = None
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Refine sample and track energy along the trajectory.
    
    Returns:
        u_K: Refined solution
        energies: List of energy values J(u) at each step
    """
    u = u0.clone()
    energies = [compute_energy(u, a, f, h, beta=1.0).detach()]
    
    for _ in range(K):
        u = preconditioned_langevin_step(u, a, f, h, step_size, noise_scale, preconditioner, grad_clip=grad_clip)
        energies.append(compute_energy(u, a, f, h, beta=1.0).detach())
    
    return u, energies


if __name__ == "__main__":
    """Test the 2D preconditioned Langevin implementation."""
    import matplotlib.pyplot as plt
    from data import sample_forcing_f, sample_coefficient_a
    from utils import solve_darcy_exact, compute_relative_l2_error_2d
    
    torch.manual_seed(42)
    
    # Setup
    N = 32
    h = 1.0 / (N + 1)
    device = 'cpu'
    
    # Problem: -Δu = f (constant a=1)
    a = torch.ones(1, N + 2, N + 2, device=device)
    f = sample_forcing_f(N, 1, method='sinusoidal', amplitude=10.0, device=device)
    
    # Initial guess
    u0 = torch.zeros(1, N, N, device=device)
    
    # Parameters (simplified!)
    step_size = 10.0   # Gradient descent step size
    noise_scale = 0.0  # Deterministic (no noise)
    K = 500
    
    kappa = 1.0
    alpha = 1.0
    
    # Create preconditioner
    precond = MaternPreconditioner2d(
        N, kappa=kappa, alpha=alpha,
        mode='matern',
        device=device
    )
    
    print(f"Testing 2D preconditioned Langevin dynamics")
    print(f"  Grid: {N}×{N}")
    print(f"  step_size: {step_size}, noise_scale: {noise_scale}")
    print(f"  Preconditioner: mode=matern (κ={kappa}, α={alpha})")
    print()
    
    # Test DST roundtrip
    test_mat = torch.randn(N, N)
    reconstructed = idst2_type1(dst2_type1(test_mat))
    dst_error = (test_mat - reconstructed).abs().max().item()
    print(f"2D DST roundtrip error: {dst_error:.2e}")
    
    # Find LBFGS optimum for reference
    u_opt = torch.zeros(1, N, N, requires_grad=True)
    optimizer = torch.optim.LBFGS([u_opt], max_iter=100, line_search_fn='strong_wolfe')
    for _ in range(10):
        def closure():
            optimizer.zero_grad()
            loss = compute_energy(u_opt, a, f, h, beta=1.0)
            loss.backward()
            return loss
        optimizer.step(closure)
    u_opt = u_opt.detach()
    energy_opt = compute_energy(u_opt, a, f, h, beta=1.0).item()
    print(f"LBFGS optimal energy: {energy_opt:.6f}")
    
    # Run preconditioned GD
    print(f"\nRunning {K} steps of preconditioned GD")
    u_refined, energies = preconditioned_langevin_refine_with_energy(
        u0.clone(), a, f, h, step_size, noise_scale, K, precond
    )
    
    # Compute errors
    err = compute_relative_l2_error_2d(u_refined, u_opt, h).item()
    
    print(f"\nResults:")
    print(f"  Initial energy: {energies[0].item():.6f}")
    print(f"  Final energy: {energies[-1].item():.6f}")
    print(f"  Optimal energy: {energy_opt:.6f}")
    print(f"  Relative L2 error vs optimum: {err:.4f}")
    
    # Quick visualization
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    ax = axes[0]
    ax.plot([e.item() for e in energies])
    ax.axhline(y=energy_opt, color='r', linestyle='--', label='Optimal')
    ax.set_xlabel('Step')
    ax.set_ylabel('Energy J(u)')
    ax.set_title('Energy Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    im = ax.imshow(u_opt[0].numpy(), cmap='RdBu_r')
    ax.set_title('LBFGS Optimal')
    plt.colorbar(im, ax=ax)
    
    ax = axes[2]
    im = ax.imshow(u_refined[0].numpy(), cmap='RdBu_r')
    ax.set_title(f'Precond GD (L² err={err:.4f})')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('preconditioned_langevin_2d_test.png', dpi=150)
    print(f"\nSaved visualization to preconditioned_langevin_2d_test.png")
