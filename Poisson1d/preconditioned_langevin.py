"""
Preconditioned Langevin dynamics for smooth PDE sampling.

The preconditioned update rule is:
    u_{k+1} = u_k - step_size * M ∇J(u_k) + noise_scale * M^{1/2} ξ_k,   ξ_k ~ N(0, I)

where M = (κ² I - Δ)^{-α} is a Matérn-type covariance operator.

In the sine basis (for Dirichlet BCs), M acts diagonally with eigenvalues:
    λ_m = (κ² + (mπ)²)^{-α}

This ensures the injected noise has H^1 regularity and satisfies boundary conditions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List

from energy import compute_energy_gradient


def dst_type1(x: torch.Tensor) -> torch.Tensor:
    """
    Discrete Sine Transform (Type-I) using FFT.
    
    DST-I: X_k = sum_{n=0}^{N-1} x_n * sin(π(n+1)(k+1)/(N+1))
    
    For a signal of length N, we create an extended signal of length 2(N+1)
    with odd symmetry and compute FFT.
    
    Args:
        x: Input tensor, shape (..., n)
        
    Returns:
        X: DST coefficients, shape (..., n)
    """
    n = x.shape[-1]
    
    # Create extended array: [0, x_0, x_1, ..., x_{n-1}, 0, -x_{n-1}, ..., -x_0]
    # Length: 2(n+1)
    zeros = torch.zeros_like(x[..., :1])
    x_extended = torch.cat([zeros, x, zeros, -x.flip(-1)], dim=-1)  # length 2(n+1)
    
    # FFT
    X_fft = torch.fft.fft(x_extended, dim=-1)
    
    # Extract imaginary part of relevant coefficients (indices 1 to n)
    # The DST coefficients are -0.5 * imag(X_fft[1:n+1])
    X = -0.5 * X_fft[..., 1:n+1].imag
    
    return X


def idst_type1(X: torch.Tensor) -> torch.Tensor:
    """
    Inverse Discrete Sine Transform (Type-I).
    
    DST-I is its own inverse up to a scaling factor: DST-I(DST-I(x)) = (N+1)/2 * x
    
    Args:
        X: DST coefficients, shape (..., n)
        
    Returns:
        x: Reconstructed signal, shape (..., n)
    """
    n = X.shape[-1]
    # DST-I is self-inverse up to scaling
    return dst_type1(X) * (2.0 / (n + 1))


class MaternPreconditioner:
    """
    Matérn-type preconditioner M = (κ² I - Δ)^{-α} for 1D Dirichlet BCs.
    
    In the sine basis, this operator is diagonal with eigenvalues:
        λ_m = (κ² + (mπ)²)^{-α}   for m = 1, 2, ..., n
        
    The key insight is that this preconditioner should:
    1. Damp high-frequency noise (via small eigenvalues for large m)
    2. Preserve gradient descent in low-frequency directions
    
    For optimal convergence on Poisson-type problems, use mode='inverse_laplacian'
    which sets eigenvalues proportional to 1/(mπ)², matching the inverse Hessian.
    
    Args:
        n_grid: Number of interior grid points
        kappa: Correlation length parameter κ (larger = shorter correlation)
        alpha: Smoothing exponent α (larger = stronger smoothing)
        normalize: If True, scale eigenvalues so max(λ) = 1.0
        mode: 'matern' (default) or 'inverse_laplacian' 
              'inverse_laplacian' uses λ_m = 1/(mπ)², optimal for Poisson
        device: Torch device
        dtype: Torch dtype
    """
    
    def __init__(
        self,
        n_grid: int,
        kappa: float = 1.0,
        alpha: float = 1.5,
        normalize: bool = True,
        scaling: str = 'max',
        mode: str = 'matern',
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32
    ):
        self.n_grid = n_grid
        self.kappa = kappa
        self.alpha = alpha
        self.mode = mode
        self.device = device
        self.dtype = dtype
        
        # Mode indices
        m = torch.arange(1, n_grid + 1, device=device, dtype=dtype)
        omega_m = m * np.pi  # Eigenvalues of -Δ in sine basis (continuous)
        
        if mode == 'inverse_laplacian':
            # Eigenvalues of (-Δ)^{-α}: λ_m = 1/(mπ)^{2α}
            eigenvalues = 1.0 / (omega_m ** (2 * alpha))
        else:
            # Matérn: λ_m = (κ² + (mπ)²)^{-α}
            eigenvalues = (kappa**2 + m**2) ** (-alpha)
        
        # Apply scaling
        if scaling == 'trace':
            # Scale so sum(lambda) = N (like white noise)
            scale = n_grid / eigenvalues.sum()
            eigenvalues = eigenvalues * scale
        elif normalize or scaling == 'max':
            # Scale so max(lambda) = 1 (default)
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
        Apply M to vector v: M v = iDST( λ · DST(v) )
        
        Args:
            v: Input vector, shape (..., n_grid)
        Returns:
            Mv: Result of applying M, shape (..., n_grid)
        """
        # Transform to sine basis
        v_hat = dst_type1(v)
        # Multiply by eigenvalues
        Mv_hat = v_hat * self.eigenvalues
        # Transform back
        Mv = idst_type1(Mv_hat)
        
        return Mv
    
    def apply_sqrt(self, v: torch.Tensor) -> torch.Tensor:
        """
        Apply M^{1/2} to vector v: M^{1/2} v = iDST( √λ · DST(v) )
        
        Args:
            v: Input vector, shape (..., n_grid)
        Returns:
            M_sqrt_v: Result of applying M^{1/2}, shape (..., n_grid)
        """
        # Transform to sine basis
        v_hat = dst_type1(v)
        # Multiply by sqrt(eigenvalues)
        Mv_hat = v_hat * self.sqrt_eigenvalues
        # Transform back
        M_sqrt_v = idst_type1(Mv_hat)
        
        return M_sqrt_v
    
    def sample_noise(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Sample colored noise M^{1/2} ξ where ξ ~ N(0, I).
        
        The result is Gaussian with covariance M in physical space.
        
        Procedure:
        1. Sample white noise ξ ~ N(0, I) in physical space
        2. Apply M^{1/2} using spectral decomposition
           M^{1/2} ξ = iDST( √λ · DST(ξ) )
        
        Args:
            shape: Shape of the output, last dimension should be n_grid
            
        Returns:
            noise: Colored noise sample, shape = shape
        """
        assert shape[-1] == self.n_grid, f"Last dimension must be {self.n_grid}"
        
        # Sample white noise in physical space
        xi = torch.randn(shape, device=self.device, dtype=self.dtype)
        
        # Apply M^{1/2}
        return self.apply_sqrt(xi)


def preconditioned_langevin_step(
    u: torch.Tensor,
    a: torch.Tensor,
    f: torch.Tensor,
    h: float,
    step_size: float,
    noise_scale: float,
    preconditioner: MaternPreconditioner,
    grad_clip: float = None
) -> torch.Tensor:
    """
    Perform one step of Preconditioned Langevin dynamics.
    
    u' = u - step_size * M ∇J(u) + noise_scale * M^{1/2} ξ
    
    Args:
        u: Current solution, shape (batch, n) or (n,)
        a: Coefficient function, shape (batch, n+2) or (n+2,)
        f: Forcing term, shape (batch, n) or (n,)
        h: Grid spacing
        step_size: Gradient step size
        noise_scale: Noise magnitude (0 = deterministic)
        preconditioner: MaternPreconditioner instance
        grad_clip: Maximum absolute value for preconditioned gradient
        
    Returns:
        u_next: Updated solution, same shape as u
    """
    # Compute gradient
    grad = compute_energy_gradient(u, a, f, h)
    
    # Apply preconditioner to gradient
    M_grad = preconditioner.apply(grad)
    
    # Clip gradient if requested
    if grad_clip is not None:
        M_grad = torch.clamp(M_grad, -grad_clip, grad_clip)
    
    # Sample colored noise
    if noise_scale > 0:
        colored_noise = preconditioner.sample_noise(u.shape)
        u_next = u - step_size * M_grad + noise_scale * colored_noise
    else:
        u_next = u - step_size * M_grad
    
    return u_next


def preconditioned_langevin_refine(
    u0: torch.Tensor,
    a: torch.Tensor,
    f: torch.Tensor,
    h: float,
    step_size: float,
    noise_scale: float,
    K: int,
    preconditioner: MaternPreconditioner,
    return_trajectory: bool = False,
    grad_clip: float = None
) -> torch.Tensor:
    """
    Refine initial sample using K steps of preconditioned Langevin dynamics.
    
    Args:
        u0: Initial solution, shape (batch, n) or (n,)
        a: Coefficient function, shape (batch, n+2) or (n+2,)
        f: Forcing term, shape (batch, n) or (n,)
        h: Grid spacing
        step_size: Gradient step size
        noise_scale: Noise magnitude
        K: Number of Langevin steps
        preconditioner: MaternPreconditioner instance
        return_trajectory: If True, return all intermediate samples
        grad_clip: Gradient clipping value (default: None)
        
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
    preconditioner: MaternPreconditioner,
    grad_clip: float = None
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Refine sample and track energy along the trajectory.
    
    Returns:
        u_K: Refined solution
        energies: List of energy values at each step
    """
    from energy import compute_energy
    
    u = u0.clone()
    energies = [compute_energy(u, a, f, h).detach()]
    
    for _ in range(K):
        u = preconditioned_langevin_step(u, a, f, h, step_size, noise_scale, preconditioner, grad_clip=grad_clip)
        energies.append(compute_energy(u, a, f, h).detach())
    
    return u, energies


if __name__ == "__main__":
    """Test the preconditioned Langevin implementation."""
    import matplotlib.pyplot as plt
    from energy import compute_energy
    from data import sample_forcing_f
    from utils import solve_poisson_exact, get_grid_points, compute_relative_l2_error
    
    torch.manual_seed(42)
    
    # Setup
    n_grid = 64
    h = 1.0 / (n_grid + 1)
    device = 'cpu'
    
    # Problem: -u'' = f, u(0)=u(1)=0, with a=1
    a = torch.ones(1, n_grid + 2, device=device)
    f = sample_forcing_f(n_grid, 1, method='grf', amplitude=10.0, device=device)
    
    # Exact solution
    u_exact = solve_poisson_exact(a, f, h)
    
    # Initial guess (random)
    u0 = torch.randn(1, n_grid, device=device)
    
    # Parameters
    kappa = 1.0
    alpha = 1.0
    
    # Create preconditioner with inverse_laplacian mode (optimal for Poisson)
    precond = MaternPreconditioner(
        n_grid, kappa=kappa, alpha=alpha, 
        mode='inverse_laplacian',  # Use inverse Hessian eigenvalues
        device=device
    )
    
    print(f"Testing preconditioned Langevin dynamics")
    print(f"  Grid: {n_grid}")
    print(f"  Preconditioner: mode=inverse_laplacian (κ={kappa}, α={alpha})")
    print()
    
    # Print eigenvalue range
    lambda_min = precond.eigenvalues.min().item()
    lambda_max = precond.eigenvalues.max().item()
    print(f"Preconditioner eigenvalue range: [{lambda_min:.4e}, {lambda_max:.4e}]")
    print(f"Ratio (condition number of M): {lambda_max/lambda_min:.2f}")
    
    # Test DST roundtrip
    test_vec = torch.randn(n_grid)
    reconstructed = idst_type1(dst_type1(test_vec))
    dst_error = (test_vec - reconstructed).abs().max().item()
    print(f"DST roundtrip error: {dst_error:.2e}")
    
    step_size_standard = 1e-4
    step_size_precond = 1e-3
    noise_scale = 1e-3
    
    K_standard = 1000
    K_precond = 1000
    
    print(f"\nStep sizes: Standard={step_size_standard}, Preconditioned={step_size_precond}")
    print(f"Noise scale: {noise_scale}")
    print(f"Steps: Standard K={K_standard}, Preconditioned K={K_precond}")
    
    # Run preconditioned Langevin
    u_precond, energies_precond = preconditioned_langevin_refine_with_energy(
        u0.clone(), a, f, h, step_size_precond, noise_scale, K_precond, precond
    )
    
    # For comparison, run standard Langevin
    from langevin import langevin_refine_with_energy
    u_standard, energies_standard = langevin_refine_with_energy(
        u0.clone(), a, f, h, step_size_standard, noise_scale, K_standard
    )
    
    # Debug: Check gradient and M*gradient spectral properties
    grad = compute_energy_gradient(u0, a, f, h)
    M_grad = precond.apply(grad)
    print(f"\nSpectral analysis:")
    grad_hat = dst_type1(grad[0])
    M_grad_hat = dst_type1(M_grad[0])
    print(f"  Gradient L2 norm: {grad.norm().item():.4f}")
    print(f"  M*Gradient L2 norm: {M_grad.norm().item():.4f}")
    print(f"  Ratio: {M_grad.norm().item() / grad.norm().item():.4f}")
    print(f"  First 5 gradient modes: {grad_hat[:5].tolist()}")
    print(f"  First 5 M*grad modes:   {M_grad_hat[:5].tolist()}")
    
    # Check noise magnitude
    noise_sample = precond.sample_noise(u0.shape)
    print(f"\nNoise analysis:")
    print(f"  Noise L2 norm: {noise_sample.norm().item():.4f}")
    print(f"  Gradient step: step_size * ||M*grad|| = {step_size_precond * M_grad.norm().item():.6f}")
    print(f"  Noise step: noise_scale * ||noise|| = {noise_scale * noise_sample.norm().item():.6f}")
    
    # Compute errors
    err_precond = compute_relative_l2_error(u_precond, u_exact, h).item()
    err_standard = compute_relative_l2_error(u_standard, u_exact, h).item()
    
    energy_exact = compute_energy(u_exact, a, f, h).item()
    
    print(f"\nResults:")
    print(f"  Standard Langevin:       L2 error = {err_standard:.4f}, Final energy = {energies_standard[-1].item():.4f}")
    print(f"  Preconditioned Langevin: L2 error = {err_precond:.4f}, Final energy = {energies_precond[-1].item():.4f}")
    print(f"  Exact solution energy: {energy_exact:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Energy trajectories
    ax = axes[0, 0]
    ax.plot([e.item() for e in energies_standard], label='Standard', alpha=0.8)
    ax.plot([e.item() for e in energies_precond], label='Preconditioned', alpha=0.8)
    ax.axhline(y=energy_exact, color='k', linestyle='--', label='Exact')
    ax.set_xlabel('Step')
    ax.set_ylabel('Energy J(u)')
    ax.set_title('Energy Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Eigenvalue spectrum
    ax = axes[0, 1]
    modes = np.arange(1, n_grid + 1)
    ax.semilogy(modes, precond.eigenvalues.numpy(), 'b-', label='λ_m')
    ax.semilogy(modes, precond.sqrt_eigenvalues.numpy(), 'r--', label='√λ_m')
    ax.set_xlabel('Mode m')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(f'Preconditioner Spectrum (κ={kappa}, α={alpha})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Solutions
    ax = axes[0, 2]
    x_full = get_grid_points(n_grid, include_boundary=True).numpy()
    u_ex_plot = np.concatenate(([0], u_exact[0].numpy(), [0]))
    u_std_plot = np.concatenate(([0], u_standard[0].numpy(), [0]))
    u_pre_plot = np.concatenate(([0], u_precond[0].numpy(), [0]))
    
    ax.plot(x_full, u_ex_plot, 'k-', linewidth=2, label='Exact')
    ax.plot(x_full, u_std_plot, 'r--', alpha=0.8, label=f'Standard (err={err_standard:.3f})')
    ax.plot(x_full, u_pre_plot, 'b--', alpha=0.8, label=f'Precond (err={err_precond:.3f})')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')
    ax.set_title('Final Solutions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Noise samples comparison
    ax = axes[1, 0]
    white_noise = torch.randn(n_grid)
    colored_noise = precond.apply_sqrt(white_noise.unsqueeze(0)).squeeze(0)
    x_interior = get_grid_points(n_grid, include_boundary=False).numpy()
    
    ax.plot(x_interior, white_noise.numpy(), 'r-', alpha=0.6, label='White noise')
    ax.plot(x_interior, colored_noise.numpy(), 'b-', alpha=0.8, label='Colored noise (M^{1/2} ξ)')
    ax.set_xlabel('x')
    ax.set_ylabel('Value')
    ax.set_title('Noise Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Error distribution
    ax = axes[1, 1]
    err_std = u_standard[0].numpy() - u_exact[0].numpy()
    err_pre = u_precond[0].numpy() - u_exact[0].numpy()
    
    ax.plot(x_interior, err_std, 'r-', label='Standard')
    ax.plot(x_interior, err_pre, 'b-', label='Preconditioned')
    ax.set_xlabel('x')
    ax.set_ylabel('u - u_exact')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Spectral content
    ax = axes[1, 2]
    u_std_hat = dst_type1(u_standard[0]).abs().numpy()
    u_pre_hat = dst_type1(u_precond[0]).abs().numpy()
    u_ex_hat = dst_type1(u_exact[0]).abs().numpy()
    
    ax.semilogy(modes, u_ex_hat, 'k-', linewidth=2, label='Exact')
    ax.semilogy(modes, u_std_hat, 'r--', alpha=0.8, label='Standard')
    ax.semilogy(modes, u_pre_hat, 'b--', alpha=0.8, label='Preconditioned')
    ax.set_xlabel('Mode m')
    ax.set_ylabel('|Coefficient|')
    ax.set_title('Spectral Content (DST)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('preconditioned_langevin_test.png', dpi=150)
    print(f"\nSaved visualization to preconditioned_langevin_test.png")

