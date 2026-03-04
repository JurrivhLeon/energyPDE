"""
Preconditioned Langevin dynamics for 1D Burgers equation (Periodic BCs).

Based on implementations in Poisson1d and Darcy2d.

Update rule:
    u_{k+1} = u_k - step_size * M ∇J(u_k) + noise_scale * M^{1/2} ξ_k

where:
    - M = (κ² I - Δ)^{-α} is the Matérn-type preconditioner
    - ξ_k ~ N(0, I) is white noise in physical space

For Periodic BCs, we use the Fourier basis. M is diagonal in Fourier space:
    λ_k = (κ² + k²)^{-α}
    
where k are integer wavenumbers (L-independent definition).
"""

import torch
import numpy as np
import math

from energy import compute_energy_gradient, compute_energy


class FourierPreconditioner:
    """
    Matérn-type preconditioner M = (κ² I - Δ)^{-α} for 1D Periodic BCs.
    
    In the Fourier basis, this operator is diagonal with eigenvalues:
        λ_k = (κ² + k²)^{-α}
        
    Args:
        n_grid: Number of grid points
        kappa: Correlation length parameter κ (larger = shorter correlation)
        alpha: Smoothing exponent α (larger = stronger smoothing)
        normalize: If True, scale eigenvalues so max(λ) = 1.0
        device: Torch device
        dtype: Torch dtype
    """
    
    def __init__(
        self,
        n_grid: int,
        kappa: float = 1.0,
        alpha: float = 2.0,
        L: float = 1.0,
        normalize: bool = True,
        scaling: str = 'max',
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32
    ):
        self.n = n_grid
        self.kappa = kappa
        self.alpha = alpha
        self.L = L
        self.device = device
        self.dtype = dtype
        
        # 1. Integer wavenumbers for Periodic domain
        # k = 0, 1, ..., n/2, -n/2+1, ..., -1
        # Corresponds to 2πk/L physical wavenumbers, but we use integer indices
        # to be consistent with Poisson1d/Darcy2d implementations and avoid
        # L-dependence in the preconditioner strength relative to the grid.
        self.k_int = torch.fft.fftfreq(n_grid, d=1.0/n_grid, device=device).to(dtype)
        # fftfreq returns f = k/n (if d=1). With d=1/n, it returns k.
        # Actually fftfreq(n, d) returns f = [0, 1, ..., -1] / (n*d).
        # with d=1/n, n*d=1, so it returns integers [0, 1, ..., n/2, -n/2+1, ..., -1]
        
        # 2. Eigenvalues: (κ² + k²)^{-α}
        k_sq = self.k_int ** 2
        eigenvalues = (kappa**2 + k_sq) ** (-alpha)
        
        # 3. Scaling
        if scaling == 'trace':
            # Scale so sum(lambda) = n (preserving total variance of white noise)
            scale = n_grid / eigenvalues.sum()
            eigenvalues = eigenvalues * scale
        elif normalize or scaling == 'max':
            # Scale so max(lambda) = 1.0 (default)
            eigenvalues = eigenvalues / eigenvalues.max()
            
        self.eigenvalues = eigenvalues
        self.eigenvalues_sqrt = eigenvalues.sqrt()
        
        # Complex versions for broadcasting with FFT results
        # (Eigenvalues are real and symmetric for even functions, so valid for FFT)
        self.eigenvalues_c = self.eigenvalues.to(torch.complex64 if dtype==torch.float32 else torch.complex128)
        self.eigenvalues_sqrt_c = self.eigenvalues_sqrt.to(torch.complex64 if dtype==torch.float32 else torch.complex128)

    def to(self, device: str):
        self.device = device
        self.eigenvalues = self.eigenvalues.to(device)
        self.eigenvalues_sqrt = self.eigenvalues_sqrt.to(device)
        self.eigenvalues_c = self.eigenvalues_c.to(device)
        self.eigenvalues_sqrt_c = self.eigenvalues_sqrt_c.to(device)
        return self
        
    def apply(self, v: torch.Tensor) -> torch.Tensor:
        """Apply M to vector v: M v = IFFT( λ · FFT(v) )"""
        v_hat = torch.fft.fft(v)
        Mv_hat = v_hat * self.eigenvalues_c
        return torch.fft.ifft(Mv_hat).real
    
    def apply_sqrt(self, v: torch.Tensor) -> torch.Tensor:
        """Apply M^{1/2} to vector v: M^{1/2} v = IFFT( √λ · FFT(v) )"""
        v_hat = torch.fft.fft(v)
        Mv_hat = v_hat * self.eigenvalues_sqrt_c
        return torch.fft.ifft(Mv_hat).real
        
    def sample_noise(self, shape: tuple) -> torch.Tensor:
        """
        Sample colored noise M^{1/2} ξ where ξ ~ N(0, I).
        Returns noise in physical space.
        """
        # 1. Sample white noise in physical space
        xi = torch.randn(shape, device=self.device, dtype=self.dtype)
        
        # 2. Apply M^{1/2}
        return self.apply_sqrt(xi)


def preconditioned_langevin_step(
    u: torch.Tensor,
    u_curr: torch.Tensor,
    nu: float,
    dt: float,
    L: float,
    step_size: float,
    noise_scale: float,
    preconditioner: FourierPreconditioner,
    grad_clip: float = None
) -> torch.Tensor:
    """
    One step of Preconditioned Langevin: u_{k+1} = u_k - step_size * M ∇J + noise_scale * M^{1/2} ξ
    """
    # 1. Gradient of Energy J(u)
    # Energy is defined in energy.py. Gradient is computed via backprop or analytical formula
    grad_J = compute_energy_gradient(u, u_curr, nu, dt, L)
    
    # 2. Precondition Gradient: M ∇J
    M_grad = preconditioner.apply(grad_J)
    
    # 3. Clip Gradient (optional)
    if grad_clip is not None:
        M_grad = torch.clamp(M_grad, -grad_clip, grad_clip)
        
    # 4. Noise: noise_scale * M^{1/2} ξ
    if noise_scale > 0.0:
        noise = preconditioner.sample_noise(u.shape)
        noise_term = noise_scale * noise
    else:
        noise_term = 0.0
        
    # 5. Update
    u_new = u - step_size * M_grad + noise_term
    
    return u_new


def preconditioned_langevin_refine(
    u0: torch.Tensor,
    u_curr: torch.Tensor,
    nu: float,
    dt: float,
    L: float,
    step_size: float,
    noise_scale: float,
    steps: int,
    preconditioner: FourierPreconditioner,
    return_trajectory: bool = False,
    grad_clip: float = None
) -> torch.Tensor:
    """Run Langevin dynamics for k steps."""
    u = u0.clone()
    
    if return_trajectory:
        trajectory = [u.clone()]
        
    for _ in range(steps):
        u = preconditioned_langevin_step(
            u, u_curr, nu, dt, L,
            step_size, noise_scale, preconditioner,
            grad_clip=grad_clip
        )
        if return_trajectory:
            trajectory.append(u.clone())
            
    if return_trajectory:
        return u, trajectory
    return u


# Self-test
if __name__ == '__main__':
    from utils import get_grid_points
    import matplotlib.pyplot as plt
    
    n_grid = 128
    L = 1.0
    u0 = torch.randn(1, n_grid)
    precond = FourierPreconditioner(n_grid, kappa=1.0, alpha=2.0)
    
    # Check noise smoothness
    noise = precond.sample_noise((5, n_grid))
    
    plt.figure()
    for i in range(5):
        plt.plot(noise[i].numpy(), label=f'Sample {i}')
    plt.title("Smoothed Noise Samples (alpha=2.0, Matern)")
    plt.legend()
    plt.savefig("debug_noise_samples_reimpl.png")
    print("Saved debug_noise_samples_reimpl.png")
