"""
Data generation utilities for the 1D nonlinear Poisson (Ginzburg-Landau) equation.

Provides functions to sample:
- Source terms s(x) using Gaussian Random Fields or sinusoidal functions
- PyTorch Dataset for training
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def sample_grf_1d(
    n_points: int,
    n_samples: int = 1,
    length_scale: float = 0.2,
    variance: float = 1.0,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Sample 1D Gaussian Random Field using spectral method.
    
    Args:
        n_points: Number of grid points
        n_samples: Number of samples to generate
        length_scale: Correlation length scale
        variance: Marginal variance
        device: Torch device
        
    Returns:
        samples: GRF samples, shape (n_samples, n_points)
    """
    # Grid points
    x = np.linspace(0, 1, n_points)
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    
    # Frequency domain
    k = np.fft.fftfreq(n_points, d=dx)
    
    # Squared exponential spectral density
    S = variance * length_scale * np.sqrt(2 * np.pi) * np.exp(-0.5 * (2 * np.pi * k * length_scale)**2)
    
    # Sample in frequency domain
    np.random.seed(None)  # Ensure randomness
    phase = np.random.uniform(0, 2 * np.pi, (n_samples, n_points))
    amplitude = np.sqrt(S) * np.random.randn(n_samples, n_points)
    
    # Complex coefficients
    coeffs = amplitude * np.exp(1j * phase)
    
    # Transform to physical space
    samples_np = np.real(np.fft.ifft(coeffs, axis=-1)) * n_points
    
    samples = torch.tensor(samples_np, dtype=torch.float32, device=device)
    
    return samples



def sample_matern_gp_1d(
    n_points: int,
    n_samples: int = 1,
    length_scale: float = 0.2,
    nu: float = 2.5,
    variance: float = 1.0,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Sample 1D Gaussian Process with Matern Kernel.
    
    Covariance K(x, y) based on Matern kernel.
    We compute the full covariance matrix and sample using Cholesky.

    Args:
        n_points: Number of grid points
        n_samples: Number of samples
        length_scale: Length scale parameter
        nu: Smoothness parameter (1.5 for Matern 3/2, 2.5 for Matern 5/2)
        variance: Marginal variance (scale factor)
        device: Torch device

    Returns:
        samples: shape (n_samples, n_points)
    """
    # Grid points
    x = torch.linspace(0, 1, n_points, device=device)
    
    # Compute dist matrix: |x_i - x_j|
    dist = torch.abs(x.unsqueeze(0) - x.unsqueeze(1)) # (n, n)
    
    # Matern Kernel
    # d_scaled = sqrt(2*nu) * d / length_scale
    if nu == 0.5:
        # Exponential
        K = variance * torch.exp(-dist / length_scale)
    elif nu == 1.5:
        # Matern 3/2
        sqrt3 = np.sqrt(3.0)
        d_scaled = sqrt3 * dist / length_scale
        K = variance * (1 + d_scaled) * torch.exp(-d_scaled)
    elif nu == 2.5:
        # Matern 5/2
        sqrt5 = np.sqrt(5.0)
        d_scaled = sqrt5 * dist / length_scale
        K = variance * (1 + d_scaled + (d_scaled**2)/3.0) * torch.exp(-d_scaled)
    else:
        # General form using Bessel functions (if needed, but 1.5/2.5 are most common)
        # For simplicity in this context, we fallback to RBF/SE if nu is very large, 
        # or error if not implemented. Let's assume nu=inf -> RBF
        if nu > 10.0:
            K = variance * torch.exp(-0.5 * (dist/length_scale)**2)
        else:
            raise NotImplementedError(f"Matern nu={nu} not implemented (use 0.5, 1.5, 2.5, or >10)")
            
    # Add jitter for stability
    K = K + 1e-6 * torch.eye(n_points, device=device)
    
    # Cholesky decomposition: K = L L^T
    try:
        L = torch.linalg.cholesky(K)
    except RuntimeError:
        # Larger jitter fallback
        K = K + 1e-4 * torch.eye(n_points, device=device)
        L = torch.linalg.cholesky(K)
        
    # Sample Z ~ N(0, I)
    z = torch.randn(n_samples, n_points, device=device)
    
    # X = Z L^T  (since covariance acts on last dim)
    # But usually x = L z. 
    # Cov(x) = L E[zz^T] L^T = L L^T = K.
    # If z is (n_samples, n), we want (n_samples, n).
    # so we want x.T = L z.T => x = z L^T
    
    samples = torch.matmul(z, L.t())
    
    return samples


def sample_source_s(
    n_interior: int,
    n_samples: int = 1,
    method: str = 'grf',
    amplitude: float = 1.0,
    length_scale: float = 0.3,
    nu: float = 2.5,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Sample source term s(x) for the nonlinear Poisson equation.
    
    Args:
        n_interior: Number of interior grid points
        n_samples: Number of samples
        method: 'grf', 'sinusoidal', 'constant', or 'matern'
        amplitude: Scale factor for the source term
        length_scale: GRF/Matern length scale
        nu: Matern smoothness (only for method='matern')
        device: Torch device
        
    Returns:
        s: Source term on interior grid, shape (n_samples, n_interior)
    """
    if method == 'mixed':
        # Generate mixed samples with random parameters for each
        # For efficiency, we can't fully vectorize if every sample has unique kernel params.
        # But we can loop.
        s_list = []
        for _ in range(n_samples):
            u = np.random.rand()
            if u < 0.5:
                m_i = 'matern'
            elif u < 0.85:
                m_i = 'grf'
            else:
                m_i = 'sinusoidal'
            
            # Random params
            amp = float(amplitude * 10**np.random.uniform(0, 1.5)) # Reduced range for stability
            ls = np.random.uniform(0.05, 0.4)
            nu_i = 1.5 if np.random.rand() < 0.5 else 2.5
            
            s_i = sample_source_s(
                n_interior, 1, 
                method=m_i, 
                amplitude=amplitude, 
                length_scale=ls, 
                nu=nu_i, 
                device=device
            )
            s_list.append(s_i)
        
        s = torch.cat(s_list, dim=0)
        
    elif method == 'grf':
        # Spectral method, implies periodic boundaries or close to it
        s = sample_grf_1d(
            n_interior, n_samples, 
            length_scale=length_scale,
            variance=1.0,
            device=device
        )
        s = amplitude * s
    
    elif method == 'matern':
        # Gaussian Process with Matern kernel (no boundary constraints)
        s = sample_matern_gp_1d(
            n_interior, n_samples,
            length_scale=length_scale,
            nu=nu,
            variance=1.0,
            device=device
        )
        s = amplitude * s
        
    elif method == 'sinusoidal':
        # Random sinusoidal source terms
        x = torch.linspace(0, 1, n_interior + 2, device=device)[1:-1]  # Interior points
        
        # Random frequency and phase
        freq = torch.rand(n_samples, 1, device=device) * 3 + 1  # Frequency in [1, 4]
        phase = torch.rand(n_samples, 1, device=device) * 2 * np.pi
        
        s = torch.sin(2 * np.pi * freq * x.unsqueeze(0) + phase) * freq 
        s = amplitude * s
        
    elif method == 'constant':
        s = amplitude * torch.ones(n_samples, n_interior, device=device)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return s


class GinzburgLandauDataset(Dataset):
    """
    Dataset for 1D nonlinear Poisson (Ginzburg-Landau) equation training.
    
    Generates source terms s on the fly or from a pre-generated buffer.
    """
    
    def __init__(
        self,
        n_interior: int,
        n_samples: int,
        method: str = 'grf',
        amplitude: float = 1.0,
        length_scale: float = 0.3,
        pregenerate: bool = True,
        device: str = 'cpu'
    ):
        """
        Args:
            n_interior: Number of interior grid points
            n_samples: Number of samples in dataset
            method: Method for sampling source ('grf', 'sinusoidal')
            amplitude: Source amplitude
            length_scale: GRF length scale
            pregenerate: If True, pre-generate all samples
            device: Torch device
        """
        self.n_interior = n_interior
        self.n_samples = n_samples
        self.method = method
        self.amplitude = amplitude
        self.length_scale = length_scale
        self.device = device
        
        if pregenerate:
            self.s_data = sample_source_s(
                n_interior, n_samples, 
                method=method,
                amplitude=amplitude,
                length_scale=length_scale,
                device='cpu'  # Pre-generate on CPU
            )
        else:
            self.s_data = None
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        if self.s_data is not None:
            return self.s_data[idx]
        else:
            # Generate on the fly
            
            if self.method == 'mixed':
                # Randomly choose method
                u = np.random.rand()
                # 40% GRF, 30% Matern, 30% Sinusoidal
                if u < 0.4:
                    method_i = 'grf'
                elif u < 0.7:
                    method_i = 'matern'
                else:
                    method_i = 'sinusoidal'
                    
                # Random parameters
                # Amplitude log-uniform in [1, 100]
                amp = float(self.amplitude * 10**np.random.uniform(0, 2))
                
                # Length scale uniform in [0.05, 0.4]
                ls = np.random.uniform(0.05, 0.4)
                
                # Nu for Matern (1.5 or 2.5)
                nu = 1.5 if np.random.rand() < 0.5 else 2.5
                
                s = sample_source_s(
                    self.n_interior, 1,
                    method=method_i,
                    amplitude=amp,
                    length_scale=ls,
                    nu=nu,
                    device='cpu'
                )[0]
                return s
            else:
                s = sample_source_s(
                    self.n_interior, 1,
                    method=self.method,
                    amplitude=self.amplitude,
                    length_scale=self.length_scale,
                    device='cpu'
                )[0]
                return s


def create_dataloader(
    n_interior: int,
    n_samples: int,
    batch_size: int,
    **dataset_kwargs
) -> DataLoader:
    """
    Create a DataLoader for training.
    """
    dataset = GinzburgLandauDataset(n_interior, n_samples, **dataset_kwargs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    n_interior = 50
    
    # Sample source terms
    s_grf = sample_source_s(n_interior, 5, method='grf', amplitude=2.0)
    s_sin = sample_source_s(n_interior, 5, method='sinusoidal', amplitude=2.0)
    
    print(f"GRF source shape: {s_grf.shape}")
    print(f"Sinusoidal source shape: {s_sin.shape}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    x = np.linspace(0, 1, n_interior + 2)[1:-1]
    
    ax = axes[0]
    for i in range(5):
        ax.plot(x, s_grf[i].numpy(), alpha=0.7)
    ax.set_xlabel('x')
    ax.set_ylabel('s(x)')
    ax.set_title('GRF Source Terms')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    for i in range(5):
        ax.plot(x, s_sin[i].numpy(), alpha=0.7)
    ax.set_xlabel('x')
    ax.set_ylabel('s(x)')
    ax.set_title('Sinusoidal Source Terms')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('source_samples.png', dpi=150)
    print("Saved source_samples.png")
    
    # Test dataset
    dataset = GinzburgLandauDataset(n_interior, 100)
    loader = DataLoader(dataset, batch_size=16)
    s_batch = next(iter(loader))
    print(f"Batch shape: s={s_batch.shape}")
