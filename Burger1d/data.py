"""
Data generation utilities for the 1D viscous Burgers equation.

Provides functions to sample initial conditions u^n(x) on a periodic domain [0, L).
The initial conditions serve as the input to the one-step solution operator.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def sample_grf_periodic(
    n_points: int,
    n_samples: int = 1,
    length_scale: float = 0.3,
    variance: float = 1.0,
    L: float = 1.0,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Sample 1D periodic Gaussian Random Field using spectral method.
    
    The GRF is sampled in Fourier space with power spectrum corresponding
    to a squared-exponential (RBF) kernel, ensuring periodic boundary conditions.
    
    Args:
        n_points: Number of grid points
        n_samples: Number of samples to generate
        length_scale: Correlation length scale
        variance: Marginal variance
        L: Domain length
        device: Torch device
        
    Returns:
        samples: GRF samples, shape (n_samples, n_points)
    """
    # Wavenumbers for periodic domain
    k = torch.fft.fftfreq(n_points, d=L / n_points, device=device) * (2 * np.pi)
    
    # Power spectrum for RBF kernel on periodic domain
    # S(k) ~ exp(-l^2 k^2 / 2)
    power = variance * length_scale * np.sqrt(2 * np.pi) * torch.exp(
        -0.5 * (length_scale * k) ** 2
    )
    
    # Sample complex coefficients in frequency domain
    real_part = torch.randn(n_samples, n_points, device=device)
    imag_part = torch.randn(n_samples, n_points, device=device)
    
    spectrum = (real_part + 1j * imag_part) * power.sqrt().unsqueeze(0)
    
    # Transform to spatial domain
    samples = torch.fft.ifft(spectrum).real * n_points ** 0.5
    
    return samples


def sample_sinusoidal(
    n_points: int,
    n_samples: int = 1,
    amplitude: float = 1.0,
    max_modes: int = 5,
    L: float = 1.0,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Sample periodic initial conditions as a sum of sinusoidal modes.
    
    u(x) = sum_k a_k sin(2*pi*k*x/L + phi_k)
    
    Args:
        n_points: Number of grid points
        n_samples: Number of samples
        amplitude: Overall amplitude scale
        max_modes: Maximum number of Fourier modes
        L: Domain length
        device: Torch device
        
    Returns:
        samples: Shape (n_samples, n_points)
    """
    x = torch.linspace(0, L, n_points + 1, device=device)[:-1]  # Exclude endpoint
    
    samples = torch.zeros(n_samples, n_points, device=device)
    
    for _ in range(max_modes):
        # Random wavenumber (integer for periodicity)
        k = torch.randint(1, max_modes + 1, (n_samples, 1), device=device).float()
        # Random amplitude (decaying with frequency)
        a = torch.randn(n_samples, 1, device=device) / k
        # Random phase
        phi = torch.rand(n_samples, 1, device=device) * 2 * np.pi
        
        samples += a * torch.sin(2 * np.pi * k * x.unsqueeze(0) / L + phi)
    
    # Normalize to target amplitude
    max_val = samples.abs().max(dim=-1, keepdim=True)[0] + 1e-8
    samples = amplitude * samples / max_val
    
    return samples


def sample_initial_condition(
    n_grid: int,
    n_samples: int = 1,
    method: str = 'grf',
    amplitude: float = 1.0,
    length_scale: float = 0.3,
    L: float = 1.0,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Sample initial conditions u^n(x) for the Burgers equation.
    
    Args:
        n_grid: Number of grid points
        n_samples: Number of samples
        method: 'grf', 'sinusoidal', or 'mixed'
        amplitude: Scale factor
        length_scale: GRF length scale
        L: Domain length
        device: Torch device
        
    Returns:
        u: Initial conditions, shape (n_samples, n_grid)
    """
    if method == 'mixed':
        u_list = []
        for _ in range(n_samples):
            r = np.random.rand()
            if r < 0.6:
                m_i = 'grf'
            else:
                m_i = 'sinusoidal'
            
            # Random amplitude variation
            amp = float(amplitude * 10 ** np.random.uniform(-0.3, 0.3))
            ls = np.random.uniform(0.1, 0.5)
            
            u_i = sample_initial_condition(
                n_grid, 1, method=m_i, amplitude=amp,
                length_scale=ls, L=L, device=device
            )
            u_list.append(u_i)
        
        u = torch.cat(u_list, dim=0)
        
    elif method == 'grf':
        raw = sample_grf_periodic(
            n_grid, n_samples, length_scale=length_scale,
            variance=1.0, L=L, device=device
        )
        # Normalize to target amplitude
        max_val = raw.abs().max(dim=-1, keepdim=True)[0] + 1e-8
        u = amplitude * raw / max_val
        
    elif method == 'sinusoidal':
        u = sample_sinusoidal(
            n_grid, n_samples, amplitude=amplitude,
            L=L, device=device
        )
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return u


class BurgersDataset(Dataset):
    """
    Dataset for 1D Burgers equation training.
    
    Generates initial conditions u^n on the fly or from a pre-generated buffer.
    """
    
    def __init__(
        self,
        n_grid: int,
        n_samples: int,
        method: str = 'grf',
        amplitude: float = 1.0,
        length_scale: float = 0.3,
        L: float = 1.0,
        pregenerate: bool = True,
        device: str = 'cpu'
    ):
        """
        Args:
            n_grid: Number of grid points
            n_samples: Number of samples in dataset
            method: Method for sampling initial conditions
            amplitude: IC amplitude
            length_scale: GRF length scale
            L: Domain length
            pregenerate: If True, pre-generate all samples
            device: Torch device
        """
        self.n_grid = n_grid
        self.n_samples = n_samples
        self.method = method
        self.amplitude = amplitude
        self.length_scale = length_scale
        self.L = L
        self.device = device
        
        if pregenerate:
            self.u_data = sample_initial_condition(
                n_grid, n_samples, method=method,
                amplitude=amplitude, length_scale=length_scale,
                L=L, device='cpu'  # Pre-generate on CPU
            )
        else:
            self.u_data = None
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        if self.u_data is not None:
            return self.u_data[idx]
        else:
            u = sample_initial_condition(
                self.n_grid, 1, method=self.method,
                amplitude=self.amplitude, length_scale=self.length_scale,
                L=self.L, device='cpu'
            )[0]
            return u


def create_dataloader(
    n_grid: int,
    n_samples: int,
    batch_size: int,
    **dataset_kwargs
) -> DataLoader:
    """Create a DataLoader for training."""
    dataset = BurgersDataset(n_grid, n_samples, **dataset_kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    n_grid = 64
    L = 1.0
    x = np.linspace(0, L, n_grid, endpoint=False)
    
    # Sample initial conditions
    u_grf = sample_initial_condition(n_grid, 5, method='grf', amplitude=1.0, L=L)
    u_sin = sample_initial_condition(n_grid, 5, method='sinusoidal', amplitude=1.0, L=L)
    u_mix = sample_initial_condition(n_grid, 5, method='mixed', amplitude=1.0, L=L)
    
    print(f"GRF shape: {u_grf.shape}")
    print(f"Sinusoidal shape: {u_sin.shape}")
    print(f"Mixed shape: {u_mix.shape}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i in range(5):
        axes[0].plot(x, u_grf[i].numpy(), alpha=0.7)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u(x)')
    axes[0].set_title('GRF Initial Conditions')
    axes[0].grid(True, alpha=0.3)
    
    for i in range(5):
        axes[1].plot(x, u_sin[i].numpy(), alpha=0.7)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('u(x)')
    axes[1].set_title('Sinusoidal Initial Conditions')
    axes[1].grid(True, alpha=0.3)
    
    for i in range(5):
        axes[2].plot(x, u_mix[i].numpy(), alpha=0.7)
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('u(x)')
    axes[2].set_title('Mixed Initial Conditions')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_samples_burgers.png', dpi=150)
    print("Saved data_samples_burgers.png")
    
    # Test dataset
    dataset = BurgersDataset(n_grid, 100)
    loader = DataLoader(dataset, batch_size=16)
    u_batch = next(iter(loader))
    print(f"Batch shape: u={u_batch.shape}")
