"""
Data generation utilities for 2D Darcy flow equation.

Provides functions to sample:
- Permeability fields a(x,y) using 2D Gaussian Random Fields
- Forcing terms f(x,y)
- PyTorch Dataset for training
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def sample_grf_2d(
    N: int,
    n_samples: int = 1,
    length_scale: float = 0.2,
    variance: float = 1.0,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Sample 2D Gaussian Random Field using spectral method.
    
    Uses isotropic RBF covariance in Fourier domain.
    
    Args:
        N: Grid size (N×N points on interior)
        n_samples: Number of samples to generate
        length_scale: Correlation length scale (float or tensor of shape (n_samples,))
        variance: Marginal variance
        device: Device to create tensors on
        
    Returns:
        samples: GRF samples, shape (n_samples, N, N)
    """
    if n_samples == 0:
        return torch.zeros(0, N, N, device=device)
    
    # 2D frequency grid
    freqs_x = torch.fft.fftfreq(N, d=1.0/N, device=device)
    freqs_y = torch.fft.fftfreq(N, d=1.0/N, device=device)
    Fx, Fy = torch.meshgrid(freqs_x, freqs_y, indexing='ij')
    
    # 2D power spectrum for isotropic RBF kernel
    # k(r) = variance * exp(-r²/(2*l²))
    # Power spectrum: P(f) ∝ exp(-2π² l² |f|²)
    freq_sq = Fx**2 + Fy**2
    
    # Handle length_scale
    if isinstance(length_scale, torch.Tensor):
        # length_scale: (n_samples, 1, 1) or (n_samples,)
        l = length_scale.view(n_samples, 1, 1).to(device)
    else:
        l = length_scale
        
    power = variance * (2 * np.pi * l**2) * torch.exp(
        -2 * (np.pi * l)**2 * freq_sq.unsqueeze(0)
    )
    
    # Sample in frequency domain
    real_part = torch.randn(n_samples, N, N, device=device)
    imag_part = torch.randn(n_samples, N, N, device=device)
    
    spectrum = (real_part + 1j * imag_part) * power.sqrt()
    
    # Transform to spatial domain
    samples = torch.fft.ifft2(spectrum).real * N
    
    return samples


def sample_matern_2d(
    N: int,
    n_samples: int = 1,
    length_scale: float = 0.2,
    alpha: float = 2.0,
    variance: float = 1.0,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Sample 2D Matérn field using spectral method.
    
    Power spectrum: P(f) ∝ (κ² + |f|²)^(-α)
    where κ ≈ 1/length_scale.
    
    Args:
        N: Grid size
        n_samples: Number of samples
        length_scale: Correlation length (float or tensor of shape (n_samples,))
        alpha: Smoothness parameter
        variance: Scaling factor
        device: Device
    """
    if n_samples == 0:
        return torch.zeros(0, N, N, device=device)
        
    freqs_x = torch.fft.fftfreq(N, d=1.0/N, device=device)
    freqs_y = torch.fft.fftfreq(N, d=1.0/N, device=device)
    Fx, Fy = torch.meshgrid(freqs_x, freqs_y, indexing='ij')
    
    # Wave number |k|^2 = |2π f|^2
    k_sq = (2 * np.pi * Fx)**2 + (2 * np.pi * Fy)**2
    
    # Handle length_scale
    if isinstance(length_scale, torch.Tensor):
        l = length_scale.view(n_samples, 1, 1).to(device)
    else:
        l = length_scale
        
    # kappa ≈ 1/l
    kappa = 1.0 / (l + 1e-8)
    
    # Spectrum
    power = (kappa**2 + k_sq.unsqueeze(0))**(-alpha)
    
    # Sample in frequency domain
    real_part = torch.randn(n_samples, N, N, device=device)
    imag_part = torch.randn(n_samples, N, N, device=device)
    
    spectrum = (real_part + 1j * imag_part) * power.sqrt()
    
    # Transform
    samples = torch.fft.ifft2(spectrum).real * N
    
    # Rescale to desired variance
    current_std = samples.std(dim=(-2,-1), keepdim=True)
    samples = samples / (current_std + 1e-8) * (variance**0.5)
    
    return samples


def sample_coefficient_a(
    N: int,
    n_samples: int = 1,
    method: str = 'grf',
    a_min: float = 0.1,
    a_max: float = 2.0,
    length_scale: float = 0.3,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Sample permeability field a(x,y) for 2D Darcy flow.
    
    Args:
        N: Interior grid size (N×N)
        n_samples: Number of samples
        method: 'grf', 'lognormal', 'matern', 'constant', or 'mixed'
        a_min: Minimum permeability value
        a_max: Maximum permeability value
        length_scale: Correlation length
        device: Device
        
    Returns:
        a: Permeability on full grid (including boundaries), shape (n_samples, N+2, N+2)
    """
    N_full = N + 2
    
    if method == 'constant':
        a = torch.ones(n_samples, N_full, N_full, device=device)
        
    elif method == 'grf':
        # Sample GRF on full grid
        raw = sample_grf_2d(N_full, n_samples, length_scale=length_scale, device=device)
        # Scale for sigmoid
        raw_scaled = 3.0 * raw / (raw.std() + 1e-8)
        a = a_min + (a_max - a_min) * torch.sigmoid(raw_scaled)
        
    elif method == 'lognormal':
        # Log-normal permeability
        raw = sample_grf_2d(N_full, n_samples, length_scale=length_scale, device=device)
        log_mean = 0.5 * (np.log(a_min) + np.log(a_max))
        log_std = 0.3 * (np.log(a_max) - np.log(a_min))
        a = torch.exp(log_mean + log_std * raw / (raw.std() + 1e-8))
        a = torch.clamp(a, a_min, a_max)
        
    elif method == 'matern':
        # Matern field with alpha=2.5
        raw = sample_matern_field_wrapper(N_full, n_samples, length_scale, alpha=2.5, device=device)
        # Scale for sigmoid (using same logic as GRF)
        raw_scaled = 3.0 * raw / (raw.std() + 1e-8)
        a = a_min + (a_max - a_min) * torch.sigmoid(raw_scaled)
        
    elif method == 'mixed':
        # Mix of GRF, LogNormal, and Matern
        n1 = n_samples // 3
        n2 = n_samples // 3
        n3 = n_samples - n1 - n2
        
        # Randomize length scales for diversity: U[0.1, 0.5]
        # (Default was 0.3)
        l1 = torch.rand(n1, device=device) * 0.4 + 0.1
        l2 = torch.rand(n2, device=device) * 0.4 + 0.1
        l3 = torch.rand(n3, device=device) * 0.4 + 0.1
        
        a1 = sample_coefficient_a(N, n1, 'grf', a_min, a_max, length_scale=l1, device=device)
        a2 = sample_coefficient_a(N, n2, 'lognormal', a_min, a_max, length_scale=l2, device=device)
        a3 = sample_coefficient_a(N, n3, 'matern', a_min, a_max, length_scale=l3, device=device)
        
        a = torch.cat([a1, a2, a3], dim=0)
        # Shuffle
        perm = torch.randperm(n_samples, device=device)
        a = a[perm]
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return a


def sample_matern_field_wrapper(N, n_samples, length_scale, alpha, device):
    """Helper to call sample_matern_2d with correct params"""
    return sample_matern_2d(N, n_samples, length_scale=length_scale, alpha=alpha, device=device)


def sample_forcing_f(
    N: int,
    n_samples: int = 1,
    method: str = 'grf',
    amplitude: float = 1.0,
    length_scale: float = 0.3,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Sample forcing term f(x,y).
    
    Args:
        N: Interior grid size (N×N)
        n_samples: Number of samples
        method: 'grf', 'sinusoidal', or 'point_source'
        amplitude: Amplitude scaling
        length_scale: Correlation length for GRF
        device: Device
        
    Returns:
        f: Forcing on interior grid, shape (n_samples, N, N)
    """
    if method == 'grf':
        raw = sample_grf_2d(N, n_samples, length_scale=length_scale, device=device)
        # Normalize and scale
        max_val = raw.abs().amax(dim=(-2, -1), keepdim=True) + 1e-8
        f = amplitude * (raw / max_val)
        
    elif method == 'matern':
        # Matern field with alpha=2.5 (similar to mixed permeability)
        raw = sample_matern_field_wrapper(N, n_samples, length_scale, alpha=2.5, device=device)
        # Normalize and scale
        max_val = raw.abs().amax(dim=(-2, -1), keepdim=True) + 1e-8
        f = amplitude * (raw / max_val)
    
    elif method == 'constant':
        f = amplitude * torch.ones(n_samples, N, N, device=device)
        
    elif method == 'sinusoidal':
        h = 1.0 / (N + 1)
        x = torch.linspace(h, 1 - h, N, device=device)
        y = torch.linspace(h, 1 - h, N, device=device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Random frequencies for each sample
        freq_x = torch.randint(1, 4, (n_samples, 1, 1), device=device).float()
        freq_y = torch.randint(1, 4, (n_samples, 1, 1), device=device).float()
        
        f = amplitude * torch.sin(np.pi * freq_x * X) * torch.sin(np.pi * freq_y * Y)
        
    elif method == 'point_source':
        # Gaussian point sources at random locations
        f = torch.zeros(n_samples, N, N, device=device)
        h = 1.0 / (N + 1)
        x = torch.linspace(h, 1 - h, N, device=device)
        y = torch.linspace(h, 1 - h, N, device=device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        for i in range(n_samples):
            n_sources = np.random.randint(1, 4)
            for _ in range(n_sources):
                cx, cy = np.random.uniform(0.2, 0.8, 2)
                sigma = np.random.uniform(0.05, 0.15)
                # sign = 1 if np.random.rand() > 0.5 else -1 # Keep positive or mixed? 
                # Forcing can be negative. 
                sign = 1 if np.random.rand() > 0.5 else -1
                f[i] += sign * amplitude * torch.exp(
                    -((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2)
                )
    
    elif method == 'mixed':
        # Mix of GRF, Matern, and Point Source
        n1 = n_samples // 3
        n2 = n_samples // 3
        n3 = n_samples - n1 - n2
        
        # Randomize length scales for diversity where applicable (GRF, Matern)
        l1 = torch.rand(n1, device=device) * 0.4 + 0.1
        l2 = torch.rand(n2, device=device) * 0.4 + 0.1
        # Point Source doesn't use length_scale directly in the same way (uses sigma), 
        # but we can pass it anyway or ignore.
        
        f1 = sample_forcing_f(N, n1, 'grf', amplitude, length_scale=l1, device=device)
        f2 = sample_forcing_f(N, n2, 'matern', amplitude, length_scale=l2, device=device)
        f3 = sample_forcing_f(N, n3, 'point_source', amplitude, device=device)
        
        f = torch.cat([f1, f2, f3], dim=0)
        perm = torch.randperm(n_samples, device=device)
        f = f[perm]
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return f


class DarcyDataset(Dataset):
    """
    Dataset for 2D Darcy flow equation training.
    
    Generates pairs (a, f) on the fly or from a pre-generated buffer.
    """
    
    def __init__(
        self,
        N: int,
        n_samples: int,
        a_method: str = 'grf',
        f_method: str = 'grf',
        a_min: float = 0.1,
        a_max: float = 2.0,
        a_length_scale: float = 0.3,
        f_length_scale: float = 0.3,
        f_amplitude: float = 1.0,
        pregenerate: bool = True,
        device: str = 'cpu'
    ):
        """
        Args:
            N: Interior grid size (N×N points)
            n_samples: Number of samples in dataset
            a_method: Method for sampling permeability
            f_method: Method for sampling forcing
            pregenerate: If True, generate all samples upfront
        """
        self.N = N
        self.n_samples = n_samples
        self.pregenerate = pregenerate
        self.device = device
        
        # Store params for on-the-fly generation
        self.a_params = {
            'method': a_method,
            'a_min': a_min,
            'a_max': a_max,
            'length_scale': a_length_scale
        }
        self.f_params = {
            'method': f_method,
            'amplitude': f_amplitude,
            'length_scale': f_length_scale
        }
        
        if pregenerate:
            self.a_data = sample_coefficient_a(
                N, n_samples, device='cpu', **self.a_params
            )
            self.f_data = sample_forcing_f(
                N, n_samples, device='cpu', **self.f_params
            )
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        if self.pregenerate:
            return self.a_data[idx], self.f_data[idx]
        else:
            a = sample_coefficient_a(
                self.N, 1, device='cpu', **self.a_params
            ).squeeze(0)
            f = sample_forcing_f(
                self.N, 1, device='cpu', **self.f_params
            ).squeeze(0)
            return a, f


def create_dataloader(
    N: int,
    n_samples: int,
    batch_size: int,
    **dataset_kwargs
) -> DataLoader:
    """
    Create a DataLoader for training.
    """
    dataset = DarcyDataset(N, n_samples, **dataset_kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    torch.manual_seed(42)
    
    N = 32  # Interior grid size
    
    # Sample permeabilities
    a_grf = sample_coefficient_a(N, 4, method='grf')
    a_const = sample_coefficient_a(N, 4, method='constant')
    
    # Sample forcings
    f_grf = sample_forcing_f(N, 4, method='grf')
    f_sin = sample_forcing_f(N, 4, method='sinusoidal')
    
    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i in range(4):
        axes[0, i].imshow(a_grf[i].numpy(), cmap='viridis')
        axes[0, i].set_title(f'Permeability a (GRF) #{i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(f_grf[i].numpy(), cmap='RdBu_r')
        axes[1, i].set_title(f'Forcing f (GRF) #{i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig("data_samples_2d.png", dpi=150)
    print("Saved plot to data_samples_2d.png")
    
    # Test dataset
    dataset = DarcyDataset(N, 100)
    loader = DataLoader(dataset, batch_size=16)
    a_batch, f_batch = next(iter(loader))
    print(f"Batch shapes: a={a_batch.shape}, f={f_batch.shape}")
    print(f"Permeability range: [{a_batch.min():.3f}, {a_batch.max():.3f}]")
