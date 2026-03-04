"""
Data generation utilities for the 1D Poisson equation.

Provides functions to sample:
- Coefficient functions a(x) using Gaussian Random Fields
- Forcing terms f(x)
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
        device: Device to create tensors on
        
    Returns:
        samples: GRF samples, shape (n_samples, n_points)
    """
    # Grid points
    x = torch.linspace(0, 1, n_points, device=device)
    
    # Spectral method: sample in frequency domain
    # Use RBF (Gaussian) covariance: k(d) = variance * exp(-d^2 / (2 * l^2))
    freqs = torch.fft.fftfreq(n_points, d=1.0/n_points, device=device)
    
    # Power spectrum for RBF kernel
    power = variance * (2 * np.pi * length_scale**2) ** 0.5 * torch.exp(
        -2 * (np.pi * length_scale * freqs) ** 2
    )
    
    # Sample in frequency domain
    # Note: real FFT would be more efficient but complex FFT is simpler
    real_part = torch.randn(n_samples, n_points, device=device)
    imag_part = torch.randn(n_samples, n_points, device=device)
    
    spectrum = (real_part + 1j * imag_part) * power.sqrt().unsqueeze(0)
    
    # Transform to spatial domain
    samples = torch.fft.ifft(spectrum).real * n_points ** 0.5
    
    return samples


def sample_coefficient_a(
    n_interior: int,
    n_samples: int = 1,
    method: str = 'grf',
    a_min: float = 0.1,
    a_max: float = 2.0,
    length_scale: float = 0.3,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Sample coefficient function a(x) for the Poisson equation.
    
    Args:
        n_interior: Number of interior grid points
        n_samples: Number of samples
        method: 'grf' (Gaussian Random Field), 'constant', 'piecewise'
        a_min: Minimum coefficient value (positivity constraint)
        a_max: Maximum coefficient value
        length_scale: Correlation length for GRF
        device: Device
        
    Returns:
        a: Coefficient on full grid (including boundaries), shape (n_samples, n_interior+2)
    """
    n_full = n_interior + 2
    
    if method == 'constant':
        # Constant coefficient a ≡ 1 (other values just scale the problem)
        a = torch.ones(n_samples, n_full, device=device)
        
    elif method == 'grf':
        # GRF-based coefficient, transformed to be positive
        raw = sample_grf_1d(n_full, n_samples, length_scale=length_scale, device=device)
        # Transform to [a_min, a_max] using sigmoid-like mapping
        a = a_min + (a_max - a_min) * torch.sigmoid(raw)
        
    elif method == 'piecewise':
        # Piecewise constant (2-3 pieces)
        a = torch.zeros(n_samples, n_full, device=device)
        n_pieces = torch.randint(2, 4, (n_samples,))
        for i in range(n_samples):
            boundaries = torch.sort(torch.rand(n_pieces[i].item() - 1))[0]
            boundaries = torch.cat([torch.zeros(1), boundaries, torch.ones(1)])
            x_grid = torch.linspace(0, 1, n_full)
            values = torch.rand(n_pieces[i].item()) * (a_max - a_min) + a_min
            for j in range(n_pieces[i].item()):
                mask = (x_grid >= boundaries[j]) & (x_grid < boundaries[j+1])
                a[i, mask] = values[j]
            a[i, -1] = values[-1]  # Handle last point
        a = a.to(device)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return a



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
    """
    x = torch.linspace(0, 1, n_points, device=device)
    dist = torch.abs(x.unsqueeze(0) - x.unsqueeze(1))
    
    if nu == 0.5:
        K = variance * torch.exp(-dist / length_scale)
    elif nu == 1.5:
        sqrt3 = np.sqrt(3.0)
        d_scaled = sqrt3 * dist / length_scale
        K = variance * (1 + d_scaled) * torch.exp(-d_scaled)
    elif nu == 2.5:
        sqrt5 = np.sqrt(5.0)
        d_scaled = sqrt5 * dist / length_scale
        K = variance * (1 + d_scaled + (d_scaled**2)/3.0) * torch.exp(-d_scaled)
    else:
        # Fallback to RBF for large nu
        K = variance * torch.exp(-0.5 * (dist/length_scale)**2)
            
    K = K + 1e-6 * torch.eye(n_points, device=device)
    
    try:
        L = torch.linalg.cholesky(K)
    except RuntimeError:
        K = K + 1e-4 * torch.eye(n_points, device=device)
        L = torch.linalg.cholesky(K)
        
    z = torch.randn(n_samples, n_points, device=device)
    samples = torch.matmul(z, L.t())
    
    return samples


def sample_forcing_f(
    n_interior: int,
    n_samples: int = 1,
    method: str = 'grf',
    amplitude: float = 1.0,
    length_scale: float = 0.3,
    nu: float = 2.5,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Sample forcing term f(x).
    Methods: 'grf', 'sinusoidal', 'polynomial', 'matern', 'mixed'
    """
    if method == 'mixed':
        f_list = []
        for _ in range(n_samples):
            u = np.random.rand()
            # 50% Matern, 40% GRF, 10% Sinusoidal
            if u < 0.5:
                m_i = 'matern'
            elif u < 0.9:
                m_i = 'grf'
            else:
                m_i = 'sinusoidal'
            
            # Random parameters
            # Amplitude log-uniform variation around base amplitude
            amp = float(amplitude * 10**np.random.uniform(-0.5, 0.5))
            ls = np.random.uniform(0.05, 0.4)
            nu_i = 1.5 if np.random.rand() < 0.5 else 2.5
            
            f_i = sample_forcing_f(
                n_interior, 1, 
                method=m_i, 
                amplitude=amp, 
                length_scale=ls, 
                nu=nu_i, 
                device=device
            )
            f_list.append(f_i)
        
        f = torch.cat(f_list, dim=0)
        
    elif method == 'grf':
        raw = sample_grf_1d(n_interior, n_samples, length_scale=length_scale, device=device)
        max_val = raw.abs().max(dim=1, keepdim=True)[0] + 1e-8
        f = amplitude * (raw / max_val)
        
    elif method == 'matern':
        raw = sample_matern_gp_1d(
            n_interior, n_samples,
            length_scale=length_scale,
            nu=nu,
            device=device
        )
        # Normalize
        max_val = raw.abs().max(dim=1, keepdim=True)[0] + 1e-8
        f = amplitude * (raw / max_val)

    elif method == 'sinusoidal':
        x = torch.linspace(0, 1, n_interior, device=device)
        freq = torch.rand(n_samples, 1, device=device) * 4 + 1
        phase = torch.rand(n_samples, 1, device=device) * 2 * np.pi
        f = amplitude * torch.sin(2 * np.pi * freq * x.unsqueeze(0) + phase)
        
    elif method == 'polynomial':
        x = torch.linspace(0, 1, n_interior, device=device)
        coeffs = torch.randn(n_samples, 3, device=device)
        poly = coeffs[:, 0:1] + coeffs[:, 1:2] * x + coeffs[:, 2:3] * x**2
        envelope = x * (1 - x) * 4
        f = amplitude * envelope.unsqueeze(0) * poly
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return f


class PoissonDataset(Dataset):
    """
    Dataset for 1D Poisson equation training.
    
    Generates pairs (a, f) on the fly or from a pre-generated buffer.
    """
    
    def __init__(
        self,
        n_interior: int,
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
            n_interior: Number of interior grid points
            n_samples: Number of samples in dataset
            a_method: Method for sampling coefficient
            f_method: Method for sampling forcing
            pregenerate: If True, generate all samples upfront
        """
        self.n_interior = n_interior
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
                n_interior, n_samples, device='cpu', **self.a_params
            )
            self.f_data = sample_forcing_f(
                n_interior, n_samples, device='cpu', **self.f_params
            )
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        if self.pregenerate:
            return self.a_data[idx], self.f_data[idx]
        else:
            a = sample_coefficient_a(
                self.n_interior, 1, device='cpu', **self.a_params
            ).squeeze(0)
            f = sample_forcing_f(
                self.n_interior, 1, device='cpu', **self.f_params
            ).squeeze(0)
            return a, f


def create_dataloader(
    n_interior: int,
    n_samples: int,
    batch_size: int,
    **dataset_kwargs
) -> DataLoader:
    """
    Create a DataLoader for training.
    """
    dataset = PoissonDataset(n_interior, n_samples, **dataset_kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    torch.manual_seed(42)
    
    n_interior = 50
    
    # Sample coefficients
    a_grf = sample_coefficient_a(n_interior, 5, method='grf')
    a_const = sample_coefficient_a(n_interior, 5, method='constant')
    
    # Sample forcings
    f_grf = sample_forcing_f(n_interior, 5, method='grf')
    f_sin = sample_forcing_f(n_interior, 5, method='sinusoidal')
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    x_full = torch.linspace(0, 1, n_interior + 2)
    x_int = torch.linspace(0, 1, n_interior + 2)[1:-1]
    
    axes[0, 0].set_title("Coefficient a(x) - GRF")
    for i in range(5):
        axes[0, 0].plot(x_full, a_grf[i])
    
    axes[0, 1].set_title("Coefficient a(x) - Constant")
    for i in range(5):
        axes[0, 1].plot(x_full, a_const[i])
    
    axes[1, 0].set_title("Forcing f(x) - GRF")
    for i in range(5):
        axes[1, 0].plot(x_int, f_grf[i])
    
    axes[1, 1].set_title("Forcing f(x) - Sinusoidal")
    for i in range(5):
        axes[1, 1].plot(x_int, f_sin[i])
    
    plt.tight_layout()
    plt.savefig("data_samples.png", dpi=150)
    print("Saved plot to data_samples.png")
    
    # Test dataset
    dataset = PoissonDataset(n_interior, 100)
    loader = DataLoader(dataset, batch_size=16)
    a_batch, f_batch = next(iter(loader))
    print(f"Batch shapes: a={a_batch.shape}, f={f_batch.shape}")
