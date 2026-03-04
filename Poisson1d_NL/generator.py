"""
MLP-based generator network for the nonlinear Poisson (Ginzburg-Landau) solver.

The generator T_theta(s, xi) -> u maps:
    - Source term s (on interior grid)
    - Gaussian noise xi
to the interior solution u.

Unlike the linear Poisson case, there is no coefficient function a(x) here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
    MLP-based generator for producing PDE solutions.
    
    Architecture: Concatenate (s, xi) -> MLP -> u_interior
    """
    
    def __init__(
        self,
        n_grid: int,
        noise_dim: int = 32,
        hidden_dims: list = None,
        activation: str = 'gelu'
    ):
        """
        Args:
            n_grid: Number of interior grid points
            noise_dim: Dimension of the noise input
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('relu', 'gelu', 'silu')
        """
        super().__init__()
        
        self.n_grid = n_grid
        self.noise_dim = noise_dim
        
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]
        
        # Input dimension: s (n) + xi (noise_dim)
        input_dim = n_grid + noise_dim
        output_dim = n_grid
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'silu':
                layers.append(nn.SiLU())
            prev_dim = hdim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize final layer with small weights
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
    
    def forward(
        self,
        s: torch.Tensor,
        xi: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Generate interior solution values.
        
        Args:
            s: Source term on interior grid, shape (batch, n)
            xi: Noise vector, shape (batch, noise_dim). If None, samples fresh noise.
            
        Returns:
            u: Interior solution, shape (batch, n)
        """
        batch_size = s.shape[0]
        device = s.device
        
        # Sample noise if not provided
        if xi is None:
            xi = torch.randn(batch_size, self.noise_dim, device=device)
        
        # Concatenate inputs
        x = torch.cat([s, xi], dim=-1)
        
        # Forward through MLP
        u = self.mlp(x)
        
        return u
    
    def sample(
        self,
        s: torch.Tensor,
        n_samples: int = 1
    ) -> torch.Tensor:
        """
        Generate multiple solution samples for the same input.
        
        Args:
            s: Source term, shape (batch, n) or (n,)
            n_samples: Number of samples per input
            
        Returns:
            u: Solutions, shape (batch, n_samples, n) or (n_samples, n)
        """
        squeeze_batch = False
        if s.dim() == 1:
            s = s.unsqueeze(0)
            squeeze_batch = True
        
        batch_size = s.shape[0]
        device = s.device
        
        # Expand inputs for multiple samples
        s_exp = s.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, s.shape[-1])
        
        # Sample noise
        xi = torch.randn(batch_size * n_samples, self.noise_dim, device=device)
        
        # Generate
        u = self.forward(s_exp, xi)
        
        # Reshape
        u = u.reshape(batch_size, n_samples, -1)
        
        if squeeze_batch:
            u = u.squeeze(0)
        
        return u


if __name__ == "__main__":
    torch.manual_seed(42)
    
    n_grid = 20
    batch_size = 8
    
    # Create generator
    gen = Generator(n_grid=n_grid, noise_dim=16)
    print(f"Generator parameters: {sum(p.numel() for p in gen.parameters()):,}")
    
    # Test forward pass
    s = torch.sin(torch.linspace(0, 3.14, n_grid)).unsqueeze(0).expand(batch_size, -1)
    
    u = gen(s)
    print(f"Output shape: {u.shape}")
    
    # Test sampling multiple solutions
    u_samples = gen.sample(s[0], n_samples=5)
    print(f"Samples shape: {u_samples.shape}")
    
    # Test batch sampling
    u_batch_samples = gen.sample(s, n_samples=3)
    print(f"Batch samples shape: {u_batch_samples.shape}")
