"""
MLP-based generator network for the 1D Burgers equation solver.

The generator T_theta(u^n, xi) -> u^{n+1} maps:
    - Current solution u^n (on periodic grid)
    - Gaussian noise xi
to the next-step solution u^{n+1}.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
    MLP-based generator for producing next-step Burgers solutions.
    
    Architecture: Concatenate (u_curr, xi) -> MLP -> u_next
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
            n_grid: Number of grid points
            noise_dim: Dimension of the noise input
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('relu', 'gelu', 'silu')
        """
        super().__init__()
        
        self.n_grid = n_grid
        self.noise_dim = noise_dim
        
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]
        
        # Input dimension: u_curr (n) + xi (noise_dim)
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
        u_curr: torch.Tensor,
        xi: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Generate next-step solution.
        
        Args:
            u_curr: Current solution, shape (batch, n)
            xi: Noise vector, shape (batch, noise_dim). If None, samples fresh noise.
            
        Returns:
            u_next: Next-step solution, shape (batch, n)
        """
        batch_size = u_curr.shape[0]
        device = u_curr.device
        
        # Sample noise if not provided
        if xi is None:
            xi = torch.randn(batch_size, self.noise_dim, device=device)
        
        # Concatenate inputs
        x = torch.cat([u_curr, xi], dim=-1)
        
        # Forward through MLP
        u_next = self.mlp(x)
        
        return u_next
    
    def sample(
        self,
        u_curr: torch.Tensor,
        n_samples: int = 1
    ) -> torch.Tensor:
        """
        Generate multiple solution samples for the same input.
        
        Args:
            u_curr: Current solution, shape (batch, n) or (n,)
            n_samples: Number of samples per input
            
        Returns:
            u: Solutions, shape (batch, n_samples, n) or (n_samples, n)
        """
        squeeze_batch = False
        if u_curr.dim() == 1:
            u_curr = u_curr.unsqueeze(0)
            squeeze_batch = True
        
        batch_size = u_curr.shape[0]
        device = u_curr.device
        
        # Expand inputs for multiple samples
        u_exp = u_curr.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, u_curr.shape[-1])
        
        # Sample noise
        xi = torch.randn(batch_size * n_samples, self.noise_dim, device=device)
        
        # Generate
        u_next = self.forward(u_exp, xi)
        
        # Reshape
        u_next = u_next.reshape(batch_size, n_samples, -1)
        
        if squeeze_batch:
            u_next = u_next.squeeze(0)
        
        return u_next


if __name__ == "__main__":
    torch.manual_seed(42)
    
    n_grid = 64
    batch_size = 8
    
    # Create generator
    gen = Generator(n_grid=n_grid, noise_dim=16)
    print(f"Generator parameters: {sum(p.numel() for p in gen.parameters()):,}")
    
    # Test forward pass
    u_curr = torch.sin(torch.linspace(0, 2 * 3.14159, n_grid + 1)[:-1]).unsqueeze(0).expand(batch_size, -1)
    
    u_next = gen(u_curr)
    print(f"Output shape: {u_next.shape}")
    
    # Test sampling multiple solutions
    u_samples = gen.sample(u_curr[0], n_samples=5)
    print(f"Samples shape: {u_samples.shape}")
    
    # Test batch sampling
    u_batch_samples = gen.sample(u_curr, n_samples=3)
    print(f"Batch samples shape: {u_batch_samples.shape}")
