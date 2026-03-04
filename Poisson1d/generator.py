"""
MLP-based generator network for the variational PDE solver.

The generator T_theta(a, f, xi) -> u maps:
    - Coefficient function a (on full grid)
    - Forcing term f (on interior grid)  
    - Gaussian noise xi
to the interior solution u.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
    MLP-based generator for producing PDE solutions.
    
    Architecture: Concatenate (a, f, xi) -> MLP -> u_interior
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
        
        # Input dimension: a (n+2) + f (n) + xi (noise_dim)
        input_dim = (n_grid + 2) + n_grid + noise_dim
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
        a: torch.Tensor,
        f: torch.Tensor,
        xi: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Generate interior solution values.
        
        Args:
            a: Coefficient on full grid, shape (batch, n+2)
            f: Forcing on interior grid, shape (batch, n)
            xi: Noise vector, shape (batch, noise_dim). If None, samples fresh noise.
            
        Returns:
            u: Interior solution, shape (batch, n)
        """
        batch_size = a.shape[0]
        device = a.device
        
        # Sample noise if not provided
        if xi is None:
            xi = torch.randn(batch_size, self.noise_dim, device=device)
        
        # Concatenate inputs
        x = torch.cat([a, f, xi], dim=-1)
        
        # Forward through MLP
        u = self.mlp(x)
        
        return u
    
    def sample(
        self,
        a: torch.Tensor,
        f: torch.Tensor,
        n_samples: int = 1
    ) -> torch.Tensor:
        """
        Generate multiple solution samples for the same input.
        
        Args:
            a: Coefficient, shape (batch, n+2) or (n+2,)
            f: Forcing, shape (batch, n) or (n,)
            n_samples: Number of samples per input
            
        Returns:
            u: Solutions, shape (batch, n_samples, n) or (n_samples, n)
        """
        squeeze_batch = False
        if a.dim() == 1:
            a = a.unsqueeze(0)
            f = f.unsqueeze(0)
            squeeze_batch = True
        
        batch_size = a.shape[0]
        device = a.device
        
        # Expand inputs for multiple samples
        a_exp = a.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, a.shape[-1])
        f_exp = f.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, f.shape[-1])
        
        # Sample noise
        xi = torch.randn(batch_size * n_samples, self.noise_dim, device=device)
        
        # Generate
        u = self.forward(a_exp, f_exp, xi)
        
        # Reshape
        u = u.reshape(batch_size, n_samples, -1)
        
        if squeeze_batch:
            u = u.squeeze(0)
        
        return u


class ConditionalGenerator(nn.Module):
    """
    Generator with separate encoders for coefficient and forcing.
    More expressive architecture for complex PDEs.
    """
    
    def __init__(
        self,
        n_grid: int,
        noise_dim: int = 32,
        encoder_dims: list = None,
        decoder_dims: list = None
    ):
        super().__init__()
        
        self.n_grid = n_grid
        self.noise_dim = noise_dim
        
        if encoder_dims is None:
            encoder_dims = [128, 128]
        if decoder_dims is None:
            decoder_dims = [256, 256, 256]
        
        latent_dim = encoder_dims[-1]
        
        # Encoder for coefficient a
        a_layers = [nn.Linear(n_grid + 2, encoder_dims[0]), nn.GELU()]
        for i in range(len(encoder_dims) - 1):
            a_layers.extend([nn.Linear(encoder_dims[i], encoder_dims[i+1]), nn.GELU()])
        self.a_encoder = nn.Sequential(*a_layers)
        
        # Encoder for forcing f
        f_layers = [nn.Linear(n_grid, encoder_dims[0]), nn.GELU()]
        for i in range(len(encoder_dims) - 1):
            f_layers.extend([nn.Linear(encoder_dims[i], encoder_dims[i+1]), nn.GELU()])
        self.f_encoder = nn.Sequential(*f_layers)
        
        # Decoder: takes encoded a, encoded f, and noise
        decoder_input_dim = 2 * latent_dim + noise_dim
        dec_layers = [nn.Linear(decoder_input_dim, decoder_dims[0]), nn.GELU()]
        for i in range(len(decoder_dims) - 1):
            dec_layers.extend([nn.Linear(decoder_dims[i], decoder_dims[i+1]), nn.GELU()])
        dec_layers.append(nn.Linear(decoder_dims[-1], n_grid))
        self.decoder = nn.Sequential(*dec_layers)
        
        # Initialize final layer
        nn.init.zeros_(self.decoder[-1].bias)
        nn.init.normal_(self.decoder[-1].weight, std=0.01)
    
    def forward(self, a, f, xi=None):
        batch_size = a.shape[0]
        device = a.device
        
        if xi is None:
            xi = torch.randn(batch_size, self.noise_dim, device=device)
        
        # Encode
        a_enc = self.a_encoder(a)
        f_enc = self.f_encoder(f)
        
        # Decode
        z = torch.cat([a_enc, f_enc, xi], dim=-1)
        u = self.decoder(z)
        
        return u


if __name__ == "__main__":
    torch.manual_seed(42)
    
    n_grid = 20
    batch_size = 8
    
    # Create generator
    gen = Generator(n_grid=n_grid, noise_dim=16)
    print(f"Generator parameters: {sum(p.numel() for p in gen.parameters()):,}")
    
    # Test forward pass
    a = torch.ones(batch_size, n_grid + 2)
    f = torch.sin(torch.linspace(0, 3.14, n_grid)).unsqueeze(0).expand(batch_size, -1)
    
    u = gen(a, f)
    print(f"Output shape: {u.shape}")
    
    # Test sampling multiple solutions
    u_samples = gen.sample(a[0], f[0], n_samples=5)
    print(f"Samples shape: {u_samples.shape}")
    
    # Test conditional generator
    cond_gen = ConditionalGenerator(n_grid=n_grid)
    print(f"Conditional generator parameters: {sum(p.numel() for p in cond_gen.parameters()):,}")
    u_cond = cond_gen(a, f)
    print(f"Conditional output shape: {u_cond.shape}")
