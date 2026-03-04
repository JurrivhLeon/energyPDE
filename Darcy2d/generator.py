"""
CNN-based generator network for 2D Darcy flow.

The generator T_theta(a, f, xi) -> u maps:
    - Permeability field a (on full grid)
    - Forcing term f (on interior grid)
    - Gaussian noise xi
to the interior solution u.

Uses a U-Net-style encoder-decoder architecture for efficiency on 2D grids.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution block with batch norm and activation."""
    
    def __init__(self, in_channels, out_channels, activation='gelu'):
        super().__init__()
        
        act_fn = {
            'relu': nn.ReLU(inplace=True),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(inplace=True)
        }[activation]
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            act_fn,
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            act_fn
        )
    
    def forward(self, x):
        return self.conv(x)


class Generator2d(nn.Module):
    """
    U-Net-style generator for 2D PDE solutions.
    
    Encodes input (a, f, noise) and decodes to solution u.
    Uses skip connections for better gradient flow.
    """
    
    def __init__(
        self,
        N: int,
        noise_channels: int = 8,
        base_channels: int = 32,
        depth: int = 3,
        activation: str = 'gelu'
    ):
        """
        Args:
            N: Interior grid size (N×N)
            noise_channels: Number of noise channels
            base_channels: Base channel count (doubles at each level)
            depth: Number of encoder/decoder levels
            activation: Activation function
        """
        super().__init__()
        
        self.N = N
        self.noise_channels = noise_channels
        self.depth = depth
        
        # Input: a (1 ch, N+2×N+2) + f (1 ch, N×N) + noise (noise_channels, N×N)
        # We'll pad f and noise to (N+2)×(N+2) for concatenation
        in_channels = 1 + 1 + noise_channels  # a, f, noise
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.encoders.append(ConvBlock(ch, out_ch, activation))
            self.pools.append(nn.MaxPool2d(2))
            ch = out_ch
        
        # Bottleneck
        bottleneck_ch = base_channels * (2 ** depth)
        self.bottleneck = ConvBlock(ch, bottleneck_ch, activation)
        
        # Decoder
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        ch = bottleneck_ch
        for i in range(depth - 1, -1, -1):
            skip_ch = base_channels * (2 ** i)
            self.upsamples.append(nn.ConvTranspose2d(ch, ch // 2, 2, stride=2))
            self.decoders.append(ConvBlock(ch // 2 + skip_ch, skip_ch, activation))
            ch = skip_ch
        
        # Output layer
        self.output = nn.Conv2d(base_channels, 1, 1)
        
        # Initialize output with small weights
        nn.init.zeros_(self.output.bias)
        nn.init.normal_(self.output.weight, std=0.01)
    
    def forward(
        self,
        a: torch.Tensor,
        f: torch.Tensor,
        xi: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Generate interior solution values.
        
        Args:
            a: Permeability on full grid, shape (batch, N+2, N+2)
            f: Forcing on interior grid, shape (batch, N, N)
            xi: Noise tensor, shape (batch, noise_channels, N, N). If None, samples fresh noise.
            
        Returns:
            u: Interior solution, shape (batch, N, N)
        """
        batch_size = a.shape[0]
        device = a.device
        N = self.N
        
        # Sample noise if not provided
        if xi is None:
            xi = torch.randn(batch_size, self.noise_channels, N, N, device=device)
        
        # Pad f and noise to full grid size (N+2 × N+2)
        f_padded = F.pad(f.unsqueeze(1), (1, 1, 1, 1), mode='constant', value=0)  # (batch, 1, N+2, N+2)
        xi_padded = F.pad(xi, (1, 1, 1, 1), mode='constant', value=0)  # (batch, noise_ch, N+2, N+2)
        
        # Concatenate inputs
        x = torch.cat([
            a.unsqueeze(1),  # (batch, 1, N+2, N+2)
            f_padded,        # (batch, 1, N+2, N+2)
            xi_padded        # (batch, noise_ch, N+2, N+2)
        ], dim=1)
        
        # Encoder path
        skip_connections = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skip_connections)):
            x = upsample(x)
            # Handle potential size mismatch due to pooling
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        
        # Output
        u_full = self.output(x).squeeze(1)  # (batch, N+2, N+2)
        
        # Extract interior (enforce Dirichlet BC)
        u = u_full[:, 1:-1, 1:-1]  # (batch, N, N)
        
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
            a: Permeability, shape (batch, N+2, N+2) or (N+2, N+2)
            f: Forcing, shape (batch, N, N) or (N, N)
            n_samples: Number of samples per input
            
        Returns:
            u: Solutions, shape (batch, n_samples, N, N) or (n_samples, N, N)
        """
        squeeze_batch = False
        if a.dim() == 2:
            a = a.unsqueeze(0)
            f = f.unsqueeze(0)
            squeeze_batch = True
        
        batch_size = a.shape[0]
        device = a.device
        
        # Expand inputs for multiple samples
        a_exp = a.unsqueeze(1).expand(-1, n_samples, -1, -1).reshape(-1, a.shape[-2], a.shape[-1])
        f_exp = f.unsqueeze(1).expand(-1, n_samples, -1, -1).reshape(-1, f.shape[-2], f.shape[-1])
        
        # Sample noise
        xi = torch.randn(batch_size * n_samples, self.noise_channels, self.N, self.N, device=device)
        
        # Generate
        u = self.forward(a_exp, f_exp, xi)
        
        # Reshape
        u = u.reshape(batch_size, n_samples, self.N, self.N)
        
        if squeeze_batch:
            u = u.squeeze(0)
        
        return u


class MLPGenerator2d(nn.Module):
    """
    Simple MLP generator for small 2D grids (fallback for small N).
    Flattens the 2D grid and uses fully-connected layers.
    """
    
    def __init__(
        self,
        N: int,
        noise_dim: int = 32,
        hidden_dims: list = None,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        self.N = N
        self.noise_dim = noise_dim
        
        if hidden_dims is None:
            hidden_dims = [512, 512, 512]
        
        # Input: a (flattened, (N+2)²) + f (flattened, N²) + xi (noise_dim)
        input_dim = (N + 2) ** 2 + N ** 2 + noise_dim
        output_dim = N ** 2
        
        # Build MLP
        layers = []
        prev_dim = input_dim
        
        act_fn = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'silu': nn.SiLU
        }[activation]
        
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(act_fn())
            prev_dim = hdim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize final layer
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
    
    def forward(self, a, f, xi=None):
        batch_size = a.shape[0]
        device = a.device
        
        if xi is None:
            xi = torch.randn(batch_size, self.noise_dim, device=device)
        
        # Flatten inputs
        a_flat = a.reshape(batch_size, -1)
        f_flat = f.reshape(batch_size, -1)
        
        # Concatenate
        x = torch.cat([a_flat, f_flat, xi], dim=-1)
        
        # Forward
        u_flat = self.mlp(x)
        
        # Reshape to 2D
        u = u_flat.reshape(batch_size, self.N, self.N)
        
        return u


class SpectralConv2d(nn.Module):
    """
    2D Spectral Convolution layer (core FNO building block).
    
    Multiplies truncated Fourier coefficients by a learnable complex weight
    tensor R of shape (in_channels, out_channels, modes1, modes2).
    
    Uses rfft2/irfft2 since input is real-valued.
    """
    
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to keep (dim -2)
        self.modes2 = modes2  # Number of Fourier modes to keep (dim -1)
        
        scale = 1.0 / (in_channels * out_channels)
        # Two sets of weights: one for positive freq in dim -2, one for negative
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
    
    @staticmethod
    def _compl_mul2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Complex multiplication via einsum: (batch, in, x, y) * (in, out, x, y) -> (batch, out, x, y)."""
        return torch.einsum("bixy,ioxy->boxy", a, b)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, H, W)  — real-valued
        Returns:
            (batch, out_channels, H, W)  — real-valued
        """
        batch_size = x.shape[0]
        H, W = x.shape[-2], x.shape[-1]
        
        # rfft2 along last two dims → (batch, in_ch, H, W//2+1) complex
        x_ft = torch.fft.rfft2(x)
        
        # Allocate output spectrum
        out_ft = torch.zeros(
            batch_size, self.out_channels, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        # Positive frequencies along dim -2 (modes 0 .. modes1-1)
        out_ft[:, :, :self.modes1, :self.modes2] = self._compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        # Negative frequencies along dim -2 (modes -modes1 .. -1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self._compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )
        
        # irfft2 back to spatial domain
        return torch.fft.irfft2(out_ft, s=(H, W))


class FNOGenerator2d(nn.Module):
    """
    Fourier Neural Operator generator for 2D PDE solutions.
    
    Architecture:
        1. Concatenate inputs (a, f, xi) as channels on the (N+2)×(N+2) full grid.
        2. Lift to `width` channels via pointwise linear.
        3. Apply `n_layers` Fourier layers (spectral conv + 1×1 conv + BN + GELU).
        4. Project to 1 channel via two pointwise linears.
        5. Extract interior N×N to enforce Dirichlet BCs.
    
    Same forward(a, f, xi) / sample(a, f, n_samples) API as Generator2d.
    """
    
    def __init__(
        self,
        N: int,
        noise_channels: int = 8,
        width: int = 32,
        modes: int = 12,
        n_layers: int = 4,
        activation: str = 'gelu'
    ):
        """
        Args:
            N: Interior grid size (N×N)
            noise_channels: Number of noise channels in xi
            width: Hidden channel width in Fourier layers
            modes: Number of Fourier modes to keep per spatial dimension
            n_layers: Number of Fourier layers
            activation: Activation function
        """
        super().__init__()
        
        self.N = N
        self.noise_channels = noise_channels
        self.width = width
        self.n_layers = n_layers
        
        # Input channels: a (1) + f (1) + noise (noise_channels)
        in_channels = 1 + 1 + noise_channels
        
        # Lifting layer: pointwise (1×1 conv)
        self.lift = nn.Conv2d(in_channels, width, 1)
        
        # Fourier layers
        self.spectral_convs = nn.ModuleList()
        self.pointwise_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(n_layers):
            self.spectral_convs.append(SpectralConv2d(width, width, modes, modes))
            self.pointwise_convs.append(nn.Conv2d(width, width, 1))
            self.norms.append(nn.BatchNorm2d(width))
        
        # Projection: two pointwise layers (width -> width -> 1)
        self.proj1 = nn.Conv2d(width, width, 1)
        self.proj2 = nn.Conv2d(width, 1, 1)
        
        # Activation
        self.act = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU()
        }[activation]
        
        # Initialize output with small weights
        nn.init.zeros_(self.proj2.bias)
        nn.init.normal_(self.proj2.weight, std=0.01)
    
    def forward(
        self,
        a: torch.Tensor,
        f: torch.Tensor,
        xi: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Generate interior solution values.
        
        Args:
            a: Permeability on full grid, shape (batch, N+2, N+2)
            f: Forcing on interior grid, shape (batch, N, N)
            xi: Noise tensor, shape (batch, noise_channels, N, N). If None, samples fresh noise.
            
        Returns:
            u: Interior solution, shape (batch, N, N)
        """
        batch_size = a.shape[0]
        device = a.device
        N = self.N
        
        # Sample noise if not provided
        if xi is None:
            xi = torch.randn(batch_size, self.noise_channels, N, N, device=device)
        
        # Pad f and noise to full grid size (N+2 × N+2)
        f_padded = F.pad(f.unsqueeze(1), (1, 1, 1, 1), mode='constant', value=0)
        xi_padded = F.pad(xi, (1, 1, 1, 1), mode='constant', value=0)
        
        # Concatenate inputs: (batch, in_channels, N+2, N+2)
        x = torch.cat([
            a.unsqueeze(1),  # (batch, 1, N+2, N+2)
            f_padded,        # (batch, 1, N+2, N+2)
            xi_padded        # (batch, noise_ch, N+2, N+2)
        ], dim=1)
        
        # Lift to hidden width
        x = self.lift(x)  # (batch, width, N+2, N+2)
        
        # Fourier layers
        for spec_conv, pw_conv, norm in zip(self.spectral_convs, self.pointwise_convs, self.norms):
            x_spec = spec_conv(x)
            x_pw = pw_conv(x)
            x = norm(x_spec + x_pw)
            x = self.act(x)
        
        # Project to output
        x = self.act(self.proj1(x))
        x = self.proj2(x)  # (batch, 1, N+2, N+2)
        
        # Extract interior (enforce Dirichlet BC)
        u = x.squeeze(1)[:, 1:-1, 1:-1]  # (batch, N, N)
        
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
            a: Permeability, shape (batch, N+2, N+2) or (N+2, N+2)
            f: Forcing, shape (batch, N, N) or (N, N)
            n_samples: Number of samples per input
            
        Returns:
            u: Solutions, shape (batch, n_samples, N, N) or (n_samples, N, N)
        """
        squeeze_batch = False
        if a.dim() == 2:
            a = a.unsqueeze(0)
            f = f.unsqueeze(0)
            squeeze_batch = True
        
        batch_size = a.shape[0]
        device = a.device
        
        # Expand inputs for multiple samples
        a_exp = a.unsqueeze(1).expand(-1, n_samples, -1, -1).reshape(-1, a.shape[-2], a.shape[-1])
        f_exp = f.unsqueeze(1).expand(-1, n_samples, -1, -1).reshape(-1, f.shape[-2], f.shape[-1])
        
        # Sample noise
        xi = torch.randn(batch_size * n_samples, self.noise_channels, self.N, self.N, device=device)
        
        # Generate
        u = self.forward(a_exp, f_exp, xi)
        
        # Reshape
        u = u.reshape(batch_size, n_samples, self.N, self.N)
        
        if squeeze_batch:
            u = u.squeeze(0)
        
        return u


if __name__ == "__main__":
    torch.manual_seed(42)
    
    N = 32
    batch_size = 4
    
    # --- Test Generator2d (U-Net) ---
    gen = Generator2d(N=N, noise_channels=8, base_channels=32, depth=3)
    print(f"Generator2d parameters: {sum(p.numel() for p in gen.parameters()):,}")
    
    a = torch.ones(batch_size, N + 2, N + 2)
    f = torch.randn(batch_size, N, N)
    
    u = gen(a, f)
    print(f"Output shape: {u.shape}")
    
    u_samples = gen.sample(a[0], f[0], n_samples=5)
    print(f"Samples shape: {u_samples.shape}")
    
    xi1 = torch.randn(1, 8, N, N)
    xi2 = torch.randn(1, 8, N, N)
    u1 = gen(a[:1], f[:1], xi1)
    u2 = gen(a[:1], f[:1], xi2)
    print(f"Different noise outputs differ: {(u1 - u2).abs().mean().item():.4f}")
    
    # --- Test MLPGenerator2d ---
    mlp_gen = MLPGenerator2d(N=N, noise_dim=32)
    print(f"\nMLPGenerator2d parameters: {sum(p.numel() for p in mlp_gen.parameters()):,}")
    u_mlp = mlp_gen(a, f)
    print(f"MLP output shape: {u_mlp.shape}")
    
    # --- Test FNOGenerator2d ---
    fno_gen = FNOGenerator2d(N=N, noise_channels=8, width=32, modes=12, n_layers=4)
    print(f"\nFNOGenerator2d parameters: {sum(p.numel() for p in fno_gen.parameters()):,}")
    
    u_fno = fno_gen(a, f)
    print(f"FNO output shape: {u_fno.shape}")
    
    u_fno_samples = fno_gen.sample(a[0], f[0], n_samples=5)
    print(f"FNO samples shape: {u_fno_samples.shape}")
    
    # Verify different noise gives different FNO outputs
    xi1 = torch.randn(1, 8, N, N)
    xi2 = torch.randn(1, 8, N, N)
    u1_fno = fno_gen(a[:1], f[:1], xi1)
    u2_fno = fno_gen(a[:1], f[:1], xi2)
    print(f"FNO different noise diff: {(u1_fno - u2_fno).abs().mean().item():.4f}")
    
    # Verify gradient flows
    u_fno_grad = fno_gen(a, f)
    loss = u_fno_grad.sum()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in fno_gen.parameters() if p.grad is not None)
    print(f"FNO gradient norm (sanity): {grad_norm:.4f}")
