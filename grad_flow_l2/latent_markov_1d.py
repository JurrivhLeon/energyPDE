"""
Shared deterministic 1D latent Markov model components.

This is the non-probabilistic counterpart to ``latent_flow_VAE.py``:
  z_k = E(u_k)
  z_{k+1} = z_k + G_FNO(z_k, f, dt)
  u_{k+1} = D(z_{k+1})
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .latent_flow_VAE import FNOLatentTransition1D, StateDecoder1D


def _padding_mode_1d(boundary_condition: str) -> str:
    bc = boundary_condition.strip().lower()
    if bc in {"periodic", "circular", "torus"}:
        return "circular"
    if bc in {"dirichlet", "zero", "zeros", "constant"}:
        return "zeros"
    raise ValueError("boundary_condition must be one of {'periodic','dirichlet'}")


def _pad_1d(x: torch.Tensor, padding: int, padding_mode: str) -> torch.Tensor:
    if padding <= 0:
        return x
    if padding_mode == "zeros":
        return F.pad(x, (padding, padding), mode="constant", value=0.0)
    return F.pad(x, (padding, padding), mode=padding_mode)


class ResidualConvBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, padding_mode: str = "zeros"):
        super().__init__()
        self.padding = kernel_size // 2
        self.padding_mode = padding_mode
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.conv1(_pad_1d(x, self.padding, self.padding_mode)))
        y = self.conv2(_pad_1d(y, self.padding, self.padding_mode))
        return x + y


class DeterministicStateEncoder1D(nn.Module):
    def __init__(
        self,
        n_x: int,
        latent_channels: int = 16,
        hidden_channels: int = 64,
        n_blocks: int = 4,
        kernel_size: int = 5,
        use_grad_features: bool = True,
        boundary_condition: str = "periodic",
    ):
        super().__init__()
        self.n_x = int(n_x)
        self.latent_channels = int(latent_channels)
        self.use_grad_features = bool(use_grad_features)
        self.padding_mode = _padding_mode_1d(boundary_condition)
        self.padding = kernel_size // 2

        in_channels = 2 if self.use_grad_features else 1
        self.in_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size)
        self.blocks = nn.ModuleList(
            [
                ResidualConvBlock1D(hidden_channels, kernel_size=kernel_size, padding_mode=self.padding_mode)
                for _ in range(n_blocks)
            ]
        )
        self.out_proj = nn.Conv1d(hidden_channels, latent_channels, kernel_size=1)
        self.act = nn.GELU()

    def _grad_feat(self, u: torch.Tensor) -> torch.Tensor:
        u_pad = _pad_1d(u.unsqueeze(1), 1, self.padding_mode).squeeze(1)
        return 0.5 * (u_pad[:, 2:] - u_pad[:, :-2])

    def forward(self, u: torch.Tensor, f: torch.Tensor | None = None) -> torch.Tensor:
        squeeze = False
        if u.dim() == 1:
            u = u.unsqueeze(0)
            squeeze = True
        if u.dim() != 2:
            raise ValueError(f"u must have shape (n_x,) or (batch,n_x), got {tuple(u.shape)}")
        if u.shape[1] != self.n_x:
            raise ValueError(f"u width must be {self.n_x}, got {u.shape[1]}")

        if self.use_grad_features:
            x = torch.stack([u, self._grad_feat(u)], dim=1)
        else:
            x = u.unsqueeze(1)

        h = self.act(self.in_proj(_pad_1d(x, self.padding, self.padding_mode)))
        for block in self.blocks:
            h = self.act(block(h))
        z = self.out_proj(h)
        if squeeze:
            return z.squeeze(0)
        return z


class LatentMarkovFNO1D(nn.Module):
    def __init__(
        self,
        encoder: DeterministicStateEncoder1D,
        decoder: StateDecoder1D,
        transition: FNOLatentTransition1D,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.transition = transition

    def encode(self, u: torch.Tensor, f: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(u, f=f)

    def decode(self, z: torch.Tensor, f: torch.Tensor | None = None) -> torch.Tensor:
        return self.decoder(z, f=f)

    def predict_latent_step(self, z_k: torch.Tensor, f: torch.Tensor, dt=None) -> torch.Tensor:
        return self.transition(z_k, f, dt=dt)

    def predict_step(self, u_k: torch.Tensor, f: torch.Tensor, dt=None, return_latent: bool = False):
        z_k = self.encode(u_k)
        z_next = self.predict_latent_step(z_k, f, dt=dt)
        u_next = self.decode(z_next)
        if return_latent:
            return u_next, z_k, z_next
        return u_next

    def forward(self, u_k: torch.Tensor, f: torch.Tensor, dt=None) -> torch.Tensor:
        return self.predict_step(u_k, f, dt=dt)


def build_latent_markov_fno_1d(
    n_x: int,
    dt: float,
    latent_channels: int = 16,
    hidden_channels: int = 64,
    enc_blocks: int = 4,
    dec_blocks: int = 4,
    fno_width: int | None = None,
    fno_layers: int = 6,
    fno_modes: int = 16,
    use_forcing_channel: bool = True,
    use_dt_channel: bool = False,
    use_grid_features: bool = True,
    use_grad_features: bool = True,
    boundary_condition: str = "periodic",
) -> LatentMarkovFNO1D:
    width = int(hidden_channels if fno_width is None else fno_width)
    encoder = DeterministicStateEncoder1D(
        n_x=n_x,
        latent_channels=latent_channels,
        hidden_channels=hidden_channels,
        n_blocks=enc_blocks,
        use_grad_features=use_grad_features,
        boundary_condition=boundary_condition,
    )
    decoder = StateDecoder1D(
        n_x=n_x,
        latent_channels=latent_channels,
        hidden_channels=hidden_channels,
        n_blocks=dec_blocks,
        boundary_condition=boundary_condition,
    )
    transition = FNOLatentTransition1D(
        n_x=n_x,
        latent_channels=latent_channels,
        width=width,
        n_layers=fno_layers,
        modes=fno_modes,
        use_forcing_channel=use_forcing_channel,
        use_dt_channel=use_dt_channel,
        use_grid_features=use_grid_features,
        default_dt=dt,
    )
    return LatentMarkovFNO1D(encoder=encoder, decoder=decoder, transition=transition)


__all__ = [
    "ResidualConvBlock1D",
    "DeterministicStateEncoder1D",
    "LatentMarkovFNO1D",
    "build_latent_markov_fno_1d",
]
