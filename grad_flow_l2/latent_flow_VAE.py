"""
Variational latent-flow model components for Burgers and periodic 2D Navier-Stokes.

This module implements a VAE-style latent Markov model:
  - q(z_n | u_n): diagonal-Gaussian spatial posterior
  - p(z_{n+1} | z_n, f): FNO mean transition with scalar-amplitude
    low-pass stochastic perturbation
  - p(u_n | z_n): deterministic decoder
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .latent_markov import (
    FNOBlock2D,
    ResidualConvBlock2D,
    StateDecoder2D,
    _fpad_mode_from_boundary_condition,
    _grid_coordinates_2d,
    _normalize_boundary_condition,
    _padding_mode_from_boundary_condition,
    _to_interior_forcing_2d,
)


def _to_interior_forcing_1d(f: torch.Tensor, n_x: int) -> torch.Tensor:
    if f.dim() == 1:
        if f.shape[0] == n_x:
            return f.unsqueeze(0)
        if f.shape[0] == n_x + 2:
            return f[1:-1].unsqueeze(0)
        raise ValueError(f"forcing width must be {n_x} or {n_x + 2}, got {f.shape[0]}")
    if f.dim() == 2:
        if f.shape[1] == n_x:
            return f
        if f.shape[1] == n_x + 2:
            return f[:, 1:-1]
        raise ValueError(f"forcing width must be {n_x} or {n_x + 2}, got {f.shape[1]}")
    raise ValueError("forcing must have shape (n_x,), (n_x+2,), (batch,n_x), or (batch,n_x+2)")


def _grid_coordinates_1d(n_x: int, device, dtype) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, int(n_x) + 2, device=device, dtype=dtype)[1:-1]


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


class VariationalStateEncoder1D(nn.Module):
    """
    Resolution-preserving variational encoder for 1D spatial latent fields.
    """

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

        in_channels = 2 if self.use_grad_features else 1
        self.padding = kernel_size // 2
        self.in_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size)
        self.blocks = nn.ModuleList(
            [
                ResidualConvBlock1D(hidden_channels, kernel_size=kernel_size, padding_mode=self.padding_mode)
                for _ in range(n_blocks)
            ]
        )
        self.mu_head = nn.Conv1d(hidden_channels, latent_channels, kernel_size=1)
        self.logvar_head = nn.Conv1d(hidden_channels, latent_channels, kernel_size=1)
        self.act = nn.GELU()

    def _grad_feat(self, u: torch.Tensor) -> torch.Tensor:
        u_pad = _pad_1d(u.unsqueeze(1), 1, self.padding_mode).squeeze(1)
        return 0.5 * (u_pad[:, 2:] - u_pad[:, :-2])

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        if squeeze:
            return mu.squeeze(0), logvar.squeeze(0)
        return mu, logvar


class StateDecoder1D(nn.Module):
    """
    Decode a 1D latent field back to a physical state with configurable padding.
    """

    def __init__(
        self,
        n_x: int,
        latent_channels: int = 16,
        hidden_channels: int = 64,
        n_blocks: int = 4,
        kernel_size: int = 5,
        boundary_condition: str = "periodic",
    ):
        super().__init__()
        self.n_x = int(n_x)
        self.latent_channels = int(latent_channels)
        self.padding_mode = _padding_mode_1d(boundary_condition)
        self.padding = kernel_size // 2

        self.in_proj = nn.Conv1d(latent_channels, hidden_channels, kernel_size=kernel_size)
        self.blocks = nn.ModuleList(
            [
                ResidualConvBlock1D(hidden_channels, kernel_size=kernel_size, padding_mode=self.padding_mode)
                for _ in range(n_blocks)
            ]
        )
        self.out_proj = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, z: torch.Tensor, f: torch.Tensor | None = None) -> torch.Tensor:
        squeeze = False
        if z.dim() == 2:
            z = z.unsqueeze(0)
            squeeze = True
        if z.dim() != 3:
            raise ValueError("z must have shape (batch, latent_channels, n_x) or (latent_channels, n_x)")
        if z.shape[1] != self.latent_channels or z.shape[2] != self.n_x:
            raise ValueError(f"z must have shape (*,{self.latent_channels},{self.n_x}), got {tuple(z.shape)}")

        h = self.act(self.in_proj(_pad_1d(z, self.padding, self.padding_mode)))
        for block in self.blocks:
            h = self.act(block(h))
        u = self.out_proj(h).squeeze(1)
        if squeeze:
            return u.squeeze(0)
        return u


class SpectralConv1d(nn.Module):
    """
    1D Fourier layer retaining a fixed number of low positive-frequency modes.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        if modes <= 0:
            raise ValueError("modes must be positive")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = int(modes)

        scale = 1.0 / max(1, in_channels * out_channels)
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.modes, 2))

    @staticmethod
    def _compl_mul1d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bim,iom->bom", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"x must have shape (batch,channels,n_x), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {x.shape[1]}")

        batch_size, _, n_x = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        max_modes = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(batch_size, self.out_channels, x_ft.shape[-1], dtype=x_ft.dtype, device=x.device)
        weight = torch.view_as_complex(self.weight[:, :, :max_modes, :].contiguous())
        out_ft[:, :, :max_modes] = self._compl_mul1d(x_ft[:, :, :max_modes], weight)
        return torch.fft.irfft(out_ft, n=n_x, dim=-1)


class FNOBlock1D(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes=modes)
        self.local = nn.Conv1d(width, width, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.spectral(x) + self.local(x))


class FNOLatentTransition1D(nn.Module):
    """
    FNO latent transition mean:
      mu_p(z_n, f) = z_n + G_theta(z_n, f, dt, x).
    """

    def __init__(
        self,
        n_x: int,
        latent_channels: int = 16,
        width: int = 64,
        n_layers: int = 6,
        modes: int = 16,
        use_forcing_channel: bool = True,
        use_dt_channel: bool = False,
        use_grid_features: bool = True,
        default_dt: Optional[float] = None,
    ):
        super().__init__()
        self.n_x = int(n_x)
        self.latent_channels = int(latent_channels)
        self.use_forcing_channel = bool(use_forcing_channel)
        self.use_dt_channel = bool(use_dt_channel)
        self.use_grid_features = bool(use_grid_features)
        self.default_dt = default_dt

        in_channels = (
            self.latent_channels
            + (1 if self.use_forcing_channel else 0)
            + (1 if self.use_dt_channel else 0)
            + (1 if self.use_grid_features else 0)
        )
        self.in_proj = nn.Conv1d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock1D(width, modes=modes) for _ in range(n_layers)])
        self.out_proj = nn.Conv1d(width, self.latent_channels, kernel_size=1)

        nn.init.zeros_(self.out_proj.bias)
        nn.init.normal_(self.out_proj.weight, std=0.01)

    def _dt_channel(self, batch_size: int, dt, device, dtype) -> torch.Tensor:
        if dt is None:
            if self.default_dt is None:
                raise ValueError("dt must be provided when use_dt_channel=True and default_dt is None")
            dt_value = self.default_dt
        else:
            dt_value = dt

        if torch.is_tensor(dt_value):
            if dt_value.dim() == 0:
                dt_ch = dt_value.to(device=device, dtype=dtype).expand(batch_size, self.n_x)
            elif dt_value.dim() == 1 and dt_value.shape[0] == batch_size:
                dt_ch = dt_value.to(device=device, dtype=dtype).view(batch_size, 1).expand(batch_size, self.n_x)
            else:
                raise ValueError("dt tensor must be scalar or shape (batch,)")
        else:
            dt_ch = torch.full((batch_size, self.n_x), float(dt_value), device=device, dtype=dtype)
        return dt_ch.unsqueeze(1)

    def _grid_features(self, batch_size: int, device, dtype) -> torch.Tensor:
        x = _grid_coordinates_1d(n_x=self.n_x, device=device, dtype=dtype)
        return x.view(1, 1, self.n_x).expand(batch_size, -1, -1)

    def forward(self, z: torch.Tensor, f: torch.Tensor, dt=None) -> torch.Tensor:
        squeeze = False
        if z.dim() == 2:
            z = z.unsqueeze(0)
            f = f.unsqueeze(0)
            squeeze = True
        if z.dim() != 3:
            raise ValueError(f"z must have shape (latent_channels,n_x) or (batch,latent_channels,n_x), got {tuple(z.shape)}")
        if z.shape[1] != self.latent_channels:
            raise ValueError(f"Expected latent_channels={self.latent_channels}, got {z.shape[1]}")
        if z.shape[2] != self.n_x:
            raise ValueError(f"z width must be {self.n_x}, got {z.shape[2]}")

        batch_size = z.shape[0]
        feat = [z]
        if self.use_forcing_channel:
            f_int = _to_interior_forcing_1d(f, n_x=self.n_x)
            if f_int.shape[0] == 1 and batch_size > 1:
                f_int = f_int.expand(batch_size, -1)
            if f_int.shape[0] != batch_size:
                raise ValueError("forcing batch size must match z batch size or be 1")
            feat.append(f_int.unsqueeze(1))
        if self.use_dt_channel:
            feat.append(self._dt_channel(batch_size, dt, z.device, z.dtype))
        if self.use_grid_features:
            feat.append(self._grid_features(batch_size, z.device, z.dtype))

        h = self.in_proj(torch.cat(feat, dim=1))
        for block in self.blocks:
            h = block(h)
        mu = z + self.out_proj(h)
        if squeeze:
            return mu.squeeze(0)
        return mu


class TransitionAmplitudeHead1D(nn.Module):
    """
    Predict a nonnegative scalar transition noise amplitude per sample.
    """

    def __init__(
        self,
        n_x: int,
        latent_channels: int = 16,
        hidden_channels: int = 32,
        use_forcing_channel: bool = True,
        boundary_condition: str = "periodic",
    ):
        super().__init__()
        self.n_x = int(n_x)
        self.latent_channels = int(latent_channels)
        self.use_forcing_channel = bool(use_forcing_channel)
        self.padding_mode = _padding_mode_1d(boundary_condition)

        in_channels = self.latent_channels + (1 if self.use_forcing_channel else 0)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2, padding_mode=self.padding_mode),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2, padding_mode=self.padding_mode),
            nn.GELU(),
        )
        self.out = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        self.softplus = nn.Softplus()

    def forward(self, z: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if z.dim() == 2:
            z = z.unsqueeze(0)
            f = f.unsqueeze(0)
            squeeze = True
        if z.dim() != 3:
            raise ValueError(f"z must have shape (latent_channels,n_x) or (batch,latent_channels,n_x), got {tuple(z.shape)}")
        if z.shape[1] != self.latent_channels or z.shape[2] != self.n_x:
            raise ValueError(f"z must have shape (*,{self.latent_channels},{self.n_x}), got {tuple(z.shape)}")

        batch_size = z.shape[0]
        feat = [z]
        if self.use_forcing_channel:
            f_int = _to_interior_forcing_1d(f, n_x=self.n_x)
            if f_int.shape[0] == 1 and batch_size > 1:
                f_int = f_int.expand(batch_size, -1)
            if f_int.shape[0] != batch_size:
                raise ValueError("forcing batch size must match z batch size or be 1")
            feat.append(f_int.unsqueeze(1))

        h = self.net(torch.cat(feat, dim=1))
        pooled = h.mean(dim=-1, keepdim=True)
        alpha = self.softplus(self.out(pooled)).view(batch_size)
        if squeeze:
            return alpha.squeeze(0)
        return alpha


class LatentVAE1D(nn.Module):
    """
    1D latent VAE with stochastic training transitions and deterministic rollout.
    """

    def __init__(
        self,
        encoder: VariationalStateEncoder1D,
        decoder: nn.Module,
        transition: FNOLatentTransition1D,
        amplitude_head: TransitionAmplitudeHead1D,
        noise_corr_length: float = 1.0,
        noise_decay_s: float = 2.0,
    ):
        super().__init__()
        if noise_corr_length <= 0.0:
            raise ValueError("noise_corr_length must be > 0")
        if noise_decay_s <= 0.0:
            raise ValueError("noise_decay_s must be > 0")
        self.encoder = encoder
        self.decoder = decoder
        self.transition = transition
        self.amplitude_head = amplitude_head
        self.noise_corr_length = float(noise_corr_length)
        self.noise_decay_s = float(noise_decay_s)

    @property
    def latent_channels(self) -> int:
        return self.encoder.latent_channels

    @property
    def n_x(self) -> int:
        return self.encoder.n_x

    def encode_stats(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(u)

    def sample_posterior(self, mu_q: torch.Tensor, logvar_q: torch.Tensor) -> torch.Tensor:
        logvar_q = torch.clamp(logvar_q, min=-8.0, max=2.0)
        std = torch.exp(0.5 * logvar_q)
        return mu_q + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def _spectral_filter(self, device, dtype) -> torch.Tensor:
        k = 2.0 * torch.pi * torch.fft.rfftfreq(self.n_x, d=1.0 / float(self.n_x), device=device).to(dtype=dtype)
        return (1.0 + (self.noise_corr_length ** 2) * k.square()).pow(-0.5 * self.noise_decay_s)

    def _filtered_noise(self, shape: Tuple[int, int, int], device, dtype) -> torch.Tensor:
        xi = torch.randn(shape, device=device, dtype=dtype)
        xi_hat = torch.fft.rfft(xi, dim=-1, norm="ortho")
        filt = self._spectral_filter(device=device, dtype=dtype).view(1, 1, -1)
        return torch.fft.irfft(xi_hat * filt, n=self.n_x, dim=-1, norm="ortho")

    def prior_stats(self, z: torch.Tensor, f: torch.Tensor, dt=None) -> tuple[torch.Tensor, torch.Tensor]:
        mu_p = self.transition(z, f, dt=dt)
        alpha = self.amplitude_head(z, f)
        prior_logvar_scalar = torch.log(alpha.square() + 1e-12)
        return mu_p, prior_logvar_scalar

    def sample_prior(self, z: torch.Tensor, f: torch.Tensor, dt=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_p, prior_logvar_scalar = self.prior_stats(z, f, dt=dt)
        alpha = torch.exp(0.5 * prior_logvar_scalar)
        noise = self._filtered_noise(mu_p.shape, device=mu_p.device, dtype=mu_p.dtype)
        z_next = mu_p + alpha.view(-1, 1, 1) * noise
        return z_next, mu_p, alpha

    def predict_step(self, u_k: torch.Tensor, f: torch.Tensor, dt=None, sample: bool = True, return_stats: bool = False):
        mu_q, logvar_q = self.encode_stats(u_k)
        z_k = self.sample_posterior(mu_q, logvar_q) if sample else mu_q
        if sample:
            z_next, mu_p, alpha = self.sample_prior(z_k, f, dt=dt)
        else:
            mu_p, prior_logvar_scalar = self.prior_stats(z_k, f, dt=dt)
            alpha = torch.exp(0.5 * prior_logvar_scalar)
            z_next = mu_p
        u_next = self.decode(z_next)
        if return_stats:
            return {
                "u_next": u_next,
                "z_k": z_k,
                "z_next": z_next,
                "mu_q": mu_q,
                "logvar_q": logvar_q,
                "mu_p": mu_p,
                "alpha": alpha,
            }
        return u_next

    def rollout_step(self, u_k: torch.Tensor, f: torch.Tensor, dt=None) -> torch.Tensor:
        mu_q, _ = self.encode_stats(u_k)
        mu_p, _ = self.prior_stats(mu_q, f, dt=dt)
        return self.decode(mu_p)

    def forward(self, u_k: torch.Tensor, f: torch.Tensor, dt=None) -> torch.Tensor:
        return self.predict_step(u_k, f, dt=dt, sample=False, return_stats=False)


class VariationalStateEncoder2D(nn.Module):
    """
    Resolution-preserving variational encoder for spatial latent fields.
    """

    def __init__(
        self,
        n_x: int,
        n_y: int,
        latent_channels: int = 16,
        hidden_channels: int = 64,
        n_blocks: int = 4,
        kernel_size: int = 3,
        use_grad_features: bool = True,
        boundary_condition: str = "periodic",
    ):
        super().__init__()
        self.n_x = int(n_x)
        self.n_y = int(n_y)
        self.latent_channels = int(latent_channels)
        self.use_grad_features = bool(use_grad_features)
        self.boundary_condition = _normalize_boundary_condition(boundary_condition)
        self.padding_mode = _padding_mode_from_boundary_condition(self.boundary_condition)

        in_channels = 3 if self.use_grad_features else 1
        padding = kernel_size // 2

        self.in_proj = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=self.padding_mode,
        )
        self.blocks = nn.ModuleList(
            [
                ResidualConvBlock2D(hidden_channels, kernel_size=kernel_size, padding_mode=self.padding_mode)
                for _ in range(n_blocks)
            ]
        )
        self.mu_head = nn.Conv2d(hidden_channels, latent_channels, kernel_size=1)
        self.logvar_head = nn.Conv2d(hidden_channels, latent_channels, kernel_size=1)
        self.act = nn.GELU()

    def _grad_feats(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pad_kwargs = {"mode": _fpad_mode_from_boundary_condition(self.boundary_condition)}
        if pad_kwargs["mode"] == "constant":
            pad_kwargs["value"] = 0.0
        u_pad = torch.nn.functional.pad(u, (1, 1, 1, 1), **pad_kwargs)
        du_dx = 0.5 * (u_pad[:, 2:, 1:-1] - u_pad[:, :-2, 1:-1])
        du_dy = 0.5 * (u_pad[:, 1:-1, 2:] - u_pad[:, 1:-1, :-2])
        return du_dx, du_dy

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        squeeze = False
        if u.dim() == 2:
            u = u.unsqueeze(0)
            squeeze = True
        if u.dim() != 3:
            raise ValueError(f"u must have shape (n_x,n_y) or (batch,n_x,n_y), got {tuple(u.shape)}")
        if u.shape[1:] != (self.n_x, self.n_y):
            raise ValueError(f"u spatial shape must be ({self.n_x},{self.n_y}), got {tuple(u.shape[1:])}")

        if self.use_grad_features:
            du_dx, du_dy = self._grad_feats(u)
            x = torch.stack([u, du_dx, du_dy], dim=1)
        else:
            x = u.unsqueeze(1)

        h = self.act(self.in_proj(x))
        for block in self.blocks:
            h = self.act(block(h))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        if squeeze:
            return mu.squeeze(0), logvar.squeeze(0)
        return mu, logvar


class FNOLatentTransition2D(nn.Module):
    """
    FNO latent transition mean:
      mu_p(z_n, f) = z_n + G_theta(z_n, f, dt).
    """

    def __init__(
        self,
        n_x: int,
        n_y: int,
        latent_channels: int = 16,
        width: int = 64,
        n_layers: int = 6,
        modes_x: int = 16,
        modes_y: int = 16,
        use_forcing_channel: bool = True,
        use_dt_channel: bool = False,
        use_grid_features: bool = True,
        default_dt: Optional[float] = None,
        boundary_condition: str = "periodic",
    ):
        super().__init__()
        self.n_x = int(n_x)
        self.n_y = int(n_y)
        self.latent_channels = int(latent_channels)
        self.use_forcing_channel = bool(use_forcing_channel)
        self.use_dt_channel = bool(use_dt_channel)
        self.use_grid_features = bool(use_grid_features)
        self.default_dt = default_dt
        self.boundary_condition = _normalize_boundary_condition(boundary_condition)

        in_channels = (
            self.latent_channels
            + (1 if self.use_forcing_channel else 0)
            + (1 if self.use_dt_channel else 0)
            + (2 if self.use_grid_features else 0)
        )
        self.in_proj = nn.Conv2d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock2D(width, modes_x=modes_x, modes_y=modes_y) for _ in range(n_layers)])
        self.out_proj = nn.Conv2d(width, self.latent_channels, kernel_size=1)

        nn.init.zeros_(self.out_proj.bias)
        nn.init.normal_(self.out_proj.weight, std=0.01)

    def _dt_channel(self, batch_size: int, dt, device, dtype) -> torch.Tensor:
        if dt is None:
            if self.default_dt is None:
                raise ValueError("dt must be provided when use_dt_channel=True and default_dt is None")
            dt_value = self.default_dt
        else:
            dt_value = dt

        if torch.is_tensor(dt_value):
            if dt_value.dim() == 0:
                dt_ch = dt_value.to(device=device, dtype=dtype).expand(batch_size, self.n_x, self.n_y)
            elif dt_value.dim() == 1 and dt_value.shape[0] == batch_size:
                dt_ch = dt_value.to(device=device, dtype=dtype).view(batch_size, 1, 1).expand(
                    batch_size, self.n_x, self.n_y
                )
            else:
                raise ValueError("dt tensor must be scalar or shape (batch,)")
        else:
            dt_ch = torch.full((batch_size, self.n_x, self.n_y), float(dt_value), device=device, dtype=dtype)
        return dt_ch.unsqueeze(1)

    def _grid_features(self, batch_size: int, device, dtype) -> torch.Tensor:
        gx, gy = _grid_coordinates_2d(
            n_x=self.n_x,
            n_y=self.n_y,
            boundary_condition=self.boundary_condition,
            device=device,
            dtype=dtype,
        )
        return torch.stack([gx, gy], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)

    def forward(self, z: torch.Tensor, f: torch.Tensor, dt=None) -> torch.Tensor:
        squeeze = False
        if z.dim() == 3:
            z = z.unsqueeze(0)
            f = f.unsqueeze(0)
            squeeze = True
        if z.dim() != 4:
            raise ValueError(
                f"z must have shape (latent_channels,n_x,n_y) or (batch,latent_channels,n_x,n_y), got {tuple(z.shape)}"
            )
        if z.shape[1] != self.latent_channels:
            raise ValueError(f"Expected latent_channels={self.latent_channels}, got {z.shape[1]}")
        if z.shape[2:] != (self.n_x, self.n_y):
            raise ValueError(f"z spatial shape must be ({self.n_x},{self.n_y}), got {tuple(z.shape[2:])}")

        batch_size = z.shape[0]
        feat = [z]
        if self.use_forcing_channel:
            f_int = _to_interior_forcing_2d(f, n_x=self.n_x, n_y=self.n_y)
            if f_int.shape[0] == 1 and batch_size > 1:
                f_int = f_int.expand(batch_size, -1, -1)
            if f_int.shape[0] != batch_size:
                raise ValueError("forcing batch size must match z batch size or be 1")
            feat.append(f_int.unsqueeze(1))
        if self.use_dt_channel:
            feat.append(self._dt_channel(batch_size, dt, z.device, z.dtype))
        if self.use_grid_features:
            feat.append(self._grid_features(batch_size, z.device, z.dtype))

        x = torch.cat(feat, dim=1)
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        delta = self.out_proj(h)
        mu = z + delta
        if squeeze:
            return mu.squeeze(0)
        return mu


class TransitionAmplitudeHead2D(nn.Module):
    """
    Predict a nonnegative scalar transition noise amplitude per sample.
    """

    def __init__(
        self,
        n_x: int,
        n_y: int,
        latent_channels: int = 16,
        hidden_channels: int = 32,
        use_forcing_channel: bool = True,
        boundary_condition: str = "periodic",
    ):
        super().__init__()
        self.n_x = int(n_x)
        self.n_y = int(n_y)
        self.latent_channels = int(latent_channels)
        self.use_forcing_channel = bool(use_forcing_channel)
        self.boundary_condition = _normalize_boundary_condition(boundary_condition)
        self.padding_mode = _padding_mode_from_boundary_condition(self.boundary_condition)

        in_channels = self.latent_channels + (1 if self.use_forcing_channel else 0)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, padding_mode=self.padding_mode),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, padding_mode=self.padding_mode),
            nn.GELU(),
        )
        self.out = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.softplus = nn.Softplus()

    def forward(self, z: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if z.dim() == 3:
            z = z.unsqueeze(0)
            f = f.unsqueeze(0)
            squeeze = True
        if z.dim() != 4:
            raise ValueError(
                f"z must have shape (latent_channels,n_x,n_y) or (batch,latent_channels,n_x,n_y), got {tuple(z.shape)}"
            )
        batch_size = z.shape[0]
        feat = [z]
        if self.use_forcing_channel:
            f_int = _to_interior_forcing_2d(f, n_x=self.n_x, n_y=self.n_y)
            if f_int.shape[0] == 1 and batch_size > 1:
                f_int = f_int.expand(batch_size, -1, -1)
            if f_int.shape[0] != batch_size:
                raise ValueError("forcing batch size must match z batch size or be 1")
            feat.append(f_int.unsqueeze(1))

        h = self.net(torch.cat(feat, dim=1))
        pooled = h.mean(dim=(-2, -1), keepdim=True)
        alpha = self.softplus(self.out(pooled)).view(batch_size)
        if squeeze:
            return alpha.squeeze(0)
        return alpha


class PeriodicLatentVAE2D(nn.Module):
    """
    Periodic 2D Navier-Stokes latent VAE with deterministic rollout.
    """

    def __init__(
        self,
        encoder: VariationalStateEncoder2D,
        decoder: StateDecoder2D,
        transition: FNOLatentTransition2D,
        amplitude_head: TransitionAmplitudeHead2D,
        noise_corr_length: float = 1.0,
        noise_decay_s: float = 2.0,
    ):
        super().__init__()
        if noise_corr_length <= 0.0:
            raise ValueError("noise_corr_length must be > 0")
        if noise_decay_s <= 0.0:
            raise ValueError("noise_decay_s must be > 0")

        self.encoder = encoder
        self.decoder = decoder
        self.transition = transition
        self.amplitude_head = amplitude_head
        self.noise_corr_length = float(noise_corr_length)
        self.noise_decay_s = float(noise_decay_s)

    @property
    def latent_channels(self) -> int:
        return self.encoder.latent_channels

    @property
    def n_x(self) -> int:
        return self.encoder.n_x

    @property
    def n_y(self) -> int:
        return self.encoder.n_y

    def encode_stats(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(u)

    def sample_posterior(self, mu_q: torch.Tensor, logvar_q: torch.Tensor) -> torch.Tensor:
        logvar_q = torch.clamp(logvar_q, min=-8.0, max=2.0)
        std = torch.exp(0.5 * logvar_q)
        eps = torch.randn_like(std)
        return mu_q + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def _spectral_filter(self, device, dtype) -> torch.Tensor:
        kx = 2.0 * torch.pi * torch.fft.fftfreq(self.n_x, d=1.0 / float(self.n_x), device=device).to(dtype=dtype)
        ky = 2.0 * torch.pi * torch.fft.rfftfreq(self.n_y, d=1.0 / float(self.n_y), device=device).to(dtype=dtype)
        kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing="ij")
        radius_sq = kx_grid.square() + ky_grid.square()
        return (1.0 + (self.noise_corr_length ** 2) * radius_sq).pow(-0.5 * self.noise_decay_s)

    def _filtered_noise(self, shape: Tuple[int, int, int, int], device, dtype) -> torch.Tensor:
        xi = torch.randn(shape, device=device, dtype=dtype)
        xi_hat = torch.fft.rfft2(xi, dim=(-2, -1), norm="ortho")
        filt = self._spectral_filter(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        return torch.fft.irfft2(xi_hat * filt, s=(self.n_x, self.n_y), dim=(-2, -1), norm="ortho")

    def prior_stats(self, z: torch.Tensor, f: torch.Tensor, dt=None) -> tuple[torch.Tensor, torch.Tensor]:
        mu_p = self.transition(z, f, dt=dt)
        alpha = self.amplitude_head(z, f)
        prior_logvar_scalar = torch.log(alpha.square() + 1e-12)
        return mu_p, prior_logvar_scalar

    def sample_prior(self, z: torch.Tensor, f: torch.Tensor, dt=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_p, prior_logvar_scalar = self.prior_stats(z, f, dt=dt)
        alpha = torch.exp(0.5 * prior_logvar_scalar)
        noise = self._filtered_noise(mu_p.shape, device=mu_p.device, dtype=mu_p.dtype)
        z_next = mu_p + alpha.view(-1, 1, 1, 1) * noise
        return z_next, mu_p, alpha

    def predict_step(self, u_k: torch.Tensor, f: torch.Tensor, dt=None, sample: bool = True, return_stats: bool = False):
        mu_q, logvar_q = self.encode_stats(u_k)
        z_k = self.sample_posterior(mu_q, logvar_q) if sample else mu_q
        if sample:
            z_next, mu_p, alpha = self.sample_prior(z_k, f, dt=dt)
        else:
            mu_p, prior_logvar_scalar = self.prior_stats(z_k, f, dt=dt)
            alpha = torch.exp(0.5 * prior_logvar_scalar)
            z_next = mu_p
        u_next = self.decode(z_next)
        if return_stats:
            return {
                "u_next": u_next,
                "z_k": z_k,
                "z_next": z_next,
                "mu_q": mu_q,
                "logvar_q": logvar_q,
                "mu_p": mu_p,
                "alpha": alpha,
            }
        return u_next

    def rollout_step(self, u_k: torch.Tensor, f: torch.Tensor, dt=None) -> torch.Tensor:
        mu_q, _ = self.encode_stats(u_k)
        mu_p, _ = self.prior_stats(mu_q, f, dt=dt)
        return self.decode(mu_p)

    def forward(self, u_k: torch.Tensor, f: torch.Tensor, dt=None) -> torch.Tensor:
        return self.predict_step(u_k, f, dt=dt, sample=False, return_stats=False)


__all__ = [
    "VariationalStateEncoder1D",
    "StateDecoder1D",
    "SpectralConv1d",
    "FNOBlock1D",
    "FNOLatentTransition1D",
    "TransitionAmplitudeHead1D",
    "LatentVAE1D",
    "VariationalStateEncoder2D",
    "FNOLatentTransition2D",
    "TransitionAmplitudeHead2D",
    "PeriodicLatentVAE2D",
]
