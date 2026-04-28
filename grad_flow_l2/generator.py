"""
Model components for L2 gradient-flow heat-equation learning.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_interior_forcing(f: torch.Tensor, n_x: int) -> torch.Tensor:
    if f.shape[-1] == n_x:
        return f
    if f.shape[-1] == n_x + 2:
        return f[..., 1:-1]
    raise ValueError(f"forcing width must be {n_x} or {n_x+2}, got {f.shape[-1]}")


class ResidualConvBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.act(y)
        y = self.conv2(y)
        return x + y


class EnergyHead1D(nn.Module):
    """
    Local-density energy head E(u; f).

    Inputs:
        u: (batch, n_x)
        f: (batch, n_x) or (batch, n_x+2)

    Output:
        scalar energy per sample: (batch,)
    """

    def __init__(
        self,
        n_x: int,
        h: float,
        hidden_channels: int = 64,
        n_layers: int = 4,
        use_ux_feature: bool = True,
    ):
        super().__init__()
        self.n_x = n_x
        self.h = h
        self.use_ux_feature = use_ux_feature

        in_channels = 3 if use_ux_feature else 2

        layers = [
            nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2),
            nn.GELU(),
        ]
        for _ in range(max(1, n_layers - 1)):
            layers.extend(
                [
                    nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
                    nn.GELU(),
                ]
            )

        self.backbone = nn.Sequential(*layers)
        self.density_head = nn.Conv1d(hidden_channels, 1, kernel_size=1)

    def _estimate_ux(self, u: torch.Tensor) -> torch.Tensor:
        # Central-difference with zero Dirichlet boundary padding.
        u_full = F.pad(u, (1, 1), mode="constant", value=0.0)
        ux = (u_full[:, 2:] - u_full[:, :-2]) / (2.0 * self.h)
        return ux

    def forward(self, u: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if u.dim() == 1:
            u = u.unsqueeze(0)
            f = f.unsqueeze(0)
            squeeze = True
        f = _to_interior_forcing(f, self.n_x)

        if self.use_ux_feature:
            ux = self._estimate_ux(u)
            feat = torch.stack([u, f, ux], dim=1)  # (batch, c, n_x)
        else:
            feat = torch.stack([u, f], dim=1)

        hidden = self.backbone(feat)
        density = F.softplus(self.density_head(hidden)).squeeze(1)  # (batch, n_x)
        energy = self.h * torch.sum(density, dim=-1)

        if squeeze:
            return energy.squeeze(0)
        return energy


class ProximalMap1D(nn.Module):
    """
    Deterministic residual Conv1d one-step map:
        u_{k+1} = u_k + Delta(u_k, f, dt).
    """

    def __init__(
        self,
        n_x: int,
        hidden_channels: int = 64,
        n_blocks: int = 6,
        kernel_size: int = 5,
        use_dt_channel: bool = False,
        default_dt: Optional[float] = None,
    ):
        super().__init__()
        self.n_x = n_x
        self.use_dt_channel = use_dt_channel
        self.default_dt = default_dt

        in_channels = 3 if use_dt_channel else 2

        self.in_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.blocks = nn.ModuleList(
            [ResidualConvBlock1D(hidden_channels, kernel_size=kernel_size) for _ in range(n_blocks)]
        )
        self.out_proj = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        self.act = nn.GELU()

        nn.init.zeros_(self.out_proj.bias)
        nn.init.normal_(self.out_proj.weight, std=0.01)

    def _dt_channel(self, batch_size: int, n_x: int, dt, device, dtype):
        if dt is None:
            if self.default_dt is None:
                raise ValueError("dt must be provided when use_dt_channel=True and default_dt is None")
            dt_value = self.default_dt
        else:
            dt_value = dt

        if torch.is_tensor(dt_value):
            if dt_value.dim() == 0:
                dt_ch = dt_value.to(device=device, dtype=dtype).expand(batch_size, n_x)
            elif dt_value.dim() == 1 and dt_value.shape[0] == batch_size:
                dt_ch = dt_value.to(device=device, dtype=dtype).unsqueeze(-1).expand(batch_size, n_x)
            else:
                raise ValueError("dt tensor must be scalar or shape (batch,)")
        else:
            dt_ch = torch.full((batch_size, n_x), float(dt_value), device=device, dtype=dtype)

        return dt_ch

    def forward(self, u_k: torch.Tensor, f: torch.Tensor, dt=None) -> torch.Tensor:
        squeeze = False
        if u_k.dim() == 1:
            u_k = u_k.unsqueeze(0)
            f = f.unsqueeze(0)
            squeeze = True

        batch_size, n_x = u_k.shape
        f = _to_interior_forcing(f, n_x)

        if self.use_dt_channel:
            dt_ch = self._dt_channel(batch_size, n_x, dt, u_k.device, u_k.dtype)
            x = torch.stack([u_k, f, dt_ch], dim=1)
        else:
            x = torch.stack([u_k, f], dim=1)

        h = self.act(self.in_proj(x))
        for block in self.blocks:
            h = self.act(block(h))
        delta = self.out_proj(h).squeeze(1)

        u_next = u_k + delta
        if squeeze:
            return u_next.squeeze(0)
        return u_next


class GradientFlowModel(nn.Module):
    """
    Coupled model with proximal map and energy head.
    """

    def __init__(self, prox_map: ProximalMap1D, energy_head: EnergyHead1D):
        super().__init__()
        self.prox_map = prox_map
        self.energy_head = energy_head

    def predict_step(self, u_k: torch.Tensor, f: torch.Tensor, dt=None) -> torch.Tensor:
        return self.prox_map(u_k, f, dt=dt)

    def energy(self, u: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        return self.energy_head(u, f)

    def forward(self, u_k: torch.Tensor, f: torch.Tensor, dt=None) -> torch.Tensor:
        return self.predict_step(u_k, f, dt=dt)


class LatentStateEncoder1D(nn.Module):
    """
    Encode physical state to a latent field z.

    Inputs:
        u: (batch, n_x) or (n_x,)
        f: ignored (kept in signature for compatibility with existing call sites)
    Output:
        z: (batch, latent_channels, n_x) or (latent_channels, n_x)
    """

    def __init__(
        self,
        n_x: int,
        latent_channels: int = 16,
        hidden_channels: int = 64,
        n_blocks: int = 4,
        kernel_size: int = 5,
        use_ux_feature: bool = True,
    ):
        super().__init__()
        self.n_x = n_x
        self.latent_channels = latent_channels
        self.use_ux_feature = use_ux_feature

        in_channels = 2 if use_ux_feature else 1
        padding = kernel_size // 2

        self.in_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding)
        self.blocks = nn.ModuleList(
            [ResidualConvBlock1D(hidden_channels, kernel_size=kernel_size) for _ in range(n_blocks)]
        )
        self.out_proj = nn.Conv1d(hidden_channels, latent_channels, kernel_size=1)
        self.act = nn.GELU()

    def _estimate_ux(self, u: torch.Tensor) -> torch.Tensor:
        u_full = F.pad(u, (1, 1), mode="constant", value=0.0)
        return (u_full[:, 2:] - u_full[:, :-2]) * 0.5

    def forward(self, u: torch.Tensor, f: torch.Tensor | None = None) -> torch.Tensor:
        squeeze = False
        if u.dim() == 1:
            u = u.unsqueeze(0)
            squeeze = True

        if self.use_ux_feature:
            ux = self._estimate_ux(u)
            x = torch.stack([u, ux], dim=1)
        else:
            x = u.unsqueeze(1)

        h = self.act(self.in_proj(x))
        for block in self.blocks:
            h = self.act(block(h))
        z = self.out_proj(h)

        if squeeze:
            return z.squeeze(0)
        return z


class LatentStateDecoder1D(nn.Module):
    """
    Decode latent field z back to physical state u.
    """

    def __init__(
        self,
        n_x: int,
        latent_channels: int = 16,
        hidden_channels: int = 64,
        n_blocks: int = 4,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.n_x = n_x
        self.latent_channels = latent_channels

        in_channels = latent_channels
        padding = kernel_size // 2

        self.in_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding)
        self.blocks = nn.ModuleList(
            [ResidualConvBlock1D(hidden_channels, kernel_size=kernel_size) for _ in range(n_blocks)]
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

        x = z

        h = self.act(self.in_proj(x))
        for block in self.blocks:
            h = self.act(block(h))
        u = self.out_proj(h).squeeze(1)

        if squeeze:
            return u.squeeze(0)
        return u


class LatentGradientStep1D(nn.Module):
    """
    One-step latent evolution:
        z_{k+1} = z_k + Delta(z_k, f, dt).
    """

    def __init__(
        self,
        n_x: int,
        latent_channels: int = 16,
        hidden_channels: int = 64,
        n_blocks: int = 6,
        kernel_size: int = 5,
        use_forcing_channel: bool = True,
        use_dt_channel: bool = False,
        default_dt: Optional[float] = None,
    ):
        super().__init__()
        self.n_x = n_x
        self.latent_channels = latent_channels
        self.use_forcing_channel = use_forcing_channel
        self.use_dt_channel = use_dt_channel
        self.default_dt = default_dt

        in_channels = latent_channels + (1 if use_forcing_channel else 0) + (1 if use_dt_channel else 0)
        padding = kernel_size // 2

        self.in_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding)
        self.blocks = nn.ModuleList(
            [ResidualConvBlock1D(hidden_channels, kernel_size=kernel_size) for _ in range(n_blocks)]
        )
        self.out_proj = nn.Conv1d(hidden_channels, latent_channels, kernel_size=1)
        self.act = nn.GELU()

        nn.init.zeros_(self.out_proj.bias)
        nn.init.normal_(self.out_proj.weight, std=0.01)

    def _dt_channel(self, batch_size: int, n_x: int, dt, device, dtype) -> torch.Tensor:
        if dt is None:
            if self.default_dt is None:
                raise ValueError("dt must be provided when use_dt_channel=True and default_dt is None")
            dt_value = self.default_dt
        else:
            dt_value = dt

        if torch.is_tensor(dt_value):
            if dt_value.dim() == 0:
                dt_ch = dt_value.to(device=device, dtype=dtype).expand(batch_size, n_x)
            elif dt_value.dim() == 1 and dt_value.shape[0] == batch_size:
                dt_ch = dt_value.to(device=device, dtype=dtype).unsqueeze(-1).expand(batch_size, n_x)
            else:
                raise ValueError("dt tensor must be scalar or shape (batch,)")
        else:
            dt_ch = torch.full((batch_size, n_x), float(dt_value), device=device, dtype=dtype)

        return dt_ch.unsqueeze(1)

    def forward(self, z_k: torch.Tensor, f: torch.Tensor, dt=None) -> torch.Tensor:
        squeeze = False
        if z_k.dim() == 2:
            z_k = z_k.unsqueeze(0)
            f = f.unsqueeze(0)
            squeeze = True

        if z_k.dim() != 3:
            raise ValueError("z_k must have shape (batch, latent_channels, n_x) or (latent_channels, n_x)")
        if z_k.shape[1] != self.latent_channels:
            raise ValueError(f"Expected latent_channels={self.latent_channels}, got {z_k.shape[1]}")

        batch_size = z_k.shape[0]
        f_in = _to_interior_forcing(f, self.n_x)

        feat = [z_k]
        if self.use_forcing_channel:
            feat.append(f_in.unsqueeze(1))
        if self.use_dt_channel:
            feat.append(self._dt_channel(batch_size, self.n_x, dt, z_k.device, z_k.dtype))
        x = torch.cat(feat, dim=1)

        h = self.act(self.in_proj(x))
        for block in self.blocks:
            h = self.act(block(h))
        delta = self.out_proj(h)
        z_next = z_k + delta

        if squeeze:
            return z_next.squeeze(0)
        return z_next


class LatentEnergyHead1D(nn.Module):
    """
    Local-density latent energy E(z; f).

    Inputs:
        z: (batch, latent_channels, n_x) or (latent_channels, n_x)
        f: (batch, n_x) or (batch, n_x+2) or (n_x,) or (n_x+2,)
    Output:
        scalar energy per sample: (batch,) or scalar
    """

    def __init__(
        self,
        n_x: int,
        h: float,
        latent_channels: int = 16,
        hidden_channels: int = 64,
        n_layers: int = 4,
        use_forcing_channel: bool = True,
        use_zx_norm_feature: bool = True,
    ):
        super().__init__()
        self.n_x = n_x
        self.h = h
        self.latent_channels = latent_channels
        self.use_forcing_channel = use_forcing_channel
        self.use_zx_norm_feature = use_zx_norm_feature

        in_channels = latent_channels
        if use_forcing_channel:
            in_channels += 1
        if use_zx_norm_feature:
            in_channels += 1

        layers = [
            nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2),
            nn.GELU(),
        ]
        for _ in range(max(1, n_layers - 1)):
            layers.extend(
                [
                    nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
                    nn.GELU(),
                ]
            )

        self.backbone = nn.Sequential(*layers)
        self.density_head = nn.Conv1d(hidden_channels, 1, kernel_size=1)

    def _estimate_zx_norm(self, z: torch.Tensor) -> torch.Tensor:
        z_full = F.pad(z, (1, 1), mode="constant", value=0.0)
        z_x = (z_full[:, :, 2:] - z_full[:, :, :-2]) * 0.5
        return torch.sqrt(torch.sum(z_x * z_x, dim=1, keepdim=True) + 1e-12)

    def forward(self, z: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if z.dim() == 2:
            z = z.unsqueeze(0)
            f = f.unsqueeze(0)
            squeeze = True

        if z.dim() != 3:
            raise ValueError("z must have shape (batch, latent_channels, n_x) or (latent_channels, n_x)")
        if z.shape[1] != self.latent_channels:
            raise ValueError(f"Expected latent_channels={self.latent_channels}, got {z.shape[1]}")

        feat = [z]
        if self.use_forcing_channel:
            f_in = _to_interior_forcing(f, self.n_x)
            feat.append(f_in.unsqueeze(1))
        if self.use_zx_norm_feature:
            feat.append(self._estimate_zx_norm(z))
        x = torch.cat(feat, dim=1)

        hidden = self.backbone(x)
        density = F.softplus(self.density_head(hidden)).squeeze(1)
        energy = self.h * torch.sum(density, dim=-1)

        if squeeze:
            return energy.squeeze(0)
        return energy


class HiddenGradientFlowModel1D(nn.Module):
    """
    Latent-space model:
        z_k = Enc(u_k)
        z_{k+1} = Phi(z_k, f, dt)
        u_{k+1} = Dec(z_{k+1})

    Energy regularization can be applied in latent space via latent_energy().
    """

    def __init__(
        self,
        encoder: LatentStateEncoder1D,
        latent_step: LatentGradientStep1D,
        decoder: LatentStateDecoder1D,
        latent_energy_head: LatentEnergyHead1D,
    ):
        super().__init__()
        self.encoder = encoder
        self.latent_step = latent_step
        self.decoder = decoder
        self.latent_energy_head = latent_energy_head

    def encode(self, u: torch.Tensor, f: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(u)

    def decode(self, z: torch.Tensor, f: torch.Tensor | None = None) -> torch.Tensor:
        return self.decoder(z)

    def predict_latent_step(self, z_k: torch.Tensor, f: torch.Tensor, dt=None) -> torch.Tensor:
        return self.latent_step(z_k, f, dt=dt)

    def predict_step(self, u_k: torch.Tensor, f: torch.Tensor, dt=None, return_latent: bool = False):
        z_k = self.encode(u_k)
        z_next = self.predict_latent_step(z_k, f, dt=dt)
        u_next = self.decode(z_next)
        if return_latent:
            return u_next, z_k, z_next
        return u_next

    def latent_energy(self, z: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        return self.latent_energy_head(z, f)

    def energy(self, u: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        z = self.encode(u)
        return self.latent_energy(z, f)

    def forward(self, u_k: torch.Tensor, f: torch.Tensor, dt=None) -> torch.Tensor:
        return self.predict_step(u_k, f, dt=dt, return_latent=False)
