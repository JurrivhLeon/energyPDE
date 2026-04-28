"""
Model components for 2D Navier-Stokes hidden-space gradient-flow learning.

Key constraint:
  - Encoder/decoder consume only the state u (not forcing f).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_interior_forcing_2d(f: torch.Tensor, n_x: int, n_y: int) -> torch.Tensor:
    if f.dim() == 2:
        if f.shape == (n_x, n_y):
            return f.unsqueeze(0)
        if f.shape == (n_x + 2, n_y + 2):
            return f[1:-1, 1:-1].unsqueeze(0)
        raise ValueError(f"forcing shape must be ({n_x},{n_y}) or ({n_x+2},{n_y+2}), got {tuple(f.shape)}")
    if f.dim() == 3:
        if f.shape[1:] == (n_x, n_y):
            return f
        if f.shape[1:] == (n_x + 2, n_y + 2):
            return f[:, 1:-1, 1:-1]
        raise ValueError(
            f"forcing shape must be (batch,{n_x},{n_y}) or (batch,{n_x+2},{n_y+2}), got {tuple(f.shape)}"
        )
    raise ValueError("forcing must have shape (n_x,n_y), (n_x+2,n_y+2), (batch,n_x,n_y), or (batch,n_x+2,n_y+2)")


def _normalize_boundary_condition(boundary_condition: str) -> str:
    bc = boundary_condition.strip().lower()
    if bc in {"dirichlet", "zero", "zeros", "constant"}:
        return "dirichlet"
    if bc in {"neumann", "reflect", "replicate"}:
        return "neumann"
    if bc in {"periodic", "circular", "torus"}:
        return "periodic"
    raise ValueError(
        "boundary_condition must be one of {'dirichlet','neumann','periodic'} "
        f"(got {boundary_condition!r})"
    )


def _padding_mode_from_boundary_condition(boundary_condition: str) -> str:
    bc = _normalize_boundary_condition(boundary_condition)
    if bc == "dirichlet":
        return "zeros"
    if bc == "neumann":
        return "replicate"
    return "circular"


def _fpad_mode_from_boundary_condition(boundary_condition: str) -> str:
    bc = _normalize_boundary_condition(boundary_condition)
    if bc == "dirichlet":
        return "constant"
    if bc == "neumann":
        return "replicate"
    return "circular"


def _grid_coordinates_2d(
    n_x: int,
    n_y: int,
    boundary_condition: str,
    device,
    dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    bc = _normalize_boundary_condition(boundary_condition)
    if bc == "periodic":
        x = torch.arange(n_x, device=device, dtype=dtype) / float(n_x)
        y = torch.arange(n_y, device=device, dtype=dtype) / float(n_y)
    else:
        x = (torch.arange(n_x, device=device, dtype=dtype) + 1.0) / float(n_x + 1)
        y = (torch.arange(n_y, device=device, dtype=dtype) + 1.0) / float(n_y + 1)
    return torch.meshgrid(x, y, indexing="ij")


def _ensure_batch_state_2d(u: torch.Tensor, name: str) -> tuple[torch.Tensor, bool]:
    if u.dim() == 2:
        return u.unsqueeze(0), True
    if u.dim() == 3:
        return u, False
    raise ValueError(f"{name} must have shape (n_x,n_y) or (batch,n_x,n_y), got {tuple(u.shape)}")


class ResidualConvBlock2D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, padding_mode: str = "zeros"):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
        )
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return x + y


class StateEncoder2D(nn.Module):
    """
    Encode state u -> latent field z.
    Does not take forcing input.
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
        boundary_condition: str = "dirichlet",
    ):
        super().__init__()
        self.n_x = n_x
        self.n_y = n_y
        self.latent_channels = latent_channels
        self.use_grad_features = use_grad_features
        self.boundary_condition = _normalize_boundary_condition(boundary_condition)
        self.padding_mode = _padding_mode_from_boundary_condition(self.boundary_condition)

        in_channels = 3 if use_grad_features else 1
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
        self.out_proj = nn.Conv2d(hidden_channels, latent_channels, kernel_size=1)
        self.act = nn.GELU()

    def _grad_feats(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pad_kwargs = {"mode": _fpad_mode_from_boundary_condition(self.boundary_condition)}
        if pad_kwargs["mode"] == "constant":
            pad_kwargs["value"] = 0.0
        u_pad = F.pad(u, (1, 1, 1, 1), **pad_kwargs)
        du_dx = 0.5 * (u_pad[:, 2:, 1:-1] - u_pad[:, :-2, 1:-1])
        du_dy = 0.5 * (u_pad[:, 1:-1, 2:] - u_pad[:, 1:-1, :-2])
        return du_dx, du_dy

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        u_b, squeeze = _ensure_batch_state_2d(u, name="u")
        if u_b.shape[1:] != (self.n_x, self.n_y):
            raise ValueError(f"u spatial shape must be ({self.n_x},{self.n_y}), got {tuple(u_b.shape[1:])}")

        if self.use_grad_features:
            du_dx, du_dy = self._grad_feats(u_b)
            x = torch.stack([u_b, du_dx, du_dy], dim=1)
        else:
            x = u_b.unsqueeze(1)

        h = self.act(self.in_proj(x))
        for block in self.blocks:
            h = self.act(block(h))
        z = self.out_proj(h)
        if squeeze:
            return z.squeeze(0)
        return z


class StateDecoder2D(nn.Module):
    """
    Decode latent field z -> physical state u.
    Does not take forcing input.
    """

    def __init__(
        self,
        n_x: int,
        n_y: int,
        latent_channels: int = 16,
        hidden_channels: int = 64,
        n_blocks: int = 4,
        kernel_size: int = 3,
        boundary_condition: str = "dirichlet",
    ):
        super().__init__()
        self.n_x = n_x
        self.n_y = n_y
        self.latent_channels = latent_channels
        self.boundary_condition = _normalize_boundary_condition(boundary_condition)
        self.padding_mode = _padding_mode_from_boundary_condition(self.boundary_condition)

        padding = kernel_size // 2
        self.in_proj = nn.Conv2d(
            latent_channels,
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
        self.out_proj = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if z.dim() == 3:
            z = z.unsqueeze(0)
            squeeze = True
        if z.dim() != 4:
            raise ValueError(
                f"z must have shape (latent_channels,n_x,n_y) or (batch,latent_channels,n_x,n_y), got {tuple(z.shape)}"
            )
        if z.shape[1] != self.latent_channels:
            raise ValueError(f"Expected latent_channels={self.latent_channels}, got {z.shape[1]}")
        if z.shape[2:] != (self.n_x, self.n_y):
            raise ValueError(f"z spatial shape must be ({self.n_x},{self.n_y}), got {tuple(z.shape[2:])}")

        h = self.act(self.in_proj(z))
        for block in self.blocks:
            h = self.act(block(h))
        u = self.out_proj(h).squeeze(1)
        if squeeze:
            return u.squeeze(0)
        return u


class ProximalStepSimulator2D(nn.Module):
    """
    Latent proximal step simulator:
        z_{k+1} = z_k + Delta(z_k, f, dt)
    """

    def __init__(
        self,
        n_x: int,
        n_y: int,
        latent_channels: int = 16,
        hidden_channels: int = 64,
        n_blocks: int = 6,
        kernel_size: int = 3,
        use_forcing_channel: bool = True,
        use_dt_channel: bool = False,
        default_dt: Optional[float] = None,
        boundary_condition: str = "dirichlet",
    ):
        super().__init__()
        self.n_x = n_x
        self.n_y = n_y
        self.latent_channels = latent_channels
        self.use_forcing_channel = use_forcing_channel
        self.use_dt_channel = use_dt_channel
        self.default_dt = default_dt
        self.boundary_condition = _normalize_boundary_condition(boundary_condition)
        self.padding_mode = _padding_mode_from_boundary_condition(self.boundary_condition)

        in_channels = latent_channels + (1 if use_forcing_channel else 0) + (1 if use_dt_channel else 0)
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
        self.out_proj = nn.Conv2d(hidden_channels, latent_channels, kernel_size=1)
        self.act = nn.GELU()

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

    def forward(self, z_k: torch.Tensor, f: torch.Tensor, dt=None) -> torch.Tensor:
        squeeze = False
        if z_k.dim() == 3:
            z_k = z_k.unsqueeze(0)
            f = f.unsqueeze(0)
            squeeze = True
        if z_k.dim() != 4:
            raise ValueError(
                f"z_k must have shape (latent_channels,n_x,n_y) or (batch,latent_channels,n_x,n_y), got {tuple(z_k.shape)}"
            )
        if z_k.shape[1] != self.latent_channels:
            raise ValueError(f"Expected latent_channels={self.latent_channels}, got {z_k.shape[1]}")
        if z_k.shape[2:] != (self.n_x, self.n_y):
            raise ValueError(f"z_k spatial shape must be ({self.n_x},{self.n_y}), got {tuple(z_k.shape[2:])}")

        batch_size = z_k.shape[0]
        feat = [z_k]
        if self.use_forcing_channel:
            f_int = _to_interior_forcing_2d(f, n_x=self.n_x, n_y=self.n_y)
            if f_int.shape[0] == 1 and batch_size > 1:
                f_int = f_int.expand(batch_size, -1, -1)
            if f_int.shape[0] != batch_size:
                raise ValueError("forcing batch size must match z batch size or be 1")
            feat.append(f_int.unsqueeze(1))
        if self.use_dt_channel:
            feat.append(self._dt_channel(batch_size, dt, z_k.device, z_k.dtype))
        x = torch.cat(feat, dim=1)

        h = self.act(self.in_proj(x))
        for block in self.blocks:
            h = self.act(block(h))
        delta = self.out_proj(h)
        z_next = z_k + delta
        if squeeze:
            return z_next.squeeze(0)
        return z_next


class SpectralConv2d(nn.Module):
    """
    2D Fourier layer:
      - FFT to frequency domain.
      - Linear transform on retained low-frequency modes.
      - Inverse FFT back to physical domain.
    """

    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int):
        super().__init__()
        if modes_x <= 0 or modes_y <= 0:
            raise ValueError("modes_x and modes_y must be positive")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes_x = int(modes_x)
        self.modes_y = int(modes_y)

        scale = 1.0 / max(1, in_channels * out_channels)
        self.weight_pos = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes_x, self.modes_y, 2)
        )
        self.weight_neg = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes_x, self.modes_y, 2)
        )

    @staticmethod
    def _compl_mul2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_c, mx, my), w: (in_c, out_c, mx, my) -> (batch, out_c, mx, my)
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"x must have shape (batch,channels,n_x,n_y), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {x.shape[1]}")

        batch_size, _, n_x, n_y = x.shape
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))

        max_modes_x = min(self.modes_x, n_x)
        max_modes_y = min(self.modes_y, x_ft.shape[-1])

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            n_x,
            x_ft.shape[-1],
            dtype=x_ft.dtype,
            device=x.device,
        )

        w_pos = torch.view_as_complex(self.weight_pos[:, :, :max_modes_x, :max_modes_y, :].contiguous())
        w_neg = torch.view_as_complex(self.weight_neg[:, :, :max_modes_x, :max_modes_y, :].contiguous())
        out_ft[:, :, :max_modes_x, :max_modes_y] = self._compl_mul2d(
            x_ft[:, :, :max_modes_x, :max_modes_y], w_pos
        )
        out_ft[:, :, -max_modes_x:, :max_modes_y] = self._compl_mul2d(
            x_ft[:, :, -max_modes_x:, :max_modes_y], w_neg
        )
        x_out = torch.fft.irfft2(out_ft, s=(n_x, n_y), dim=(-2, -1))
        return x_out


class FNOBlock2D(nn.Module):
    def __init__(self, width: int, modes_x: int, modes_y: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_x=modes_x, modes_y=modes_y)
        self.local = nn.Conv2d(width, width, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.spectral(x) + self.local(x))


class FNOProximalStepSimulator2D(nn.Module):
    """
    FNO-based latent proximal step simulator:
        z_{k+1} = z_k + Delta_FNO(z_k, f, dt)
    """

    def __init__(
        self,
        n_x: int,
        n_y: int,
        latent_channels: int = 16,
        width: int = 64,
        n_layers: int = 4,
        modes_x: int = 16,
        modes_y: int = 16,
        use_forcing_channel: bool = True,
        use_dt_channel: bool = False,
        use_grid_features: bool = True,
        default_dt: Optional[float] = None,
        boundary_condition: str = "dirichlet",
    ):
        super().__init__()
        self.n_x = n_x
        self.n_y = n_y
        self.latent_channels = latent_channels
        self.use_forcing_channel = use_forcing_channel
        self.use_dt_channel = use_dt_channel
        self.use_grid_features = use_grid_features
        self.default_dt = default_dt
        self.boundary_condition = _normalize_boundary_condition(boundary_condition)
        self.modes_x = int(modes_x)
        self.modes_y = int(modes_y)

        in_channels = (
            latent_channels
            + (1 if use_forcing_channel else 0)
            + (1 if use_dt_channel else 0)
            + (2 if use_grid_features else 0)
        )
        self.in_proj = nn.Conv2d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock2D(width, modes_x=self.modes_x, modes_y=self.modes_y) for _ in range(n_layers)])
        self.out_proj = nn.Conv2d(width, latent_channels, kernel_size=1)

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

    def forward(self, z_k: torch.Tensor, f: torch.Tensor, dt=None) -> torch.Tensor:
        squeeze = False
        if z_k.dim() == 3:
            z_k = z_k.unsqueeze(0)
            f = f.unsqueeze(0)
            squeeze = True
        if z_k.dim() != 4:
            raise ValueError(
                f"z_k must have shape (latent_channels,n_x,n_y) or (batch,latent_channels,n_x,n_y), got {tuple(z_k.shape)}"
            )
        if z_k.shape[1] != self.latent_channels:
            raise ValueError(f"Expected latent_channels={self.latent_channels}, got {z_k.shape[1]}")
        if z_k.shape[2:] != (self.n_x, self.n_y):
            raise ValueError(f"z_k spatial shape must be ({self.n_x},{self.n_y}), got {tuple(z_k.shape[2:])}")

        batch_size = z_k.shape[0]
        feat = [z_k]
        if self.use_forcing_channel:
            f_int = _to_interior_forcing_2d(f, n_x=self.n_x, n_y=self.n_y)
            if f_int.shape[0] == 1 and batch_size > 1:
                f_int = f_int.expand(batch_size, -1, -1)
            if f_int.shape[0] != batch_size:
                raise ValueError("forcing batch size must match z batch size or be 1")
            feat.append(f_int.unsqueeze(1))
        if self.use_dt_channel:
            feat.append(self._dt_channel(batch_size, dt, z_k.device, z_k.dtype))
        if self.use_grid_features:
            feat.append(self._grid_features(batch_size, z_k.device, z_k.dtype))

        x = torch.cat(feat, dim=1)
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        delta = self.out_proj(h)
        z_next = z_k + delta
        if squeeze:
            return z_next.squeeze(0)
        return z_next


class EnergyHead2D(nn.Module):
    """
    Latent energy head.

    local mode:
      E(z; f) = integral softplus(rho_local(z,f))

    fno mode:
      E(z; f) = integral [softplus(rho_local(z,f)) + softplus(rho_spec(z,f))]
    """

    def __init__(
        self,
        n_x: int,
        n_y: int,
        h_x: float,
        h_y: float,
        latent_channels: int = 16,
        hidden_channels: int = 64,
        n_layers: int = 4,
        use_forcing_channel: bool = True,
        use_grad_norm_feature: bool = True,
        boundary_condition: str = "dirichlet",
        head_type: str = "local",
        energy_fno_modes_x: int = 16,
        energy_fno_modes_y: int = 16,
        energy_fno_width: Optional[int] = None,
        energy_fno_layers: Optional[int] = None,
    ):
        super().__init__()
        self.n_x = n_x
        self.n_y = n_y
        self.area = float(h_x) * float(h_y)
        self.latent_channels = latent_channels
        self.use_forcing_channel = use_forcing_channel
        self.use_grad_norm_feature = use_grad_norm_feature
        self.boundary_condition = _normalize_boundary_condition(boundary_condition)
        self.padding_mode = _padding_mode_from_boundary_condition(self.boundary_condition)
        self.head_type = head_type.lower().strip()
        if self.head_type not in ("local", "fno"):
            raise ValueError("head_type must be one of {'local', 'fno'}")

        in_channels = latent_channels + (1 if use_forcing_channel else 0) + (1 if use_grad_norm_feature else 0)

        # Local branch.
        local_layers = [
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                padding_mode=self.padding_mode,
            ),
            nn.GELU(),
        ]
        for _ in range(max(1, n_layers - 1)):
            local_layers.extend(
                [
                    nn.Conv2d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=3,
                        padding=1,
                        padding_mode=self.padding_mode,
                    ),
                    nn.GELU(),
                ]
            )
        self.local_backbone = nn.Sequential(*local_layers)
        self.local_density_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

        # Optional spectral branch.
        self.use_spectral_branch = self.head_type == "fno"
        if self.use_spectral_branch:
            spectral_width = int(energy_fno_width if energy_fno_width is not None else hidden_channels)
            spectral_layers = int(energy_fno_layers if energy_fno_layers is not None else n_layers)
            if spectral_width <= 0:
                raise ValueError("energy_fno_width must be > 0")
            if spectral_layers < 1:
                raise ValueError("energy_fno_layers must be >= 1")
            if energy_fno_modes_x < 1 or energy_fno_modes_y < 1:
                raise ValueError("energy_fno_modes_x/y must be >= 1")

            self.spectral_in_proj = nn.Conv2d(in_channels, spectral_width, kernel_size=1)
            self.spectral_blocks = nn.ModuleList(
                [FNOBlock2D(spectral_width, modes_x=energy_fno_modes_x, modes_y=energy_fno_modes_y) for _ in range(spectral_layers)]
            )
            self.spectral_density_head = nn.Conv2d(spectral_width, 1, kernel_size=1)
        else:
            self.spectral_in_proj = None
            self.spectral_blocks = None
            self.spectral_density_head = None

    def _estimate_grad_norm(self, z: torch.Tensor) -> torch.Tensor:
        pad_kwargs = {"mode": _fpad_mode_from_boundary_condition(self.boundary_condition)}
        if pad_kwargs["mode"] == "constant":
            pad_kwargs["value"] = 0.0
        z_pad = F.pad(z, (1, 1, 1, 1), **pad_kwargs)
        z_x = 0.5 * (z_pad[:, :, 2:, 1:-1] - z_pad[:, :, :-2, 1:-1])
        z_y = 0.5 * (z_pad[:, :, 1:-1, 2:] - z_pad[:, :, 1:-1, :-2])
        return torch.sqrt(torch.sum(z_x * z_x + z_y * z_y, dim=1, keepdim=True) + 1e-12)

    def _build_input_features(self, z: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        feat = [z]
        if self.use_forcing_channel:
            f_int = _to_interior_forcing_2d(f, n_x=self.n_x, n_y=self.n_y)
            if f_int.shape[0] == 1 and z.shape[0] > 1:
                f_int = f_int.expand(z.shape[0], -1, -1)
            if f_int.shape[0] != z.shape[0]:
                raise ValueError("forcing batch size must match z batch size or be 1")
            feat.append(f_int.unsqueeze(1))
        if self.use_grad_norm_feature:
            feat.append(self._estimate_grad_norm(z))
        return torch.cat(feat, dim=1)

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
        if z.shape[1] != self.latent_channels:
            raise ValueError(f"Expected latent_channels={self.latent_channels}, got {z.shape[1]}")
        if z.shape[2:] != (self.n_x, self.n_y):
            raise ValueError(f"z spatial shape must be ({self.n_x},{self.n_y}), got {tuple(z.shape[2:])}")

        x = self._build_input_features(z, f)

        local_hidden = self.local_backbone(x)
        local_density = F.softplus(self.local_density_head(local_hidden)).squeeze(1)

        if self.use_spectral_branch:
            spec_hidden = self.spectral_in_proj(x)
            for block in self.spectral_blocks:
                spec_hidden = block(spec_hidden)
            spectral_density = F.softplus(self.spectral_density_head(spec_hidden)).squeeze(1)
            density = local_density + spectral_density
        else:
            density = local_density

        energy = self.area * torch.sum(density, dim=(1, 2))
        if squeeze:
            return energy.squeeze(0)
        return energy


class HiddenGradientFlowModel2D(nn.Module):
    """
    Hidden-space model:
      z_k = Enc(u_k)
      z_{k+1} = Prox(z_k, f, dt)
      u_{k+1} = Dec(z_{k+1})
    """

    def __init__(
        self,
        encoder: StateEncoder2D,
        decoder: StateDecoder2D,
        prox_step: nn.Module,
        energy_head: EnergyHead2D,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prox_step = prox_step
        self.energy_head = energy_head

    def encode(self, u: torch.Tensor) -> torch.Tensor:
        return self.encoder(u)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def latent_energy(self, z: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        return self.energy_head(z, f)

    def predict_latent_step(self, z_k: torch.Tensor, f: torch.Tensor, dt=None) -> torch.Tensor:
        return self.prox_step(z_k, f, dt=dt)

    def predict_step(self, u_k: torch.Tensor, f: torch.Tensor, dt=None, return_latent: bool = False):
        z_k = self.encode(u_k)
        z_next = self.predict_latent_step(z_k, f, dt=dt)
        u_next = self.decode(z_next)
        if return_latent:
            return u_next, z_k, z_next
        return u_next

    def energy(self, u: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        z = self.encode(u)
        return self.latent_energy(z, f)

    def forward(self, u_k: torch.Tensor, f: torch.Tensor, dt=None) -> torch.Tensor:
        return self.predict_step(u_k, f, dt=dt, return_latent=False)


__all__ = [
    "ResidualConvBlock2D",
    "StateEncoder2D",
    "StateDecoder2D",
    "ProximalStepSimulator2D",
    "SpectralConv2d",
    "FNOBlock2D",
    "FNOProximalStepSimulator2D",
    "EnergyHead2D",
    "HiddenGradientFlowModel2D",
]
