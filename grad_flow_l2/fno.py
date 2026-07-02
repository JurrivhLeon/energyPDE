"""Physical-space Fourier neural operators for 2D time-step prediction."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _periodic_grid_2d(n_x: int, n_y: int, device, dtype) -> torch.Tensor:
    x = torch.arange(n_x, device=device, dtype=dtype) / float(n_x)
    y = torch.arange(n_y, device=device, dtype=dtype) / float(n_y)
    gx, gy = torch.meshgrid(x, y, indexing="ij")
    return torch.stack([gx, gy], dim=0)


def _ensure_state_2d(
    u: torch.Tensor,
    state_channels: int,
    n_x: int,
    n_y: int,
    name: str = "u",
) -> tuple[torch.Tensor, bool]:
    if state_channels == 1:
        if u.dim() == 2:
            u = u.unsqueeze(0)
            squeeze = True
        elif u.dim() == 3:
            squeeze = False
        elif u.dim() == 4 and u.shape[1] == 1:
            return u, False
        else:
            raise ValueError(
                f"{name} must have shape (n_x,n_y), (batch,n_x,n_y), "
                f"or (batch,1,n_x,n_y), got {tuple(u.shape)}"
            )
        if u.shape[-2:] != (n_x, n_y):
            raise ValueError(f"{name} spatial shape must be ({n_x},{n_y}), got {tuple(u.shape[-2:])}")
        return u.unsqueeze(1), squeeze

    if u.dim() == 3:
        if u.shape != (state_channels, n_x, n_y):
            raise ValueError(
                f"{name} must have shape ({state_channels},{n_x},{n_y}) or "
                f"(batch,{state_channels},{n_x},{n_y}), got {tuple(u.shape)}"
            )
        return u.unsqueeze(0), True
    if u.dim() == 4:
        if u.shape[1:] != (state_channels, n_x, n_y):
            raise ValueError(
                f"{name} must have shape (batch,{state_channels},{n_x},{n_y}), got {tuple(u.shape)}"
            )
        return u, False
    raise ValueError(
        f"{name} must have shape ({state_channels},{n_x},{n_y}) or "
        f"(batch,{state_channels},{n_x},{n_y}), got {tuple(u.shape)}"
    )


def _ensure_forcing_2d(
    f: torch.Tensor,
    batch_size: int,
    forcing_channels: int,
    n_x: int,
    n_y: int,
) -> torch.Tensor:
    if forcing_channels == 1:
        if f.dim() == 2:
            f = f.unsqueeze(0)
        if f.dim() == 3:
            if f.shape[-2:] != (n_x, n_y):
                raise ValueError(f"forcing spatial shape must be ({n_x},{n_y}), got {tuple(f.shape[-2:])}")
            f = f.unsqueeze(1)
        elif f.dim() == 4:
            if f.shape[1:] != (1, n_x, n_y):
                raise ValueError(f"forcing must have shape (batch,1,{n_x},{n_y}), got {tuple(f.shape)}")
        else:
            raise ValueError(
                f"forcing must have shape ({n_x},{n_y}), (batch,{n_x},{n_y}), "
                f"or (batch,1,{n_x},{n_y}), got {tuple(f.shape)}"
            )
    else:
        if f.dim() == 3:
            if f.shape != (forcing_channels, n_x, n_y):
                raise ValueError(
                    f"forcing must have shape ({forcing_channels},{n_x},{n_y}) or "
                    f"(batch,{forcing_channels},{n_x},{n_y}), got {tuple(f.shape)}"
                )
            f = f.unsqueeze(0)
        elif f.dim() == 4:
            if f.shape[1:] != (forcing_channels, n_x, n_y):
                raise ValueError(
                    f"forcing must have shape (batch,{forcing_channels},{n_x},{n_y}), got {tuple(f.shape)}"
                )
        else:
            raise ValueError(
                f"forcing must have shape ({forcing_channels},{n_x},{n_y}) or "
                f"(batch,{forcing_channels},{n_x},{n_y}), got {tuple(f.shape)}"
            )

    if f.shape[0] == 1 and batch_size > 1:
        f = f.expand(batch_size, -1, -1, -1)
    if f.shape[0] != batch_size:
        raise ValueError("forcing batch size must match state batch size or be 1")
    return f


class SpectralConv2d(nn.Module):
    """2D spectral convolution retaining low Fourier modes."""

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
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"x must have shape (batch,channels,n_x,n_y), got {tuple(x.shape)}")
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
        return torch.fft.irfft2(out_ft, s=(n_x, n_y), dim=(-2, -1))


class FNOBlock2D(nn.Module):
    def __init__(self, width: int, modes_x: int, modes_y: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_x=modes_x, modes_y=modes_y)
        self.local = nn.Conv2d(width, width, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.spectral(x) + self.local(x))


class FNO2D(nn.Module):
    """
    Standard physical-space 2D FNO stepper.

    The model predicts one time step in physical variables. With
    ``residual=True`` it learns an increment, ``u_{k+1}=u_k+G(u_k,f,dt,x,y)``.
    """

    def __init__(
        self,
        n_x: int,
        n_y: int,
        state_channels: int = 1,
        forcing_channels: int = 1,
        width: int = 64,
        n_layers: int = 4,
        modes_x: int = 16,
        modes_y: int = 16,
        use_forcing_channel: bool = True,
        use_dt_channel: bool = False,
        use_grid_features: bool = True,
        default_dt: Optional[float] = None,
        residual: bool = True,
    ):
        super().__init__()
        self.n_x = int(n_x)
        self.n_y = int(n_y)
        self.state_channels = int(state_channels)
        self.forcing_channels = int(forcing_channels)
        self.use_forcing_channel = bool(use_forcing_channel)
        self.use_dt_channel = bool(use_dt_channel)
        self.use_grid_features = bool(use_grid_features)
        self.default_dt = default_dt
        self.residual = bool(residual)
        if self.state_channels < 1:
            raise ValueError("state_channels must be >= 1")
        if self.forcing_channels < 1:
            raise ValueError("forcing_channels must be >= 1")

        in_channels = self.state_channels
        if self.use_forcing_channel:
            in_channels += self.forcing_channels
        if self.use_dt_channel:
            in_channels += 1
        if self.use_grid_features:
            in_channels += 2

        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [FNOBlock2D(width, modes_x=modes_x, modes_y=modes_y) for _ in range(n_layers)]
        )
        self.project = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width * 2, self.state_channels, kernel_size=1),
        )
        if self.residual:
            nn.init.zeros_(self.project[-1].bias)
            nn.init.normal_(self.project[-1].weight, std=0.01)

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
        grid = _periodic_grid_2d(self.n_x, self.n_y, device=device, dtype=dtype)
        return grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

    def _restore_state_shape(self, u: torch.Tensor, squeeze: bool) -> torch.Tensor:
        if self.state_channels == 1:
            u = u.squeeze(1)
            if squeeze:
                return u.squeeze(0)
            return u
        if squeeze:
            return u.squeeze(0)
        return u

    def predict_step(self, u_k: torch.Tensor, f: torch.Tensor | None = None, dt=None) -> torch.Tensor:
        u, squeeze = _ensure_state_2d(u_k, self.state_channels, self.n_x, self.n_y, name="u_k")
        feat = [u]
        if self.use_forcing_channel:
            if f is None:
                raise ValueError("forcing f must be provided when use_forcing_channel=True")
            feat.append(_ensure_forcing_2d(f, u.shape[0], self.forcing_channels, self.n_x, self.n_y))
        if self.use_dt_channel:
            feat.append(self._dt_channel(u.shape[0], dt, u.device, u.dtype))
        if self.use_grid_features:
            feat.append(self._grid_features(u.shape[0], u.device, u.dtype))

        h = self.lift(torch.cat(feat, dim=1))
        for block in self.blocks:
            h = block(h)
        out = self.project(h)
        if self.residual:
            out = u + out
        return self._restore_state_shape(out, squeeze=squeeze)

    def forward(self, u_k: torch.Tensor, f: torch.Tensor | None = None, dt=None) -> torch.Tensor:
        return self.predict_step(u_k, f=f, dt=dt)



def _periodic_grid_1d(n_x: int, device, dtype) -> torch.Tensor:
    return (torch.arange(n_x, device=device, dtype=dtype) / float(n_x)).view(1, n_x)


def _ensure_state_1d(
    u: torch.Tensor,
    state_channels: int,
    n_x: int,
    name: str = "u",
) -> tuple[torch.Tensor, bool]:
    if state_channels == 1:
        if u.dim() == 1:
            u = u.unsqueeze(0)
            squeeze = True
        elif u.dim() == 2:
            squeeze = False
        elif u.dim() == 3 and u.shape[1] == 1:
            return u, False
        else:
            raise ValueError(
                f"{name} must have shape (n_x,), (batch,n_x), or (batch,1,n_x), got {tuple(u.shape)}"
            )
        if u.shape[-1] != n_x:
            raise ValueError(f"{name} width must be {n_x}, got {u.shape[-1]}")
        return u.unsqueeze(1), squeeze

    if u.dim() == 2:
        if u.shape != (state_channels, n_x):
            raise ValueError(
                f"{name} must have shape ({state_channels},{n_x}) or "
                f"(batch,{state_channels},{n_x}), got {tuple(u.shape)}"
            )
        return u.unsqueeze(0), True
    if u.dim() == 3:
        if u.shape[1:] != (state_channels, n_x):
            raise ValueError(f"{name} must have shape (batch,{state_channels},{n_x}), got {tuple(u.shape)}")
        return u, False
    raise ValueError(
        f"{name} must have shape ({state_channels},{n_x}) or (batch,{state_channels},{n_x}), got {tuple(u.shape)}"
    )


def _ensure_forcing_1d(
    f: torch.Tensor,
    batch_size: int,
    forcing_channels: int,
    n_x: int,
) -> torch.Tensor:
    if forcing_channels == 1:
        if f.dim() == 1:
            f = f.unsqueeze(0)
        if f.dim() == 2:
            if f.shape[-1] != n_x:
                raise ValueError(f"forcing width must be {n_x}, got {f.shape[-1]}")
            f = f.unsqueeze(1)
        elif f.dim() == 3:
            if f.shape[1:] != (1, n_x):
                raise ValueError(f"forcing must have shape (batch,1,{n_x}), got {tuple(f.shape)}")
        else:
            raise ValueError(f"forcing must have shape (n_x,), (batch,n_x), or (batch,1,n_x), got {tuple(f.shape)}")
    else:
        if f.dim() == 2:
            if f.shape != (forcing_channels, n_x):
                raise ValueError(
                    f"forcing must have shape ({forcing_channels},{n_x}) or "
                    f"(batch,{forcing_channels},{n_x}), got {tuple(f.shape)}"
                )
            f = f.unsqueeze(0)
        elif f.dim() == 3:
            if f.shape[1:] != (forcing_channels, n_x):
                raise ValueError(f"forcing must have shape (batch,{forcing_channels},{n_x}), got {tuple(f.shape)}")
        else:
            raise ValueError(
                f"forcing must have shape ({forcing_channels},{n_x}) or "
                f"(batch,{forcing_channels},{n_x}), got {tuple(f.shape)}"
            )
    if f.shape[0] == 1 and batch_size > 1:
        f = f.expand(batch_size, -1, -1)
    if f.shape[0] != batch_size:
        raise ValueError("forcing batch size must match state batch size or be 1")
    return f


class SpectralConv1d(nn.Module):
    """1D spectral convolution retaining low positive Fourier modes."""

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


class FNO1D(nn.Module):
    """Standard physical-space 1D FNO stepper."""

    def __init__(
        self,
        n_x: int,
        state_channels: int = 1,
        forcing_channels: int = 1,
        width: int = 64,
        n_layers: int = 4,
        modes: int = 64,
        use_forcing_channel: bool = True,
        use_dt_channel: bool = False,
        use_grid_features: bool = True,
        default_dt: Optional[float] = None,
        residual: bool = True,
    ):
        super().__init__()
        self.n_x = int(n_x)
        self.state_channels = int(state_channels)
        self.forcing_channels = int(forcing_channels)
        self.use_forcing_channel = bool(use_forcing_channel)
        self.use_dt_channel = bool(use_dt_channel)
        self.use_grid_features = bool(use_grid_features)
        self.default_dt = default_dt
        self.residual = bool(residual)
        if self.state_channels < 1:
            raise ValueError("state_channels must be >= 1")
        if self.forcing_channels < 1:
            raise ValueError("forcing_channels must be >= 1")

        in_channels = self.state_channels
        if self.use_forcing_channel:
            in_channels += self.forcing_channels
        if self.use_dt_channel:
            in_channels += 1
        if self.use_grid_features:
            in_channels += 1
        self.lift = nn.Conv1d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock1D(width, modes=modes) for _ in range(n_layers)])
        self.project = nn.Sequential(
            nn.Conv1d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(width * 2, self.state_channels, kernel_size=1),
        )
        if self.residual:
            nn.init.zeros_(self.project[-1].bias)
            nn.init.normal_(self.project[-1].weight, std=0.01)

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
        return _periodic_grid_1d(self.n_x, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)

    def _restore_state_shape(self, u: torch.Tensor, squeeze: bool) -> torch.Tensor:
        if self.state_channels == 1:
            u = u.squeeze(1)
            if squeeze:
                return u.squeeze(0)
            return u
        if squeeze:
            return u.squeeze(0)
        return u

    def predict_step(self, u_k: torch.Tensor, f: torch.Tensor | None = None, dt=None) -> torch.Tensor:
        u, squeeze = _ensure_state_1d(u_k, self.state_channels, self.n_x, name="u_k")
        feat = [u]
        if self.use_forcing_channel:
            if f is None:
                raise ValueError("forcing f must be provided when use_forcing_channel=True")
            feat.append(_ensure_forcing_1d(f, u.shape[0], self.forcing_channels, self.n_x))
        if self.use_dt_channel:
            feat.append(self._dt_channel(u.shape[0], dt, u.device, u.dtype))
        if self.use_grid_features:
            feat.append(self._grid_features(u.shape[0], u.device, u.dtype))
        h = self.lift(torch.cat(feat, dim=1))
        for block in self.blocks:
            h = block(h)
        out = self.project(h)
        if self.residual:
            out = u + out
        return self._restore_state_shape(out, squeeze=squeeze)

    def forward(self, u_k: torch.Tensor, f: torch.Tensor | None = None, dt=None) -> torch.Tensor:
        return self.predict_step(u_k, f=f, dt=dt)


__all__ = ["FNO1D", "FNO2D", "FNOBlock1D", "FNOBlock2D", "SpectralConv1d", "SpectralConv2d"]
