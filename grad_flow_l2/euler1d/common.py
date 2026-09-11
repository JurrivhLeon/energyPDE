"""Shared utilities for 1D Euler latent Markov experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F

STATE_CHANNELS = 3
CHANNEL_NAMES = ["rho", "u", "p"]


def resolve_device(cpu: bool = False, device: str | None = None) -> str:
    if cpu:
        return "cpu"
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def channel_weights_from_split(train_split, provided=None) -> torch.Tensor:
    if provided is not None:
        weights = torch.as_tensor(provided, dtype=torch.float32)
    else:
        u_flat = train_split["u_traj"].permute(2, 0, 1, 3).reshape(STATE_CHANNELS, -1)
        weights = 1.0 / u_flat.var(dim=1).clamp(min=1e-8)
    return weights / weights.mean()


def channel_weighted_mse(
    u_pred: torch.Tensor, u_ref: torch.Tensor, channel_weights=None
) -> torch.Tensor:
    if u_pred.shape != u_ref.shape:
        raise ValueError(
            f"u_pred and u_ref must have same shape, got {tuple(u_pred.shape)} vs {tuple(u_ref.shape)}"
        )
    if channel_weights is None:
        return F.mse_loss(u_pred, u_ref)
    if u_pred.dim() != 3:
        raise ValueError(
            f"Expected state shape (batch,channels,n_x), got {tuple(u_pred.shape)}"
        )
    weights = torch.as_tensor(channel_weights, device=u_pred.device, dtype=u_pred.dtype)
    if weights.dim() != 1 or int(weights.shape[0]) != int(u_pred.shape[1]):
        raise ValueError(f"channel_weights must have {u_pred.shape[1]} entries")
    return ((u_pred - u_ref).square() * weights.view(1, -1, 1)).mean()


@torch.no_grad()
def rollout_model_1d(
    model, u0: torch.Tensor, f: torch.Tensor, n_steps: int, dt: float, delta_clip=None
) -> torch.Tensor:
    squeeze = False
    if u0.dim() == 2:
        u0 = u0.unsqueeze(0)
        f = f.unsqueeze(0)
        squeeze = True
    states = [u0]
    u = u0
    for _ in range(int(n_steps)):
        u_tilde = model.predict_step(u, f, dt=dt)
        delta = u_tilde - u
        if delta_clip is not None and float(delta_clip) > 0.0:
            delta = delta.clamp(-float(delta_clip), float(delta_clip))
        u_next = u + delta
        finite = torch.isfinite(u_next).flatten(1).all(dim=1)
        u = torch.where(finite.view(-1, 1, 1), u_next, states[-1])
        states.append(u)
    traj = torch.stack(states, dim=1)
    if squeeze:
        return traj.squeeze(0)
    return traj


@torch.no_grad()
def rollout_vae_mean_1d(
    model, u0: torch.Tensor, f: torch.Tensor, n_steps: int, dt: float, delta_clip=None
) -> torch.Tensor:
    squeeze = False
    if u0.dim() == 2:
        u0 = u0.unsqueeze(0)
        f = f.unsqueeze(0)
        squeeze = True
    states = [u0]
    u = u0
    for _ in range(int(n_steps)):
        u_tilde = model.rollout_step(u, f, dt=dt)
        delta = u_tilde - u
        if delta_clip is not None and float(delta_clip) > 0.0:
            delta = delta.clamp(-float(delta_clip), float(delta_clip))
        u_next = u + delta
        finite = torch.isfinite(u_next).flatten(1).all(dim=1)
        u = torch.where(finite.view(-1, 1, 1), u_next, states[-1])
        states.append(u)
    traj = torch.stack(states, dim=1)
    if squeeze:
        return traj.squeeze(0)
    return traj


@torch.no_grad()
def rollout_vae_latent_mean_1d(
    model,
    u0: torch.Tensor,
    f: torch.Tensor,
    n_steps: int,
    dt: float,
    delta_clip=None,
) -> torch.Tensor:
    """Deterministic VAE rollout that marches once-encoded states in latent space."""
    del delta_clip
    squeeze = False
    if u0.dim() == 2:
        u0 = u0.unsqueeze(0)
        f = f.unsqueeze(0)
        squeeze = True

    z, _ = model.encode_stats(u0)
    z_states = [z]
    for _ in range(int(n_steps)):
        z_next = model.transition(z, f, dt=dt)
        finite = torch.isfinite(z_next).flatten(1).all(dim=1)
        z = torch.where(finite.view(-1, 1, 1), z_next, z)
        z_states.append(z)

    z_traj = torch.stack(z_states, dim=1)
    batch, traj_len, channels, n_x = z_traj.shape
    traj = model.decode(z_traj.reshape(batch * traj_len, channels, n_x))
    traj = traj.reshape(batch, traj_len, *traj.shape[1:])

    states = [u0]
    previous = u0
    finite = torch.isfinite(traj).flatten(2).all(dim=2)
    for step in range(1, traj_len):
        u_step = torch.where(finite[:, step].view(-1, 1, 1), traj[:, step], previous)
        states.append(u_step)
        previous = u_step
    traj = torch.stack(states, dim=1)
    if squeeze:
        return traj.squeeze(0)
    return traj


def relative_l2_error_1d(
    u_pred: torch.Tensor, u_ref: torch.Tensor, h: float
) -> torch.Tensor:
    if u_pred.shape != u_ref.shape:
        raise ValueError(
            f"u_pred and u_ref must have same shape, got {tuple(u_pred.shape)} vs {tuple(u_ref.shape)}"
        )
    diff = u_pred - u_ref
    num = torch.sqrt(float(h) * diff.square().sum(dim=-1))
    den = torch.sqrt(float(h) * u_ref.square().sum(dim=-1))
    return num / (den + 1e-8)


def spectral_h1_squared_1d(u: torch.Tensor, domain_length: float = 1.0) -> torch.Tensor:
    n_x = int(u.shape[-1])
    dx = float(domain_length) / float(n_x)
    u_hat = torch.fft.fft(u, dim=-1, norm="ortho")
    real_dtype = u.real.dtype
    k = (
        2.0
        * torch.pi
        * torch.fft.fftfreq(n_x, d=dx, device=u.device).to(dtype=real_dtype)
    )
    power = u_hat.real.square() + u_hat.imag.square()
    return (power * (1.0 + k.square()).view(*([1] * (u.dim() - 1)), n_x)).sum(dim=-1)


def safe_torch_load(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location)
    except RuntimeError as exc:
        if "weights_only=True" not in str(exc) or "legacy .tar format" not in str(exc):
            raise
        return torch.load(path, map_location=map_location, weights_only=False)
