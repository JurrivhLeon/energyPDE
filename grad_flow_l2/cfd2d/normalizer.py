"""Per-channel normalisation for 2D compressible NS states.

Statistics are computed from the training u_traj tensor and applied to
all splits so the model trains on zero-mean unit-variance data per channel.
The stored (mean, std) are used by eval scripts to report metrics in physical
units when needed.
"""

from __future__ import annotations
from typing import Dict, Tuple

import torch


def compute_channel_stats(
    u_traj: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-channel mean and std from a trajectory tensor.

    Parameters
    ----------
    u_traj : (N, T+1, C, nx, ny)
        Training trajectories in physical units.

    Returns
    -------
    mean : (C,)
    std  : (C,)
    """
    # Flatten all but the channel dimension and compute statistics
    C = u_traj.shape[2]
    flat = u_traj.permute(2, 0, 1, 3, 4).reshape(C, -1)   # (C, N*T*nx*ny)
    mean = flat.mean(dim=1)                                 # (C,)
    std  = flat.std(dim=1).clamp(min=1e-8)                  # (C,)
    return mean, std


def normalize_split(
    split: Dict[str, torch.Tensor],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Return a new split with u0 and u_traj normalised channel-wise.

    mean / std broadcast over (N, T+1, C, nx, ny) by unsqueezing to (1, 1, C, 1, 1).
    u0 uses the same statistics (it equals u_traj[:, 0]).
    f is left unchanged (zero placeholder).
    """
    m = mean.view(1, -1, 1, 1)    # (1, C, 1, 1) for u0 shape (N, C, nx, ny)
    s = std.view(1, -1, 1, 1)
    mt = mean.view(1, 1, -1, 1, 1)  # (1, 1, C, 1, 1) for u_traj shape (N, T, C, nx, ny)
    st = std.view(1, 1, -1, 1, 1)
    return {
        "f":      split["f"],
        "u0":     (split["u0"]     - m) / s,
        "u_traj": (split["u_traj"] - mt) / st,
    }


def denormalize(
    u: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """
    Inverse of normalisation.  u can be (N, C, nx, ny) or (N, T, C, nx, ny).
    """
    if u.dim() == 4:
        m, s = mean.view(1, -1, 1, 1), std.view(1, -1, 1, 1)
    else:
        m, s = mean.view(1, 1, -1, 1, 1), std.view(1, 1, -1, 1, 1)
    return u * s + m
