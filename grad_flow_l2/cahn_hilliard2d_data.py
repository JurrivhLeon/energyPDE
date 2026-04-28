"""
Data generation for 2D Cahn-Hilliard dynamics with homogeneous Neumann BC:

    u_t = M * Delta(mu),
    mu = -epsilon^2 * Delta(u) + (u^3 - u).

Dataset format (operator-learning friendly and model-compatible):
    split["f"]        : (n_samples, n_x, n_y)     conditioning map (here: epsilon field)
    split["u0"]       : (n_samples, n_x, n_y)     initial condition
    split["u_traj"]   : (n_samples, K+1, n_x, n_y)
    split["epsilon"]  : (n_samples,)              per-sample epsilon
    split["mobility"] : (n_samples,)              per-sample mobility
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .cahn_hilliard2d_solver import (
        compute_ch_free_energy_2d,
        compute_total_mass_2d,
        prepare_ch2d_spectral_cache,
        solve_cahn_hilliard_trajectory,
    )
    from .heat_data import save_dataset_splits
except ImportError:
    from grad_flow_l2.cahn_hilliard2d_solver import (
        compute_ch_free_energy_2d,
        compute_total_mass_2d,
        prepare_ch2d_spectral_cache,
        solve_cahn_hilliard_trajectory,
    )
    from grad_flow_l2.heat_data import save_dataset_splits


DATASET_VERSION = 1


def _iter_with_progress(
    iterable: Iterable,
    total: int,
    desc: str,
    enabled: bool,
):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm  # type: ignore

        return tqdm(iterable, total=total, desc=desc, leave=False)
    except Exception:
        return iterable


class CahnHilliard2DTrajectoryTensorDataset(Dataset):
    """
    Dataset wrapper around precomputed tensors:
      f: (n_samples, n_x, n_y)
      u0: (n_samples, n_x, n_y)
      u_traj: (n_samples, K+1, n_x, n_y)
    """

    def __init__(
        self,
        f_data: torch.Tensor,
        u0_data: torch.Tensor,
        u_traj_data: torch.Tensor,
        epsilon_data: torch.Tensor | None = None,
        mobility_data: torch.Tensor | None = None,
    ):
        if f_data.dim() != 3:
            raise ValueError("f_data must have shape (n_samples,n_x,n_y)")
        if u0_data.dim() != 3:
            raise ValueError("u0_data must have shape (n_samples,n_x,n_y)")
        if u_traj_data.dim() != 4:
            raise ValueError("u_traj_data must have shape (n_samples,K+1,n_x,n_y)")

        n_samples = int(u_traj_data.shape[0])
        n_x = int(u_traj_data.shape[2])
        n_y = int(u_traj_data.shape[3])
        if (
            int(f_data.shape[0]) != n_samples
            or tuple(f_data.shape[1:]) != (n_x, n_y)
            or int(u0_data.shape[0]) != n_samples
            or tuple(u0_data.shape[1:]) != (n_x, n_y)
        ):
            raise ValueError("inconsistent tensor shapes for CahnHilliard2D trajectory dataset")

        if epsilon_data is not None and (epsilon_data.dim() != 1 or int(epsilon_data.shape[0]) != n_samples):
            raise ValueError("epsilon_data must have shape (n_samples,)")
        if mobility_data is not None and (mobility_data.dim() != 1 or int(mobility_data.shape[0]) != n_samples):
            raise ValueError("mobility_data must have shape (n_samples,)")

        self.f_data = f_data
        self.u0_data = u0_data
        self.u_traj_data = u_traj_data
        self.epsilon_data = epsilon_data
        self.mobility_data = mobility_data

    def __len__(self) -> int:
        return int(self.u0_data.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        out = {
            "f": self.f_data[idx],
            "u0": self.u0_data[idx],
            "u_traj": self.u_traj_data[idx],
        }
        if self.epsilon_data is not None:
            out["epsilon"] = self.epsilon_data[idx]
        if self.mobility_data is not None:
            out["mobility"] = self.mobility_data[idx]
        return out


class CahnHilliard2DStepDataset(Dataset):
    """
    Flattened one-step dataset from trajectories.
    Returns tuple: (u_k, u_{k+1}, f).
    """

    def __init__(self, f_data: torch.Tensor, u_traj_data: torch.Tensor):
        if f_data.dim() != 3:
            raise ValueError("f_data must have shape (n_samples,n_x,n_y)")
        if u_traj_data.dim() != 4:
            raise ValueError("u_traj_data must have shape (n_samples,K+1,n_x,n_y)")
        if int(f_data.shape[0]) != int(u_traj_data.shape[0]) or tuple(f_data.shape[1:]) != tuple(u_traj_data.shape[2:]):
            raise ValueError("f_data and u_traj_data shapes are inconsistent")

        self.f_data = f_data
        self.u_traj_data = u_traj_data
        self.n_samples = int(u_traj_data.shape[0])
        self.n_steps = int(u_traj_data.shape[1] - 1)

    def __len__(self) -> int:
        return self.n_samples * self.n_steps

    def __getitem__(self, idx: int):
        i = idx // self.n_steps
        k = idx % self.n_steps
        return self.u_traj_data[i, k], self.u_traj_data[i, k + 1], self.f_data[i]


def build_cahn_hilliard2d_step_dataset(split_or_dataset) -> CahnHilliard2DStepDataset:
    if isinstance(split_or_dataset, CahnHilliard2DTrajectoryTensorDataset):
        f_data = split_or_dataset.f_data
        u_traj_data = split_or_dataset.u_traj_data
    elif isinstance(split_or_dataset, dict):
        f_data = split_or_dataset["f"]
        u_traj_data = split_or_dataset["u_traj"]
    else:
        raise TypeError("Expected split dict or CahnHilliard2DTrajectoryTensorDataset")
    return CahnHilliard2DStepDataset(f_data=f_data, u_traj_data=u_traj_data)


def build_cahn_hilliard2d_trajectory_dataset_from_split(
    split: Dict[str, torch.Tensor],
) -> CahnHilliard2DTrajectoryTensorDataset:
    return CahnHilliard2DTrajectoryTensorDataset(
        f_data=split["f"],
        u0_data=split["u0"],
        u_traj_data=split["u_traj"],
        epsilon_data=split.get("epsilon"),
        mobility_data=split.get("mobility"),
    )


def sample_neumann_cosine_field_2d(
    n_x: int,
    n_y: int,
    n_samples: int = 1,
    max_modes: int = 8,
    mean_min: float = -0.30,
    mean_max: float = 0.30,
    fluctuation_amplitude: float = 0.30,
    noise_std: float = 0.02,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Sample low-frequency cosine fields compatible with Neumann BC.
    """
    if max_modes < 1:
        raise ValueError("max_modes must be >= 1")
    if mean_min > mean_max:
        raise ValueError("mean_min must be <= mean_max")
    if fluctuation_amplitude <= 0:
        raise ValueError("fluctuation_amplitude must be > 0")
    if noise_std < 0:
        raise ValueError("noise_std must be >= 0")

    x = torch.linspace(0.0, 1.0, n_x + 2, device=device)[1:-1]  # interior
    y = torch.linspace(0.0, 1.0, n_y + 2, device=device)[1:-1]
    k = torch.arange(1, max_modes + 1, device=device).float()
    l = torch.arange(1, max_modes + 1, device=device).float()

    basis_x = torch.cos(np.pi * k.unsqueeze(1) * x.unsqueeze(0))  # (mx, n_x)
    basis_y = torch.cos(np.pi * l.unsqueeze(1) * y.unsqueeze(0))  # (my, n_y)

    k2 = k.unsqueeze(1) * k.unsqueeze(1)
    l2 = l.unsqueeze(0) * l.unsqueeze(0)
    scale = 1.0 / torch.sqrt(1.0 + k2 + l2)  # (mx,my)

    coeff = torch.randn(n_samples, max_modes, max_modes, device=device) * scale.unsqueeze(0)
    field = torch.einsum("bkl,kx,ly->bxy", coeff, basis_x, basis_y)

    if noise_std > 0:
        field = field + float(noise_std) * torch.randn_like(field)

    field = field / (field.abs().amax(dim=(1, 2), keepdim=True) + 1e-8)
    amp = float(fluctuation_amplitude) * (10.0 ** torch.empty(n_samples, 1, 1, device=device).uniform_(-0.20, 0.20))
    mean = torch.empty(n_samples, 1, 1, device=device).uniform_(float(mean_min), float(mean_max))
    u0 = mean + amp * field
    return torch.clamp(u0, -0.98, 0.98)


def _rescale_batch_l2_2d(
    x: torch.Tensor,
    area: float,
    norm_min: float,
    norm_max: float,
) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError("x must have shape (batch, n_x, n_y)")
    norms = torch.sqrt(float(area) * torch.sum(x * x, dim=(1, 2)))
    targets = torch.empty_like(norms).uniform_(norm_min, norm_max)
    scale = targets / (norms + 1e-8)
    return x * scale.view(-1, 1, 1)


def _slice_split(data: Dict[str, torch.Tensor], start: int, end: int) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in data.items():
        out[k] = v[start:end].clone()
    return out


def generate_cahn_hilliard2d_dataset_splits(
    n_x: int,
    n_y: int,
    n_steps: int,
    t_final: float,
    n_train: int,
    n_val: int,
    n_test: int,
    epsilon_min: float = 0.04,
    epsilon_max: float = 0.04,
    mobility_min: float = 1.0,
    mobility_max: float = 1.0,
    seed: int = 42,
    u0_mean_min: float = -0.30,
    u0_mean_max: float = 0.30,
    u0_fluctuation_amplitude: float = 0.30,
    u0_max_modes: int = 8,
    u0_noise_std: float = 0.02,
    norm_targeting: bool = False,
    target_u0_norm_range: tuple[float, float] = (0.20, 1.00),
    cfl_nonlinear: float = 0.20,
    max_substeps_per_step: int = 4000,
    max_dt_substep: float | None = None,
    chunk_size: int = 128,
    show_progress: bool = False,
    solver_dtype: torch.dtype = torch.float64,
    out_dtype: torch.dtype = torch.float32,
) -> Dict[str, Dict[str, torch.Tensor]]:
    if n_x < 4 or n_y < 4:
        raise ValueError("n_x and n_y must both be >= 4")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if t_final <= 0:
        raise ValueError("t_final must be > 0")
    if n_train < 1 or n_val < 1 or n_test < 1:
        raise ValueError("n_train, n_val, n_test must all be >= 1")
    if epsilon_min <= 0 or epsilon_max <= 0:
        raise ValueError("epsilon bounds must be > 0")
    if mobility_min <= 0 or mobility_max <= 0:
        raise ValueError("mobility bounds must be > 0")
    if epsilon_min > epsilon_max:
        raise ValueError("epsilon_min must be <= epsilon_max")
    if mobility_min > mobility_max:
        raise ValueError("mobility_min must be <= mobility_max")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    rng_state_torch = torch.random.get_rng_state()
    rng_state_numpy = np.random.get_state()
    torch.manual_seed(seed)
    np.random.seed(seed)

    total = int(n_train + n_val + n_test)
    h_x = 1.0 / float(n_x + 1)
    h_y = 1.0 / float(n_y + 1)
    area = h_x * h_y

    u0 = sample_neumann_cosine_field_2d(
        n_x=n_x,
        n_y=n_y,
        n_samples=total,
        max_modes=u0_max_modes,
        mean_min=u0_mean_min,
        mean_max=u0_mean_max,
        fluctuation_amplitude=u0_fluctuation_amplitude,
        noise_std=u0_noise_std,
        device="cpu",
    )
    if norm_targeting:
        u0 = _rescale_batch_l2_2d(
            x=u0,
            area=area,
            norm_min=float(target_u0_norm_range[0]),
            norm_max=float(target_u0_norm_range[1]),
        )
    u0 = u0.to(dtype=out_dtype)

    eps_values = torch.empty(total, dtype=out_dtype).uniform_(float(epsilon_min), float(epsilon_max))
    mob_values = torch.empty(total, dtype=out_dtype).uniform_(float(mobility_min), float(mobility_max))

    # Conditioning map for model interfaces expecting a scalar field input f(x,y).
    # We encode epsilon as a constant map per sample.
    f = eps_values.view(total, 1, 1).expand(total, n_x, n_y).clone()

    u_traj_chunks = []
    chunk_starts = range(0, total, int(chunk_size))
    chunk_iter = _iter_with_progress(
        chunk_starts,
        total=(total + int(chunk_size) - 1) // int(chunk_size),
        desc="solve CH trajectories",
        enabled=show_progress,
    )
    for start in chunk_iter:
        end = min(int(start) + int(chunk_size), total)
        chunk_cache = prepare_ch2d_spectral_cache(
            n_x=n_x,
            n_y=n_y,
            h_x=h_x,
            h_y=h_y,
            device="cpu",
            dtype=solver_dtype,
        )
        u_chunk = solve_cahn_hilliard_trajectory(
            u0=u0[start:end].to(dtype=solver_dtype),
            n_steps=n_steps,
            t_final=t_final,
            epsilon=eps_values[start:end].to(dtype=solver_dtype),
            mobility=mob_values[start:end].to(dtype=solver_dtype),
            cfl_nonlinear=cfl_nonlinear,
            max_substeps_per_step=max_substeps_per_step,
            max_dt_substep=max_dt_substep,
            enforce_mass_correction=True,
            solver_cache=chunk_cache,
        ).to(dtype=out_dtype)
        u_traj_chunks.append(u_chunk)
    u_traj = torch.cat(u_traj_chunks, dim=0)

    all_data = {
        "f": f,
        "u0": u0,
        "u_traj": u_traj,
        "epsilon": eps_values,
        "mobility": mob_values,
    }

    train_end = int(n_train)
    val_end = int(n_train + n_val)
    splits: Dict[str, Dict[str, torch.Tensor]] = {
        "train": _slice_split(all_data, 0, train_end),
        "val": _slice_split(all_data, train_end, val_end),
        "test": _slice_split(all_data, val_end, total),
        "meta": {
            "dataset_version": DATASET_VERSION,
            "equation": "cahn_hilliard_2d_neumann",
            "n_x": int(n_x),
            "n_y": int(n_y),
            "n_steps": int(n_steps),
            "t_final": float(t_final),
            "h_x": float(h_x),
            "h_y": float(h_y),
            "n_train": int(n_train),
            "n_val": int(n_val),
            "n_test": int(n_test),
            "seed": int(seed),
            "epsilon_min": float(epsilon_min),
            "epsilon_max": float(epsilon_max),
            "mobility_min": float(mobility_min),
            "mobility_max": float(mobility_max),
            "u0_mean_min": float(u0_mean_min),
            "u0_mean_max": float(u0_mean_max),
            "u0_fluctuation_amplitude": float(u0_fluctuation_amplitude),
            "u0_max_modes": int(u0_max_modes),
            "u0_noise_std": float(u0_noise_std),
            "norm_targeting": bool(norm_targeting),
            "target_u0_norm_range": [float(target_u0_norm_range[0]), float(target_u0_norm_range[1])],
            "cfl_nonlinear": float(cfl_nonlinear),
            "max_substeps_per_step": int(max_substeps_per_step),
            "max_dt_substep": None if max_dt_substep is None else float(max_dt_substep),
            "chunk_size": int(chunk_size),
            "solver_dtype": str(solver_dtype).replace("torch.", ""),
            "out_dtype": str(out_dtype).replace("torch.", ""),
            "f_description": "constant epsilon field per sample",
        },
    }

    torch.random.set_rng_state(rng_state_torch)
    np.random.set_state(rng_state_numpy)
    return splits


def _print_split_stats(splits: Dict[str, Dict[str, torch.Tensor]]) -> None:
    print("Dataset meta:", splits.get("meta", {}))
    n_x = int(splits["meta"]["n_x"])
    n_y = int(splits["meta"]["n_y"])
    h_x = float(splits["meta"].get("h_x", 1.0 / float(n_x + 1)))
    h_y = float(splits["meta"].get("h_y", 1.0 / float(n_y + 1)))
    cache = prepare_ch2d_spectral_cache(n_x=n_x, n_y=n_y, h_x=h_x, h_y=h_y, dtype=torch.float64)
    for split_name in ("train", "val", "test"):
        split = splits[split_name]
        if int(split["u0"].shape[0]) == 0:
            print(f"{split_name}: empty split")
            continue

        u0 = split["u0"].to(torch.float64)
        u_last = split["u_traj"][:, -1].to(torch.float64)
        eps = split["epsilon"].to(torch.float64)

        mass0 = compute_total_mass_2d(u0, h_x=h_x, h_y=h_y)
        massf = compute_total_mass_2d(u_last, h_x=h_x, h_y=h_y)
        e0 = compute_ch_free_energy_2d(u0, epsilon=eps, h_x=h_x, h_y=h_y, cache=cache)
        ef = compute_ch_free_energy_2d(u_last, epsilon=eps, h_x=h_x, h_y=h_y, cache=cache)
        print(
            f"{split_name}: "
            f"f={tuple(split['f'].shape)}, "
            f"u0={tuple(split['u0'].shape)}, "
            f"u_traj={tuple(split['u_traj'].shape)}, "
            f"mass_drift_mean={torch.mean(torch.abs(massf - mass0)).item():.3e}, "
            f"E0_mean={e0.mean().item():.4f}, "
            f"Efinal_mean={ef.mean().item():.4f}"
        )


def _parse_snapshot_times(value: str) -> list[float]:
    out = []
    for token in value.split(","):
        stripped = token.strip()
        if stripped:
            out.append(float(stripped))
    if len(out) == 0:
        raise ValueError("snapshot_times must include at least one time value")
    return out


def plot_sample_rows(
    split: Dict[str, torch.Tensor],
    split_name: str,
    out_path: str,
    t_final: float,
    snapshot_times: Sequence[float],
    n_plot_samples: int = 6,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plotting because matplotlib is unavailable: {exc}")
        return

    u_traj = split["u_traj"]
    eps = split.get("epsilon")
    total = int(u_traj.shape[0])
    if total == 0:
        print(f"Skipping plotting for {split_name}: empty split")
        return

    n_plot = min(max(1, int(n_plot_samples)), total)
    sample_indices = torch.linspace(0, total - 1, n_plot).long().tolist()

    n_steps = int(u_traj.shape[1] - 1)
    n_cols = 1 + len(snapshot_times)  # u0 + snapshots
    fig, axes = plt.subplots(
        n_plot,
        n_cols,
        figsize=(3.1 * n_cols, 2.6 * n_plot),
        squeeze=False,
        constrained_layout=True,
    )

    for row, sample_idx in enumerate(sample_indices):
        traj_i = u_traj[sample_idx]
        scale = max(float(torch.max(torch.abs(traj_i)).item()), 1e-8)
        eps_text = ""
        if eps is not None:
            eps_text = f", eps={float(eps[sample_idx].item()):.3f}"

        ax0 = axes[row, 0]
        ax0.imshow(
            traj_i[0].cpu().numpy(),
            origin="lower",
            cmap="coolwarm",
            vmin=-scale,
            vmax=scale,
            extent=[0.0, 1.0, 0.0, 1.0],
            aspect="auto",
        )
        ax0.set_title("u0")
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.set_ylabel(f"{split_name} #{sample_idx}{eps_text}")
        im_last = ax0.images[-1]

        for col, t_snap in enumerate(snapshot_times, start=1):
            frac = 0.0 if t_final <= 0 else float(t_snap) / float(t_final)
            k = int(round(frac * n_steps))
            k = max(0, min(n_steps, k))
            ax = axes[row, col]
            ax.imshow(
                traj_i[k].cpu().numpy(),
                origin="lower",
                cmap="coolwarm",
                vmin=-scale,
                vmax=scale,
                extent=[0.0, 1.0, 0.0, 1.0],
                aspect="auto",
            )
            ax.set_title(f"t={t_snap:.2f}")
            ax.set_xticks([])
            ax.set_yticks([])
            im_last = ax.images[-1]

        cbar = fig.colorbar(im_last, ax=axes[row, :], fraction=0.015, pad=0.01)
        cbar.ax.set_ylabel("u", rotation=90)

    folder = os.path.dirname(out_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig.suptitle("2D Cahn-Hilliard samples (Neumann BC)", fontsize=13)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved sample-grid plot: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cached 2D Cahn-Hilliard dataset splits")
    parser.add_argument("--n-x", type=int, default=32, help="Number of interior x-grid points")
    parser.add_argument("--n-y", type=int, default=32, help="Number of interior y-grid points")
    parser.add_argument("--n-steps", type=int, default=10, help="Number of macro time steps on [0,t_final]")
    parser.add_argument("--t-final", type=float, default=1.0, help="Final time horizon")
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epsilon-min", type=float, default=0.04)
    parser.add_argument("--epsilon-max", type=float, default=0.04)
    parser.add_argument("--mobility-min", type=float, default=1.0)
    parser.add_argument("--mobility-max", type=float, default=1.0)

    parser.add_argument("--u0-mean-min", type=float, default=-0.30)
    parser.add_argument("--u0-mean-max", type=float, default=0.30)
    parser.add_argument("--u0-fluctuation-amplitude", type=float, default=0.30)
    parser.add_argument("--u0-max-modes", type=int, default=8)
    parser.add_argument("--u0-noise-std", type=float, default=0.02)
    parser.add_argument("--norm-targeting", action="store_true")
    parser.add_argument("--target-u0-norm-min", type=float, default=0.20)
    parser.add_argument("--target-u0-norm-max", type=float, default=1.00)

    parser.add_argument("--cfl-nonlinear", type=float, default=0.20)
    parser.add_argument("--max-substeps-per-step", type=int, default=4000)
    parser.add_argument("--max-dt-substep", type=float, default=None)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--solver-float32", action="store_true", help="Use float32 solver instead of float64")

    parser.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/cahn_hilliard2d_l2_neumann_eps0p04_nx32_ny32_steps10.pt",
        help="Path to output dataset file (.pt)",
    )

    parser.add_argument("--plot-samples", action="store_true", help="Generate sample-grid visualization")
    parser.add_argument("--plot-split", type=str, default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--n-plot-samples", type=int, default=10)
    parser.add_argument("--snapshot-times", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--plot-dir", type=str, default="grad_flow_l2/outputs/data_samples")
    parser.add_argument("--plot-prefix", type=str, default="ch2d_samples")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    solver_dtype = torch.float32 if args.solver_float32 else torch.float64
    splits = generate_cahn_hilliard2d_dataset_splits(
        n_x=args.n_x,
        n_y=args.n_y,
        n_steps=args.n_steps,
        t_final=args.t_final,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        epsilon_min=args.epsilon_min,
        epsilon_max=args.epsilon_max,
        mobility_min=args.mobility_min,
        mobility_max=args.mobility_max,
        seed=args.seed,
        u0_mean_min=args.u0_mean_min,
        u0_mean_max=args.u0_mean_max,
        u0_fluctuation_amplitude=args.u0_fluctuation_amplitude,
        u0_max_modes=args.u0_max_modes,
        u0_noise_std=args.u0_noise_std,
        norm_targeting=bool(args.norm_targeting),
        target_u0_norm_range=(args.target_u0_norm_min, args.target_u0_norm_max),
        cfl_nonlinear=args.cfl_nonlinear,
        max_substeps_per_step=args.max_substeps_per_step,
        max_dt_substep=args.max_dt_substep,
        chunk_size=args.chunk_size,
        show_progress=(not args.no_progress),
        solver_dtype=solver_dtype,
        out_dtype=torch.float32,
    )

    out_dir = os.path.dirname(args.dataset_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_dataset_splits(splits, args.dataset_path)
    print(f"Saved Cahn-Hilliard dataset splits to: {args.dataset_path}")
    _print_split_stats(splits)

    if args.plot_samples:
        snapshot_times = _parse_snapshot_times(args.snapshot_times)
        split_names = ("train", "val", "test") if args.plot_split == "all" else (args.plot_split,)
        for split_name in split_names:
            out_path = os.path.join(args.plot_dir, f"{args.plot_prefix}_{split_name}.png")
            plot_sample_rows(
                split=splits[split_name],
                split_name=split_name,
                out_path=out_path,
                t_final=float(args.t_final),
                snapshot_times=snapshot_times,
                n_plot_samples=int(args.n_plot_samples),
            )


if __name__ == "__main__":
    main(parse_args())

