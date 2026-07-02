"""
Data generation for periodic 2D incompressible resistive MHD in ``(omega, a)`` form.

Dataset format:
    split["f"]      : (n_samples, n_x, n_y)
    split["u0"]     : (n_samples, 2, n_x, n_y)
    split["u_traj"] : (n_samples, n_steps+1, 2, n_x, n_y)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from ..heat_data import save_dataset_splits
    from ..ns2d_per.ns2d_data import sample_periodic_field_mixed_2d
    from ..ns2d_per.ns2d_solver import (
        sample_periodic_gaussian_field_2d,
        spectral_truncate_periodic_field_2d,
    )
    from .mhd_solver import (
        magnetic_potential_to_field_and_current,
        prepare_mhd2d_periodic_spectral_cache,
        project_zero_mean_2d,
        solve_mhd2d_trajectory_pseudospectral,
    )
except ImportError:
    from grad_flow_l2.heat_data import save_dataset_splits
    from grad_flow_l2.ns2d_per.ns2d_data import sample_periodic_field_mixed_2d
    from grad_flow_l2.ns2d_per.ns2d_solver import (
        sample_periodic_gaussian_field_2d,
        spectral_truncate_periodic_field_2d,
    )
    from grad_flow_l2.mhd2d.mhd_solver import (
        magnetic_potential_to_field_and_current,
        prepare_mhd2d_periodic_spectral_cache,
        project_zero_mean_2d,
        solve_mhd2d_trajectory_pseudospectral,
    )


DATASET_VERSION = 1
STATE_CHANNELS = 2
STATE_NAMES = ("omega", "a")


class MHD2DTrajectoryTensorDataset(Dataset):
    def __init__(self, f_data: torch.Tensor, u0_data: torch.Tensor, u_traj_data: torch.Tensor):
        if f_data.dim() != 3:
            raise ValueError("f_data must have shape (n_samples,n_x,n_y)")
        if u0_data.dim() != 4 or int(u0_data.shape[1]) != STATE_CHANNELS:
            raise ValueError("u0_data must have shape (n_samples,2,n_x,n_y)")
        if u_traj_data.dim() != 5 or int(u_traj_data.shape[2]) != STATE_CHANNELS:
            raise ValueError("u_traj_data must have shape (n_samples,K+1,2,n_x,n_y)")
        if (
            int(f_data.shape[0]) != int(u_traj_data.shape[0])
            or tuple(f_data.shape[1:]) != tuple(u_traj_data.shape[-2:])
            or int(u0_data.shape[0]) != int(u_traj_data.shape[0])
            or tuple(u0_data.shape[-2:]) != tuple(u_traj_data.shape[-2:])
        ):
            raise ValueError("inconsistent MHD2D dataset tensor shapes")
        self.f_data = f_data
        self.u0_data = u0_data
        self.u_traj_data = u_traj_data

    def __len__(self) -> int:
        return int(self.u0_data.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"f": self.f_data[idx], "u0": self.u0_data[idx], "u_traj": self.u_traj_data[idx]}


class MHD2DStepDataset(Dataset):
    def __init__(self, f_data: torch.Tensor, u_traj_data: torch.Tensor):
        if f_data.dim() != 3:
            raise ValueError("f_data must have shape (n_samples,n_x,n_y)")
        if u_traj_data.dim() != 5 or int(u_traj_data.shape[2]) != STATE_CHANNELS:
            raise ValueError("u_traj_data must have shape (n_samples,K+1,2,n_x,n_y)")
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


def build_mhd2d_step_dataset(split_or_dataset) -> MHD2DStepDataset:
    if isinstance(split_or_dataset, MHD2DTrajectoryTensorDataset):
        return MHD2DStepDataset(split_or_dataset.f_data, split_or_dataset.u_traj_data)
    if isinstance(split_or_dataset, dict):
        return MHD2DStepDataset(split_or_dataset["f"], split_or_dataset["u_traj"])
    raise TypeError("Expected split dict or MHD2DTrajectoryTensorDataset")


def build_mhd2d_trajectory_dataset_from_split(split: Dict[str, torch.Tensor]) -> MHD2DTrajectoryTensorDataset:
    return MHD2DTrajectoryTensorDataset(split["f"], split["u0"], split["u_traj"])


def _slice_split(data: Dict[str, torch.Tensor], start: int, end: int) -> Dict[str, torch.Tensor]:
    return {key: value[start:end].clone() for key, value in data.items()}


def _rescale_batch_rms(field: torch.Tensor, target_rms: float) -> torch.Tensor:
    rms = field.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-8)
    return float(target_rms) * field / rms


def sample_mhd2d_initial_conditions(
    n_x: int,
    n_y: int,
    n_samples: int,
    omega_rms: float,
    magnetic_field_rms: float,
    omega_spectrum_scale: float,
    omega_spectrum_shift: float,
    omega_spectrum_power: float,
    a_spectrum_scale: float,
    a_spectrum_shift: float,
    a_spectrum_power: float,
    device: str,
    dtype: torch.dtype,
    ic_mode: str = "potential_a",
) -> torch.Tensor:
    if ic_mode not in {"omega_a", "potential_a"}:
        raise ValueError("ic_mode must be one of {'omega_a', 'potential_a'}")
    omega_or_psi = sample_periodic_gaussian_field_2d(
        n_x=n_x,
        n_y=n_y,
        n_samples=n_samples,
        spectrum_scale=omega_spectrum_scale,
        spectrum_shift=omega_spectrum_shift,
        spectrum_power=omega_spectrum_power,
        zero_mean=True,
        device=device,
        dtype=dtype,
    )
    cache = prepare_mhd2d_periodic_spectral_cache(
        n_x=n_x,
        n_y=n_y,
        device=device,
        dtype=dtype,
        dealias_factor=1.0,
        include_dealias_cache=False,
    )
    omega_or_psi = project_zero_mean_2d(omega_or_psi)
    if ic_mode == "potential_a":
        psi_hat = torch.fft.fft2(omega_or_psi)
        omega = torch.fft.ifft2(cache["laplace_eigs"].unsqueeze(0) * psi_hat).real
    else:
        omega = omega_or_psi
    omega = _rescale_batch_rms(project_zero_mean_2d(omega), omega_rms)

    a = sample_periodic_gaussian_field_2d(
        n_x=n_x,
        n_y=n_y,
        n_samples=n_samples,
        spectrum_scale=a_spectrum_scale,
        spectrum_shift=a_spectrum_shift,
        spectrum_power=a_spectrum_power,
        zero_mean=True,
        device=device,
        dtype=dtype,
    )
    b_x, b_y, _ = magnetic_potential_to_field_and_current(a, cache)
    b_rms = (b_x.square() + b_y.square()).mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-8)
    a = project_zero_mean_2d(float(magnetic_field_rms) * a / b_rms)
    return torch.stack([omega, a], dim=1)


def _cap_linf(field: torch.Tensor, max_abs: float) -> torch.Tensor:
    if max_abs <= 0.0:
        return field
    scale = torch.clamp(float(max_abs) / field.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8), max=1.0)
    return field * scale


def generate_mhd2d_dataset_splits(
    n_x: int,
    n_y: int,
    n_steps: int,
    t_final: float,
    n_train: int,
    n_val: int,
    n_test: int,
    nu: float = 1e-3,
    eta: float = 1e-3,
    seed: int = 42,
    solver_n_x: int = 256,
    solver_n_y: int = 256,
    solver_dt: float = 1e-3,
    record_dt: float = 1e-2,
    warmup_time: float = 0.0,
    omega_rms: float = 0.5,
    magnetic_field_rms: float = 0.1,
    ic_mode: str = "potential_a",
    time_integrator: str = "rk4",
    dealias_factor: float = 1.5,
    omega_spectrum_scale: float = 8.0 ** 1.5,
    omega_spectrum_shift: float = 4.0,
    omega_spectrum_power: float = 2.5,
    a_spectrum_scale: float = 8.0 ** 1.5,
    a_spectrum_shift: float = 4.0,
    a_spectrum_power: float = 2.5,
    forcing_mode: str = "vorticity",
    f_grf_amplitude: float = 0.25,
    f_sinusoidal_amplitude: float = 0.50,
    f_sinusoidal_linf_min: float = 0.10,
    f_sinusoidal_linf_max: float = 0.20,
    f_sinusoidal_terms_min: int = 2,
    f_sinusoidal_terms_max: int = 6,
    f_max_abs: float = 0.25,
    f_grf_prob: float = 0.80,
    f_matern_prob: float = 0.0,
    f_length_scale_min: float = 0.05,
    f_length_scale_max: float = 0.15,
    f_max_modes: int = 3,
    f_allow_sinusoidal: bool = True,
    chunk_size: int = 64,
    show_progress: bool = False,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, Dict[str, torch.Tensor]]:
    if min(n_x, n_y, n_steps) < 1:
        raise ValueError("n_x, n_y, and n_steps must be positive")
    if min(nu, eta) < 0.0:
        raise ValueError("nu and eta must be >= 0")
    if min(t_final, solver_dt, record_dt) <= 0.0:
        raise ValueError("t_final, solver_dt, and record_dt must be > 0")
    if warmup_time < 0.0 or warmup_time >= t_final:
        raise ValueError("warmup_time must satisfy 0 <= warmup_time < t_final")
    if solver_n_x < n_x or solver_n_y < n_y:
        raise ValueError("solver grid must be at least as large as output grid")
    if forcing_mode not in {"zero", "vorticity"}:
        raise ValueError("forcing_mode must be one of {'zero', 'vorticity'}")
    if ic_mode not in {"omega_a", "potential_a"}:
        raise ValueError("ic_mode must be one of {'omega_a', 'potential_a'}")
    if time_integrator not in {"rk4", "cn", "crank_nicolson"}:
        raise ValueError("time_integrator must be one of {'rk4', 'cn', 'crank_nicolson'}")
    if dealias_factor < 1.0:
        raise ValueError("dealias_factor must be >= 1")

    warmup_records = int(round(float(warmup_time) / float(record_dt)))
    total_records = int(round(float(t_final) / float(record_dt)))
    if abs(float(warmup_time) - warmup_records * float(record_dt)) > 1e-10:
        raise ValueError("warmup_time must be an integer multiple of record_dt")
    if abs(float(t_final) - total_records * float(record_dt)) > 1e-10:
        raise ValueError("t_final must be an integer multiple of record_dt")
    if int(n_steps) != total_records - warmup_records:
        raise ValueError("n_steps must match (t_final - warmup_time) / record_dt")

    total = int(n_train + n_val + n_test)
    torch_state = torch.random.get_rng_state()
    np_state = np.random.get_state()
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_hr = sample_mhd2d_initial_conditions(
        n_x=solver_n_x,
        n_y=solver_n_y,
        n_samples=total,
        omega_rms=omega_rms,
        magnetic_field_rms=magnetic_field_rms,
        omega_spectrum_scale=omega_spectrum_scale,
        omega_spectrum_shift=omega_spectrum_shift,
        omega_spectrum_power=omega_spectrum_power,
        a_spectrum_scale=a_spectrum_scale,
        a_spectrum_shift=a_spectrum_shift,
        a_spectrum_power=a_spectrum_power,
        device=device,
        dtype=torch.float64,
        ic_mode=ic_mode,
    )

    if forcing_mode == "zero":
        f_hr = None
        f = torch.zeros(total, n_x, n_y, dtype=dtype)
    else:
        f_hr = sample_periodic_field_mixed_2d(
            n_x=solver_n_x,
            n_y=solver_n_y,
            n_samples=total,
            grf_amplitude=f_grf_amplitude,
            sinusoidal_amplitude=f_sinusoidal_amplitude,
            sinusoidal_linf_range=(f_sinusoidal_linf_min, f_sinusoidal_linf_max),
            sinusoidal_terms_range=(f_sinusoidal_terms_min, f_sinusoidal_terms_max),
            length_scale_range=(f_length_scale_min, f_length_scale_max),
            max_modes=f_max_modes,
            grf_prob=f_grf_prob,
            matern_prob=f_matern_prob,
            allow_sinusoidal=f_allow_sinusoidal,
            show_progress=show_progress,
            progress_desc="sample forcing",
            device=device,
        ).to(dtype=torch.float64)
        f_hr = _cap_linf(project_zero_mean_2d(f_hr), max_abs=f_max_abs)
        f = spectral_truncate_periodic_field_2d(f_hr, target_n_x=n_x, target_n_y=n_y).to(dtype=dtype)
        f = project_zero_mean_2d(f)

    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None
    starts = range(0, total, int(chunk_size))
    total_chunks = (total + int(chunk_size) - 1) // int(chunk_size)
    total_record_intervals = int(total) * int(total_records)
    chunk_bar = None
    record_bar = None
    if show_progress and tqdm is not None:
        chunk_bar = tqdm(total=total_chunks, desc="trajectory chunks", leave=True, dynamic_ncols=True)
        record_bar = tqdm(total=total_record_intervals, desc="record intervals", leave=True, dynamic_ncols=True)

    traj_chunks = []
    for start in starts:
        end = min(int(start) + int(chunk_size), total)
        traj_hr = solve_mhd2d_trajectory_pseudospectral(
            u0=state_hr[start:end],
            forcing=None if f_hr is None else f_hr[start:end],
            t_final=t_final,
            dt=solver_dt,
            record_dt=record_dt,
            nu=nu,
            eta=eta,
            progress_callback=(None if record_bar is None else lambda n, b=end-start: record_bar.update(int(n) * int(b))),
            time_integrator=time_integrator,
            dealias_factor=dealias_factor,
        )
        B, T, C, _, _ = traj_hr.shape
        traj_small = spectral_truncate_periodic_field_2d(
            traj_hr.reshape(B * T * C, solver_n_x, solver_n_y),
            target_n_x=n_x,
            target_n_y=n_y,
        ).reshape(B, T, C, n_x, n_y)
        traj_chunks.append(traj_small[:, warmup_records : warmup_records + n_steps + 1].to(dtype=dtype))
        if chunk_bar is not None:
            chunk_bar.update(1)
    if chunk_bar is not None:
        chunk_bar.close()
    if record_bar is not None:
        record_bar.close()
    u_traj = torch.cat(traj_chunks, dim=0)
    all_data = {"f": f.cpu(), "u0": u_traj[:, 0].clone().cpu(), "u_traj": u_traj.cpu()}

    train_end = int(n_train)
    val_end = int(n_train + n_val)
    splits = {
        "train": _slice_split(all_data, 0, train_end),
        "val": _slice_split(all_data, train_end, val_end),
        "test": _slice_split(all_data, val_end, total),
        "meta": {
            "dataset_version": DATASET_VERSION,
            "equation": "incompressible_resistive_mhd_2d_omega_a_periodic",
            "domain": "unit_torus",
            "periodic": True,
            "state_channels": STATE_CHANNELS,
            "state_names": list(STATE_NAMES),
            "n_x": int(n_x),
            "n_y": int(n_y),
            "solver_n_x": int(solver_n_x),
            "solver_n_y": int(solver_n_y),
            "n_steps": int(n_steps),
            "t_final": float(t_final),
            "record_dt": float(record_dt),
            "solver_dt": float(solver_dt),
            "time_integrator": str(time_integrator),
            "dealias_factor": float(dealias_factor),
            "warmup_time": float(warmup_time),
            "stored_t_start": float(warmup_time),
            "stored_t_final": float(t_final),
            "nu": float(nu),
            "eta": float(eta),
            "forcing_mode": str(forcing_mode),
            "omega_rms": float(omega_rms),
            "magnetic_field_rms": float(magnetic_field_rms),
            "ic_mode": str(ic_mode),
            "omega_spectrum_scale": float(omega_spectrum_scale),
            "omega_spectrum_shift": float(omega_spectrum_shift),
            "omega_spectrum_power": float(omega_spectrum_power),
            "a_spectrum_scale": float(a_spectrum_scale),
            "a_spectrum_shift": float(a_spectrum_shift),
            "a_spectrum_power": float(a_spectrum_power),
            "f_max_abs": float(f_max_abs),
            "n_train": int(n_train),
            "n_val": int(n_val),
            "n_test": int(n_test),
            "seed": int(seed),
            "chunk_size": int(chunk_size),
            "device": str(device),
        },
    }
    torch.random.set_rng_state(torch_state)
    np.random.set_state(np_state)
    return splits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate periodic 2D incompressible resistive MHD dataset splits")
    p.add_argument("--n-x", type=int, default=128)
    p.add_argument("--n-y", type=int, default=128)
    p.add_argument("--solver-n-x", type=int, default=128)
    p.add_argument("--solver-n-y", type=int, default=128)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--t-final", type=float, default=1.0)
    p.add_argument("--record-dt", type=float, default=1e-2)
    p.add_argument("--solver-dt", type=float, default=1e-3)
    p.add_argument("--warmup-time", type=float, default=0.0)
    p.add_argument("--nu", type=float, default=1e-3)
    p.add_argument("--eta", type=float, default=1e-3)
    p.add_argument("--omega-rms", type=float, default=0.5)
    p.add_argument("--magnetic-field-rms", type=float, default=0.1)
    p.add_argument("--ic-mode", choices=["omega_a", "potential_a"], default="potential_a")
    p.add_argument("--time-integrator", choices=["rk4", "cn", "crank_nicolson"], default="rk4")
    p.add_argument("--dealias-factor", type=float, default=1.5)
    p.add_argument("--omega-spectrum-scale", type=float, default=8.0 ** 1.5)
    p.add_argument("--omega-spectrum-shift", type=float, default=4.0)
    p.add_argument("--omega-spectrum-power", type=float, default=2.5)
    p.add_argument("--a-spectrum-scale", type=float, default=8.0 ** 1.5)
    p.add_argument("--a-spectrum-shift", type=float, default=4.0)
    p.add_argument("--a-spectrum-power", type=float, default=2.5)
    p.add_argument("--forcing-mode", choices=["zero", "vorticity"], default="vorticity")
    p.add_argument("--f-grf-amplitude", type=float, default=0.25)
    p.add_argument("--f-sinusoidal-amplitude", type=float, default=0.50)
    p.add_argument("--f-sinusoidal-linf-min", type=float, default=0.10)
    p.add_argument("--f-sinusoidal-linf-max", type=float, default=0.20)
    p.add_argument("--f-sinusoidal-terms-min", type=int, default=2)
    p.add_argument("--f-sinusoidal-terms-max", type=int, default=6)
    p.add_argument("--f-max-abs", type=float, default=0.25)
    p.add_argument("--f-grf-prob", type=float, default=0.80)
    p.add_argument("--f-matern-prob", type=float, default=0.0)
    p.add_argument("--f-length-scale-min", type=float, default=0.05)
    p.add_argument("--f-length-scale-max", type=float, default=0.15)
    p.add_argument("--f-max-modes", type=int, default=3)
    p.add_argument("--f-allow-sinusoidal", dest="f_allow_sinusoidal", action="store_true")
    p.add_argument("--f-no-sinusoidal", dest="f_allow_sinusoidal", action="store_false")
    p.set_defaults(f_allow_sinusoidal=True)
    p.add_argument("--n-train", type=int, default=1600)
    p.add_argument("--n-val", type=int, default=400)
    p.add_argument("--n-test", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--dataset-path", type=str, default="grad_flow_l2/mhd2d/datasets/mhd2d_train2000_t1_rk4_dt1e-3_outdt1e-2.pt")
    p.add_argument("--settings-path", type=str, default=None)
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    splits = generate_mhd2d_dataset_splits(
        n_x=args.n_x,
        n_y=args.n_y,
        n_steps=args.n_steps,
        t_final=args.t_final,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        nu=args.nu,
        eta=args.eta,
        seed=args.seed,
        solver_n_x=args.solver_n_x,
        solver_n_y=args.solver_n_y,
        solver_dt=args.solver_dt,
        record_dt=args.record_dt,
        warmup_time=args.warmup_time,
        omega_rms=args.omega_rms,
        magnetic_field_rms=args.magnetic_field_rms,
        ic_mode=args.ic_mode,
        time_integrator=args.time_integrator,
        dealias_factor=args.dealias_factor,
        omega_spectrum_scale=args.omega_spectrum_scale,
        omega_spectrum_shift=args.omega_spectrum_shift,
        omega_spectrum_power=args.omega_spectrum_power,
        a_spectrum_scale=args.a_spectrum_scale,
        a_spectrum_shift=args.a_spectrum_shift,
        a_spectrum_power=args.a_spectrum_power,
        forcing_mode=args.forcing_mode,
        f_grf_amplitude=args.f_grf_amplitude,
        f_sinusoidal_amplitude=args.f_sinusoidal_amplitude,
        f_sinusoidal_linf_min=args.f_sinusoidal_linf_min,
        f_sinusoidal_linf_max=args.f_sinusoidal_linf_max,
        f_sinusoidal_terms_min=args.f_sinusoidal_terms_min,
        f_sinusoidal_terms_max=args.f_sinusoidal_terms_max,
        f_max_abs=args.f_max_abs,
        f_grf_prob=args.f_grf_prob,
        f_matern_prob=args.f_matern_prob,
        f_length_scale_min=args.f_length_scale_min,
        f_length_scale_max=args.f_length_scale_max,
        f_max_modes=args.f_max_modes,
        f_allow_sinusoidal=args.f_allow_sinusoidal,
        chunk_size=args.chunk_size,
        show_progress=(not args.no_progress),
        device=args.device,
    )
    out_dir = os.path.dirname(args.dataset_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_dataset_splits(splits, args.dataset_path)
    print(f"Saved periodic MHD2D dataset splits to: {args.dataset_path}")
    if args.settings_path:
        settings_dir = os.path.dirname(args.settings_path)
        if settings_dir:
            os.makedirs(settings_dir, exist_ok=True)
        with open(args.settings_path, "w", encoding="utf-8") as f:
            json.dump({"cli_args": vars(args), "meta": splits["meta"]}, f, indent=2, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main(parse_args())

