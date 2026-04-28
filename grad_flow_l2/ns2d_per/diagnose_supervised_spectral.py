"""Spectral diagnosis for a supervised periodic Navier-Stokes checkpoint.

This script focuses on the OOD comparison samples saved under
``outputs_supervised/ood_eval/test_sample_comparisons`` and reports:
  - per-sample rollout error,
  - radial spectral error curves,
  - PCA / KL coordinates of the forcing fields,
  - group statistics for the better vs worse samples.
"""

from __future__ import annotations

import argparse
import json
import os
from argparse import Namespace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

try:
    from ..heat_data import load_dataset_splits
    from .eval import _build_model, _load_checkpoint, _rollout
except ImportError:
    from grad_flow_l2.heat_data import load_dataset_splits
    from grad_flow_l2.ns2d_per.eval import _build_model, _load_checkpoint, _rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose supervised periodic NS model with spectral tools")
    parser.add_argument(
        "--summary-path",
        type=str,
        default="grad_flow_l2/ns2d_per/outputs_supervised/ood_eval/test_summary.json",
        help="Path to the evaluation summary JSON produced by eval.py.",
    )
    parser.add_argument(
        "--sample-dir",
        type=str,
        default=None,
        help="Directory containing sample_*_comparison.png files. Defaults to the sibling sample_comparisons folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write diagnosis plots and CSVs. Defaults to a spectral_diagnostics folder next to summary.",
    )
    parser.add_argument("--n-selected", type=int, default=20, help="Number of comparison samples to diagnose.")
    parser.add_argument(
        "--n-curve-samples",
        type=int,
        default=12,
        help="Number of representative trajectories to overlay per group in the invariant curves.",
    )
    parser.add_argument(
        "--use-full-split",
        action="store_true",
        help="Ignore the comparison thumbnails and diagnose every sample in the OOD test split.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for rollout-based diagnostics.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    return parser.parse_args()


def _load_json(path: str | os.PathLike[str]) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_sample_ids(sample_dir: Path) -> List[int]:
    ids: List[int] = []
    for path in sorted(sample_dir.glob("sample_*_comparison.png")):
        parts = path.stem.split("_")
        if len(parts) < 3:
            continue
        try:
            ids.append(int(parts[1]))
        except ValueError:
            continue
    return ids


def _flatten_forcing_matrix(forcing: np.ndarray) -> np.ndarray:
    return forcing.reshape(forcing.shape[0], -1)


def _radial_shell_indices(n_x: int, n_y: int) -> Tuple[np.ndarray, np.ndarray]:
    kx = np.fft.fftfreq(n_x, d=1.0 / float(n_x))
    ky = np.fft.fftfreq(n_y, d=1.0 / float(n_y))
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    radius = np.sqrt(kx_grid * kx_grid + ky_grid * ky_grid)
    shell_idx = np.floor(radius + 1e-12).astype(np.int64)
    shell_vals = np.arange(int(shell_idx.max()) + 1, dtype=np.int64)
    return shell_idx, shell_vals


def _shell_sum(power: np.ndarray, shell_idx: np.ndarray, n_shells: int) -> np.ndarray:
    flat_power = power.reshape(-1)
    flat_shell = shell_idx.reshape(-1)
    return np.bincount(flat_shell, weights=flat_power, minlength=n_shells)


def _shell_sums_over_batch(power: np.ndarray, shell_idx: np.ndarray) -> np.ndarray:
    """Return shell sums for an array shaped (B, T, X, Y)."""
    b, t = power.shape[:2]
    n_shells = int(shell_idx.max()) + 1
    out = np.zeros((b, t, n_shells), dtype=np.float64)
    for i in range(b):
        for j in range(t):
            out[i, j] = _shell_sum(power[i, j], shell_idx, n_shells)
    return out


def _enstrophy_curve(field: np.ndarray, area: float) -> np.ndarray:
    """Return area * sum(omega^2) over the spatial dimensions."""
    return area * np.sum(field * field, axis=(-2, -1))


def _palinstrophy_curve(field: np.ndarray, area: float, n_x: int, n_y: int) -> np.ndarray:
    """Return area * sum(|grad omega|^2) using spectral derivatives."""
    kx = 2.0 * np.pi * np.fft.fftfreq(n_x, d=1.0 / float(n_x))
    ky = 2.0 * np.pi * np.fft.fftfreq(n_y, d=1.0 / float(n_y))
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    omega_hat = np.fft.fft2(field, axes=(-2, -1))
    omega_x = np.fft.ifft2(1j * kx_grid * omega_hat, axes=(-2, -1)).real
    omega_y = np.fft.ifft2(1j * ky_grid * omega_hat, axes=(-2, -1)).real
    return area * np.sum(omega_x * omega_x + omega_y * omega_y, axis=(-2, -1))


def _laplacian_norm_curve(field: np.ndarray, area: float, n_x: int, n_y: int) -> np.ndarray:
    """Return the L2 norm of the Laplacian of omega on each time slice."""
    kx = 2.0 * np.pi * np.fft.fftfreq(n_x, d=1.0 / float(n_x))
    ky = 2.0 * np.pi * np.fft.fftfreq(n_y, d=1.0 / float(n_y))
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    lap_eigs = -(kx_grid * kx_grid + ky_grid * ky_grid)
    omega_hat = np.fft.fft2(field, axes=(-2, -1))
    lap_omega = np.fft.ifft2(lap_eigs * omega_hat, axes=(-2, -1)).real
    return np.sqrt(area * np.sum(lap_omega * lap_omega, axis=(-2, -1)))


def _group_mean(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return np.zeros(arr.shape[1:], dtype=arr.dtype)
    return arr[mask].mean(axis=0)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    sx = float(np.sqrt(np.sum(x * x)))
    sy = float(np.sqrt(np.sum(y * y)))
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return float(np.sum(x * y) / (sx * sy))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _pick_representative_indices(mask: np.ndarray, values: np.ndarray, n_pick: int) -> List[int]:
    """Pick evenly spaced samples within a masked group, sorted by the given values."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    idx = idx[np.argsort(values[idx])]
    if idx.size <= n_pick:
        return idx.tolist()
    pick_pos = np.linspace(0, idx.size - 1, num=n_pick, dtype=int)
    return idx[pick_pos].tolist()


def _remap_checkpoint_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Compatibility shim for older energy-head parameter names."""
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("energy_head.backbone."):
            new_key = key.replace("energy_head.backbone.", "energy_head.local_backbone.", 1)
        elif key.startswith("energy_head.density_head."):
            new_key = key.replace("energy_head.density_head.", "energy_head.local_density_head.", 1)
        remapped[new_key] = value
    return remapped


def main(args: argparse.Namespace) -> None:
    summary_path = Path(args.summary_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    summary = _load_json(summary_path)
    checkpoint_path = Path(summary["checkpoint_path"])
    dataset_path = Path(summary["dataset_path"])
    if args.output_dir is None:
        output_dir = summary_path.parent / ("spectral_diagnostics_full" if args.use_full_split else "spectral_diagnostics")
    else:
        output_dir = Path(args.output_dir)
    _ensure_dir(output_dir)

    run_args_path = checkpoint_path.parent / "args.json"
    run_args = _load_json(run_args_path)
    splits = load_dataset_splits(str(dataset_path), map_location="cpu")
    split = splits["test"]
    meta = splits.get("meta", {})
    n_x = int(split["u0"].shape[1])
    n_y = int(split["u0"].shape[2])
    n_steps = int(split["u_traj"].shape[1] - 1)
    t_final = float(meta.get("t_final", 1.0))
    dt = t_final / float(n_steps)
    area = (1.0 / float(n_x)) * (1.0 / float(n_y))
    total_samples = int(split["u0"].shape[0])
    if args.use_full_split:
        sample_ids = list(range(total_samples))
        print(f"Using full OOD split with {len(sample_ids)} samples")
    else:
        if args.sample_dir is None:
            sample_dir = summary_path.parent / "test_sample_comparisons"
        else:
            sample_dir = Path(args.sample_dir)
        sample_ids = _load_sample_ids(sample_dir)
        if not sample_ids:
            raise RuntimeError(f"No comparison samples found in {sample_dir}")
        sample_ids = sample_ids[: max(1, min(int(args.n_selected), len(sample_ids)))]
        print(f"Loaded {len(sample_ids)} comparison samples from {sample_dir}")
    print(f"Dataset: {dataset_path}")
    print(f"Checkpoint: {checkpoint_path}")

    build_args = Namespace(
        prox_simulator_type=run_args.get("prox_simulator_type", "fno"),
        hidden_channels=int(run_args.get("hidden_channels", 64)),
        latent_channels=int(run_args.get("latent_channels", 16)),
        enc_blocks=int(run_args.get("enc_blocks", 4)),
        dec_blocks=int(run_args.get("dec_blocks", 4)),
        prox_blocks=int(run_args.get("prox_blocks", 6)),
        fno_modes_x=int(run_args.get("fno_modes_x", 16)),
        fno_modes_y=int(run_args.get("fno_modes_y", 16)),
        disable_fno_grid=bool(run_args.get("disable_fno_grid", False)),
        energy_layers=int(run_args.get("energy_layers", 4)),
        energy_head_type=str(run_args.get("energy_head_type", "local")),
        energy_fno_modes_x=int(run_args.get("energy_fno_modes_x", 16)),
        energy_fno_modes_y=int(run_args.get("energy_fno_modes_y", 16)),
        use_dt_channel=bool(run_args.get("use_dt_channel", False)),
        disable_forcing_channel=bool(run_args.get("disable_forcing_channel", False)),
        disable_z_grad_feature=bool(run_args.get("disable_z_grad_feature", False)),
        disable_u_grad_feature=bool(run_args.get("disable_u_grad_feature", False)),
    )

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(n_x=n_x, n_y=n_y, h_x=1.0 / float(n_x), h_y=1.0 / float(n_y), dt=dt, args=build_args).to(device)
    ckpt = _load_checkpoint(str(checkpoint_path), map_location=device)
    model.load_state_dict(_remap_checkpoint_state_dict(ckpt["model_state_dict"]), strict=True)
    model.eval()

    delta_clip = float(summary.get("delta_clip", 10.0))
    energy_accept_reject = bool(summary.get("energy_accept_reject", False))
    energy_reject_factor = float(summary.get("energy_reject_factor", 1.10))
    energy_reject_margin = float(summary.get("energy_reject_margin", 1e-6))
    energy_fallback_mode = str(summary.get("energy_fallback_mode", "prev_delta"))

    print(f"Device: {device}")
    print(f"Model dt inferred from data: {dt:.6f}")
    print(f"Energy gate: {energy_accept_reject}")

    u0 = split["u0"]
    f = split["f"]
    u_traj = split["u_traj"]
    batch_size = max(1, int(args.batch_size))
    n = n_x * n_y
    shell_idx, shell_vals = _radial_shell_indices(n_x, n_y)
    n_shells = int(shell_idx.max()) + 1

    rel_curve_batches: List[np.ndarray] = []
    enstrophy_ref_batches: List[np.ndarray] = []
    enstrophy_pred_batches: List[np.ndarray] = []
    palinstrophy_ref_batches: List[np.ndarray] = []
    palinstrophy_pred_batches: List[np.ndarray] = []
    lapnorm_ref_batches: List[np.ndarray] = []
    lapnorm_pred_batches: List[np.ndarray] = []
    shell_ref_batches: List[np.ndarray] = []
    shell_pred_batches: List[np.ndarray] = []
    shell_err_batches: List[np.ndarray] = []

    n_batches = (len(sample_ids) + batch_size - 1) // batch_size
    for batch_idx, batch_start in enumerate(range(0, len(sample_ids), batch_size), start=1):
        batch_ids = sample_ids[batch_start : batch_start + batch_size]
        print(f"Processing batch {batch_idx}/{n_batches} ({batch_start + 1}-{batch_start + len(batch_ids)} of {len(sample_ids)})")

        u0_b = u0[batch_ids].to(device)
        f_b = f[batch_ids].to(device)
        u_ref_b = u_traj[batch_ids]
        rollout_b = _rollout(
            model=model,
            u0=u0_b,
            f=f_b,
            n_steps=n_steps,
            dt=dt,
            delta_clip=delta_clip,
            energy_accept_reject=energy_accept_reject,
            energy_reject_factor=energy_reject_factor,
            energy_reject_margin=energy_reject_margin,
            energy_fallback_mode=energy_fallback_mode,
            return_stats=False,
        ).detach().cpu()

        u_ref_b = u_ref_b.cpu()
        diff = rollout_b - u_ref_b
        num = torch.sqrt(area * torch.sum(diff * diff, dim=(-2, -1)))
        den = torch.sqrt(area * torch.sum(u_ref_b * u_ref_b, dim=(-2, -1)))
        rel_b = (num / (den + 1e-8)).numpy()
        rel_curve_batches.append(rel_b)

        rollout_np = rollout_b.numpy()
        ref_np = u_ref_b.numpy()
        enstrophy_ref_batches.append(_enstrophy_curve(ref_np, area=area))
        enstrophy_pred_batches.append(_enstrophy_curve(rollout_np, area=area))
        palinstrophy_ref_batches.append(_palinstrophy_curve(ref_np, area=area, n_x=n_x, n_y=n_y))
        palinstrophy_pred_batches.append(_palinstrophy_curve(rollout_np, area=area, n_x=n_x, n_y=n_y))
        lapnorm_ref_batches.append(_laplacian_norm_curve(ref_np, area=area, n_x=n_x, n_y=n_y))
        lapnorm_pred_batches.append(_laplacian_norm_curve(rollout_np, area=area, n_x=n_x, n_y=n_y))

        fft_pred = np.fft.fft2(rollout_np, axes=(-2, -1))
        fft_ref = np.fft.fft2(ref_np, axes=(-2, -1))
        power_pred = np.abs(fft_pred) ** 2 / float(n * n)
        power_ref = np.abs(fft_ref) ** 2 / float(n * n)
        power_err = np.abs(fft_pred - fft_ref) ** 2 / float(n * n)
        shell_pred_batches.append(_shell_sums_over_batch(power_pred, shell_idx))
        shell_ref_batches.append(_shell_sums_over_batch(power_ref, shell_idx))
        shell_err_batches.append(_shell_sums_over_batch(power_err, shell_idx))

    rel_curve_arr = np.concatenate(rel_curve_batches, axis=0)
    enstrophy_ref_arr = np.concatenate(enstrophy_ref_batches, axis=0)
    enstrophy_pred_arr = np.concatenate(enstrophy_pred_batches, axis=0)
    palinstrophy_ref_arr = np.concatenate(palinstrophy_ref_batches, axis=0)
    palinstrophy_pred_arr = np.concatenate(palinstrophy_pred_batches, axis=0)
    lapnorm_ref_arr = np.concatenate(lapnorm_ref_batches, axis=0)
    lapnorm_pred_arr = np.concatenate(lapnorm_pred_batches, axis=0)
    shell_pred = np.concatenate(shell_pred_batches, axis=0)
    shell_ref = np.concatenate(shell_ref_batches, axis=0)
    shell_err = np.concatenate(shell_err_batches, axis=0)

    rel_final_arr = rel_curve_arr[:, -1]
    rel_mean_arr = rel_curve_arr.mean(axis=1)
    enstrophy_ref_final = enstrophy_ref_arr[:, -1]
    enstrophy_pred_final = enstrophy_pred_arr[:, -1]
    palinstrophy_ref_final = palinstrophy_ref_arr[:, -1]
    palinstrophy_pred_final = palinstrophy_pred_arr[:, -1]
    lapnorm_ref_final = lapnorm_ref_arr[:, -1]
    lapnorm_pred_final = lapnorm_pred_arr[:, -1]

    order = np.argsort(rel_final_arr)
    if args.use_full_split:
        q25, q75 = np.quantile(rel_final_arr, [0.25, 0.75])
        good_mask = rel_final_arr <= q25
        bad_mask = rel_final_arr >= q75
        good_label = "best quartile"
        bad_label = "worst quartile"
    else:
        median_rel = float(np.median(rel_final_arr))
        good_mask = rel_final_arr <= median_rel
        bad_mask = ~good_mask
        good_label = "good"
        bad_label = "bad"
    good_ids = [sample_ids[i] for i in np.where(good_mask)[0].tolist()]
    bad_ids = [sample_ids[i] for i in np.where(bad_mask)[0].tolist()]

    print("\nSamples sorted by final relative L2:")
    for rank in order[: min(len(order), 20)]:
        print(
            f"  sample {sample_ids[rank]:04d}: final_rel={rel_final_arr[rank]:.4e}, "
            f"mean_rel={rel_mean_arr[rank]:.4e}"
        )
    if args.use_full_split:
        print(f"Bottom quartile threshold: {q25:.4e}")
        print(f"Top quartile threshold: {q75:.4e}")
        print("Top 20 worst samples:")
        for rank in order[-20:][::-1]:
            print(
                f"  sample {sample_ids[rank]:04d}: final_rel={rel_final_arr[rank]:.4e}, "
                f"mean_rel={rel_mean_arr[rank]:.4e}"
            )
    else:
        print(f"Median final relative L2 among the selected samples: {median_rel:.4e}")
    if args.use_full_split:
        print(f"{good_label.capitalize()} count: {len(good_ids)}")
        print(f"{bad_label.capitalize()} count: {len(bad_ids)}")
    else:
        print(f"{good_label.capitalize()} samples: {good_ids[:20]}{' ...' if len(good_ids) > 20 else ''}")
        print(f"{bad_label.capitalize()} samples:  {bad_ids[:20]}{' ...' if len(bad_ids) > 20 else ''}")

    eps = 1e-14
    rel_shell_err = np.sqrt(shell_err / (shell_ref + eps))
    high_k = shell_vals >= 12
    low_k = (shell_vals > 0) & (shell_vals <= 3)

    good_rel_shell = rel_shell_err[good_mask]
    bad_rel_shell = rel_shell_err[bad_mask]
    good_err_final = shell_err[good_mask, -1]
    bad_err_final = shell_err[bad_mask, -1]
    good_ref_final = shell_ref[good_mask, -1]
    bad_ref_final = shell_ref[bad_mask, -1]
    good_pred_final = shell_pred[good_mask, -1]
    bad_pred_final = shell_pred[bad_mask, -1]

    def _hi_lo_fraction(shell_power: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        total = shell_power.sum(axis=-1)
        hi = shell_power[..., high_k].sum(axis=-1)
        lo = shell_power[..., low_k].sum(axis=-1)
        return hi / (total + eps), lo / (total + eps)

    state_hi_frac, state_lo_frac = _hi_lo_fraction(shell_ref[:, -1])
    state_err_hi_frac, state_err_lo_frac = _hi_lo_fraction(shell_err[:, -1])

    forcing_arr = split["f"].numpy()
    forcing_fft = np.fft.fft2(forcing_arr, axes=(-2, -1))
    forcing_power = np.abs(forcing_fft) ** 2 / float(n * n)
    forcing_shell = _shell_sums_over_batch(forcing_power[:, None, ...], shell_idx)[:, 0]
    forcing_hi_frac, forcing_lo_frac = _hi_lo_fraction(forcing_shell)
    forcing_hi_frac_sel = forcing_hi_frac[sample_ids]
    forcing_lo_frac_sel = forcing_lo_frac[sample_ids]

    print("\nGroup statistics on the selected samples:")
    print(f"  good mean final_rel = {float(rel_final_arr[good_mask].mean()):.4e}")
    print(f"  bad  mean final_rel = {float(rel_final_arr[bad_mask].mean()):.4e}")
    print(f"  good mean high-k forcing fraction = {float(forcing_hi_frac_sel[good_mask].mean()):.4e}")
    print(f"  bad  mean high-k forcing fraction = {float(forcing_hi_frac_sel[bad_mask].mean()):.4e}")
    print(f"  corr(final_rel, state_hi_k_fraction) = {_safe_corr(rel_final_arr, state_hi_frac):.4f}")
    print(f"  corr(final_rel, forcing_hi_k_fraction) = {_safe_corr(rel_final_arr, forcing_hi_frac_sel):.4f}")
    print(f"  corr(final_rel, forcing_low_k_fraction) = {_safe_corr(rel_final_arr, forcing_lo_frac_sel):.4f}")
    print(f"  corr(final_rel, state_hi_k_error_fraction) = {_safe_corr(rel_final_arr, state_err_hi_frac):.4f}")
    print(f"  corr(final_rel, ref_enstrophy_final) = {_safe_corr(rel_final_arr, enstrophy_ref_final):.4f}")
    print(f"  corr(final_rel, pred_enstrophy_final) = {_safe_corr(rel_final_arr, enstrophy_pred_final):.4f}")
    print(f"  corr(final_rel, ref_palinstrophy_final) = {_safe_corr(rel_final_arr, palinstrophy_ref_final):.4f}")
    print(f"  corr(final_rel, pred_palinstrophy_final) = {_safe_corr(rel_final_arr, palinstrophy_pred_final):.4f}")
    print(f"  corr(final_rel, ref_lapnorm_final) = {_safe_corr(rel_final_arr, lapnorm_ref_final):.4f}")
    print(f"  corr(final_rel, pred_lapnorm_final) = {_safe_corr(rel_final_arr, lapnorm_pred_final):.4f}")

    forcing_flat = _flatten_forcing_matrix(forcing_arr)
    forcing_mean = forcing_flat.mean(axis=0, keepdims=True)
    forcing_centered = forcing_flat - forcing_mean
    u_svd, s_svd, vt_svd = np.linalg.svd(forcing_centered, full_matrices=False)
    n_modes = min(6, vt_svd.shape[0])
    pcs = vt_svd[:n_modes]
    scores = forcing_centered[sample_ids] @ pcs.T

    selected_scores = scores
    good_scores = selected_scores[good_mask]
    bad_scores = selected_scores[bad_mask]
    score_corr = [_safe_corr(rel_final_arr, selected_scores[:, i]) for i in range(n_modes)]
    print("\nPCA / KL forcing diagnostics:")
    print("  explained variance ratios:", np.round((s_svd[:n_modes] ** 2) / np.sum(s_svd**2), 5).tolist())
    print("  correlation(final_rel, PC scores):", [round(x, 4) for x in score_corr])

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    full_mode = bool(args.use_full_split)

    # Spectral error figure.
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.5), constrained_layout=True)
    ax = axes[0, 0]
    if full_mode:
        ax.semilogy(shell_vals[1:], rel_shell_err.mean(axis=0)[-1, 1:], color="tab:gray", lw=2.0, label="all mean")
        ax.semilogy(shell_vals[1:], rel_shell_err[good_mask].mean(axis=0)[-1, 1:], color="tab:green", lw=2.5, label="best quartile")
        ax.semilogy(shell_vals[1:], rel_shell_err[bad_mask].mean(axis=0)[-1, 1:], color="tab:red", lw=2.5, label="worst quartile")
        ax.set_title("Final-time shell-wise spectral error")
    else:
        for i, idx in enumerate(sample_ids):
            color = "tab:green" if good_mask[i] else "tab:red"
            alpha = 0.55 if good_mask[i] else 0.75
            ax.semilogy(shell_vals[1:], rel_shell_err[i, -1, 1:], color=color, alpha=alpha, lw=1.4)
        ax.semilogy(shell_vals[1:], good_rel_shell[:, -1, 1:].mean(axis=0), color="tab:green", lw=2.5, label="good mean")
        ax.semilogy(shell_vals[1:], bad_rel_shell[:, -1, 1:].mean(axis=0), color="tab:red", lw=2.5, label="bad mean")
        ax.set_title("Final-time shell-wise spectral error")
    ax.set_xlabel("radial wavenumber k")
    ax.set_ylabel(r"$\sqrt{E_{err}(k)/E_{true}(k)}$")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    if full_mode:
        ax.semilogy(shell_vals[1:], rel_shell_err.mean(axis=(0, 1))[1:], color="tab:gray", lw=2.0, label="all mean")
        ax.semilogy(shell_vals[1:], rel_shell_err[good_mask].mean(axis=(0, 1))[1:], color="tab:green", lw=2.5, label="best quartile")
        ax.semilogy(shell_vals[1:], rel_shell_err[bad_mask].mean(axis=(0, 1))[1:], color="tab:red", lw=2.5, label="worst quartile")
        ax.set_title("Time-averaged spectral error")
    else:
        for i, idx in enumerate(sample_ids):
            color = "tab:green" if good_mask[i] else "tab:red"
            alpha = 0.55 if good_mask[i] else 0.75
            ax.semilogy(shell_vals[1:], rel_shell_err[i, 1:, 1:].mean(axis=0), color=color, alpha=alpha, lw=1.4)
        ax.semilogy(shell_vals[1:], good_rel_shell[:, 1:, 1:].mean(axis=(0, 1)), color="tab:green", lw=2.5, label="good mean")
        ax.semilogy(shell_vals[1:], bad_rel_shell[:, 1:, 1:].mean(axis=(0, 1)), color="tab:red", lw=2.5, label="bad mean")
        ax.set_title("Time-averaged spectral error")
    ax.set_xlabel("radial wavenumber k")
    ax.set_ylabel(r"$\sqrt{E_{err}(k)/E_{true}(k)}$")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    if full_mode:
        sc = ax.scatter(
            selected_scores[:, 0],
            selected_scores[:, 1],
            c=np.log10(rel_final_arr + 1e-12),
            cmap="viridis",
            s=18,
            alpha=0.9,
        )
        ax.scatter(good_scores[:, 0], good_scores[:, 1], facecolors="none", edgecolors="tab:green", s=40, lw=1.2, label="best quartile")
        ax.scatter(bad_scores[:, 0], bad_scores[:, 1], facecolors="none", edgecolors="tab:red", s=40, lw=1.2, label="worst quartile")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label="log10 final rel L2")
        ax.set_title("Forcing PCA / KL coordinates")
    else:
        ax.scatter(good_scores[:, 0], good_scores[:, 1], c=rel_final_arr[good_mask], cmap="viridis", s=75, marker="o", edgecolor="k", label="good")
        ax.scatter(bad_scores[:, 0], bad_scores[:, 1], c=rel_final_arr[bad_mask], cmap="viridis", s=85, marker="s", edgecolor="k", label="bad")
        for i, idx in enumerate(sample_ids):
            ax.text(selected_scores[i, 0], selected_scores[i, 1], str(idx), fontsize=8, alpha=0.8)
        ax.set_title("Forcing PCA / KL coordinates")
    ax.set_xlabel("PC1 score")
    ax.set_ylabel("PC2 score")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    if n_modes >= 1:
        pc1 = pcs[0].reshape(n_x, n_y)
        im = ax.imshow(pc1, origin="lower", cmap="coolwarm", extent=[0.0, 1.0, 0.0, 1.0], aspect="auto")
        ax.set_title("KL mode 1")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    else:
        ax.axis("off")

    fig.suptitle("Spectral and forcing diagnostics for selected OOD samples", fontsize=14)
    spectral_path = output_dir / "spectral_error_and_forcing_pca.png"
    fig.savefig(spectral_path, dpi=180)
    plt.close(fig)

    # Forcing mean / group comparison figure.
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.0), constrained_layout=True)
    mean_forcing = forcing_mean.reshape(n_x, n_y)
    good_mean_forcing = forcing_flat[sample_ids][good_mask].mean(axis=0).reshape(n_x, n_y)
    bad_mean_forcing = forcing_flat[sample_ids][bad_mask].mean(axis=0).reshape(n_x, n_y)
    diff_forcing = bad_mean_forcing - good_mean_forcing

    panels = [
        (axes[0, 0], mean_forcing, "mean forcing"),
        (axes[0, 1], good_mean_forcing, "good mean forcing"),
        (axes[1, 0], bad_mean_forcing, "bad mean forcing"),
        (axes[1, 1], diff_forcing, "bad - good"),
    ]
    ims = []
    for ax, field, title in panels:
        im = ax.imshow(field, origin="lower", cmap="coolwarm", extent=[0.0, 1.0, 0.0, 1.0], aspect="auto")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ims.append(im)
    for im in ims:
        fig.colorbar(im, ax=im.axes, fraction=0.046, pad=0.02)
    fig.suptitle("Forcing mean and group comparison", fontsize=14)
    kl_path = output_dir / "forcing_group_means.png"
    fig.savefig(kl_path, dpi=180)
    plt.close(fig)

    # KL modes figure.
    n_show = min(4, n_modes)
    fig, axes = plt.subplots(1, n_show, figsize=(3.4 * n_show, 3.4), constrained_layout=True)
    if n_show == 1:
        axes = np.array([axes])
    kl_ims = []
    for j in range(n_show):
        pc = pcs[j].reshape(n_x, n_y)
        im = axes[j].imshow(pc, origin="lower", cmap="coolwarm", extent=[0.0, 1.0, 0.0, 1.0], aspect="auto")
        axes[j].set_title(f"KL mode {j + 1}")
        axes[j].set_xticks([])
        axes[j].set_yticks([])
        kl_ims.append(im)
    for im in kl_ims:
        fig.colorbar(im, ax=im.axes, fraction=0.046, pad=0.02)
    fig.suptitle("Leading KL / PCA modes of the forcing", fontsize=14)
    kl_modes_path = output_dir / "forcing_kl_modes.png"
    fig.savefig(kl_modes_path, dpi=180)
    plt.close(fig)

    # Enstrophy / palinstrophy figure.
    pred_en_curve = enstrophy_pred_arr
    ref_en_curve = enstrophy_ref_arr
    pred_pal_curve = palinstrophy_pred_arr
    ref_pal_curve = palinstrophy_ref_arr
    pred_lap_curve = lapnorm_pred_arr
    ref_lap_curve = lapnorm_ref_arr

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 4.8), constrained_layout=True)
    curve_n = max(1, int(args.n_curve_samples))
    good_curve_ids = _pick_representative_indices(good_mask, rel_final_arr, curve_n)
    bad_curve_ids = _pick_representative_indices(bad_mask, rel_final_arr, curve_n)
    ax = axes[0]
    if full_mode:
        for idx in good_curve_ids:
            ax.semilogy(ref_en_curve[idx], color="tab:green", alpha=0.28, lw=1.0)
            ax.semilogy(pred_en_curve[idx], color="tab:green", alpha=0.28, lw=1.0, ls="--")
        for idx in bad_curve_ids:
            ax.semilogy(ref_en_curve[idx], color="tab:red", alpha=0.28, lw=1.0)
            ax.semilogy(pred_en_curve[idx], color="tab:red", alpha=0.28, lw=1.0, ls="--")
        ax.semilogy(ref_en_curve[good_mask].mean(axis=0), color="tab:green", lw=2.5, label="best quartile ref mean")
        ax.semilogy(pred_en_curve[good_mask].mean(axis=0), color="tab:green", lw=2.5, ls="--", label="best quartile pred mean")
        ax.semilogy(ref_en_curve[bad_mask].mean(axis=0), color="tab:red", lw=2.5, label="worst quartile ref mean")
        ax.semilogy(pred_en_curve[bad_mask].mean(axis=0), color="tab:red", lw=2.5, ls="--", label="worst quartile pred mean")
    else:
        for i, idx in enumerate(sample_ids):
            color = "tab:green" if good_mask[i] else "tab:red"
            alpha = 0.55 if good_mask[i] else 0.75
            ax.semilogy(ref_en_curve[i], color=color, alpha=alpha, lw=1.4)
            ax.semilogy(pred_en_curve[i], color=color, alpha=alpha, lw=1.4, ls="--")
        ax.semilogy(ref_en_curve[good_mask].mean(axis=0), color="tab:green", lw=2.5, label="ref good")
        ax.semilogy(pred_en_curve[good_mask].mean(axis=0), color="tab:green", lw=2.5, ls="--", label="pred good")
        ax.semilogy(ref_en_curve[bad_mask].mean(axis=0), color="tab:red", lw=2.5, label="ref bad")
        ax.semilogy(pred_en_curve[bad_mask].mean(axis=0), color="tab:red", lw=2.5, ls="--", label="pred bad")
    ax.set_title("Enstrophy vs time")
    ax.set_xlabel("time step")
    ax.set_ylabel(r"$||\omega||_2^2$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    if full_mode:
        for idx in good_curve_ids:
            ax.semilogy(ref_pal_curve[idx], color="tab:green", alpha=0.28, lw=1.0)
            ax.semilogy(pred_pal_curve[idx], color="tab:green", alpha=0.28, lw=1.0, ls="--")
        for idx in bad_curve_ids:
            ax.semilogy(ref_pal_curve[idx], color="tab:red", alpha=0.28, lw=1.0)
            ax.semilogy(pred_pal_curve[idx], color="tab:red", alpha=0.28, lw=1.0, ls="--")
        ax.semilogy(ref_pal_curve[good_mask].mean(axis=0), color="tab:green", lw=2.5, label="best quartile ref mean")
        ax.semilogy(pred_pal_curve[good_mask].mean(axis=0), color="tab:green", lw=2.5, ls="--", label="best quartile pred mean")
        ax.semilogy(ref_pal_curve[bad_mask].mean(axis=0), color="tab:red", lw=2.5, label="worst quartile ref mean")
        ax.semilogy(pred_pal_curve[bad_mask].mean(axis=0), color="tab:red", lw=2.5, ls="--", label="worst quartile pred mean")
    else:
        for i, idx in enumerate(sample_ids):
            color = "tab:green" if good_mask[i] else "tab:red"
            alpha = 0.55 if good_mask[i] else 0.75
            ax.semilogy(ref_pal_curve[i], color=color, alpha=alpha, lw=1.4)
            ax.semilogy(pred_pal_curve[i], color=color, alpha=alpha, lw=1.4, ls="--")
        ax.semilogy(ref_pal_curve[good_mask].mean(axis=0), color="tab:green", lw=2.5, label="ref good")
        ax.semilogy(pred_pal_curve[good_mask].mean(axis=0), color="tab:green", lw=2.5, ls="--", label="pred good")
        ax.semilogy(ref_pal_curve[bad_mask].mean(axis=0), color="tab:red", lw=2.5, label="ref bad")
        ax.semilogy(pred_pal_curve[bad_mask].mean(axis=0), color="tab:red", lw=2.5, ls="--", label="pred bad")
    ax.set_title("Palinstrophy vs time")
    ax.set_xlabel("time step")
    ax.set_ylabel(r"$||\nabla \omega||_2^2$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[2]
    if full_mode:
        for idx in good_curve_ids:
            ax.semilogy(ref_lap_curve[idx], color="tab:green", alpha=0.28, lw=1.0)
            ax.semilogy(pred_lap_curve[idx], color="tab:green", alpha=0.28, lw=1.0, ls="--")
        for idx in bad_curve_ids:
            ax.semilogy(ref_lap_curve[idx], color="tab:red", alpha=0.28, lw=1.0)
            ax.semilogy(pred_lap_curve[idx], color="tab:red", alpha=0.28, lw=1.0, ls="--")
        ax.semilogy(ref_lap_curve[good_mask].mean(axis=0), color="tab:green", lw=2.5, label="best quartile ref mean")
        ax.semilogy(pred_lap_curve[good_mask].mean(axis=0), color="tab:green", lw=2.5, ls="--", label="best quartile pred mean")
        ax.semilogy(ref_lap_curve[bad_mask].mean(axis=0), color="tab:red", lw=2.5, label="worst quartile ref mean")
        ax.semilogy(pred_lap_curve[bad_mask].mean(axis=0), color="tab:red", lw=2.5, ls="--", label="worst quartile pred mean")
    else:
        for i, idx in enumerate(sample_ids):
            color = "tab:green" if good_mask[i] else "tab:red"
            alpha = 0.55 if good_mask[i] else 0.75
            ax.semilogy(ref_lap_curve[i], color=color, alpha=alpha, lw=1.4)
            ax.semilogy(pred_lap_curve[i], color=color, alpha=alpha, lw=1.4, ls="--")
        ax.semilogy(ref_lap_curve[good_mask].mean(axis=0), color="tab:green", lw=2.5, label="ref good")
        ax.semilogy(pred_lap_curve[good_mask].mean(axis=0), color="tab:green", lw=2.5, ls="--", label="pred good")
        ax.semilogy(ref_lap_curve[bad_mask].mean(axis=0), color="tab:red", lw=2.5, label="ref bad")
        ax.semilogy(pred_lap_curve[bad_mask].mean(axis=0), color="tab:red", lw=2.5, ls="--", label="pred bad")
    ax.set_title(r"Laplacian norm $||\Delta\omega||_2$")
    ax.set_xlabel("time step")
    ax.set_ylabel(r"$||\Delta\omega||_2$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Enstrophy, palinstrophy, and Laplacian norm on OOD samples", fontsize=14)
    ep_path = output_dir / "enstrophy_palinstrophy_curves.png"
    fig.savefig(ep_path, dpi=180)
    plt.close(fig)

    if full_mode:
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), constrained_layout=True)
        ax = axes[0, 0]
        ax.hist(np.log10(rel_final_arr + 1e-12), bins=35, color="slateblue", alpha=0.85)
        ax.axvline(np.log10(q25 + 1e-12), color="tab:green", ls="--", lw=2.0, label="best quartile")
        ax.axvline(np.log10(q75 + 1e-12), color="tab:red", ls="--", lw=2.0, label="worst quartile")
        ax.set_title("Final relative L2 distribution")
        ax.set_xlabel("log10(final rel L2)")
        ax.set_ylabel("count")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)

        scatter_specs = [
            (axes[0, 1], enstrophy_ref_final, "ref enstrophy final", "final rel L2"),
            (axes[1, 0], palinstrophy_ref_final, "ref palinstrophy final", "final rel L2"),
            (axes[1, 1], lapnorm_ref_final, r"ref $||\Delta\omega||_2$ final", "final rel L2"),
        ]
        for ax, xval, xlabel, ylabel in scatter_specs:
            ax.scatter(xval, rel_final_arr, c=np.log10(rel_final_arr + 1e-12), cmap="viridis", s=14, alpha=0.75)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
        axes[0, 1].set_title("Error vs final enstrophy")
        axes[1, 0].set_title("Error vs final palinstrophy")
        axes[1, 1].set_title(r"Error vs final $||\Delta\omega||_2$")
        summary_path = output_dir / "full_split_error_and_invariants.png"
        fig.suptitle("Full OOD split summary", fontsize=14)
        fig.savefig(summary_path, dpi=180)
        plt.close(fig)

    # CSV diagnostics.
    csv_path = output_dir / "sample_diagnostics.csv"
    with open(csv_path, "w", encoding="utf-8") as fcsv:
        fcsv.write(
            "sample_id,final_rel_l2,mean_rel_l2,good_bad,pc1,pc2,pc3,hi_k_fraction,low_k_fraction,"
            "ref_enstrophy,pred_enstrophy,ref_palinstrophy,pred_palinstrophy\n"
        )
        for i, idx in enumerate(sample_ids):
            label = "good" if good_mask[i] else "bad"
            pc_vals = selected_scores[i]
            fcsv.write(
                f"{idx},{rel_final_arr[i]:.8e},{rel_mean_arr[i]:.8e},{label},"
                f"{pc_vals[0]:.8e},{pc_vals[1]:.8e},{pc_vals[2] if n_modes >= 3 else 0.0:.8e},"
                f"{forcing_hi_frac_sel[i]:.8e},{forcing_lo_frac_sel[i]:.8e},"
                f"{enstrophy_ref_final[i]:.8e},{enstrophy_pred_final[i]:.8e},"
                f"{palinstrophy_ref_final[i]:.8e},{palinstrophy_pred_final[i]:.8e}\n"
            )

    print(f"\nSaved spectral diagnosis figure: {spectral_path}")
    print(f"Saved forcing group comparison figure: {kl_path}")
    print(f"Saved forcing KL modes figure: {kl_modes_path}")
    print(f"Saved enstrophy/palinstrophy figure: {ep_path}")
    if full_mode:
        print(f"Saved full-split summary figure: {summary_path}")
    print(f"Saved per-sample table: {csv_path}")


if __name__ == "__main__":
    main(parse_args())
