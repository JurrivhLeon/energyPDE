"""Plot selected NS2D trajectory snapshots across methods."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

try:
    from ..heat_data import load_dataset_splits
    from .plot_physical_diagnostics import (
        FORCINGS,
        METHODS,
        _method_summary,
        _rollout_method,
        _time_metadata,
    )
except ImportError:
    from grad_flow_l2.heat_data import load_dataset_splits
    from grad_flow_l2.ns2d_per.plot_physical_diagnostics import (
        FORCINGS,
        METHODS,
        _method_summary,
        _rollout_method,
        _time_metadata,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("grad_flow_l2/ns2d_per/results"),
        help="Directory containing outputs_* result folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("grad_flow_l2/ns2d_per/results/selected_trajectory_plots"),
        help="Directory where selected trajectory figures are written.",
    )
    parser.add_argument(
        "--viscosities",
        type=int,
        nargs="+",
        default=[3],
        help="Viscosity exponents to plot, e.g. 4 means nu=1e-4.",
    )
    parser.add_argument(
        "--forcings",
        type=str,
        nargs="+",
        default=[name for name, _label in FORCINGS],
        choices=[name for name, _label in FORCINGS],
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"]
    )
    parser.add_argument(
        "--sample-indices",
        type=str,
        default="180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199",
        help="Comma-separated sample indices from the selected split.",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _parse_sample_indices(raw: str) -> List[int]:
    out: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        idx = int(token)
        if idx < 0:
            raise ValueError(f"sample indices must be nonnegative, got {idx}")
        out.append(idx)
    if not out:
        raise ValueError("at least one sample index is required")
    return out


def _forcing_labels() -> Dict[str, str]:
    return {name: label for name, label in FORCINGS}


def _load_selected_split(
    dataset_path: str,
    split_name: str,
    sample_indices: List[int],
    max_steps: int | None,
) -> Tuple[Dict[str, torch.Tensor], float, np.ndarray]:
    splits = load_dataset_splits(dataset_path, map_location="cpu")
    split = splits[split_name]
    total = int(split["u0"].shape[0])
    bad = [idx for idx in sample_indices if idx >= total]
    if bad:
        raise IndexError(f"sample indices out of range for split size {total}: {bad}")
    index = torch.as_tensor(sample_indices, dtype=torch.long)
    selected = {
        "u0": split["u0"].index_select(0, index).clone(),
        "f": split["f"].index_select(0, index).clone(),
        "u_traj": split["u_traj"].index_select(0, index).clone(),
    }
    if max_steps is not None:
        steps = min(int(max_steps), int(selected["u_traj"].shape[1] - 1))
        selected["u_traj"] = selected["u_traj"][:, : steps + 1]
    n_steps = int(selected["u_traj"].shape[1] - 1)
    dt, time_values = _time_metadata(splits.get("meta", {}), n_steps)
    return selected, dt, time_values[: n_steps + 1]


def _scalar_field(u: torch.Tensor) -> np.ndarray:
    if u.dim() == 3:
        return u.detach().cpu().numpy()
    if u.dim() == 4 and u.shape[1] == 1:
        return u[:, 0].detach().cpu().numpy()
    raise ValueError(f"Expected scalar field batch, got shape {tuple(u.shape)}")


def _scalar_traj(u: torch.Tensor) -> np.ndarray:
    if u.dim() == 4:
        return u.detach().cpu().numpy()
    if u.dim() == 5 and u.shape[2] == 1:
        return u[:, :, 0].detach().cpu().numpy()
    raise ValueError(f"Expected scalar trajectory batch, got shape {tuple(u.shape)}")


def _snapshot_indices(n_steps: int) -> np.ndarray:
    targets = np.linspace(n_steps / 5.0, float(n_steps), 5)
    indices = np.rint(targets).astype(int)
    return np.clip(indices, 1, n_steps)


def _field_scale(arrays: List[np.ndarray]) -> float:
    vmax = 0.0
    for arr in arrays:
        finite = np.asarray(arr)[np.isfinite(arr)]
        if finite.size:
            vmax = max(vmax, float(np.max(np.abs(finite))))
    return max(vmax, 1e-8)


TRAINING_HORIZONS = {3: 10.0, 4: 12.0, 5: 8.0}


def _trajectory_cmap():
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_under("#274f9d")
    cmap.set_over("#b22a2a")
    return cmap


def _draw_training_horizon_marker(
    fig: plt.Figure,
    nu: int,
    time_items: List[Tuple[int, float]],
    snap_left: float,
    cell_w: float,
    col_gap: float,
    ic_gap: float,
    first_after_train: int | None,
    train_gap_extra: float,
    bottom: float,
    top: float,
) -> None:
    train_t = TRAINING_HORIZONS.get(nu)
    if train_t is None:
        return
    times = np.asarray([t for _idx, t in time_items], dtype=float)
    if train_t >= times[-1]:
        return
    after = np.flatnonzero(times > train_t)
    if after.size == 0:
        return
    right_col = int(after[0])
    if right_col <= 0:
        return
    right_x = (
        snap_left
        if right_col == 0
        else snap_left + cell_w + ic_gap + (right_col - 1) * (cell_w + col_gap)
    )
    if first_after_train is not None and right_col >= first_after_train:
        right_x += train_gap_extra
    left_col = right_col - 1
    left_right_edge = (
        snap_left + cell_w
        if left_col == 0
        else snap_left + cell_w + ic_gap + (left_col - 1) * (cell_w + col_gap) + cell_w
    )
    if first_after_train is not None and left_col >= first_after_train:
        left_right_edge += train_gap_extra
    x = 0.5 * (left_right_edge + right_x)
    fig.add_artist(
        plt.Line2D(
            [x, x],
            [bottom, top],
            transform=fig.transFigure,
            color="0.35",
            linestyle=(0, (4, 4)),
            linewidth=2.0,
            alpha=0.9,
            zorder=20,
        )
    )
    fig.text(
        x,
        bottom - 0.016,
        "Training horizon",
        ha="center",
        va="top",
        fontsize=17.5,
        color="0.30",
    )


def _plot_sample(
    output_dir: Path,
    nu: int,
    forcing: str,
    forcing_label: str,
    sample_id: int,
    local_index: int,
    split: Dict[str, torch.Tensor],
    time_values: np.ndarray,
    pred_by_method: List[Tuple[str, torch.Tensor]],
    dpi: int,
) -> Path:
    true_traj = _scalar_traj(split["u_traj"])[local_index]
    u0 = _scalar_field(split["u0"])[local_index]
    forcing_field = _scalar_field(split["f"])[local_index]
    pred_arrays = [
        (label, _scalar_traj(pred)[local_index]) for label, pred in pred_by_method
    ]

    n_steps = int(true_traj.shape[0] - 1)
    snap_idx = _snapshot_indices(n_steps)
    snap_times = time_values[snap_idx]
    ic_scale = _field_scale([u0])
    force_scale = _field_scale([forcing_field])
    ref_state_scale = _field_scale([true_traj[snap_idx]])
    trajectory_cmap = _trajectory_cmap()

    fig_w, fig_h = 21.6, 10.8
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)

    n_rows = 1 + len(pred_arrays)
    n_cols = 1 + len(snap_idx)
    top = 0.925
    bottom = 0.05
    row_gap = 0.004
    col_gap = 0.004 * fig_h / fig_w
    ic_gap = 0.016
    train_gap_extra = 0.018
    cell_h = (top - bottom - (n_rows - 1) * row_gap) / n_rows
    cell_w = cell_h * fig_h / fig_w

    left_x = 0.040
    forcing_scale = 1.20
    left_w = forcing_scale * cell_w
    left_h = forcing_scale * cell_h
    left_y = bottom + 0.5 * (top - bottom - left_h)
    left_cbar_x = left_x + left_w + 0.010
    left_cbar_w = 0.010
    label_x = left_cbar_x + left_cbar_w + 0.05
    label_w = 0.125
    snap_left = label_x + label_w + 0.020
    train_t = TRAINING_HORIZONS.get(nu)
    shown_times = np.asarray([0.0] + [float(t) for t in snap_times], dtype=float)
    first_after_train = int(np.flatnonzero(shown_times > train_t)[0]) if train_t is not None and np.any(shown_times > train_t) else None
    grid_w = n_cols * cell_w + (n_cols - 2) * col_gap + ic_gap + (train_gap_extra if first_after_train is not None else 0.0)
    row_cbar_x = snap_left + grid_w + 0.014
    row_cbar_w = 0.012

    ax_force = fig.add_axes([left_x, left_y, left_w, left_h])
    cax_force = fig.add_axes([left_cbar_x, left_y, left_cbar_w, left_h])

    im_force = ax_force.imshow(
        forcing_field,
        origin="lower",
        cmap="coolwarm",
        vmin=-force_scale,
        vmax=force_scale,
        extent=[0, 1, 0, 1],
    )
    ax_force.set_title("Forcing", fontsize=25, pad=11)
    ax_force.set_xticks([])
    ax_force.set_yticks([])
    ax_force.set_frame_on(False)
    for spine in ax_force.spines.values():
        spine.set_visible(False)
    cbar_force = fig.colorbar(im_force, cax=cax_force)
    cbar_force.ax.tick_params(labelsize=12)

    row_items: List[Tuple[str, np.ndarray]] = [("Groundtruth", true_traj)] + pred_arrays
    time_items = [(0, 0.0)] + [(int(idx), float(t_val)) for idx, t_val in zip(snap_idx, snap_times)]
    last_im = None
    for row, (row_label, traj) in enumerate(row_items):
        row_y = top - (row + 1.0) * cell_h - row * row_gap
        label_ax = fig.add_axes([label_x, row_y, label_w, cell_h])
        label_ax.axis("off")
        label_ax.text(
            0.98,
            0.5,
            row_label,
            ha="right",
            va="center",
            fontsize=25,
            fontweight="semibold" if row == 0 else "normal",
        )
        for col, (idx, t_val) in enumerate(time_items):
            col_x = snap_left if col == 0 else snap_left + cell_w + ic_gap + (col - 1) * (cell_w + col_gap)
            if first_after_train is not None and col >= first_after_train:
                col_x += train_gap_extra
            ax = fig.add_axes([col_x, row_y, cell_w, cell_h])
            if col == 0 and row > 0:
                ax.axis("off")
                continue
            field = u0 if col == 0 else traj[idx]
            last_im = ax.imshow(
                field,
                origin="lower",
                cmap=trajectory_cmap,
                vmin=-ref_state_scale,
                vmax=ref_state_scale,
                extent=[0, 1, 0, 1],
            )
            if row == 0:
                title = "IC" if col == 0 else f"t={t_val:g}"
                ax.set_title(title, fontsize=20, pad=13)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            for spine in ax.spines.values():
                spine.set_visible(False)

    if last_im is None:
        raise RuntimeError("No trajectory snapshots were plotted")
    _draw_training_horizon_marker(
        fig,
        nu,
        time_items,
        snap_left,
        cell_w,
        col_gap,
        ic_gap,
        first_after_train,
        train_gap_extra,
        bottom,
        top,
    )
    cax_state = fig.add_axes([row_cbar_x, bottom, row_cbar_w, top - bottom])
    cbar_state = fig.colorbar(last_im, cax=cax_state, extend="both")
    cbar_state.ax.set_ylabel("vorticity (groundtruth scale)", rotation=90, fontsize=16, labelpad=12)
    cbar_state.ax.tick_params(labelsize=11)

    """fig.suptitle(
        rf"NS2D trajectories, $\nu=10^{{-{nu}}}$, {forcing_label}, sample {sample_id}",
        y=0.985,
        fontsize=22,
    )"""

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        output_dir / f"ns2d_nu{nu}_{forcing}_sample_{sample_id:04d}_trajectory.png"
    )
    pdf_path = (
        output_dir / f"ns2d_nu{nu}_{forcing}_sample_{sample_id:04d}_trajectory.pdf"
    )
    fig.savefig(out_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    return out_path


def make_plots_for_setting(
    results_dir: Path,
    output_dir: Path,
    nu: int,
    forcing: str,
    forcing_label: str,
    split_name: str,
    sample_indices: List[int],
    max_steps: int | None,
    device: str,
    dpi: int,
) -> List[Path]:
    method_infos = []
    for patterns, method_label, kind, _color in METHODS:
        run_dir, summary = _method_summary(results_dir, nu, forcing, patterns)
        method_infos.append((method_label, kind, run_dir, summary))

    dataset_path = str(method_infos[0][3]["dataset_path"])
    split, dt, time_values = _load_selected_split(
        dataset_path, split_name, sample_indices, max_steps
    )

    pred_by_method = []
    for method_label, kind, run_dir, summary in method_infos:
        pred = _rollout_method(kind, run_dir, summary, split, dt, device)
        pred_by_method.append((method_label, pred))

    written = []
    setting_dir = output_dir / f"nu{nu}_{forcing}"
    for local_index, sample_id in enumerate(sample_indices):
        written.append(
            _plot_sample(
                setting_dir,
                nu,
                forcing,
                forcing_label,
                sample_id,
                local_index,
                split,
                time_values,
                pred_by_method,
                dpi,
            )
        )
    return written


def main() -> None:
    args = parse_args()
    sample_indices = _parse_sample_indices(args.sample_indices)
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    forcing_labels = _forcing_labels()
    written = []
    for nu in args.viscosities:
        for forcing in args.forcings:
            written.extend(
                make_plots_for_setting(
                    args.results_dir,
                    args.output_dir,
                    nu,
                    forcing,
                    forcing_labels[forcing],
                    args.split,
                    sample_indices,
                    args.max_steps,
                    device,
                    args.dpi,
                )
            )
    print("Wrote selected trajectory plots:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
