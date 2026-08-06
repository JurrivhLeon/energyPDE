"""Make NS2D OOD metric trend comparison plots from saved eval CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np


METHODS: List[Tuple[Tuple[str, ...], str, str, str, str]] = [
    (("outputs_fno_nu{nu}_mixed",), "Vanilla FNO", "tab:blue", "-", "o"),
    (("outputs_fno_nu{nu}_noisy_mixed",), "FNO+Noise", "tab:orange", "--", "s"),
    (
        ("outputs_ae_nu{nu}_mixed", "outputs_sv_nu{nu}_mixed"),
        "FNO-AE",
        "tab:green",
        ":",
        "^",
    ),
    (("outputs_vae_nu{nu}_mixed",), "VAMO-FNO", "tab:red", "-.", "D"),
]

METRICS: List[Tuple[str, str]] = [
    ("rel_l2_mean", r"$L^2$ error"),
    ("rel_h1_mean", r"$H^1$ error"),
    ("enstrophy_rel_mean", r"Enstrophy error"),
    ("palinstrophy_rel_mean", r"Palinstrophy error"),
]

FORCINGS: List[Tuple[str, str]] = [
    ("grf", "GRF forcing"),
    ("sinusoidal", "Sinusoidal forcing"),
]

TRAINING_HORIZONS: Dict[int, float] = {
    3: 10.0,
    4: 12.0,
    5: 8.0,
}


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
        default=Path("grad_flow_l2/ns2d_per/results/comparison_plots"),
        help="Directory where comparison figures are written.",
    )
    parser.add_argument("--viscosities", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _apply_sparse_log_ticks(ax) -> None:
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=10.0 ** np.asarray([0.2, 0.4, 0.6, 0.8]))
    )
    ax.yaxis.set_minor_formatter(NullFormatter())


def _format_time_tick(value: float) -> str:
    if abs(value - round(value)) < 1e-8:
        return str(int(round(value)))
    return f"{value:g}"


def _latest_run(method_dir: Path) -> Path:
    runs = sorted(method_dir.glob("run_*"))
    if not runs:
        raise FileNotFoundError(f"No run_* directory found under {method_dir}")
    return runs[-1]


def _resolve_method_dir(results_dir: Path, nu: int, patterns: Tuple[str, ...]) -> Path:
    for pattern in patterns:
        method_dir = results_dir / pattern.format(nu=nu)
        if method_dir.exists():
            return method_dir
    raise FileNotFoundError(
        f"None of the method directories exist for nu={nu}: "
        + ", ".join(pattern.format(nu=nu) for pattern in patterns)
    )


def _read_curve(csv_path: Path) -> Dict[str, np.ndarray]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    if "step" in fieldnames:
        rows = [row for row in rows if int(float(row["step"])) > 0]
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in fieldnames
    }


def _load_method_curve(
    results_dir: Path, nu: int, forcing: str, patterns: Tuple[str, ...]
) -> Dict[str, np.ndarray]:
    method_dir = _resolve_method_dir(results_dir, nu, patterns)
    run_dir = _latest_run(method_dir)
    csv_path = run_dir / f"ood_eval_{forcing}" / "test_rollout_error_curve.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    return _read_curve(csv_path)


def _marker_indices(steps: np.ndarray) -> List[int]:
    if steps.size == 0:
        return []
    t_max = float(np.max(steps))
    targets = np.concatenate(([1.0], np.linspace(t_max / 10.0, t_max, 10)))
    indices: List[int] = []
    for target in targets:
        if target < float(np.min(steps)) or target > t_max:
            continue
        index = int(np.argmin(np.abs(steps - target)))
        if index not in indices:
            indices.append(index)
    return indices


def _time_values(curve: Dict[str, np.ndarray], metric_key: str) -> np.ndarray:
    time = curve.get("time")
    if time is None:
        return np.arange(1, len(curve[metric_key]) + 1, dtype=np.float64)
    return np.asarray(time, dtype=np.float64)


def make_plot(
    results_dir: Path,
    output_dir: Path,
    nu: int,
    forcing: str,
    forcing_label: str,
    dpi: int,
) -> Path:
    curves = []
    for patterns, label, color, linestyle, marker in METHODS:
        curve = _load_method_curve(results_dir, nu, forcing, patterns)
        curves.append((label, color, linestyle, marker, curve))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes_flat = axes.ravel()
    base_time = _time_values(curves[0][4], METRICS[0][0])
    t_min = 0.0
    t_max = float(base_time[-1])
    x_right = t_max + 0.05 * max(t_max - t_min, 1.0)
    x_ticks = np.linspace(t_min, t_max, 11)
    train_horizon = min(TRAINING_HORIZONS.get(nu, t_max), t_max)

    for ax, (metric_key, metric_label) in zip(axes_flat, METRICS):
        ax.axvspan(t_min, train_horizon, color="0.9", alpha=0.45, zorder=0)
        for method_label, color, linestyle, marker, curve in curves:
            time = _time_values(curve, metric_key)
            steps = np.asarray(
                curve.get("step", np.arange(1, len(time) + 1)), dtype=np.float64
            )
            marker_indices = _marker_indices(steps)
            y = np.asarray(curve[metric_key], dtype=np.float64)
            y = np.where(y > 0.0, y, np.nan)
            ax.plot(
                time,
                y,
                label=method_label,
                color=color,
                linewidth=1.35,
                linestyle=linestyle,
                marker=marker,
                markersize=3.0,
                markevery=marker_indices,
                zorder=3,
            )
        ax.set_yscale("log")
        ax.set_xlim(t_min, x_right)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([_format_time_tick(x) for x in x_ticks])
        ax.set_title(metric_label, fontsize=15, pad=10)
        ax.set_xlabel("time", fontsize=13, labelpad=5)
        ax.set_ylabel("Relative error", fontsize=13, labelpad=5)
        ax.grid(True, which="both", alpha=0.25, zorder=2)
        _apply_sparse_log_ticks(ax)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    handles.append(Patch(facecolor="0.9", edgecolor="none", alpha=0.45))
    labels.append("training horizon")
    fig.suptitle(
        rf"NS2D Rollout Errors, $\nu=10^{{-{nu}}}$, {forcing_label}",
        y=0.985,
        fontsize=17.5,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=5,
        frameon=False,
        fontsize=15,
    )
    fig.subplots_adjust(
        left=0.08, right=0.98, bottom=0.06, top=0.82, hspace=0.32, wspace=0.25
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"ns2d_nu{nu}_{forcing}_metric_trends.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    written = []
    for nu in args.viscosities:
        for forcing, forcing_label in FORCINGS:
            written.append(
                make_plot(
                    args.results_dir,
                    args.output_dir,
                    nu,
                    forcing,
                    forcing_label,
                    args.dpi,
                )
            )
    print("Wrote comparison plots:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
