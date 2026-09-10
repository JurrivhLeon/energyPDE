"""Plot Euler1D rollout metric comparisons from trained checkpoints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter


SETTINGS = (5, 10, 15, 20)
TRAINING_HORIZON_END = 15.0
METHODS = (
    ("outputs_fno_L{setting}", "Vanilla FNO", "-", "o"),
    ("outputs_fno_noisy_L{setting}", "FNO+Noise", "--", "s"),
    ("outputs_ae_L{setting}", "FNO-AE", ":", "^"),
    ("outputs_vamo_L{setting}", "VAMO-FNO", "-.", "D"),
)
METRICS = (
    ("rel_l2_mean_all", "Relative L2"),
    ("rel_l1_mean_all", "Relative L1"),
)


def _latest_run(method_dir: Path) -> Path:
    runs = sorted(p for p in method_dir.glob("run_*") if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"No run_* directories found under {method_dir}")
    return runs[-1]


def _read_curve(csv_path: Path) -> dict[str, np.ndarray]:
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    columns = {name: [] for name in rows[0].keys()}
    for row in rows:
        step = int(float(row["step"]))
        if step <= 0:
            continue
        for name, value in row.items():
            columns[name].append(float(value))
    return {name: np.asarray(values, dtype=float) for name, values in columns.items()}


def _marker_indices(steps: np.ndarray) -> list[int]:
    targets = np.asarray([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
    indices: list[int] = []
    for target in targets:
        if (
            not steps.size
            or target < float(np.min(steps))
            or target > float(np.max(steps))
        ):
            continue
        index = int(np.argmin(np.abs(steps - target)))
        if not indices or index != indices[-1]:
            indices.append(index)
    return indices


def _apply_axes_style(ax: plt.Axes, x_values: np.ndarray) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=10.0 ** np.array([0.2, 0.4, 0.6, 0.8]))
    )
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.8)
    ax.grid(True, which="minor", color="#eeeeee", linewidth=0.5, axis="y")
    ax.set_xlabel("Rollout step", fontsize=12, labelpad=5)
    if x_values.size:
        x_max = float(np.max(x_values))
        ax.set_xlim(0.0, x_max * 1.03 if x_max > 0 else 1.0)
        ax.set_xticks(np.linspace(0.0, x_max, 11))


def plot_setting(root: Path, output_dir: Path, setting: int) -> Path:
    fig, axes = plt.subplots(1, len(METRICS), figsize=(10, 4.5), sharex=True)
    all_x_values: list[np.ndarray] = []

    for method_dir_template, method_label, linestyle, marker in METHODS:
        method_dir = root / method_dir_template.format(setting=setting)
        csv_path = _latest_run(method_dir) / "ood_eval" / "test_rollout_error_curve.csv"
        curve = _read_curve(csv_path)
        x_values = curve["step"]
        marker_indices = _marker_indices(x_values)
        all_x_values.append(x_values)

        for ax, (column, metric_label) in zip(axes, METRICS):
            if column not in curve:
                raise KeyError(f"Missing column {column!r} in {csv_path}")
            ax.plot(
                x_values,
                curve[column],
                linewidth=1.5,
                linestyle=linestyle,
                marker=marker,
                markersize=3.0,
                markevery=marker_indices,
                label=method_label,
            )
            ax.set_title(metric_label, fontsize=14, pad=10)
            ax.set_ylabel("Error", fontsize=12, labelpad=5)

    longest_x_values = max(all_x_values, key=len)
    for index, ax in enumerate(axes):
        ax.axvspan(
            0.0,
            TRAINING_HORIZON_END,
            color="#d9d9d9",
            alpha=0.45,
            linewidth=0,
            label="Training horizon" if index == 0 else None,
            zorder=0,
        )
        _apply_axes_style(ax, longest_x_values)

    fig.suptitle(f"Euler1D L={setting}", y=0.99, fontsize=15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(METHODS) + 1,
        bbox_to_anchor=(0.5, 0.925),
        fontsize=12,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.925))

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"euler1d_L{setting}_metric_trends.png"
    pdf_path = output_dir / f"euler1d_L{setting}_metric_trends.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("grad_flow_l2/euler1d/trained_checkpoints"),
        help="Directory containing Euler1D trained checkpoint/result folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for comparison plots. Defaults to ROOT/comparison_plots.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (args.root / "comparison_plots")
    for setting in SETTINGS:
        path = plot_setting(args.root, output_dir, setting)
        print(path)


if __name__ == "__main__":
    main()
