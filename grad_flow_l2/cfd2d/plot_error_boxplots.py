"""Make CFD2D per-sample channel error boxplots from saved evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter
import numpy as np


CHANNELS = (
    ("rho", "Density"),
    ("p", "Pressure"),
    ("velocity", "Velocity"),
    ("mean", "Mean"),
)
METHODS = (
    ("outputs_fno_nu8", "ood_eval", "FNO", "#4C78A8"),
    ("outputs_fno_noisy_nu8", "ood_eval", "FNO+Noise", "#F58518"),
    ("outputs_ae_nu8", "ood_eval_t10", "FNO-AE", "#54A24B"),
    ("outputs_vamo_nu8", "ood_eval_t10", "FNO-VAMO", "#E45756"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("grad_flow_l2/cfd2d/trained_checkpoints"),
        help="Directory containing CFD2D checkpoint result folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for boxplots. Defaults to ROOT/error_boxplots.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--hide-fliers", action="store_true")
    return parser.parse_args()


def _latest_run(method_dir: Path) -> Path:
    runs = sorted(p for p in method_dir.glob("run_*") if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"No run_* directories found under {method_dir}")
    return runs[-1]


def _last_positive(values) -> float:
    if isinstance(values, list):
        value = values[-1]
    else:
        value = values
    return float(value)


def _load_channel_errors(json_path: Path, metric_key: str) -> dict[str, np.ndarray]:
    with json_path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    if not items:
        raise ValueError(f"No per-sample entries found in {json_path}")

    channel_errors: dict[str, list[float]] = {label: [] for _, label in CHANNELS}
    for item in items:
        metric = item[metric_key]
        for key, label in CHANNELS:
            value = _last_positive(metric[key])
            if np.isfinite(value) and value > 0.0:
                channel_errors[label].append(value)

    arrays = {
        label: np.asarray(values, dtype=np.float64)
        for label, values in channel_errors.items()
    }
    empty = [label for label, values in arrays.items() if values.size == 0]
    if empty:
        raise ValueError(f"No positive finite {metric_key} entries for {empty} in {json_path}")
    return arrays


def _load_method(root: Path, method_dir_name: str, eval_dir_name: str):
    json_path = (
        _latest_run(root / method_dir_name)
        / eval_dir_name
        / "test_per_sample_errors.json"
    )
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    return {
        "l2": _load_channel_errors(json_path, "overall_rel_l2"),
        "h1": _load_channel_errors(json_path, "overall_rel_h1"),
    }, json_path


def _apply_sparse_log_ticks(ax) -> None:
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 4.0, 6.0, 8.0)))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _boxplot_panel(ax, method_data, metric_key: str, show_fliers: bool) -> None:
    n_methods = len(method_data)
    data = []
    colors = []
    positions = []

    for channel_index, (_, channel_label) in enumerate(CHANNELS):
        base = channel_index * (n_methods + 1) + 1
        for method_index, method in enumerate(method_data):
            data.append(method["metrics"][metric_key][channel_label])
            colors.append(method["color"])
            positions.append(base + method_index)

    box = ax.boxplot(
        data,
        positions=positions,
        patch_artist=True,
        showfliers=show_fliers,
        widths=0.82,
        medianprops={"color": "black", "linewidth": 1.4},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
        boxprops={"linewidth": 1.0},
        flierprops={
            "marker": "o",
            "markersize": 5.2,
            "markeredgewidth": 0.7,
            "alpha": 0.45,
        },
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    for flier, color in zip(box["fliers"], colors):
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)
        flier.set_alpha(0.45)

    centers = [
        channel_index * (n_methods + 1) + 1 + (n_methods - 1) / 2
        for channel_index in range(len(CHANNELS))
    ]
    ax.set_xticks(centers)
    ax.set_xticklabels([label for _, label in CHANNELS], fontsize=16)
    ax.tick_params(axis="x", pad=10)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_yscale("log")
    ax.set_xlim(min(positions) - 0.8, max(positions) + 0.8)
    _apply_sparse_log_ticks(ax)
    ax.grid(True, axis="y", which="major", color="0.48", linewidth=1.05, alpha=0.60)
    ax.grid(True, axis="y", which="minor", color="0.78", linewidth=0.55, alpha=0.40)
    ax.set_axisbelow(True)
    for boundary in range(1, len(CHANNELS)):
        ax.axvline(boundary * (n_methods + 1), color="0.78", linewidth=0.9)


def make_plot(root: Path, output_dir: Path, dpi: int, show_fliers: bool) -> Path:
    method_data = []
    sources = []
    for method_dir_name, eval_dir_name, label, color in METHODS:
        metrics, source = _load_method(root, method_dir_name, eval_dir_name)
        method_data.append({"label": label, "color": color, "metrics": metrics})
        sources.append(source)

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 9.0), sharex=True, sharey=False)
    _boxplot_panel(axes[0], method_data, "l2", show_fliers)
    _boxplot_panel(axes[1], method_data, "h1", show_fliers)
    axes[0].set_ylabel(r"Aggr. rel. $L^2$ error", fontsize=18, labelpad=9)
    axes[1].set_ylabel(r"Aggr. rel. $H^1$ error", fontsize=18, labelpad=9)

    #fig.suptitle(r"CFD2D per-sample channel errors, $\nu=10^{-8}$", fontsize=21, y=0.96)
    legend_handles = [
        Patch(facecolor=m["color"], edgecolor="black", alpha=0.72, label=m["label"])
        for m in method_data
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 1.0),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=18,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "cfd2d_nu8_channel_error_boxplots.png"
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), h_pad=2.0)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    print(f"Wrote {out_path}")
    for source in sources:
        print(f"  source: {source}")
    return out_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.root / "error_boxplots")
    path = make_plot(args.root, output_dir, args.dpi, not args.hide_fliers)
    print("Wrote CFD2D error boxplots:")
    print(f"  {path}")


if __name__ == "__main__":
    main()
