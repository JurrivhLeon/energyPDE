"""Make NS2D per-sample L2/H1 boxplots from saved evaluation JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter
import numpy as np


METHODS = [
    (("outputs_fno_nu{nu}_mixed",), "FNO", "#4C78A8"),
    (("outputs_fno_nu{nu}_noisy_mixed",), "FNO+Noise", "#F58518"),
    (("outputs_ae_nu{nu}_mixed", "outputs_sv_nu{nu}_mixed"), "FNO-AE", "#54A24B"),
    (("outputs_vae_nu{nu}_mixed",), "FNO-VAMO", "#E45756"),
]

FORCINGS = [
    ("grf", "Forcing type: GRF"),
    ("sinusoidal", "Forcing type: Wave"),
]


def _apply_sparse_log_ticks(ax) -> None:
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 4.0, 6.0, 8.0)))
    ax.yaxis.set_minor_formatter(NullFormatter())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("grad_flow_l2/ns2d_per/results"),
        help="Directory containing NS2D outputs_* result folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("grad_flow_l2/ns2d_per/results/error_boxplots"),
        help="Directory where boxplot figures are written.",
    )
    parser.add_argument("--viscosities", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--hide-fliers",
        action="store_true",
        help="Hide outlier markers.",
    )
    return parser.parse_args()


def _latest_run(method_dir: Path) -> Path:
    runs = sorted(method_dir.glob("run_*"))
    if not runs:
        raise FileNotFoundError(f"No run_* directory found under {method_dir}")
    return runs[-1]


def _resolve_method_dir(results_dir: Path, nu: int, patterns: Iterable[str]) -> Path:
    for pattern in patterns:
        method_dir = results_dir / pattern.format(nu=nu)
        if method_dir.exists():
            return method_dir
    names = ", ".join(pattern.format(nu=nu) for pattern in patterns)
    raise FileNotFoundError(f"No method directory found for nu={nu}: {names}")


def _scalar_stat(item: dict, key: str) -> float:
    value = item[key]
    if isinstance(value, dict):
        value = value.get("mean", value.get("vorticity"))
    if isinstance(value, list):
        value = value[-1]
    return float(value)


def _load_overall_errors(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    if not items:
        raise ValueError(f"No per-sample entries found in {path}")
    rel_l2 = np.asarray(
        [_scalar_stat(item, "overall_rel_l2") for item in items], dtype=np.float64
    )
    rel_h1 = np.asarray(
        [_scalar_stat(item, "overall_rel_h1") for item in items], dtype=np.float64
    )
    rel_l2 = rel_l2[np.isfinite(rel_l2) & (rel_l2 > 0.0)]
    rel_h1 = rel_h1[np.isfinite(rel_h1) & (rel_h1 > 0.0)]
    if rel_l2.size == 0 or rel_h1.size == 0:
        raise ValueError(f"No positive finite L2/H1 entries found in {path}")
    return rel_l2, rel_h1


def _load_method_errors(
    results_dir: Path,
    nu: int,
    forcing: str,
    patterns: Iterable[str],
) -> tuple[np.ndarray, np.ndarray, Path]:
    method_dir = _resolve_method_dir(results_dir, nu, patterns)
    run_dir = _latest_run(method_dir)
    json_path = run_dir / f"ood_eval_{forcing}" / "test_per_sample_errors.json"
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    rel_l2, rel_h1 = _load_overall_errors(json_path)
    return rel_l2, rel_h1, json_path


def _boxplot_panel(
    ax,
    results_dir: Path,
    nu: int,
    forcing: str,
    forcing_label: str,
    show_fliers: bool,
) -> list[Path]:
    method_errors = []
    sources = []
    for patterns, method_label, color in METHODS:
        rel_l2, rel_h1, source = _load_method_errors(results_dir, nu, forcing, patterns)
        method_errors.append((method_label, color, rel_l2, rel_h1))
        sources.append(source)

    data = [entry[2] for entry in method_errors] + [entry[3] for entry in method_errors]
    colors = [entry[1] for entry in method_errors] * 2
    positions = np.asarray([1, 2, 3, 4, 6, 7, 8, 9], dtype=np.float64)

    box = ax.boxplot(
        data,
        positions=positions,
        patch_artist=True,
        showfliers=show_fliers,
        widths=0.68,
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

    ax.set_yscale("log")
    ax.set_xlim(0.25, 9.75)
    ax.set_xticks([2.5, 7.5])
    ax.set_xticklabels([r"$L^2$", r"$H^1$"], fontsize=18)
    ax.tick_params(axis="y", labelsize=15)
    ax.tick_params(axis="x", pad=8)
    ax.set_title(forcing_label, fontsize=18, pad=12)
    _apply_sparse_log_ticks(ax)
    ax.grid(True, axis="y", which="major", color="0.48", linewidth=1.05, alpha=0.60)
    ax.grid(True, axis="y", which="minor", color="0.78", linewidth=0.55, alpha=0.40)
    ax.set_axisbelow(True)
    ax.axvline(5.0, color="0.78", linewidth=0.9)
    return sources


def make_plot(
    results_dir: Path,
    output_dir: Path,
    nu: int,
    dpi: int,
    show_fliers: bool,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.5), sharey=True)
    all_sources = []
    for ax, (forcing, forcing_label) in zip(axes, FORCINGS):
        all_sources.extend(
            _boxplot_panel(
                ax,
                results_dir=results_dir,
                nu=nu,
                forcing=forcing,
                forcing_label=forcing_label,
                show_fliers=show_fliers,
            )
        )

    axes[0].set_ylabel("Aggregate relative error", fontsize=18, labelpad=9)
    legend_handles = [
        Patch(facecolor=color, edgecolor="black", alpha=0.72, label=label)
        for _, label, color in METHODS
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        frameon=False,
        fontsize=18,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"ns2d_nu{nu}_l2_h1_boxplots.png"
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.89), w_pad=2.0)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    print(f"Wrote {out_path}")
    for source in all_sources:
        print(f"  source: {source}")
    return out_path


def main() -> None:
    args = parse_args()
    written = []
    for nu in args.viscosities:
        written.append(
            make_plot(
                args.results_dir,
                args.output_dir,
                nu,
                args.dpi,
                not args.hide_fliers,
            )
        )

    print("Wrote NS2D error boxplots:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
