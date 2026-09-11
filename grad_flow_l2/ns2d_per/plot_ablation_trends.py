"""Make NS2D VAMO ablation trend plots from saved evaluation CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np


METRICS = (
    ("rel_l2_mean", r"$L^2$ error"),
    ("rel_h1_mean", r"$H^1$ error"),
)
FORCINGS = (
    ("grf", "GRF\nforcing"),
    ("sinusoidal", "Wave\nforcing"),
)
TRAINING_HORIZONS = {
    4: 12.0,
    5: 8.0,
}
ABLATIONS = (
    ("beta0", r"$\beta_{\mathrm{KL}}=0$", "#7F7F7F", (0, (3, 1, 1, 1)), "o"),
    ("disable_encoder", "w/o encoder noise", "#F58518", "--", "s"),
    ("disable_transition", "w/o transition noise", "#54A24B", ":", "^"),
    ("full", "full model", "#E45756", "-", "D"),
)


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
        default=Path("grad_flow_l2/ns2d_per/results/ablation_plots"),
        help="Directory where ablation trend figures are written.",
    )
    parser.add_argument("--viscosities", type=int, nargs="+", default=[4, 5])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise an error if an ablation result is missing.",
    )
    return parser.parse_args()


def _latest_run(method_dir: Path) -> Path:
    runs = sorted(p for p in method_dir.glob("run_*") if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"No run_* directory found under {method_dir}")
    return runs[-1]


def _load_args(run_dir: Path) -> dict:
    args_path = run_dir / "args.json"
    if not args_path.exists():
        return {}
    with args_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_zero(value) -> bool:
    try:
        return abs(float(value)) < 1e-15
    except (TypeError, ValueError):
        return False


def _find_by_args(
    results_dir: Path,
    nu: int,
    predicate: Callable[[dict], bool],
) -> Path | None:
    candidates = []
    for method_dir in sorted(results_dir.glob(f"outputs_vae_nu{nu}*")):
        if not method_dir.is_dir():
            continue
        try:
            run_dir = _latest_run(method_dir)
        except FileNotFoundError:
            continue
        if predicate(_load_args(run_dir)):
            candidates.append(run_dir)
    return candidates[-1] if candidates else None


def _resolve_run_dir(results_dir: Path, nu: int, ablation_key: str) -> Path | None:
    if ablation_key == "full":
        method_dir = results_dir / f"outputs_vae_nu{nu}_mixed"
        return _latest_run(method_dir) if method_dir.exists() else None
    if ablation_key == "disable_encoder":
        method_dir = results_dir / f"outputs_vae_nu{nu}_disable_encoder_noise_mixed"
        return _latest_run(method_dir) if method_dir.exists() else None
    if ablation_key == "disable_transition":
        method_dir = results_dir / f"outputs_vae_nu{nu}_disable_transition_noise_mixed"
        return _latest_run(method_dir) if method_dir.exists() else None
    if ablation_key == "beta0":
        for name in (
            f"outputs_vae_nu{nu}_novar",
            f"outputs_vae_nu{nu}_zkl",
            f"outputs_vae_nu{nu}_beta0_mixed",
            f"outputs_vae_nu{nu}_beta_kl0_mixed",
            f"outputs_vae_nu{nu}_kl0_mixed",
            f"outputs_vae_nu{nu}_no_kl_mixed",
        ):
            method_dir = results_dir / name
            if method_dir.exists():
                return _latest_run(method_dir)
        return _find_by_args(
            results_dir,
            nu,
            lambda args: _is_zero(args.get("beta_kl"))
            or _is_zero(args.get("kl_weight"))
            or _is_zero(args.get("beta")),
        )
    raise ValueError(f"Unknown ablation key: {ablation_key}")


def _read_curve(csv_path: Path) -> dict[str, np.ndarray]:
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


def _candidate_beta0_run_dirs(results_dir: Path, nu: int) -> list[Path]:
    run_dirs = []
    for name in (
        f"outputs_vae_nu{nu}_novar",
        f"outputs_vae_nu{nu}_zkl",
        f"outputs_vae_nu{nu}_beta0_mixed",
        f"outputs_vae_nu{nu}_beta_kl0_mixed",
        f"outputs_vae_nu{nu}_kl0_mixed",
        f"outputs_vae_nu{nu}_no_kl_mixed",
    ):
        method_dir = results_dir / name
        if method_dir.exists():
            run_dirs.extend(sorted(method_dir.glob("run_*"), reverse=True))

    args_match = _find_by_args(
        results_dir,
        nu,
        lambda args: _is_zero(args.get("beta_kl"))
        or _is_zero(args.get("kl_weight"))
        or _is_zero(args.get("beta")),
    )
    if args_match is not None and args_match not in run_dirs:
        run_dirs.append(args_match)
    return run_dirs


def _load_ablation_curve(
    results_dir: Path,
    nu: int,
    forcing: str,
    ablation_key: str,
) -> tuple[dict[str, np.ndarray], Path] | None:
    if ablation_key == "beta0":
        for run_dir in _candidate_beta0_run_dirs(results_dir, nu):
            csv_path = run_dir / f"ood_eval_{forcing}" / "test_rollout_error_curve.csv"
            if csv_path.exists():
                return _read_curve(csv_path), csv_path
        return None

    run_dir = _resolve_run_dir(results_dir, nu, ablation_key)
    if run_dir is None:
        return None
    csv_path = run_dir / f"ood_eval_{forcing}" / "test_rollout_error_curve.csv"
    if not csv_path.exists():
        return None
    return _read_curve(csv_path), csv_path


def _load_forcing_curves(
    results_dir: Path,
    nu: int,
    forcing: str,
    strict: bool,
) -> tuple[list[tuple], list[str]]:
    curves = []
    missing = []
    for key, label, color, linestyle, marker in ABLATIONS:
        loaded = _load_ablation_curve(results_dir, nu, forcing, key)
        if loaded is None:
            missing.append(label)
            if strict:
                raise FileNotFoundError(
                    f"Missing evaluated ablation {label} for nu={nu}, forcing={forcing}"
                )
            continue
        curve, source = loaded
        curves.append((label, color, linestyle, marker, curve, source))
    return curves, missing


def _apply_log_style(ax) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=10.0 ** np.asarray([0.2, 0.4, 0.6, 0.8]))
    )
    ax.yaxis.set_minor_formatter(NullFormatter())


def _marker_indices(steps: np.ndarray) -> list[int]:
    if steps.size == 0:
        return []
    t_max = float(np.max(steps))
    targets = np.concatenate(([1.0], np.linspace(t_max / 10.0, t_max, 10)))
    indices = []
    for target in targets:
        if target < float(np.min(steps)) or target > t_max:
            continue
        index = int(np.argmin(np.abs(steps - target)))
        if index not in indices:
            indices.append(index)
    return indices


def _time_values(curve: dict[str, np.ndarray], metric_key: str) -> np.ndarray:
    if "time" in curve:
        return np.asarray(curve["time"], dtype=np.float64)
    return np.arange(1, len(curve[metric_key]) + 1, dtype=np.float64)


def _format_tick(value: float) -> str:
    if abs(value - round(value)) < 1e-8:
        return str(int(round(value)))
    return f"{value:g}"


def make_plot(
    results_dir: Path,
    output_dir: Path,
    nu: int,
    dpi: int,
    strict: bool,
) -> Path | None:
    forcing_curves = []
    all_sources = []
    all_missing = []
    for forcing, forcing_label in FORCINGS:
        curves, missing = _load_forcing_curves(results_dir, nu, forcing, strict)
        if missing:
            all_missing.append((forcing_label, missing))
        for *_, source in curves:
            all_sources.append(source)
        forcing_curves.append((forcing_label, curves))

    available_curves = [curves for _, curves in forcing_curves if curves]
    if not available_curves:
        print(f"Skipping nu={nu}: no ablation curves found")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 6.0), sharex=True)
    base_time = _time_values(available_curves[0][0][4], METRICS[0][0])
    t_min = 0.0
    t_max = float(base_time[-1])
    x_right = t_max + 0.04 * max(t_max - t_min, 1.0)
    x_ticks = np.linspace(t_min, t_max, 6)
    train_horizon = min(TRAINING_HORIZONS.get(nu, t_max), t_max)

    for row, (forcing_label, curves) in enumerate(forcing_curves):
        for col, (metric_key, metric_label) in enumerate(METRICS):
            ax = axes[row, col]
            ax.axvspan(t_min, train_horizon, color="0.9", alpha=0.45, zorder=0)
            for label, color, linestyle, marker, curve, _ in curves:
                time = _time_values(curve, metric_key)
                steps = np.asarray(
                    curve.get("step", np.arange(1, len(time) + 1)), dtype=np.float64
                )
                y = np.asarray(curve[metric_key], dtype=np.float64)
                y = np.where(y > 0.0, y, np.nan)
                ax.plot(
                    time,
                    y,
                    label=label,
                    color=color,
                    linewidth=1.6,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=3.0,
                    markevery=_marker_indices(steps),
                    zorder=3,
                )
            ax.set_xlim(t_min, x_right)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([_format_tick(x) for x in x_ticks])
            if row == 0:
                ax.set_title(metric_label, fontsize=14, pad=9)
            if row == len(FORCINGS) - 1:
                ax.set_xlabel("time", fontsize=12, labelpad=5)
            ax.set_ylabel("Relative error", fontsize=12, labelpad=5)
            ax.grid(True, which="both", alpha=0.25, zorder=2)
            _apply_log_style(ax)

    for row, (forcing_label, _) in enumerate(forcing_curves):
        row_box = axes[row, 0].get_position()
        fig.text(
            0.03,
            0.5 * (row_box.y0 + row_box.y1) - 0.02,
            forcing_label,
            ha="center",
            va="center",
            fontsize=15,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Patch(facecolor="0.9", edgecolor="none", alpha=0.45))
    labels.append("training horizon")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=min(len(labels), 5),
        frameon=False,
        fontsize=13,
    )
    #fig.suptitle(rf"NS2D VAMO ablations, $\nu=10^{{-{nu}}}$", y=0.92, fontsize=15)
    fig.tight_layout(rect=(0.08, 0.0, 1.0, 0.92), h_pad=1.8, w_pad=2.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"ns2d_nu{nu}_ablation_trends.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out_path}")
    for forcing_label, missing in all_missing:
        print(f"  missing {forcing_label}: {', '.join(missing)}")
    for source in all_sources:
        print(f"  source: {source}")
    return out_path


def main() -> None:
    args = parse_args()
    written = []
    for nu in args.viscosities:
        path = make_plot(args.results_dir, args.output_dir, nu, args.dpi, args.strict)
        if path is not None:
            written.append(path)
    print("Wrote NS2D ablation trend plots:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
