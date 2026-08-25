"""Plot selected Euler1D trajectory curves across trained baselines."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
import torch

try:
    from ..heat_data import load_dataset_splits
    from . import eval_fno
    from .common import (
        CHANNEL_NAMES,
        rollout_model_1d,
        rollout_vae_mean_1d,
        safe_torch_load,
    )
    from .train import _build_model as _build_ae_model
    from .train_vae import _build_model as _build_vae_model
except ImportError:
    from grad_flow_l2.heat_data import load_dataset_splits
    from grad_flow_l2.euler1d import eval_fno
    from grad_flow_l2.euler1d.common import (
        CHANNEL_NAMES,
        rollout_model_1d,
        rollout_vae_mean_1d,
        safe_torch_load,
    )
    from grad_flow_l2.euler1d.train import _build_model as _build_ae_model
    from grad_flow_l2.euler1d.train_vae import _build_model as _build_vae_model


SETTINGS = (5, 10, 15, 20)
SNAPSHOT_TIMES = (1.0, 2.0, 4.0, 6.0, 8.0, 10.0)
CHANNEL_LABELS = {"rho": "Density", "u": "Velocity", "p": "Pressure"}
METHODS: List[Tuple[str, str, str, str, str]] = [
    ("outputs_fno_L{setting}", "ood_eval", "Vanilla FNO", "fno", "#1f77b4"),
    ("outputs_fno_noisy_L{setting}", "ood_eval", "FNO+Noise", "fno", "#ff7f0e"),
    ("outputs_ae_L{setting}", "ood_eval", "FNO-AE", "ae", "#2ca02c"),
    ("outputs_vamo_L{setting}", "ood_eval", "VAMO-FNO", "vae", "#d62728"),
]
LINESTYLES = {
    "Ground truth": "-",
    "Vanilla FNO": "--",
    "FNO+Noise": ":",
    "FNO-AE": "-.",
    "VAMO-FNO": (0, (5, 2, 1, 2)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("grad_flow_l2/euler1d/trained_checkpoints")
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--settings", type=int, nargs="+", default=list(SETTINGS))
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"]
    )
    parser.add_argument("--sample-indices", type=str, default="0")
    parser.add_argument(
        "--snapshot-times",
        type=str,
        default=",".join(str(t) for t in SNAPSHOT_TIMES),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _parse_indices(raw: str) -> List[int]:
    indices = [int(tok.strip()) for tok in raw.split(",") if tok.strip()]
    if not indices:
        raise ValueError("At least one sample index is required")
    if any(idx < 0 for idx in indices):
        raise ValueError(f"Sample indices must be nonnegative: {indices}")
    return indices


def _parse_times(raw: str) -> List[float]:
    times = [float(tok.strip()) for tok in raw.split(",") if tok.strip()]
    if len(times) != 6:
        raise ValueError(
            "The Euler1D 2x4 trajectory layout expects exactly 6 snapshot times"
        )
    if any(t <= 0.0 for t in times):
        raise ValueError(f"Snapshot times must be positive: {times}")
    return times


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _latest_run(method_dir: Path) -> Path:
    runs = sorted(p for p in method_dir.glob("run_*") if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"No run_* directories found under {method_dir}")
    return runs[-1]


def _method_summary(
    root: Path, setting: int, template: str, eval_dir: str
) -> Tuple[Path, Dict]:
    run_dir = _latest_run(root / template.format(setting=setting))
    summary_path = run_dir / eval_dir / "test_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    return run_dir, _load_json(summary_path)


def _checkpoint_path(run_dir: Path, summary: Dict) -> str:
    value = summary.get("checkpoint_path") or summary.get("checkpoint")
    if value is not None and Path(value).exists():
        return str(value)
    fallback = run_dir / "best_model.pt"
    if fallback.exists():
        return str(fallback)
    if value is not None:
        return str(value)
    raise FileNotFoundError(f"No checkpoint found for {run_dir}")


def _train_args(checkpoint_path: str, summary: Dict) -> Namespace:
    candidates = [Path(checkpoint_path).parent / "args.json"]
    if summary.get("args_json") is not None:
        candidates.append(Path(str(summary["args_json"])))
    for path in candidates:
        if path.exists():
            return Namespace(**_load_json(path))
    return Namespace()


def _dt_from_meta(meta: Dict, n_steps: int) -> float:
    return float(
        meta.get("dataset_dt", meta.get("t_final", float(n_steps)) / float(n_steps))
    )


def _domain_length(meta: Dict, args: Namespace) -> float:
    return float(meta.get("domain_length", getattr(args, "domain_length", 1.0)))


def _boundary_condition(meta: Dict, args: Namespace) -> str:
    return str(
        meta.get("boundary_condition", getattr(args, "boundary_condition", "periodic"))
    )


def _select_split(
    dataset_path: str, split_name: str, indices: List[int], max_step: int
):
    splits = load_dataset_splits(dataset_path, map_location="cpu")
    split = splits[split_name]
    total = int(split["u0"].shape[0])
    bad = [idx for idx in indices if idx >= total]
    if bad:
        raise IndexError(f"Sample indices out of range for split size {total}: {bad}")
    available = int(split["u_traj"].shape[1] - 1)
    if max_step > available:
        raise ValueError(
            f"Requested step {max_step}, but trajectory has only {available} steps"
        )
    idx = torch.as_tensor(indices, dtype=torch.long)
    out = {
        "u0": split["u0"].index_select(0, idx).clone(),
        "f": split["f"].index_select(0, idx).clone(),
        "u_traj": split["u_traj"].index_select(0, idx)[:, : max_step + 1].clone(),
    }
    out["u0"] = out["u_traj"][:, 0].clone()
    return out, splits.get("meta", {})


@torch.no_grad()
def _rollout_fno(
    run_dir: Path, summary: Dict, split: Dict, dt: float, device: str
) -> torch.Tensor:
    checkpoint = _checkpoint_path(run_dir, summary)
    train_args = _train_args(checkpoint, summary)
    fallback = Namespace(
        width=64,
        fno_layers=6,
        fno_modes=32,
        disable_fno_grid=False,
        use_dt_channel=False,
        disable_forcing_channel=False,
        no_residual=False,
        lift_noise_std=0.0,
        lift_noise_corr_length=1.0,
        lift_noise_decay_s=2.0,
    )
    model = eval_fno._build_model(
        n_x=int(split["u0"].shape[-1]), dt=dt, args=fallback, train_args=train_args
    ).to(device)
    ckpt = safe_torch_load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    delta_clip = ckpt.get("rollout_delta_clip", summary.get("delta_clip", None))
    delta_clip = (
        None if delta_clip is None or float(delta_clip) <= 0 else float(delta_clip)
    )
    return (
        rollout_model_1d(
            model,
            split["u0"].to(device),
            split["f"].to(device),
            n_steps=int(split["u_traj"].shape[1] - 1),
            dt=dt,
            delta_clip=delta_clip,
        )
        .detach()
        .cpu()
    )


@torch.no_grad()
def _rollout_ae(
    run_dir: Path, summary: Dict, split: Dict, dt: float, meta: Dict, device: str
) -> torch.Tensor:
    checkpoint = _checkpoint_path(run_dir, summary)
    train_args = _train_args(checkpoint, summary)
    model = _build_ae_model(
        n_x=int(split["u0"].shape[-1]),
        dt=dt,
        boundary_condition=_boundary_condition(meta, train_args),
        args=train_args,
    ).to(device)
    ckpt = safe_torch_load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    delta_clip = ckpt.get("rollout_delta_clip", summary.get("delta_clip", None))
    delta_clip = (
        None if delta_clip is None or float(delta_clip) <= 0 else float(delta_clip)
    )
    return (
        rollout_model_1d(
            model,
            split["u0"].to(device),
            split["f"].to(device),
            n_steps=int(split["u_traj"].shape[1] - 1),
            dt=dt,
            delta_clip=delta_clip,
        )
        .detach()
        .cpu()
    )


@torch.no_grad()
def _rollout_vae(
    run_dir: Path, summary: Dict, split: Dict, dt: float, meta: Dict, device: str
) -> torch.Tensor:
    checkpoint = _checkpoint_path(run_dir, summary)
    train_args = _train_args(checkpoint, summary)
    model = _build_vae_model(
        n_x=int(split["u0"].shape[-1]),
        dt=dt,
        boundary_condition=_boundary_condition(meta, train_args),
        args=train_args,
    ).to(device)
    ckpt = safe_torch_load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    delta_clip = ckpt.get("rollout_delta_clip", summary.get("delta_clip", None))
    delta_clip = (
        None if delta_clip is None or float(delta_clip) <= 0 else float(delta_clip)
    )
    return (
        rollout_vae_mean_1d(
            model,
            split["u0"].to(device),
            split["f"].to(device),
            n_steps=int(split["u_traj"].shape[1] - 1),
            dt=dt,
            delta_clip=delta_clip,
        )
        .detach()
        .cpu()
    )


def _rollout_method(
    kind: str,
    run_dir: Path,
    summary: Dict,
    split: Dict,
    dt: float,
    meta: Dict,
    device: str,
):
    if kind == "fno":
        return _rollout_fno(run_dir, summary, split, dt, device)
    if kind == "ae":
        return _rollout_ae(run_dir, summary, split, dt, meta, device)
    if kind == "vae":
        return _rollout_vae(run_dir, summary, split, dt, meta, device)
    raise ValueError(f"Unknown method kind: {kind}")


def _style_axes(ax: plt.Axes, x: np.ndarray) -> None:
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=10, pad=2.5)
    ax.grid(True, color="#d9d9d9", linewidth=0.5, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#999999")


def _plot_sample(
    output_dir: Path,
    setting: int,
    sample_id: int,
    local_index: int,
    split: Dict,
    meta: Dict,
    dt: float,
    steps: List[int],
    preds: List[Tuple[str, str, torch.Tensor]],
    dpi: int,
) -> Path:
    ref = split["u_traj"][local_index].detach().cpu().numpy()
    pred_arrays = [
        (label, color, pred[local_index].detach().cpu().numpy())
        for label, color, pred in preds
    ]
    domain_length = _domain_length(meta, Namespace())
    x = np.linspace(0.0, domain_length, int(ref.shape[-1]), endpoint=False)

    fig = plt.figure(figsize=(16.5, 13.0), constrained_layout=False)
    outer_left, outer_right, outer_top, outer_bottom = 0.085, 0.985, 0.900, 0.055
    panel_hspace = 0.30
    outer = GridSpec(
        len(CHANNEL_NAMES),
        1,
        figure=fig,
        left=outer_left,
        right=outer_right,
        top=outer_top,
        bottom=outer_bottom,
        hspace=panel_hspace,
    )
    slots = [
        ("IC", 0),
        (f"t={steps[0] * dt:.1f}", steps[0]),
        (f"t={steps[1] * dt:.1f}", steps[1]),
        (f"t={steps[2] * dt:.1f}", steps[2]),
        ("", None),
        (f"t={steps[3] * dt:.1f}", steps[3]),
        (f"t={steps[4] * dt:.1f}", steps[4]),
        (f"t={steps[5] * dt:.1f}", steps[5]),
    ]
    panel_height = (outer_top - outer_bottom) / (
        len(CHANNEL_NAMES) + panel_hspace * (len(CHANNEL_NAMES) - 1)
    )
    panel_gap = panel_hspace * panel_height
    legend_source_ax = None

    for c, name in enumerate(CHANNEL_NAMES):
        channel_label = CHANNEL_LABELS.get(name, name)
        panel_top = outer_top - c * (panel_height + panel_gap)
        fig.text(
            outer_left - 0.045,
            panel_top - 0.5 * panel_height,
            channel_label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=18,
        )
        inner = GridSpecFromSubplotSpec(
            2, 4, subplot_spec=outer[c], wspace=0.16, hspace=0.48
        )
        values = [ref[0, c]] + [ref[step, c] for step in steps]
        for _label, _color, pred in pred_arrays:
            values.extend(pred[step, c] for step in steps)
        finite = np.concatenate([np.asarray(v)[np.isfinite(v)].ravel() for v in values])
        ymin = float(np.min(finite)) if finite.size else -1.0
        ymax = float(np.max(finite)) if finite.size else 1.0
        pad = 0.08 * max(ymax - ymin, 1e-8)

        for slot_idx, (title, step) in enumerate(slots):
            row, col = divmod(slot_idx, 4)
            ax = fig.add_subplot(inner[row, col])
            if step is None:
                ax.axis("off")
                continue
            ax.plot(
                x,
                ref[step, c],
                color="black",
                linestyle="-",
                linewidth=1.7,
                alpha=0.42,
                label="Ground truth",
            )
            if step > 0:
                for method_label, color, pred in pred_arrays:
                    ax.plot(
                        x,
                        pred[step, c],
                        color=color,
                        linestyle=LINESTYLES[method_label],
                        linewidth=1.25,
                        alpha=0.96,
                        label=method_label,
                    )
                if legend_source_ax is None:
                    legend_source_ax = ax
            ax.set_ylim(ymin - pad, ymax + pad)
            ax.set_title(title, fontsize=15, pad=8)
            _style_axes(ax, x)
            ax.set_xticks(np.linspace(0.2 * domain_length, 0.8 * domain_length, 4))
            show_x_labels = col == 0 or (c == len(CHANNEL_NAMES) - 1 and row == 1)
            if show_x_labels:
                ax.tick_params(axis="x", labelsize=8)
                if c == len(CHANNEL_NAMES) - 1 and row == 1:
                    ax.set_xlabel("x", fontsize=15, labelpad=6)
            else:
                ax.tick_params(axis="x", labelbottom=False)

    if legend_source_ax is not None:
        handles, labels = legend_source_ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=5,
            frameon=False,
            fontsize=18,
            bbox_to_anchor=(0.5, 0.985),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = (
        output_dir / f"euler1d_L{setting}_sample_{sample_id:04d}_trajectory_curves.png"
    )
    pdf_path = (
        output_dir / f"euler1d_L{setting}_sample_{sample_id:04d}_trajectory_curves.pdf"
    )
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def make_plots_for_setting(
    root: Path,
    output_dir: Path,
    setting: int,
    split_name: str,
    sample_indices: List[int],
    snapshot_times: List[float],
    device: str,
    dpi: int,
) -> List[Path]:
    method_infos = []
    for template, eval_dir, label, kind, color in METHODS:
        run_dir, summary = _method_summary(root, setting, template, eval_dir)
        method_infos.append((label, kind, color, run_dir, summary))

    dataset_path = str(method_infos[0][4]["dataset_path"])
    splits = load_dataset_splits(dataset_path, map_location="cpu")
    meta = splits.get("meta", {})
    dt = _dt_from_meta(meta, int(splits[split_name]["u_traj"].shape[1] - 1))
    steps = [int(round(t / dt)) for t in snapshot_times]
    split, meta = _select_split(dataset_path, split_name, sample_indices, max(steps))
    dt = _dt_from_meta(meta, int(split["u_traj"].shape[1] - 1))
    steps = [int(round(t / dt)) for t in snapshot_times]

    preds = []
    for label, kind, color, run_dir, summary in method_infos:
        pred = _rollout_method(kind, run_dir, summary, split, dt, meta, device)
        preds.append((label, color, pred))

    written = []
    setting_dir = output_dir / f"L{setting}"
    for local_index, sample_id in enumerate(sample_indices):
        written.append(
            _plot_sample(
                setting_dir,
                setting,
                sample_id,
                local_index,
                split,
                meta,
                dt,
                steps,
                preds,
                dpi,
            )
        )
    return written


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.root / "selected_trajectory_plots")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    written = []
    for setting in args.settings:
        written.extend(
            make_plots_for_setting(
                args.root,
                output_dir,
                setting,
                args.split,
                _parse_indices(args.sample_indices),
                _parse_times(args.snapshot_times),
                device,
                args.dpi,
            )
        )
    print("Wrote Euler1D selected trajectory plots:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
