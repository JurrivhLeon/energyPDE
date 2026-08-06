"""Plot selected CFD2D trajectory snapshots across baselines."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from ..heat_data import load_dataset_splits
    from ..latent_markov_trainer_mc import rollout_latent_markov_2d
    from . import eval as eval_ae
    from . import eval_fno
    from .cfd_data import STATE_CHANNELS
    from .train import _build_model as _build_ae_model
    from .train_vae import _build_model as _build_vae_model
    from .train_vae import _rollout_vae_mean
except ImportError:
    from grad_flow_l2.heat_data import load_dataset_splits
    from grad_flow_l2.latent_markov_trainer_mc import rollout_latent_markov_2d
    from grad_flow_l2.cfd2d import eval as eval_ae
    from grad_flow_l2.cfd2d import eval_fno
    from grad_flow_l2.cfd2d.cfd_data import STATE_CHANNELS
    from grad_flow_l2.cfd2d.train import _build_model as _build_ae_model
    from grad_flow_l2.cfd2d.train_vae import _build_model as _build_vae_model
    from grad_flow_l2.cfd2d.train_vae import _rollout_vae_mean


CHANNEL_NAMES = ["rho", "vx", "vy", "p"]
CHANNEL_CMAP = {
    "rho": "sciviz.blue_medb717b",
    "vx": "coolwarm",
    "vy": "coolwarm",
    "p": "crest",
}
CHANNEL_SYMMETRIC = {"rho": False, "vx": True, "vy": True, "p": False}
CHANNEL_DISPLAY_NAMES = {
    "rho": "Density",
    "vx": "Velocity (x)",
    "vy": "Velocity (y)",
    "p": "Pressure",
}
SNAPSHOT_STEPS = tuple(range(10, 101, 10))
METHODS: List[Tuple[str, str, str, str]] = [
    ("outputs_fno_nu8", "ood_eval", "Vanilla FNO", "fno"),
    ("outputs_fno_noisy_nu8", "ood_eval", "FNO+Noise", "fno"),
    ("outputs_ae_nu8", "ood_eval_t10", "FNO-AE", "ae"),
    ("outputs_vamo_nu8", "ood_eval_t10", "VAMO-FNO", "vae"),
]


def _install_register_cmap_compat() -> None:
    import matplotlib as mpl
    import matplotlib.cm as cm

    if hasattr(cm, "register_cmap"):
        return

    def register_cmap(name=None, cmap=None, **kwargs):
        if cmap is None:
            cmap = name
            name = getattr(cmap, "name", None)
        force = bool(kwargs.get("override_builtin", False)) or bool(
            kwargs.get("force", False)
        )
        try:
            mpl.colormaps.register(cmap, name=name, force=force)
        except ValueError:
            if force:
                raise
        return None

    cm.register_cmap = register_cmap


def _get_channel_cmap(name: str):
    cmap_name = CHANNEL_CMAP[name]
    if cmap_name == "sciviz.blue_medb717b":
        try:
            import colormaps as cmaps

            return cmaps.blue_medb717b
        except Exception:
            cmap_name = "scientific.oslo"
    if cmap_name == "scientific.oslo":
        try:
            from cmcrameri import cm as cmc

            return cmc.oslo
        except Exception:
            try:
                import seaborn as sns

                return sns.color_palette("light:b", as_cmap=True)
            except Exception:
                return "Blues"
    if cmap_name in {"carbonplan.water", "carbonplan.water_light"}:
        try:
            _install_register_cmap_compat()
            from carbonplan_styles.mpl import colormaps

            return getattr(colormaps, cmap_name.split(".")[-1])
        except Exception:
            try:
                import seaborn as sns

                return sns.color_palette("light:b", as_cmap=True)
            except Exception:
                return "Blues"
    if cmap_name in {"mako", "crest", "light:b"}:
        try:
            import seaborn as sns

            return sns.color_palette(cmap_name, as_cmap=True)
        except Exception:
            return "Blues" if name == "rho" else "viridis"
    return cmap_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("grad_flow_l2/cfd2d/trained_checkpoints"),
        help="Directory containing CFD2D trained checkpoint/result folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "grad_flow_l2/cfd2d/trained_checkpoints/selected_trajectory_plots"
        ),
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"]
    )
    parser.add_argument("--sample-indices", type=str, default="0")
    parser.add_argument(
        "--snapshot-steps", type=str, default=",".join(str(s) for s in SNAPSHOT_STEPS)
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


def _parse_steps(raw: str) -> List[int]:
    steps = [int(tok.strip()) for tok in raw.split(",") if tok.strip()]
    if not steps:
        raise ValueError("At least one snapshot step is required")
    if any(step <= 0 for step in steps):
        raise ValueError(f"Snapshot steps must be positive: {steps}")
    return steps


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _latest_run(method_dir: Path) -> Path:
    runs = sorted(p for p in method_dir.glob("run_*") if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"No run_* directories found under {method_dir}")
    return runs[-1]


def _method_summary(
    root: Path, method_dir_name: str, eval_dir_name: str
) -> Tuple[Path, Dict]:
    run_dir = _latest_run(root / method_dir_name)
    summary_path = run_dir / eval_dir_name / "test_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    return run_dir, _load_json(summary_path)


def _load_train_args(path: Path) -> Namespace:
    if path.exists():
        return Namespace(**_load_json(path))
    return Namespace()


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


def _select_split(
    dataset_path: str, split_name: str, sample_indices: List[int]
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    splits = load_dataset_splits(dataset_path, map_location="cpu")
    split = splits[split_name]
    total = int(split["u0"].shape[0])
    bad = [idx for idx in sample_indices if idx >= total]
    if bad:
        raise IndexError(f"Sample indices out of range for split size {total}: {bad}")
    index = torch.as_tensor(sample_indices, dtype=torch.long)
    selected = {
        "u0": split["u0"].index_select(0, index).clone(),
        "f": split["f"].index_select(0, index).clone(),
        "u_traj": split["u_traj"].index_select(0, index).clone(),
    }
    return selected, splits.get("meta", {})


def _dt_from_meta(meta: Dict, n_steps: int) -> float:
    return float(meta.get("t_final", float(n_steps))) / float(n_steps)


def _truncate_for_steps(
    split: Dict[str, torch.Tensor], max_step: int
) -> Dict[str, torch.Tensor]:
    available = int(split["u_traj"].shape[1] - 1)
    if max_step > available:
        raise ValueError(
            f"Requested snapshot step {max_step}, but trajectory has only {available} steps"
        )
    out = dict(split)
    out["u_traj"] = split["u_traj"][:, : max_step + 1].clone()
    out["u0"] = out["u_traj"][:, 0].clone()
    return out


@torch.no_grad()
def _rollout_fno(
    run_dir: Path, summary: Dict, split: Dict[str, torch.Tensor], dt: float, device: str
) -> torch.Tensor:
    train_args = _load_train_args(
        Path(_checkpoint_path(run_dir, summary)).parent / "args.json"
    )
    fallback = Namespace(
        width=64,
        fno_layers=6,
        fno_modes_x=16,
        fno_modes_y=16,
        disable_fno_grid=False,
        use_dt_channel=False,
        disable_forcing_channel=False,
        no_residual=False,
        lift_noise_std=0.0,
        lift_noise_corr_length=1.0,
        lift_noise_decay_s=2.0,
    )
    u0 = split["u0"].to(device)
    f = split["f"].to(device)
    n_x, n_y = int(u0.shape[-2]), int(u0.shape[-1])
    model = eval_fno._build_model(n_x, n_y, dt, fallback, train_args).to(device)
    checkpoint = eval_ae._torch_load_checkpoint(
        _checkpoint_path(run_dir, summary), map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    delta_clip = eval_ae._resolve_delta_clip(
        None, checkpoint, default=float(summary.get("delta_clip", 10.0))
    )
    return (
        rollout_latent_markov_2d(
            model,
            u0=u0,
            f=f,
            n_steps=int(split["u_traj"].shape[1] - 1),
            dt=dt,
            delta_clip=delta_clip,
        )
        .detach()
        .cpu()
    )


@torch.no_grad()
def _rollout_ae(
    run_dir: Path, summary: Dict, split: Dict[str, torch.Tensor], dt: float, device: str
) -> torch.Tensor:
    train_args = _load_train_args(
        Path(_checkpoint_path(run_dir, summary)).parent / "args.json"
    )
    u0 = split["u0"].to(device)
    f = split["f"].to(device)
    n_x, n_y = int(u0.shape[-2]), int(u0.shape[-1])
    model = _build_ae_model(n_x=n_x, n_y=n_y, dt=dt, args=train_args).to(device)
    checkpoint = eval_ae._torch_load_checkpoint(
        _checkpoint_path(run_dir, summary), map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    delta_clip = eval_ae._resolve_delta_clip(
        None, checkpoint, default=float(summary.get("delta_clip", 10.0))
    )
    return (
        rollout_latent_markov_2d(
            model,
            u0=u0,
            f=f,
            n_steps=int(split["u_traj"].shape[1] - 1),
            dt=dt,
            delta_clip=delta_clip,
        )
        .detach()
        .cpu()
    )


@torch.no_grad()
def _rollout_vae(
    run_dir: Path, summary: Dict, split: Dict[str, torch.Tensor], dt: float, device: str
) -> torch.Tensor:
    train_args = _load_train_args(
        Path(_checkpoint_path(run_dir, summary)).parent / "args.json"
    )
    u0 = split["u0"].to(device)
    f = split["f"].to(device)
    n_x, n_y = int(u0.shape[-2]), int(u0.shape[-1])
    model = _build_vae_model(n_x=n_x, n_y=n_y, dt=dt, args=train_args).to(device)
    checkpoint = eval_ae._torch_load_checkpoint(
        _checkpoint_path(run_dir, summary), map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    delta_clip = eval_ae._resolve_delta_clip(
        None, checkpoint, default=float(summary.get("delta_clip", 1.0))
    )
    return (
        _rollout_vae_mean(
            model,
            u0=u0,
            f=f,
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
    split: Dict[str, torch.Tensor],
    dt: float,
    device: str,
) -> torch.Tensor:
    if kind == "fno":
        return _rollout_fno(run_dir, summary, split, dt, device)
    if kind == "ae":
        return _rollout_ae(run_dir, summary, split, dt, device)
    if kind == "vae":
        return _rollout_vae(run_dir, summary, split, dt, device)
    raise ValueError(f"Unknown method kind: {kind}")


def _channel_limits(
    ref: np.ndarray, channel: int, steps: List[int]
) -> Tuple[float, float]:
    values = ref[np.asarray([0] + steps), channel]
    name = CHANNEL_NAMES[channel]
    if CHANNEL_SYMMETRIC[name]:
        scale = max(float(np.nanmax(np.abs(values))), 1e-8)
        return -scale, scale
    return float(np.nanmin(values)), float(np.nanmax(values))


def _plot_sample(
    output_dir: Path,
    sample_id: int,
    local_index: int,
    split: Dict[str, torch.Tensor],
    preds: List[Tuple[str, torch.Tensor]],
    steps: List[int],
    dt: float,
    dpi: int,
) -> Path:
    ref = split["u_traj"][local_index].detach().cpu().numpy()
    pred_arrays = [
        (label, pred[local_index].detach().cpu().numpy()) for label, pred in preds
    ]

    n_fields = len(CHANNEL_NAMES)
    n_rows = 1 + len(pred_arrays)
    n_cols = 1 + len(steps)
    fig_w = 18.0
    fig_h = 24.0
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)

    left = 0.035
    label_w = 0.14
    grid_left = left + label_w + 0.015
    top = 0.960
    bottom = 0.040
    panel_gap = 0.018
    row_gap = 0.0008
    col_gap = 0.0008 * fig_h / fig_w
    ic_gap = 0.012
    cbar_gap = 0.018
    cbar_w = 0.010

    panel_h = (top - bottom - (n_fields - 1) * panel_gap) / n_fields
    cell_h = (panel_h - (n_rows - 1) * row_gap) / n_rows
    cell_w = cell_h * fig_h / fig_w
    grid_w = n_cols * cell_w + (n_cols - 2) * col_gap + ic_gap
    cbar_x = grid_left + grid_w + cbar_gap

    row_labels = ["Groundtruth"] + [label for label, _traj in pred_arrays]
    col_labels = ["IC"] + ["t=" + str(step * 0.1) for step in steps]

    for field_idx, field_name in enumerate(CHANNEL_NAMES):
        panel_top = top - field_idx * (panel_h + panel_gap)
        panel_bottom = panel_top - panel_h
        field_label_ax = fig.add_axes([left, panel_bottom, label_w - 0.006, panel_h])
        field_label_ax.axis("off")
        field_label_ax.text(
            0.0,
            0.5,
            CHANNEL_DISPLAY_NAMES[field_name],
            ha="left",
            va="center",
            rotation=90,
            fontsize=25,
            # fontweight="bold",
        )

        vmin, vmax = _channel_limits(ref, field_idx, steps)
        cmap = _get_channel_cmap(field_name)
        im_last = None
        row_trajs = [ref] + [traj for _label, traj in pred_arrays]

        for row, (row_label, traj) in enumerate(zip(row_labels, row_trajs)):
            y = panel_top - (row + 1) * cell_h - row * row_gap
            row_label_ax = fig.add_axes([left + 0.028, y, label_w - 0.034, cell_h])
            row_label_ax.axis("off")
            row_label_ax.text(
                0.98,
                0.5,
                row_label,
                ha="right",
                va="center",
                fontsize=15,
                fontweight="semibold" if row == 0 else "normal",
            )

            for col in range(n_cols):
                x = (
                    grid_left
                    if col == 0
                    else grid_left + cell_w + ic_gap + (col - 1) * (cell_w + col_gap)
                )
                ax = fig.add_axes([x, y, cell_w, cell_h])
                if row > 0 and col == 0:
                    ax.axis("off")
                    continue
                step = 0 if col == 0 else steps[col - 1]
                im_last = ax.imshow(
                    traj[step, field_idx],
                    origin="lower",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_frame_on(False)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if field_idx == 0 and row == 0:
                    ax.set_title(col_labels[col], fontsize=15, pad=9)

        if im_last is not None:
            cax = fig.add_axes([cbar_x, panel_bottom, cbar_w, panel_h])
            cbar = fig.colorbar(im_last, cax=cax)
            cbar.ax.tick_params(labelsize=10)

    """fig.suptitle(
        "CFD2D trajectory comparison, sample "
        + str(sample_id)
        + " | snapshots "
        + ", ".join(map(str, steps)),
        y=0.985,
        fontsize=25,
    )"""

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"cfd2d_sample_{sample_id:04d}_trajectory_comparison.png"
    pdf_path = output_dir / f"cfd2d_sample_{sample_id:04d}_trajectory_comparison.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def main() -> None:
    args = parse_args()
    sample_indices = _parse_indices(args.sample_indices)
    steps = _parse_steps(args.snapshot_steps)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    method_infos = []
    for method_dir, eval_dir, label, kind in METHODS:
        run_dir, summary = _method_summary(args.root, method_dir, eval_dir)
        method_infos.append((label, kind, run_dir, summary))

    dataset_path = str(method_infos[0][3]["dataset_path"])
    split, meta = _select_split(dataset_path, args.split, sample_indices)
    split = _truncate_for_steps(split, max(steps))
    dt = _dt_from_meta(meta, int(split["u_traj"].shape[1] - 1))

    preds = []
    for label, kind, run_dir, summary in method_infos:
        pred = _rollout_method(kind, run_dir, summary, split, dt, device)
        preds.append((label, pred))

    written = []
    for local_index, sample_id in enumerate(sample_indices):
        written.append(
            _plot_sample(
                args.output_dir,
                sample_id,
                local_index,
                split,
                preds,
                steps,
                dt,
                args.dpi,
            )
        )
    print("Wrote CFD2D selected trajectory plots:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
