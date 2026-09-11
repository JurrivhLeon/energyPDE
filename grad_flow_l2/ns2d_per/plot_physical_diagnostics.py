"""Plot NS2D true/predicted enstrophy and palinstrophy trajectory diagnostics."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter, NullLocator
import numpy as np
import torch

try:
    from ..heat_data import load_dataset_splits
    from . import eval as eval_ae
    from . import eval_fno
    from . import eval_vae
    from .train_vae import rollout_vae_latent_mean, rollout_vae_mean
except ImportError:
    from grad_flow_l2.heat_data import load_dataset_splits
    from grad_flow_l2.ns2d_per import eval as eval_ae
    from grad_flow_l2.ns2d_per import eval_fno
    from grad_flow_l2.ns2d_per import eval_vae
    from grad_flow_l2.ns2d_per.train_vae import (
        rollout_vae_latent_mean,
        rollout_vae_mean,
    )


METHODS: List[Tuple[Tuple[str, ...], str, str, str]] = [
    (("outputs_fno_nu{nu}_mixed",), "Vanilla FNO", "fno", "tab:blue"),
    (("outputs_fno_nu{nu}_noisy_mixed",), "FNO+Noise", "fno", "tab:orange"),
    (
        ("outputs_ae_nu{nu}_mixed", "outputs_sv_nu{nu}_mixed"),
        "FNO-AE",
        "ae",
        "tab:green",
    ),
    (("outputs_vae_nu{nu}_mixed",), "VAMO-FNO", "vae", "tab:red"),
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
        default=Path("grad_flow_l2/ns2d_per/results/physical_diagnostic_plots"),
        help="Directory where diagnostic figures are written.",
    )
    parser.add_argument("--viscosities", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"]
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _latest_run(method_dir: Path) -> Path:
    runs = sorted(p for p in method_dir.glob("run_*") if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"No run_* directories found under {method_dir}")
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


def _method_summary(
    results_dir: Path, nu: int, forcing: str, patterns: Tuple[str, ...]
) -> Tuple[Path, Dict]:
    method_dir = _resolve_method_dir(results_dir, nu, patterns)
    run_dir = _latest_run(method_dir)
    summary_path = run_dir / f"ood_eval_{forcing}" / "test_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    return run_dir, _load_json(summary_path)


def _namespace_with_defaults(values: Dict, defaults: Dict) -> Namespace:
    merged = dict(defaults)
    merged.update(values)
    return Namespace(**merged)


def _load_train_args(run_dir: Path) -> Dict:
    args_path = run_dir / "args.json"
    if not args_path.exists():
        return {}
    return _load_json(args_path)


def _time_metadata(meta: Dict, n_steps: int) -> Tuple[float, np.ndarray]:
    dt, _t_start, _t_end, time_values = eval_fno._time_metadata(meta, n_steps)
    return dt, time_values


def _slice_batch(
    split: Dict[str, torch.Tensor], batch_size: int, max_steps: int | None
) -> Dict[str, torch.Tensor]:
    n = min(int(batch_size), int(split["u0"].shape[0]))
    out = {
        "u0": split["u0"][:n].clone(),
        "f": split["f"][:n].clone(),
        "u_traj": split["u_traj"][:n].clone(),
    }
    if max_steps is not None:
        steps = min(int(max_steps), int(out["u_traj"].shape[1] - 1))
        out["u_traj"] = out["u_traj"][:, : steps + 1]
    return out


def _prepare_batch(
    dataset_path: str, split_name: str, batch_size: int, max_steps: int | None
) -> Tuple[Dict[str, torch.Tensor], Dict, float, np.ndarray]:
    splits = load_dataset_splits(dataset_path, map_location="cpu")
    split = _slice_batch(splits[split_name], batch_size=batch_size, max_steps=max_steps)
    meta = splits.get("meta", {})
    n_steps = int(split["u_traj"].shape[1] - 1)
    dt, time_values = _time_metadata(meta, n_steps)
    return split, meta, dt, time_values[: n_steps + 1]


def _scalar_vorticity(u: torch.Tensor) -> torch.Tensor:
    if u.dim() == 4:
        return u
    if u.dim() == 5 and u.shape[2] == 1:
        return u[:, :, 0]
    raise ValueError(
        f"Expected scalar vorticity trajectory, got shape {tuple(u.shape)}"
    )


def _enstrophy(u: torch.Tensor, area: float) -> torch.Tensor:
    u = _scalar_vorticity(u)
    return 0.5 * float(area) * torch.sum(u.square(), dim=(-2, -1))


def _palinstrophy(u: torch.Tensor, area: float) -> torch.Tensor:
    u = _scalar_vorticity(u)
    n_x, n_y = int(u.shape[-2]), int(u.shape[-1])
    u_hat = torch.fft.fft2(u, dim=(-2, -1), norm="ortho")
    kx = (
        2.0
        * torch.pi
        * torch.fft.fftfreq(n_x, d=1.0 / float(n_x), device=u.device).to(
            dtype=u.real.dtype
        )
    )
    ky = (
        2.0
        * torch.pi
        * torch.fft.fftfreq(n_y, d=1.0 / float(n_y), device=u.device).to(
            dtype=u.real.dtype
        )
    )
    kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing="ij")
    grad_weight = kx_grid.square() + ky_grid.square()
    power = u_hat.real.square() + u_hat.imag.square()
    return 0.5 * float(area) * torch.sum(power * grad_weight, dim=(-2, -1))


@torch.no_grad()
def _rollout_fno(
    run_dir: Path,
    checkpoint_path: str,
    split: Dict[str, torch.Tensor],
    dt: float,
    device: str,
    delta_clip: float | None,
) -> torch.Tensor:
    train_args = Namespace(**_load_train_args(run_dir))
    fallback = Namespace(
        state_channels=None,
        forcing_channels=None,
        width=64,
        fno_layers=4,
        fno_modes_x=16,
        fno_modes_y=16,
        disable_fno_grid=False,
        use_dt_channel=False,
        disable_forcing_channel=False,
        no_residual=False,
    )
    u0 = split["u0"].to(device)
    f = split["f"].to(device)
    n_x, n_y = int(u0.shape[-2]), int(u0.shape[-1])
    model = eval_fno._build_model(n_x, n_y, dt, split, fallback, train_args).to(device)
    ckpt = eval_fno._load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return (
        eval_fno._rollout(
            model, u0, f, int(split["u_traj"].shape[1] - 1), dt, delta_clip
        )
        .detach()
        .cpu()
    )


@torch.no_grad()
def _rollout_ae(
    run_dir: Path,
    checkpoint_path: str,
    split: Dict[str, torch.Tensor],
    dt: float,
    device: str,
    delta_clip: float | None,
) -> torch.Tensor:
    defaults = dict(
        prox_simulator_type="fno",
        hidden_channels=64,
        latent_channels=16,
        enc_blocks=4,
        dec_blocks=4,
        prox_blocks=6,
        fno_modes_x=16,
        fno_modes_y=16,
        disable_fno_grid=False,
        use_dt_channel=False,
        disable_forcing_channel=False,
        disable_u_grad_feature=False,
    )
    build_args = _namespace_with_defaults(_load_train_args(run_dir), defaults)
    u0 = split["u0"].to(device)
    f = split["f"].to(device)
    n_x, n_y = int(u0.shape[-2]), int(u0.shape[-1])
    model = eval_ae._build_model(n_x, n_y, 1.0 / n_x, 1.0 / n_y, dt, build_args).to(
        device
    )
    ckpt = eval_ae._load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(
        eval_ae._checkpoint_state_for_model(model, ckpt["model_state_dict"]),
        strict=True,
    )
    model.eval()
    return (
        eval_ae._rollout(
            model,
            u0,
            f,
            int(split["u_traj"].shape[1] - 1),
            dt,
            delta_clip=delta_clip,
            rollout_mode="physical",
        )
        .detach()
        .cpu()
    )


@torch.no_grad()
def _rollout_vae(
    run_dir: Path,
    checkpoint_path: str,
    split: Dict[str, torch.Tensor],
    dt: float,
    device: str,
    delta_clip: float | None,
    state_clip: float | None,
    rollout_mode: str,
) -> torch.Tensor:
    train_args = Namespace(**_load_train_args(run_dir))
    u0 = split["u0"].to(device)
    f = split["f"].to(device)
    n_x, n_y = int(u0.shape[-2]), int(u0.shape[-1])
    model = eval_vae._build_model(n_x=n_x, n_y=n_y, dt=dt, args=train_args).to(device)
    ckpt = eval_vae._torch_load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    n_steps = int(split["u_traj"].shape[1] - 1)
    if rollout_mode == "latent":
        pred = rollout_vae_latent_mean(
            model, u0=u0, f=f, n_steps=n_steps, dt=dt, state_clip=state_clip or 0.0
        )
    else:
        pred = rollout_vae_mean(
            model,
            u0=u0,
            f=f,
            n_steps=n_steps,
            dt=dt,
            delta_clip=delta_clip or 0.0,
            state_clip=state_clip or 0.0,
        )
    return pred.detach().cpu()


def _rollout_method(
    kind: str,
    run_dir: Path,
    summary: Dict,
    split: Dict[str, torch.Tensor],
    dt: float,
    device: str,
) -> torch.Tensor:
    checkpoint_path = summary.get("checkpoint_path") or summary.get("checkpoint")
    if checkpoint_path is None:
        checkpoint_path = str(run_dir / "best_model.pt")
    delta_clip = float(summary.get("delta_clip", 10.0))
    if delta_clip <= 0.0:
        delta_clip_arg = None
    else:
        delta_clip_arg = delta_clip
    if kind == "fno":
        return _rollout_fno(run_dir, checkpoint_path, split, dt, device, delta_clip_arg)
    if kind == "ae":
        return _rollout_ae(run_dir, checkpoint_path, split, dt, device, delta_clip_arg)
    if kind == "vae":
        state_clip = float(summary.get("state_clip", 20.0))
        rollout_mode = str(summary.get("rollout_mode", "latent"))
        return _rollout_vae(
            run_dir,
            checkpoint_path,
            split,
            dt,
            device,
            delta_clip_arg,
            state_clip,
            rollout_mode,
        )
    raise ValueError(f"Unknown method kind: {kind}")


def _format_log_tick_label(exponent: float) -> str:
    exponent = 0.0 if abs(float(exponent)) < 1e-8 else float(exponent)
    rounded = round(exponent)
    if abs(exponent - rounded) < 1e-8:
        if rounded == 0:
            return "1.0"
        return rf"$10^{{{rounded}}}$"
    return rf"$10^{{{exponent:g}}}$"


def _set_log_equally_spaced_yticks(
    ax: plt.Axes,
    values: np.ndarray,
    exponent_step: float = 0.4,
    max_labels: int = 5,
) -> None:
    positive = np.asarray(values, dtype=np.float64)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size == 0:
        return
    y_min = float(np.min(positive))
    y_max = float(np.max(positive))
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min <= 0.0 or y_max <= 0.0:
        return
    if y_min == y_max:
        y_min *= 0.8
        y_max *= 1.25

    log_min_data = np.log10(y_min)
    log_max_data = np.log10(y_max)
    log_min = np.floor(log_min_data / exponent_step) * exponent_step
    log_max = np.ceil(log_max_data / exponent_step) * exponent_step
    exponents = np.arange(log_min, log_max + 0.5 * exponent_step, exponent_step)
    exponents = np.where(np.abs(exponents) < 1e-8, 0.0, exponents)
    ticks = 10.0**exponents
    ax.set_ylim(10.0**log_min, 10.0**log_max)
    ax.set_yticks(ticks)

    stride = max(1, int(np.ceil(len(exponents) / max(1, int(max_labels)))))
    labels = []
    for idx, exponent in enumerate(exponents):
        if idx % stride == 0:
            labels.append(_format_log_tick_label(float(exponent)))
        else:
            labels.append("")
    ax.set_yticklabels(labels)
    ax.yaxis.set_minor_locator(NullLocator())


def _apply_sparse_log_ticks(ax: plt.Axes) -> None:
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=10.0 ** np.asarray([0.2, 0.4, 0.6, 0.8]))
    )
    ax.yaxis.set_minor_formatter(NullFormatter())


def _plot_panel(
    ax: plt.Axes,
    time_values: np.ndarray,
    true_values: np.ndarray,
    pred_values: np.ndarray,
    title: str,
    ylabel: str,
    color: str,
    train_horizon: float | None,
) -> None:
    if train_horizon is not None:
        ax.axvspan(
            float(time_values[0]),
            float(min(train_horizon, time_values[-1])),
            color="0.88",
            alpha=0.45,
            linewidth=0,
            zorder=0,
        )
    true_values = np.where(true_values > 0.0, true_values, np.nan)
    pred_values = np.where(pred_values > 0.0, pred_values, np.nan)
    for i in range(true_values.shape[0]):
        ax.plot(
            time_values,
            true_values[i],
            color=color,
            alpha=0.38,
            linewidth=0.95,
            zorder=2,
        )
        ax.plot(
            time_values,
            pred_values[i],
            color=color,
            alpha=0.95,
            linestyle="--",
            linewidth=1.15,
            zorder=3,
        )
    ax.set_yscale("log")
    _set_log_equally_spaced_yticks(
        ax, np.concatenate([true_values.ravel(), pred_values.ravel()])
    )
    ax.set_title(title, fontsize=14, pad=8)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25)


def make_plot(
    results_dir: Path,
    output_dir: Path,
    nu: int,
    forcing: str,
    forcing_label: str,
    split_name: str,
    batch_size: int,
    max_steps: int | None,
    device: str,
    dpi: int,
) -> Path:
    method_infos = []
    for patterns, method_label, kind, color in METHODS:
        run_dir, summary = _method_summary(results_dir, nu, forcing, patterns)
        method_infos.append((method_label, kind, color, run_dir, summary))

    dataset_path = str(method_infos[0][4]["dataset_path"])
    split, _meta, dt, time_values = _prepare_batch(
        dataset_path, split_name, batch_size, max_steps
    )
    u_ref = split["u_traj"]
    n_x, n_y = int(u_ref.shape[-2]), int(u_ref.shape[-1])
    area = (1.0 / float(n_x)) * (1.0 / float(n_y))
    true_ens = _enstrophy(u_ref, area).numpy()
    true_pal = _palinstrophy(u_ref, area).numpy()

    fig, axes = plt.subplots(2, 4, figsize=(14.0, 5.0), sharex=True)
    train_horizon = TRAINING_HORIZONS.get(nu)
    for col, (method_label, kind, color, run_dir, summary) in enumerate(method_infos):
        pred = _rollout_method(kind, run_dir, summary, split, dt, device)
        pred_ens = _enstrophy(pred, area).numpy()
        pred_pal = _palinstrophy(pred, area).numpy()
        _plot_panel(
            axes[0, col],
            time_values,
            true_ens,
            pred_ens,
            method_label,
            "Enstrophy",
            color,
            train_horizon,
        )
        _plot_panel(
            axes[1, col],
            time_values,
            true_pal,
            pred_pal,
            "",
            "Palinstrophy",
            color,
            train_horizon,
        )

    true_proxy = plt.Line2D(
        [0],
        [0],
        color="0.2",
        alpha=0.5,
        linewidth=1.2,
        label="Groundtruth",
    )
    pred_proxy = plt.Line2D(
        [0],
        [0],
        color="0.2",
        linestyle="--",
        linewidth=1.4,
        label="Predicted",
    )
    train_proxy = Patch(
        facecolor="0.88", edgecolor="none", alpha=0.45, label="Training horizon"
    )
    #fig.suptitle(
    #    rf"NS2D physical diagnostics, $\nu=10^{{-{nu}}}$, {forcing_label}, N={int(u_ref.shape[0])}",
    #    y=0.99,
    #    fontsize=15,
    #)
    fig.legend(
        handles=[true_proxy, pred_proxy, train_proxy],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.99),
        frameon=False,
        fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"ns2d_nu{nu}_{forcing}_physical_diagnostics.png"
    pdf_path = output_dir / f"ns2d_nu{nu}_{forcing}_physical_diagnostics.pdf"
    fig.savefig(out_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
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
                    args.split,
                    args.batch_size,
                    args.max_steps,
                    device,
                    args.dpi,
                )
            )
    print("Wrote physical diagnostic plots:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
