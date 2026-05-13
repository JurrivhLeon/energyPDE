"""Evaluation entrypoint for periodic 2D FitzHugh-Nagumo hidden-space checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
from argparse import Namespace
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from ..fhn2d.fhn_data import build_fhn2d_step_dataset, build_fhn2d_trajectory_dataset_from_split
    from ..heat_data import load_dataset_splits
    from ..navier_stokes2d.trainer import HiddenGradientFlowTrainer2D, rollout_model_2d
    from .train import _build_model
except ImportError:
    from grad_flow_l2.fhn2d.fhn_data import build_fhn2d_step_dataset, build_fhn2d_trajectory_dataset_from_split
    from grad_flow_l2.heat_data import load_dataset_splits
    from grad_flow_l2.navier_stokes2d.trainer import HiddenGradientFlowTrainer2D, rollout_model_2d
    from grad_flow_l2.fhn2d.train import _build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate periodic 2D FitzHugh-Nagumo hidden-space checkpoint")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=str, default="grad_flow_l2/fhn2d/outputs/eval")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-plot-samples", type=int, default=6)
    parser.add_argument("--snapshot-times", type=str, default="")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def _torch_load_checkpoint(checkpoint_path: str, map_location):
    try:
        return torch.load(checkpoint_path, map_location=map_location)
    except RuntimeError as exc:
        msg = str(exc)
        if "weights_only=True" not in msg or "legacy .tar format" not in msg:
            raise
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def _parse_snapshot_times(raw: str, t_final: float) -> List[float]:
    if not raw.strip():
        return [0.0, 0.25 * t_final, 0.5 * t_final, 0.75 * t_final, t_final]
    vals = [float(tok.strip()) for tok in raw.split(",") if tok.strip()]
    for val in vals:
        if val < 0.0 or val > t_final:
            raise ValueError(f"Snapshot time must be in [0,{t_final}], got {val}")
    return vals


@torch.no_grad()
def _evaluate_rollout_curves(model, traj_loader: DataLoader, device: str, dt: float, area: float) -> Dict[str, np.ndarray]:
    mse_batches = []
    rel_batches = []
    for batch in traj_loader:
        u0 = batch["u0"].to(device)
        f = batch["f"].to(device)
        u_ref = batch["u_traj"].to(device)
        u_pred = rollout_model_2d(model, u0=u0, f=f, n_steps=int(u_ref.shape[1] - 1), dt=dt)
        diff = u_pred - u_ref
        mse_batches.append(torch.mean(diff * diff, dim=(2, 3, 4)).detach().cpu())
        num = torch.sqrt(float(area) * torch.sum(diff * diff, dim=(2, 3, 4)))
        den = torch.sqrt(float(area) * torch.sum(u_ref * u_ref, dim=(2, 3, 4)))
        rel_batches.append((num / (den + 1e-8)).detach().cpu())
    mse = torch.cat(mse_batches, dim=0)
    rel = torch.cat(rel_batches, dim=0)
    return {
        "mse_curve_mean": torch.nanmean(mse, dim=0).numpy().astype(np.float64),
        "mse_curve_median": np.nanmedian(mse.numpy(), axis=0).astype(np.float64),
        "rel_curve_mean": torch.nanmean(rel, dim=0).numpy().astype(np.float64),
        "rel_curve_median": np.nanmedian(rel.numpy(), axis=0).astype(np.float64),
        "rollout_rel_mean": float(torch.nanmean(rel.mean(dim=1)).item()),
        "rollout_rel_median": float(np.nanmedian(rel.mean(dim=1).numpy())),
    }


def _save_curve_csv(curves: Dict[str, np.ndarray], dt: float, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "time", "mse_mean", "mse_median", "rel_l2_mean", "rel_l2_median"])
        for k in range(len(curves["mse_curve_mean"])):
            writer.writerow([k, f"{k * dt:.8f}", curves["mse_curve_mean"][k], curves["mse_curve_median"][k], curves["rel_curve_mean"][k], curves["rel_curve_median"][k]])


def _plot_curve(curves: Dict[str, np.ndarray], dt: float, path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping curve plot because matplotlib is unavailable: {exc}")
        return
    t = np.arange(curves["mse_curve_mean"].shape[0]) * float(dt)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
    axes[0, 0].plot(t, curves["mse_curve_mean"], label="mean")
    axes[0, 0].plot(t, curves["mse_curve_median"], linestyle="--", label="median")
    axes[0, 0].set_title("Rollout MSE")
    axes[0, 0].set_xlabel("time")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()
    axes[0, 1].plot(t, curves["rel_curve_mean"], label="mean")
    axes[0, 1].plot(t, curves["rel_curve_median"], linestyle="--", label="median")
    axes[0, 1].set_title("Rollout Relative L2")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def _plot_samples(model, split, device: str, dt: float, t_final: float, snapshot_times: List[float], n_plot_samples: int, out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping sample plots because matplotlib is unavailable: {exc}")
        return
    os.makedirs(out_dir, exist_ok=True)
    total = int(split["u0"].shape[0])
    sample_ids = torch.linspace(0, total - 1, min(max(1, n_plot_samples), total)).long().tolist()
    n_steps = int(split["u_traj"].shape[1] - 1)
    channel_names = ["u", "v"]
    for sample_id in sample_ids:
        u0 = split["u0"][sample_id : sample_id + 1].to(device)
        f = split["f"][sample_id : sample_id + 1].to(device)
        ref = split["u_traj"][sample_id]
        pred = rollout_model_2d(model, u0=u0, f=f, n_steps=n_steps, dt=dt)[0].cpu()
        cols = len(snapshot_times)
        fig, axes = plt.subplots(4, cols, figsize=(3.0 * cols, 10.0), squeeze=False, constrained_layout=True)
        for j, t_snap in enumerate(snapshot_times):
            k = max(0, min(n_steps, int(round((float(t_snap) / float(t_final)) * n_steps)) if t_final > 0 else 0))
            for c, name in enumerate(channel_names):
                vmax = max(float(ref[:, c].abs().max().item()), float(pred[:, c].abs().max().item()), 1e-8)
                axes[2 * c, j].imshow(ref[k, c].numpy(), origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
                axes[2 * c, j].set_title(f"ref {name} t={t_snap:g}")
                axes[2 * c + 1, j].imshow(pred[k, c].numpy(), origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
                axes[2 * c + 1, j].set_title(f"pred {name} t={t_snap:g}")
                axes[2 * c, j].set_xticks([])
                axes[2 * c, j].set_yticks([])
                axes[2 * c + 1, j].set_xticks([])
                axes[2 * c + 1, j].set_yticks([])
        fig.savefig(os.path.join(out_dir, f"sample_{sample_id:04d}_comparison.png"), dpi=180)
        plt.close(fig)


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.dirname(args.checkpoint_path)
    with open(os.path.join(run_dir, "args.json"), "r", encoding="utf-8") as f:
        train_args = Namespace(**json.load(f))
    splits = load_dataset_splits(args.dataset_path, map_location="cpu")
    split = splits[args.split]
    meta = splits.get("meta", {})
    n_x = int(split["u0"].shape[-2])
    n_y = int(split["u0"].shape[-1])
    n_steps = int(split["u_traj"].shape[1] - 1)
    t_final = float(meta.get("t_final", float(n_steps)))
    dt = t_final / float(n_steps)
    area = 1.0 / float(n_x * n_y)
    model = _build_model(n_x=n_x, n_y=n_y, h_x=1.0 / n_x, h_y=1.0 / n_y, dt=dt, args=train_args).to(device)
    checkpoint = _torch_load_checkpoint(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    step_loader = DataLoader(build_fhn2d_step_dataset(split), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    traj_loader = DataLoader(build_fhn2d_trajectory_dataset_from_split(split), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    trainer = HiddenGradientFlowTrainer2D(model=model, dt=dt, h_x=1.0 / n_x, h_y=1.0 / n_y, lambda_spec=0.0, device=device, output_dir=None, show_epoch_pbar=False)
    metrics = trainer.validate(step_loader, traj_loader=None)
    curves = _evaluate_rollout_curves(model, traj_loader, device=device, dt=dt, area=area)
    metrics["rollout_rel_l2"] = curves["rollout_rel_mean"]
    metrics["rollout_rel_l2_median"] = curves["rollout_rel_median"]
    print(f"Device: {device}")
    print(f"Split: {args.split}, n={int(split['u0'].shape[0])}, grid=({n_x},{n_y}), steps={n_steps}, dt={dt:.6f}")
    print("Metrics:", metrics)
    _save_curve_csv(curves, dt, os.path.join(args.output_dir, f"{args.split}_rollout_error_curve.csv"))
    _plot_curve(curves, dt, os.path.join(args.output_dir, f"{args.split}_rollout_error_curve.png"))
    _plot_samples(model, split, device, dt, t_final, _parse_snapshot_times(args.snapshot_times, t_final), args.n_plot_samples, os.path.join(args.output_dir, f"{args.split}_sample_comparisons"))


if __name__ == "__main__":
    main(parse_args())
