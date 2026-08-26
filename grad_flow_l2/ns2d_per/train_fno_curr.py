"""Train a physical-space FNO with rollout/teacher-forcing curriculum."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    from ..fno import FNO2D
    from ..fno_trainer import (
        AverageMeter,
        FNOTrainer2D,
        channel_weighted_mse,
        spectral_sobolev_loss_2d,
    )
    from ..heat_data import load_dataset_splits
    from .ns2d_data import (
        build_navier_stokes2d_periodic_step_dataset,
        build_navier_stokes2d_periodic_trajectory_dataset_from_split,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from grad_flow_l2.fno import FNO2D
    from grad_flow_l2.fno_trainer import (
        AverageMeter,
        FNOTrainer2D,
        channel_weighted_mse,
        spectral_sobolev_loss_2d,
    )
    from grad_flow_l2.heat_data import load_dataset_splits
    from grad_flow_l2.ns2d_per.ns2d_data import (
        build_navier_stokes2d_periodic_step_dataset,
        build_navier_stokes2d_periodic_trajectory_dataset_from_split,
    )


def set_seed(seed: int, seed_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda:
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train physical-space FNO with rollout curriculum on periodic 2D Navier-Stokes data"
    )
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--n-train", type=int, default=1600)
    parser.add_argument("--n-val", type=int, default=400)
    parser.add_argument("--n-test", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--train-t-start",
        type=float,
        default=None,
        help="Optional first physical time kept from stored trajectories for training/validation.",
    )
    parser.add_argument(
        "--train-t-end",
        type=float,
        default=None,
        help="Optional last physical time kept from stored trajectories for training/validation.",
    )

    parser.add_argument(
        "--state-channels",
        type=int,
        default=None,
        help="Defaults to 1 for scalar NS data.",
    )
    parser.add_argument(
        "--forcing-channels",
        type=int,
        default=None,
        help="Defaults to 1 for scalar forcing data.",
    )
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--fno-layers", type=int, default=4)
    parser.add_argument("--fno-modes-x", type=int, default=16)
    parser.add_argument("--fno-modes-y", type=int, default=16)
    parser.add_argument("--disable-fno-grid", action="store_true")
    parser.add_argument("--use-dt-channel", action="store_true")
    parser.add_argument("--disable-forcing-channel", action="store_true")
    parser.add_argument(
        "--no-residual",
        action="store_true",
        help="Predict u_{k+1} directly instead of an increment.",
    )
    parser.add_argument(
        "--lift-noise-std",
        type=float,
        default=0.0,
        help="Std of Matern-filtered Gaussian noise added after FNO lift during training only.",
    )
    parser.add_argument(
        "--lift-noise-corr-length",
        type=float,
        default=1.0,
        help="Correlation length for Matern-filtered FNO lift noise.",
    )
    parser.add_argument(
        "--lift-noise-decay-s",
        type=float,
        default=2.0,
        help="Spectral decay exponent s for Matern-filtered FNO lift noise.",
    )

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-step-size", type=int, default=100)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--rollout-delta-clip", type=float, default=10.0, help="Use <=0 to disable."
    )
    parser.add_argument(
        "--curriculum-start-rollout",
        type=int,
        default=1,
        help="Autoregressive rollout steps at the start of training. Use 0 for pure teacher forcing warmup.",
    )
    parser.add_argument(
        "--curriculum-end-rollout",
        type=int,
        default=None,
        help="Final autoregressive rollout steps. Defaults to the full training horizon.",
    )
    parser.add_argument(
        "--curriculum-warmup-epochs",
        type=int,
        default=0,
        help="Epochs kept at --curriculum-start-rollout before increasing the horizon.",
    )
    parser.add_argument(
        "--curriculum-epochs",
        type=int,
        default=None,
        help="Epochs used to increase from start to end rollout. Defaults to epochs minus warmup.",
    )
    parser.add_argument(
        "--curriculum-train-delta-clip",
        type=float,
        default=0.0,
        help="Clip increments inside curriculum training when >0. Validation still uses --rollout-delta-clip.",
    )
    parser.add_argument("--lambda-spec", type=float, default=0.0)
    parser.add_argument("--spectral-s", type=float, default=1.0)
    parser.add_argument("--channel-weights", type=float, nargs="*", default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-epoch-pbar", action="store_true")
    parser.add_argument(
        "--output-dir", type=str, default="grad_flow_l2/ns2d_per/outputs_fno_curr"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _infer_state_channels(split: dict) -> int:
    u0 = split["u0"]
    return 1 if u0.dim() == 3 else int(u0.shape[1])


def _infer_forcing_channels(split: dict) -> int:
    f = split["f"]
    return 1 if f.dim() == 3 else int(f.shape[1])


def _channel_weights_from_split(
    train_split: dict, state_channels: int, provided
) -> torch.Tensor | None:
    if provided is not None and len(provided) > 0:
        weights = torch.tensor(provided, dtype=torch.float32)
        if int(weights.numel()) != state_channels:
            raise ValueError(f"--channel-weights must have {state_channels} entries")
        return weights / weights.mean()
    if state_channels == 1:
        return None
    u = train_split["u_traj"]
    flat = u.permute(2, 0, 1, 3, 4).reshape(state_channels, -1)
    weights = 1.0 / flat.var(dim=1).clamp(min=1e-8)
    return weights / weights.mean()


def _slice_time_window(
    split: dict,
    time_values: np.ndarray,
    t_start: float | None,
    t_end: float | None,
    split_name: str,
) -> tuple[dict, np.ndarray, tuple[int, int]]:
    if t_start is None and t_end is None:
        return split, time_values, (0, int(time_values.shape[0] - 1))

    start = float(time_values[0]) if t_start is None else float(t_start)
    end = float(time_values[-1]) if t_end is None else float(t_end)
    tol = 1e-8 + 1e-6 * max(1.0, abs(float(time_values[-1] - time_values[0])))
    if start < float(time_values[0]) - tol or end > float(time_values[-1]) + tol:
        raise ValueError(
            f"Requested time window [{start},{end}] is outside stored range "
            f"[{float(time_values[0])},{float(time_values[-1])}]"
        )
    if end <= start:
        raise ValueError(
            f"--train-t-end must be greater than --train-t-start, got [{start},{end}]"
        )

    i0 = int(np.searchsorted(time_values, start - tol, side="left"))
    i1 = int(np.searchsorted(time_values, end + tol, side="right") - 1)
    i0 = max(0, min(i0, int(time_values.shape[0] - 1)))
    i1 = max(0, min(i1, int(time_values.shape[0] - 1)))
    if i1 <= i0:
        raise ValueError(
            f"Time window [{start},{end}] for {split_name} keeps fewer than two snapshots; "
            f"nearest indices are {i0}:{i1}"
        )

    u_traj = split["u_traj"][:, i0 : i1 + 1]
    sliced = dict(split)
    sliced["u_traj"] = u_traj
    sliced["u0"] = u_traj[:, 0].clone()
    return sliced, time_values[i0 : i1 + 1], (i0, i1)


def _build_model(
    n_x: int, n_y: int, dt: float, state_channels: int, forcing_channels: int, args
) -> FNO2D:
    return FNO2D(
        n_x=n_x,
        n_y=n_y,
        state_channels=state_channels,
        forcing_channels=forcing_channels,
        width=args.width,
        n_layers=args.fno_layers,
        modes_x=args.fno_modes_x,
        modes_y=args.fno_modes_y,
        use_forcing_channel=not args.disable_forcing_channel,
        use_dt_channel=args.use_dt_channel,
        use_grid_features=not args.disable_fno_grid,
        default_dt=dt,
        residual=not args.no_residual,
        lift_noise_std=args.lift_noise_std,
        lift_noise_corr_length=args.lift_noise_corr_length,
        lift_noise_decay_s=args.lift_noise_decay_s,
    )


class CurriculumFNOTrainer2D(FNOTrainer2D):
    """FNO trainer using short autoregressive prefixes plus teacher-forced tails."""

    def __init__(
        self,
        *args,
        n_steps: int,
        curriculum_start_rollout: int = 1,
        curriculum_end_rollout: int | None = None,
        curriculum_warmup_epochs: int = 0,
        curriculum_epochs: int | None = None,
        train_delta_clip: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_steps = int(n_steps)
        self.curriculum_start_rollout = max(0, int(curriculum_start_rollout))
        self.curriculum_end_rollout = (
            self.n_steps
            if curriculum_end_rollout is None
            else int(curriculum_end_rollout)
        )
        self.curriculum_end_rollout = max(
            0, min(self.n_steps, self.curriculum_end_rollout)
        )
        self.curriculum_warmup_epochs = max(0, int(curriculum_warmup_epochs))
        self.curriculum_epochs = (
            None if curriculum_epochs is None else max(1, int(curriculum_epochs))
        )
        self.train_delta_clip = (
            None
            if train_delta_clip is None or train_delta_clip <= 0.0
            else float(train_delta_clip)
        )
        if self.curriculum_start_rollout > self.curriculum_end_rollout:
            raise ValueError(
                "curriculum_start_rollout must be <= curriculum_end_rollout, got "
                f"{self.curriculum_start_rollout}>{self.curriculum_end_rollout}"
            )

    def rollout_steps_for_epoch(self, epoch: int, total_epochs: int) -> int:
        if self.curriculum_start_rollout == self.curriculum_end_rollout:
            return self.curriculum_end_rollout
        epoch = max(1, int(epoch))
        if epoch <= self.curriculum_warmup_epochs:
            return self.curriculum_start_rollout
        span = self.curriculum_epochs
        if span is None:
            span = max(1, int(total_epochs) - self.curriculum_warmup_epochs)
        progress = min(
            1.0,
            float(epoch - self.curriculum_warmup_epochs - 1) / float(max(1, span - 1)),
        )
        rollout = self.curriculum_start_rollout + progress * (
            self.curriculum_end_rollout - self.curriculum_start_rollout
        )
        return int(round(rollout))

    @staticmethod
    def _unpack_traj_batch(batch):
        if isinstance(batch, dict):
            return batch["u0"], batch["f"], batch["u_traj"]
        if isinstance(batch, (tuple, list)) and len(batch) == 3:
            return batch[0], batch[1], batch[2]
        raise ValueError(
            "Trajectory batch must be dict with keys u0,f,u_traj or tuple/list (u0,f,u_traj)"
        )

    def _maybe_clip_training_step(
        self, u_in: torch.Tensor, u_pred: torch.Tensor
    ) -> torch.Tensor:
        if self.train_delta_clip is None:
            return u_pred
        delta = torch.clamp(
            u_pred - u_in, min=-self.train_delta_clip, max=self.train_delta_clip
        )
        return u_in + delta

    def _compute_curriculum_losses(
        self,
        u_ref: torch.Tensor,
        f: torch.Tensor,
        rollout_steps: int,
    ) -> dict[str, torch.Tensor]:
        n_steps = int(u_ref.shape[1] - 1)
        rollout_steps = max(0, min(int(rollout_steps), n_steps))
        loss_terms = []
        spec_terms = []
        rollout_terms = []
        teacher_forced_terms = []
        u_ar = u_ref[:, 0]

        for k in range(n_steps):
            autoregressive = k < rollout_steps
            u_in = u_ar if autoregressive else u_ref[:, k]
            u_pred = self.model.predict_step(u_in, f, dt=self.dt)
            u_pred = self._maybe_clip_training_step(u_in, u_pred)
            u_target = u_ref[:, k + 1]
            weights = self._weights_for(u_pred)
            loss_step = channel_weighted_mse(u_pred, u_target, weights)
            loss_spec = (
                spectral_sobolev_loss_2d(u_pred, u_target, s=self.spectral_s)
                if self.lambda_spec > 0.0
                else loss_step.new_zeros(())
            )
            loss_terms.append(loss_step)
            spec_terms.append(loss_spec)
            if autoregressive:
                rollout_terms.append(loss_step)
                u_ar = u_pred
            else:
                teacher_forced_terms.append(loss_step)

        loss_step = torch.stack(loss_terms).mean()
        loss_spec = (
            torch.stack(spec_terms).mean() if spec_terms else loss_step.new_zeros(())
        )
        loss_rollout = (
            torch.stack(rollout_terms).mean()
            if rollout_terms
            else loss_step.new_zeros(())
        )
        loss_teacher_forced = (
            torch.stack(teacher_forced_terms).mean()
            if teacher_forced_terms
            else loss_step.new_zeros(())
        )
        return {
            "loss": loss_step + self.lambda_spec * loss_spec,
            "loss_step": loss_step,
            "loss_spec": loss_spec,
            "loss_rollout": loss_rollout,
            "loss_teacher_forced": loss_teacher_forced,
        }

    def train_epoch(
        self, loader: DataLoader, epoch: int, total_epochs: int
    ) -> dict[str, float]:
        self.model.train()
        rollout_steps = self.rollout_steps_for_epoch(epoch, total_epochs)
        meters = {
            k: AverageMeter()
            for k in (
                "loss",
                "loss_step",
                "loss_spec",
                "loss_rollout",
                "loss_teacher_forced",
            )
        }
        iterable = loader
        pbar = None
        if self.show_epoch_pbar and tqdm is not None:
            pbar = tqdm(
                loader,
                total=len(loader),
                desc=f"Epoch {epoch:03d} K={rollout_steps}",
                leave=False,
            )
            iterable = pbar
        for batch_idx, batch in enumerate(iterable, start=1):
            _u0, f, u_ref = self._unpack_traj_batch(batch)
            f = f.to(self.device)
            u_ref = u_ref.to(self.device)
            losses = self._compute_curriculum_losses(
                u_ref, f, rollout_steps=rollout_steps
            )
            self.optimizer.zero_grad()
            losses["loss"].backward()
            if self.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            bsz = int(u_ref.shape[0])
            for key, meter in meters.items():
                meter.update(losses[key].item(), bsz)
            if pbar is not None and (batch_idx == 1 or batch_idx % 10 == 0):
                pbar.set_postfix(
                    total=f"{meters['loss'].avg:.4f}",
                    ar=f"{meters['loss_rollout'].avg:.4f}",
                    tf=f"{meters['loss_teacher_forced'].avg:.4f}",
                )
        if pbar is not None:
            pbar.close()
        metrics = {k: v.avg for k, v in meters.items()}
        metrics["curriculum_rollout_steps"] = float(rollout_steps)
        return metrics

    def fit(
        self,
        train_traj_loader: DataLoader,
        val_step_loader: DataLoader,
        val_traj_loader: DataLoader | None = None,
        epochs: int = 200,
        eval_interval: int = 1,
        checkpoint_interval: int = 25,
    ) -> dict[str, list]:
        history = {"train": [], "val": []}
        best_metric = float("inf")
        best_epoch = 0
        best_metrics = None
        best_state_dict = None
        for epoch in range(1, int(epochs) + 1):
            train_metrics = self.train_epoch(
                train_traj_loader, epoch=epoch, total_epochs=epochs
            )
            history["train"].append({"epoch": epoch, **train_metrics})
            if epoch % int(eval_interval) == 0:
                val_metrics = self.validate(
                    val_step_loader, traj_loader=val_traj_loader
                )
                val_metrics["curriculum_rollout_steps"] = train_metrics[
                    "curriculum_rollout_steps"
                ]
                history["val"].append({"epoch": epoch, **val_metrics})
                monitor = val_metrics.get(
                    "val_rollout_rel_l2", val_metrics["val_loss_step"]
                )
                if monitor < best_metric:
                    best_metric = monitor
                    best_epoch = epoch
                    best_metrics = val_metrics
                    best_state_dict = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                    self._save_checkpoint("best_model.pt", epoch, val_metrics)
                print(
                    f"[Epoch {epoch:03d}] "
                    f"K={int(train_metrics['curriculum_rollout_steps'])} "
                    f"train_total={train_metrics['loss']:.6f} "
                    f"train_step={train_metrics['loss_step']:.6f} "
                    f"train_ar={train_metrics['loss_rollout']:.6f} "
                    f"train_tf={train_metrics['loss_teacher_forced']:.6f} "
                    f"train_spec={train_metrics['loss_spec']:.6f} "
                    f"val_total={val_metrics['val_loss']:.6f} "
                    f"val_step={val_metrics['val_loss_step']:.6f} "
                    f"val_spec={val_metrics['val_loss_spec']:.6f} "
                    f"val_rollout={val_metrics.get('val_rollout_rel_l2', float('nan')):.6f}"
                )
            else:
                print(
                    f"[Epoch {epoch:03d}] "
                    f"K={int(train_metrics['curriculum_rollout_steps'])} "
                    f"train_total={train_metrics['loss']:.6f} "
                    f"train_step={train_metrics['loss_step']:.6f} "
                    f"train_ar={train_metrics['loss_rollout']:.6f} "
                    f"train_tf={train_metrics['loss_teacher_forced']:.6f} "
                    f"train_spec={train_metrics['loss_spec']:.6f}"
                )
            if checkpoint_interval > 0 and epoch % int(checkpoint_interval) == 0:
                self._save_checkpoint(
                    f"best_model_through_epoch_{epoch:04d}.pt",
                    best_epoch if best_metrics is not None else epoch,
                    best_metrics if best_metrics is not None else train_metrics,
                    state_dict=best_state_dict,
                )
            self.scheduler.step()

        final_metrics = history["val"][-1] if history["val"] else history["train"][-1]
        self._save_checkpoint("final_model.pt", int(epochs), final_metrics)
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(
                os.path.join(self.output_dir, "history.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(history, f, indent=2)
        return history


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed, seed_cuda=not args.cpu)
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    splits = load_dataset_splits(args.dataset_path, map_location="cpu")
    train_split, val_split, test_split = splits["train"], splits["val"], splits["test"]
    sizes = (
        int(train_split["u0"].shape[0]),
        int(val_split["u0"].shape[0]),
        int(test_split["u0"].shape[0]),
    )
    if sizes != (args.n_train, args.n_val, args.n_test):
        raise ValueError(
            f"Dataset split sizes {sizes} do not match args {(args.n_train, args.n_val, args.n_test)}"
        )

    meta = splits.get("meta", {})
    n_x = int(train_split["u0"].shape[-2])
    n_y = int(train_split["u0"].shape[-1])
    n_steps = int(train_split["u_traj"].shape[1] - 1)
    t_final = float(meta.get("t_final", 1.0))
    t_start = float(meta.get("stored_t_start", meta.get("warmup_time", 0.0)))
    stored_horizon = float(meta.get("stored_time_horizon", t_final - t_start))
    dt = float(meta.get("record_dt", stored_horizon / float(n_steps)))
    time_values_full = t_start + np.arange(n_steps + 1, dtype=np.float64) * dt
    train_split, time_values, window_idx = _slice_time_window(
        train_split,
        time_values_full,
        t_start=args.train_t_start,
        t_end=args.train_t_end,
        split_name="train",
    )
    val_split, _, _ = _slice_time_window(
        val_split,
        time_values_full,
        t_start=args.train_t_start,
        t_end=args.train_t_end,
        split_name="val",
    )
    test_split, _, _ = _slice_time_window(
        test_split,
        time_values_full,
        t_start=args.train_t_start,
        t_end=args.train_t_end,
        split_name="test",
    )
    n_steps_full = n_steps
    n_steps = int(train_split["u_traj"].shape[1] - 1)
    t_window_start = float(time_values[0])
    t_window_end = float(time_values[-1])
    h_x, h_y = 1.0 / float(n_x), 1.0 / float(n_y)
    state_channels = (
        _infer_state_channels(train_split)
        if args.state_channels is None
        else int(args.state_channels)
    )
    forcing_channels = (
        _infer_forcing_channels(train_split)
        if args.forcing_channels is None
        else int(args.forcing_channels)
    )
    channel_weights = _channel_weights_from_split(
        train_split, state_channels, args.channel_weights
    )
    rollout_delta_clip = (
        args.rollout_delta_clip if args.rollout_delta_clip > 0.0 else None
    )

    print(f"Device: {device}")
    print(f"Loaded dataset: {args.dataset_path}")
    print(
        f"Grid: n_x={n_x}, n_y={n_y}, n_steps={n_steps}, dt={dt:.6f}, "
        f"train_time=[{t_window_start:.6f},{t_window_end:.6f}] "
        f"(indices {window_idx[0]}:{window_idx[1]} of {n_steps_full})"
    )
    print(f"state_channels={state_channels}, forcing_channels={forcing_channels}")
    if channel_weights is not None:
        print(f"Channel weights: {channel_weights.tolist()}")

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    val_step_loader = DataLoader(
        build_navier_stokes2d_periodic_step_dataset(val_split),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
    )
    test_step_loader = DataLoader(
        build_navier_stokes2d_periodic_step_dataset(test_split),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
    )
    train_traj_loader = DataLoader(
        build_navier_stokes2d_periodic_trajectory_dataset_from_split(train_split),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    val_traj_loader = DataLoader(
        build_navier_stokes2d_periodic_trajectory_dataset_from_split(val_split),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
    )
    test_traj_loader = DataLoader(
        build_navier_stokes2d_periodic_trajectory_dataset_from_split(test_split),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
    )

    model = _build_model(n_x, n_y, dt, state_channels, forcing_channels, args).to(
        device
    )
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    args_dict = vars(args).copy()
    args_dict.update(
        {
            "state_channels_used": state_channels,
            "forcing_channels_used": forcing_channels,
            "channel_weights_used": (
                None if channel_weights is None else channel_weights.tolist()
            ),
            "train_time_start_used": t_window_start,
            "train_time_end_used": t_window_end,
            "train_time_index_start": int(window_idx[0]),
            "train_time_index_end": int(window_idx[1]),
            "train_steps_used": int(n_steps),
            "curriculum_end_rollout_used": int(
                max(
                    0,
                    min(
                        n_steps,
                        (
                            n_steps
                            if args.curriculum_end_rollout is None
                            else args.curriculum_end_rollout
                        ),
                    ),
                )
            ),
            "curriculum_train_delta_clip_used": (
                None
                if args.curriculum_train_delta_clip <= 0.0
                else float(args.curriculum_train_delta_clip)
            ),
        }
    )
    with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=2)

    trainer = CurriculumFNOTrainer2D(
        model=model,
        dt=dt,
        h_x=h_x,
        h_y=h_y,
        lambda_spec=args.lambda_spec,
        spectral_s=args.spectral_s,
        channel_weights=channel_weights,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        rollout_delta_clip=rollout_delta_clip,
        device=device,
        output_dir=run_dir,
        show_epoch_pbar=not args.no_epoch_pbar,
        n_steps=n_steps,
        curriculum_start_rollout=args.curriculum_start_rollout,
        curriculum_end_rollout=args.curriculum_end_rollout,
        curriculum_warmup_epochs=args.curriculum_warmup_epochs,
        curriculum_epochs=args.curriculum_epochs,
        train_delta_clip=args.curriculum_train_delta_clip,
    )

    if args.dry_run:
        print(
            "Dry run val metrics:",
            trainer.validate(val_step_loader, traj_loader=val_traj_loader),
        )
        print(
            "Dry run test metrics:",
            trainer.validate(test_step_loader, traj_loader=test_traj_loader),
        )
        return

    print(
        f"Training config: epochs={args.epochs}, lr={args.lr}, lambda_spec={args.lambda_spec}, "
        f"fno_layers={args.fno_layers}, width={args.width}, modes=({args.fno_modes_x},{args.fno_modes_y}), "
        f"residual={not args.no_residual}, rollout_delta_clip={rollout_delta_clip}, "
        f"curriculum_start={args.curriculum_start_rollout}, "
        f"curriculum_end={n_steps if args.curriculum_end_rollout is None else args.curriculum_end_rollout}, "
        f"curriculum_warmup_epochs={args.curriculum_warmup_epochs}, "
        f"curriculum_epochs={args.curriculum_epochs}, "
        f"curriculum_train_delta_clip={None if args.curriculum_train_delta_clip <= 0.0 else args.curriculum_train_delta_clip}, "
        f"lift_noise_std={args.lift_noise_std}, lift_noise_corr_length={args.lift_noise_corr_length}, "
        f"lift_noise_decay_s={args.lift_noise_decay_s}, "
        f"train_time=[{t_window_start:.6f},{t_window_end:.6f}], output={run_dir}"
    )
    history = trainer.fit(
        train_traj_loader=train_traj_loader,
        val_step_loader=val_step_loader,
        val_traj_loader=val_traj_loader,
        epochs=args.epochs,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
    )
    print("Training complete.")
    print("Last train metrics:", history["train"][-1])
    if history["val"]:
        print("Last val metrics:", history["val"][-1])
    print(
        "Test metrics:",
        trainer.validate(test_step_loader, traj_loader=test_traj_loader),
    )
    print(f"Saved training artifacts to: {run_dir}")


if __name__ == "__main__":
    main(parse_args())
