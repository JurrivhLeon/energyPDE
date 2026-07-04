"""Train a standard physical-space FNO on periodic 2D compressible NS data."""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    from ..cfd2d.cfd_data import STATE_CHANNELS, build_cfd2d_step_dataset, build_cfd2d_trajectory_dataset_from_split
    from ..fno import FNO2D
    from ..heat_data import load_dataset_splits
    from ..latent_markov_trainer_mc import AverageMeter, channel_weighted_mse, relative_l2_error_2d, rollout_latent_markov_2d
except ImportError:
    from grad_flow_l2.cfd2d.cfd_data import STATE_CHANNELS, build_cfd2d_step_dataset, build_cfd2d_trajectory_dataset_from_split
    from grad_flow_l2.fno import FNO2D
    from grad_flow_l2.heat_data import load_dataset_splits
    from grad_flow_l2.latent_markov_trainer_mc import AverageMeter, channel_weighted_mse, relative_l2_error_2d, rollout_latent_markov_2d


def set_seed(seed: int, seed_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda:
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train physical-space FNO on periodic 2D compressible NS data")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--n-train", type=int, default=1200)
    parser.add_argument("--n-val", type=int, default=300)
    parser.add_argument("--n-test", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-t-start", type=float, default=None)
    parser.add_argument("--train-t-end", type=float, default=None)

    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--fno-layers", type=int, default=6)
    parser.add_argument("--fno-modes-x", type=int, default=16)
    parser.add_argument("--fno-modes-y", type=int, default=16)
    parser.add_argument("--disable-fno-grid", action="store_true")
    parser.add_argument("--use-dt-channel", action="store_true")
    parser.add_argument("--disable-forcing-channel", action="store_true")
    parser.add_argument("--no-residual", action="store_true")
    parser.add_argument("--lift-noise-std", type=float, default=0.0, help="Std of Gaussian noise added after FNO lift during training only.")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-step-size", type=int, default=100)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--rollout-delta-clip", type=float, default=10.0)
    parser.add_argument("--channel-weights", type=float, nargs=4, default=None, metavar=("W_RHO", "W_VX", "W_VY", "W_P"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-epoch-pbar", action="store_true")
    parser.add_argument("--output-dir", type=str, default="grad_flow_l2/cfd2d/outputs_fno")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _slice_time_window(split, time_values, t_start, t_end, split_name):
    if t_start is None and t_end is None:
        return split, time_values, (0, int(time_values.shape[0] - 1))
    start = float(time_values[0]) if t_start is None else float(t_start)
    end = float(time_values[-1]) if t_end is None else float(t_end)
    tol = 1e-8 + 1e-6 * max(1.0, abs(float(time_values[-1] - time_values[0])))
    if start < float(time_values[0]) - tol or end > float(time_values[-1]) + tol:
        raise ValueError(f"Requested time window [{start},{end}] is outside stored range [{float(time_values[0])},{float(time_values[-1])}]")
    if end <= start:
        raise ValueError("--train-t-end must be greater than --train-t-start")
    i0 = int(np.searchsorted(time_values, start - tol, side="left"))
    i1 = int(np.searchsorted(time_values, end + tol, side="right") - 1)
    i0 = max(0, min(i0, int(time_values.shape[0] - 1)))
    i1 = max(0, min(i1, int(time_values.shape[0] - 1)))
    if i1 <= i0:
        raise ValueError(f"Time window [{start},{end}] for {split_name} keeps fewer than two snapshots")
    u_traj = split["u_traj"][:, i0 : i1 + 1]
    out = dict(split)
    out["u_traj"] = u_traj
    out["u0"] = u_traj[:, 0].clone()
    return out, time_values[i0 : i1 + 1], (i0, i1)


def _build_model(n_x: int, n_y: int, dt: float, args: argparse.Namespace) -> FNO2D:
    return FNO2D(
        n_x=n_x,
        n_y=n_y,
        state_channels=STATE_CHANNELS,
        forcing_channels=1,
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
    )


class CFDFNOTrainer2D:
    def __init__(self, model, dt: float, h_x: float, h_y: float, channel_weights, lr=1e-4,
                 lr_step_size=100, lr_gamma=0.5, weight_decay=1e-5, grad_clip=1.0,
                 rollout_delta_clip: Optional[float] = 10.0, device="cpu", output_dir=None, show_epoch_pbar=True):
        self.model = model.to(device)
        self.dt = float(dt)
        self.h_x = float(h_x)
        self.h_y = float(h_y)
        self.area = self.h_x * self.h_y
        self.channel_weights = torch.as_tensor(channel_weights, dtype=torch.float32)
        self.grad_clip = float(grad_clip)
        self.rollout_delta_clip = None if rollout_delta_clip is None else float(rollout_delta_clip)
        self.device = device
        self.output_dir = output_dir
        self.show_epoch_pbar = bool(show_epoch_pbar)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=max(1, int(lr_step_size)), gamma=float(lr_gamma))

    def _weights(self, tensor):
        return self.channel_weights.to(device=tensor.device, dtype=tensor.dtype)

    def _compute_losses(self, u_k, u_k1, f):
        pred = self.model.predict_step(u_k, f, dt=self.dt)
        loss_step = channel_weighted_mse(pred, u_k1, self._weights(pred))
        return {"loss": loss_step, "loss_step": loss_step}

    def train_epoch(self, loader, epoch=None):
        self.model.train()
        meters = {k: AverageMeter() for k in ("loss", "loss_step")}
        iterable = loader
        pbar = None
        if self.show_epoch_pbar and tqdm is not None:
            pbar = tqdm(loader, total=len(loader), desc=f"Epoch {epoch:03d}" if epoch else "Epoch", leave=False, dynamic_ncols=True)
            iterable = pbar
        for i, (u_k, u_k1, f) in enumerate(iterable, start=1):
            u_k, u_k1, f = u_k.to(self.device), u_k1.to(self.device), f.to(self.device)
            losses = self._compute_losses(u_k, u_k1, f)
            self.optimizer.zero_grad()
            losses["loss"].backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            bsz = int(u_k.shape[0])
            for k, m in meters.items():
                m.update(losses[k].item(), bsz)
            if pbar is not None and (i == 1 or i % 10 == 0):
                pbar.set_postfix(total=f"{meters['loss'].avg:.4f}")
        if pbar is not None:
            pbar.close()
        return {k: v.avg for k, v in meters.items()}

    @torch.no_grad()
    def validate(self, step_loader, traj_loader=None):
        self.model.eval()
        meters = {k: AverageMeter() for k in ("val_loss", "val_loss_step")}
        for u_k, u_k1, f in step_loader:
            u_k, u_k1, f = u_k.to(self.device), u_k1.to(self.device), f.to(self.device)
            losses = self._compute_losses(u_k, u_k1, f)
            bsz = int(u_k.shape[0])
            meters["val_loss"].update(losses["loss"].item(), bsz)
            meters["val_loss_step"].update(losses["loss_step"].item(), bsz)
        metrics = {k: v.avg for k, v in meters.items()}
        if traj_loader is not None:
            rollout = AverageMeter()
            for batch in traj_loader:
                u0, f, u_ref = batch["u0"].to(self.device), batch["f"].to(self.device), batch["u_traj"].to(self.device)
                pred = rollout_latent_markov_2d(self.model, u0, f, n_steps=int(u_ref.shape[1]-1), dt=self.dt, delta_clip=self.rollout_delta_clip)
                rel = relative_l2_error_2d(pred, u_ref, area=self.area)
                rollout.update(rel.mean(dim=(-1, -2)).mean().item(), int(u0.shape[0]))
            metrics["val_rollout_rel_l2"] = rollout.avg
        return metrics

    def _save_checkpoint(self, name, epoch, metrics, state_dict=None):
        if self.output_dir is None:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict() if state_dict is None else state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "dt": self.dt,
            "h_x": self.h_x,
            "h_y": self.h_y,
            "rollout_delta_clip": self.rollout_delta_clip,
            "channel_weights": self.channel_weights.detach().cpu(),
        }, os.path.join(self.output_dir, name))

    def fit(self, train_step_loader, val_step_loader, val_traj_loader=None, epochs=200, eval_interval=1, checkpoint_interval=25):
        history = {"train": [], "val": []}
        best = float("inf")
        best_state = None
        best_metrics = None
        best_epoch = 0
        for epoch in range(1, int(epochs)+1):
            tr = self.train_epoch(train_step_loader, epoch=epoch)
            history["train"].append({"epoch": epoch, **tr})
            if epoch % int(eval_interval) == 0:
                val = self.validate(val_step_loader, val_traj_loader)
                history["val"].append({"epoch": epoch, **val})
                monitor = val.get("val_rollout_rel_l2", val["val_loss_step"])
                if monitor < best:
                    best = monitor
                    best_epoch = epoch
                    best_metrics = val
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    self._save_checkpoint("best_model.pt", epoch, val)
                print(f"[Epoch {epoch:03d}] train_total={tr['loss']:.6f} train_step={tr['loss_step']:.6f} val_total={val['val_loss']:.6f} val_step={val['val_loss_step']:.6f} val_rollout={val.get('val_rollout_rel_l2', float('nan')):.6f}")
            else:
                print(f"[Epoch {epoch:03d}] train_total={tr['loss']:.6f} train_step={tr['loss_step']:.6f}")
            if checkpoint_interval > 0 and epoch % int(checkpoint_interval) == 0:
                self._save_checkpoint(f"best_model_through_epoch_{epoch:04d}.pt", best_epoch or epoch, best_metrics or tr, state_dict=best_state)
            self.scheduler.step()
        final = history["val"][-1] if history["val"] else history["train"][-1]
        self._save_checkpoint("final_model.pt", int(epochs), final)
        if self.output_dir:
            with open(os.path.join(self.output_dir, "history.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        return history


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed, seed_cuda=not args.cpu)
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
    splits = load_dataset_splits(args.dataset_path, map_location="cpu")
    train_split, val_split, test_split = splits["train"], splits["val"], splits["test"]
    sizes = (int(train_split["u0"].shape[0]), int(val_split["u0"].shape[0]), int(test_split["u0"].shape[0]))
    if sizes != (args.n_train, args.n_val, args.n_test):
        raise ValueError(f"Dataset split sizes {sizes} do not match args {(args.n_train, args.n_val, args.n_test)}")
    meta = splits.get("meta", {})
    n_x = int(train_split["u0"].shape[-2])
    n_y = int(train_split["u0"].shape[-1])
    n_steps = int(train_split["u_traj"].shape[1] - 1)
    t_final = float(meta.get("t_final", float(n_steps)))
    dt = t_final / float(n_steps)
    time_values_full = np.arange(n_steps + 1, dtype=np.float64) * dt
    train_split, time_values, window_idx = _slice_time_window(train_split, time_values_full, args.train_t_start, args.train_t_end, "train")
    val_split, _, _ = _slice_time_window(val_split, time_values_full, args.train_t_start, args.train_t_end, "val")
    test_split, _, _ = _slice_time_window(test_split, time_values_full, args.train_t_start, args.train_t_end, "test")
    n_steps_full = n_steps
    n_steps = int(train_split["u_traj"].shape[1] - 1)
    t_window_start = float(time_values[0])
    t_window_end = float(time_values[-1])
    h_x, h_y = 1.0 / float(n_x), 1.0 / float(n_y)
    if args.channel_weights is not None:
        channel_weights = torch.tensor(args.channel_weights, dtype=torch.float32)
    else:
        u_flat = train_split["u_traj"].permute(2, 0, 1, 3, 4).reshape(STATE_CHANNELS, -1)
        channel_weights = 1.0 / u_flat.var(dim=1).clamp(min=1e-8)
    channel_weights = channel_weights / channel_weights.mean()
    rollout_delta_clip = args.rollout_delta_clip if args.rollout_delta_clip > 0 else None
    print(f"Device: {device}")
    print(f"Loaded dataset: {args.dataset_path}")
    print(f"Grid: ({n_x},{n_y}), state_channels={STATE_CHANNELS}, steps={n_steps}, dt={dt:.6f}, train_time=[{t_window_start:.6f},{t_window_end:.6f}] (indices {window_idx[0]}:{window_idx[1]} of {n_steps_full})")
    print(f"Channel weights (rho,vx,vy,p): {channel_weights.tolist()}")
    train_step_loader = DataLoader(build_cfd2d_step_dataset(train_split), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_step_loader = DataLoader(build_cfd2d_step_dataset(val_split), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_step_loader = DataLoader(build_cfd2d_step_dataset(test_split), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) if args.n_test > 0 else None
    val_traj_loader = DataLoader(build_cfd2d_trajectory_dataset_from_split(val_split), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_traj_loader = DataLoader(build_cfd2d_trajectory_dataset_from_split(test_split), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) if args.n_test > 0 else None
    model = _build_model(n_x=n_x, n_y=n_y, dt=dt, args=args).to(device)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    run_dir = os.path.join(args.output_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    args_dict = vars(args).copy()
    args_dict["channel_weights_used"] = channel_weights.tolist()
    args_dict["train_time_start_used"] = t_window_start
    args_dict["train_time_end_used"] = t_window_end
    args_dict["train_time_index_start"] = int(window_idx[0])
    args_dict["train_time_index_end"] = int(window_idx[1])
    args_dict["train_steps_used"] = int(n_steps)
    with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=2)
    trainer = CFDFNOTrainer2D(model, dt=dt, h_x=h_x, h_y=h_y, channel_weights=channel_weights,
                              lr=args.lr, lr_step_size=args.lr_step_size, lr_gamma=args.lr_gamma,
                              weight_decay=args.weight_decay, grad_clip=args.grad_clip,
                              rollout_delta_clip=rollout_delta_clip,
                              device=device, output_dir=run_dir, show_epoch_pbar=not args.no_epoch_pbar)
    if args.dry_run:
        print("Dry run val metrics:", trainer.validate(val_step_loader, val_traj_loader))
        if args.n_test > 0:
            print("Dry run test metrics:", trainer.validate(test_step_loader, test_traj_loader))
        return
    print(f"Training config: epochs={args.epochs}, lr={args.lr}, width={args.width}, fno_layers={args.fno_layers}, fno_modes=({args.fno_modes_x},{args.fno_modes_y}), residual={not args.no_residual}, train_time=[{t_window_start:.6f},{t_window_end:.6f}], output={run_dir}")
    history = trainer.fit(train_step_loader, val_step_loader, val_traj_loader, args.epochs, args.eval_interval, args.checkpoint_interval)
    print("Training complete.")
    print("Last train metrics:", history["train"][-1])
    if history["val"]:
        print("Last val metrics:", history["val"][-1])
    if args.n_test > 0:
        print("Test metrics:", trainer.validate(test_step_loader, test_traj_loader))
    print(f"Saved training artifacts to: {run_dir}")


if __name__ == "__main__":
    main(parse_args())
