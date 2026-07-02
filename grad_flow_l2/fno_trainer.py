"""Training utilities for physical-space 2D FNO steppers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


@dataclass
class AverageMeter:
    val: float = 0.0
    avg: float = 0.0
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int) -> None:
        self.val = float(value)
        self.total += float(value) * int(n)
        self.count += int(n)
        self.avg = self.total / max(1, self.count)


def _unpack_step_batch(batch):
    if isinstance(batch, (tuple, list)) and len(batch) == 3:
        return batch[0], batch[1], batch[2]
    raise ValueError("Step batch must be a tuple/list (u_k, u_k1, f)")


def _unpack_traj_batch(batch):
    if isinstance(batch, dict):
        return batch["u0"], batch["f"], batch["u_traj"]
    if isinstance(batch, (tuple, list)) and len(batch) == 3:
        return batch[0], batch[1], batch[2]
    raise ValueError("Trajectory batch must be dict with keys u0,f,u_traj or tuple/list (u0,f,u_traj)")


def _state_channel_dim(u: torch.Tensor) -> Optional[int]:
    if u.dim() in {4, 5}:
        return -3
    return None


def _channel_weights(channel_weights, tensor: torch.Tensor) -> Optional[torch.Tensor]:
    if channel_weights is None:
        return None
    w = torch.as_tensor(channel_weights, device=tensor.device, dtype=tensor.dtype)
    if w.dim() != 1:
        raise ValueError("channel_weights must be 1D")
    return w


def channel_weighted_mse(u_pred: torch.Tensor, u_ref: torch.Tensor, channel_weights=None) -> torch.Tensor:
    if u_pred.shape != u_ref.shape:
        raise ValueError(f"u_pred and u_ref must have identical shape, got {tuple(u_pred.shape)} vs {tuple(u_ref.shape)}")
    w = _channel_weights(channel_weights, u_pred)
    if w is None:
        return F.mse_loss(u_pred, u_ref)
    if u_pred.dim() not in {4, 5}:
        if int(w.numel()) != 1:
            raise ValueError("scalar states require exactly one channel weight")
        return F.mse_loss(u_pred, u_ref) * w[0]
    if int(w.numel()) != int(u_pred.shape[-3]):
        raise ValueError(f"channel_weights has {w.numel()} entries for {u_pred.shape[-3]} channels")
    view_shape = [1] * u_pred.dim()
    view_shape[-3] = int(w.numel())
    return ((u_pred - u_ref).square() * w.view(*view_shape)).mean()


def spectral_sobolev_loss_2d(u_pred: torch.Tensor, u_ref: torch.Tensor, s: float) -> torch.Tensor:
    if u_pred.shape != u_ref.shape:
        raise ValueError(f"u_pred and u_ref must have identical shape, got {tuple(u_pred.shape)} vs {tuple(u_ref.shape)}")
    diff = u_pred - u_ref
    n_x = int(diff.shape[-2])
    n_y = int(diff.shape[-1])
    diff_hat = torch.fft.rfft2(diff, dim=(-2, -1), norm="ortho")
    real_dtype = diff.real.dtype
    kx = 2.0 * torch.pi * torch.fft.fftfreq(n_x, d=1.0 / float(n_x), device=diff.device).to(dtype=real_dtype)
    ky = 2.0 * torch.pi * torch.fft.rfftfreq(n_y, d=1.0 / float(n_y), device=diff.device).to(dtype=real_dtype)
    kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing="ij")
    weight = (1.0 + kx_grid.square() + ky_grid.square()).pow(float(s))
    power = diff_hat.real.square() + diff_hat.imag.square()
    return (power * weight).mean()


def relative_l2_error_2d(u_pred: torch.Tensor, u_ref: torch.Tensor, area: float) -> torch.Tensor:
    if u_pred.shape != u_ref.shape:
        raise ValueError(f"u_pred and u_ref must have identical shape, got {tuple(u_pred.shape)} vs {tuple(u_ref.shape)}")
    diff = u_pred - u_ref
    if u_pred.dim() == 5:
        reduce_dims = (-3, -2, -1)
    else:
        reduce_dims = (-2, -1)
    num = torch.sqrt(float(area) * torch.sum(diff * diff, dim=reduce_dims))
    den = torch.sqrt(float(area) * torch.sum(u_ref * u_ref, dim=reduce_dims))
    return num / (den + 1e-8)


@torch.no_grad()
def rollout_fno_2d(
    model,
    u0: torch.Tensor,
    f: torch.Tensor | None,
    n_steps: int,
    dt: float,
    delta_clip: Optional[float] = None,
) -> torch.Tensor:
    states = [u0]
    u = u0
    for _ in range(int(n_steps)):
        u_tilde = model.predict_step(u, f, dt=dt)
        delta = u_tilde - u
        if delta_clip is not None and float(delta_clip) > 0.0:
            delta = torch.clamp(delta, min=-float(delta_clip), max=float(delta_clip))
        u_next = u + delta
        finite = torch.isfinite(u_next).flatten(1).all(dim=1)
        finite_view = [u_next.shape[0]] + [1] * (u_next.dim() - 1)
        u = torch.where(finite.view(*finite_view), u_next, states[-1])
        states.append(u)
    return torch.stack(states, dim=1)


class FNOTrainer2D:
    def __init__(
        self,
        model: torch.nn.Module,
        dt: float,
        h_x: float,
        h_y: float,
        lambda_spec: float = 0.0,
        spectral_s: float = 1.0,
        channel_weights=None,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
        rollout_delta_clip: Optional[float] = 10.0,
        lr_step_size: int = 100,
        lr_gamma: float = 0.5,
        device: str = "cpu",
        output_dir: Optional[str] = None,
        show_epoch_pbar: bool = True,
    ):
        self.model = model.to(device)
        self.dt = float(dt)
        self.h_x = float(h_x)
        self.h_y = float(h_y)
        self.area = self.h_x * self.h_y
        self.lambda_spec = float(lambda_spec)
        self.spectral_s = float(spectral_s)
        self.channel_weights = None if channel_weights is None else torch.as_tensor(channel_weights, dtype=torch.float32)
        self.grad_clip = float(grad_clip)
        self.rollout_delta_clip = None if rollout_delta_clip is None else float(rollout_delta_clip)
        self.device = device
        self.output_dir = output_dir
        self.show_epoch_pbar = bool(show_epoch_pbar)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=max(1, int(lr_step_size)),
            gamma=float(lr_gamma),
        )

    def _weights_for(self, tensor: torch.Tensor):
        return _channel_weights(self.channel_weights, tensor)

    def _compute_losses(self, u_k: torch.Tensor, u_k1: torch.Tensor, f: torch.Tensor) -> Dict[str, torch.Tensor]:
        u_pred = self.model.predict_step(u_k, f, dt=self.dt)
        weights = self._weights_for(u_pred)
        loss_step = channel_weighted_mse(u_pred, u_k1, weights)
        loss_spec = (
            spectral_sobolev_loss_2d(u_pred, u_k1, s=self.spectral_s)
            if self.lambda_spec > 0.0
            else loss_step.new_zeros(())
        )
        return {"loss": loss_step + self.lambda_spec * loss_spec, "loss_step": loss_step, "loss_spec": loss_spec}

    def train_epoch(self, loader: DataLoader, epoch: Optional[int] = None) -> Dict[str, float]:
        self.model.train()
        meters = {k: AverageMeter() for k in ("loss", "loss_step", "loss_spec")}
        iterable = loader
        pbar = None
        if self.show_epoch_pbar and tqdm is not None:
            pbar = tqdm(loader, total=len(loader), desc=f"Epoch {epoch:03d}" if epoch else "Epoch", leave=False)
            iterable = pbar
        for batch_idx, batch in enumerate(iterable, start=1):
            u_k, u_k1, f = _unpack_step_batch(batch)
            u_k = u_k.to(self.device)
            u_k1 = u_k1.to(self.device)
            f = f.to(self.device)
            losses = self._compute_losses(u_k, u_k1, f)
            self.optimizer.zero_grad()
            losses["loss"].backward()
            if self.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            bsz = int(u_k.shape[0])
            for key, meter in meters.items():
                meter.update(losses[key].item(), bsz)
            if pbar is not None and (batch_idx == 1 or batch_idx % 10 == 0):
                pbar.set_postfix(total=f"{meters['loss'].avg:.4f}", step=f"{meters['loss_step'].avg:.4f}")
        if pbar is not None:
            pbar.close()
        return {k: v.avg for k, v in meters.items()}

    def validate(self, step_loader: DataLoader, traj_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        self.model.eval()
        meters = {k: AverageMeter() for k in ("val_loss", "val_loss_step", "val_loss_spec")}
        with torch.no_grad():
            for batch in step_loader:
                u_k, u_k1, f = _unpack_step_batch(batch)
                u_k = u_k.to(self.device)
                u_k1 = u_k1.to(self.device)
                f = f.to(self.device)
                losses = self._compute_losses(u_k, u_k1, f)
                bsz = int(u_k.shape[0])
                meters["val_loss"].update(losses["loss"].item(), bsz)
                meters["val_loss_step"].update(losses["loss_step"].item(), bsz)
                meters["val_loss_spec"].update(losses["loss_spec"].item(), bsz)
        metrics = {k: v.avg for k, v in meters.items()}
        if traj_loader is not None:
            rollout_meter = AverageMeter()
            with torch.no_grad():
                for batch in traj_loader:
                    u0, f, u_ref = _unpack_traj_batch(batch)
                    u0 = u0.to(self.device)
                    f = f.to(self.device)
                    u_ref = u_ref.to(self.device)
                    u_pred = rollout_fno_2d(
                        self.model,
                        u0=u0,
                        f=f,
                        n_steps=int(u_ref.shape[1] - 1),
                        dt=self.dt,
                        delta_clip=self.rollout_delta_clip,
                    )
                    rel = relative_l2_error_2d(u_pred, u_ref, area=self.area)
                    rollout_meter.update(rel.mean().item(), int(u0.shape[0]))
            metrics["val_rollout_rel_l2"] = rollout_meter.avg
        return metrics

    def _save_checkpoint(self, name: str, epoch: int, metrics: Dict[str, float], state_dict=None) -> None:
        if self.output_dir is None:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": state_dict if state_dict is not None else self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "dt": self.dt,
                "h_x": self.h_x,
                "h_y": self.h_y,
                "lambda_spec": self.lambda_spec,
                "spectral_s": self.spectral_s,
                "rollout_delta_clip": self.rollout_delta_clip,
                "channel_weights": None if self.channel_weights is None else self.channel_weights.detach().cpu(),
            },
            os.path.join(self.output_dir, name),
        )

    def fit(
        self,
        train_step_loader: DataLoader,
        val_step_loader: DataLoader,
        val_traj_loader: Optional[DataLoader] = None,
        epochs: int = 200,
        eval_interval: int = 1,
        checkpoint_interval: int = 25,
    ) -> Dict[str, list]:
        history = {"train": [], "val": []}
        best_metric = float("inf")
        best_epoch = 0
        best_metrics: Optional[Dict[str, float]] = None
        best_state_dict = None
        for epoch in range(1, int(epochs) + 1):
            train_metrics = self.train_epoch(train_step_loader, epoch=epoch)
            history["train"].append({"epoch": epoch, **train_metrics})
            if epoch % int(eval_interval) == 0:
                val_metrics = self.validate(val_step_loader, traj_loader=val_traj_loader)
                history["val"].append({"epoch": epoch, **val_metrics})
                monitor = val_metrics.get("val_rollout_rel_l2", val_metrics["val_loss_step"])
                if monitor < best_metric:
                    best_metric = monitor
                    best_epoch = epoch
                    best_metrics = val_metrics
                    best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    self._save_checkpoint("best_model.pt", epoch, val_metrics)
                print(
                    f"[Epoch {epoch:03d}] "
                    f"train_total={train_metrics['loss']:.6f} "
                    f"train_step={train_metrics['loss_step']:.6f} "
                    f"train_spec={train_metrics['loss_spec']:.6f} "
                    f"val_total={val_metrics['val_loss']:.6f} "
                    f"val_step={val_metrics['val_loss_step']:.6f} "
                    f"val_spec={val_metrics['val_loss_spec']:.6f} "
                    f"val_rollout={val_metrics.get('val_rollout_rel_l2', float('nan')):.6f}"
                )
            else:
                print(
                    f"[Epoch {epoch:03d}] "
                    f"train_total={train_metrics['loss']:.6f} "
                    f"train_step={train_metrics['loss_step']:.6f} "
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
            with open(os.path.join(self.output_dir, "history.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        return history


__all__ = [
    "AverageMeter",
    "FNOTrainer2D",
    "channel_weighted_mse",
    "relative_l2_error_2d",
    "rollout_fno_2d",
    "spectral_sobolev_loss_2d",
]
