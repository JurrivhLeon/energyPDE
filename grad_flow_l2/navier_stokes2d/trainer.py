"""
Trainer for 2D hidden-space gradient-flow models on Navier-Stokes-vorticity data.
"""

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


def rollout_model_2d(model, u0: torch.Tensor, f: torch.Tensor, n_steps: int, dt: float) -> torch.Tensor:
    squeeze = False
    if u0.dim() == 2:
        u0 = u0.unsqueeze(0)
        f = f.unsqueeze(0)
        squeeze = True

    states = [u0]
    u = u0
    for _ in range(n_steps):
        u = model.predict_step(u, f, dt=dt)
        states.append(u)

    traj = torch.stack(states, dim=1)  # (batch, K+1, n_x, n_y)
    if squeeze:
        return traj.squeeze(0)
    return traj


def relative_l2_error_2d(u_pred: torch.Tensor, u_ref: torch.Tensor, area: float) -> torch.Tensor:
    diff = u_pred - u_ref
    num = torch.sqrt(float(area) * torch.sum(diff * diff, dim=(-2, -1)))
    den = torch.sqrt(float(area) * torch.sum(u_ref * u_ref, dim=(-2, -1)))
    return num / (den + 1e-8)


def spectral_sobolev_loss_2d(u_pred: torch.Tensor, u_ref: torch.Tensor, s: float) -> torch.Tensor:
    """
    Weighted Fourier-domain loss:

        sum_k (1 + |k|^2)^s |û_pred(k) - û_ref(k)|^2.

    This emphasizes high-mode errors when s > 0 and reduces to an FFT-domain
    L2 loss when s = 0.
    """
    if u_pred.shape != u_ref.shape:
        raise ValueError(f"u_pred and u_ref must have identical shape, got {tuple(u_pred.shape)} vs {tuple(u_ref.shape)}")
    if u_pred.dim() != 3:
        raise ValueError(f"Expected tensors with shape (batch,n_x,n_y), got {tuple(u_pred.shape)}")

    n_x = int(u_pred.shape[-2])
    n_y = int(u_pred.shape[-1])
    diff_hat = torch.fft.rfft2(u_pred - u_ref, dim=(-2, -1), norm="ortho")

    real_dtype = u_pred.real.dtype
    kx = 2.0 * torch.pi * torch.fft.fftfreq(n_x, d=1.0 / float(n_x), device=u_pred.device).to(dtype=real_dtype)
    ky = 2.0 * torch.pi * torch.fft.rfftfreq(n_y, d=1.0 / float(n_y), device=u_pred.device).to(dtype=real_dtype)
    kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing="ij")
    weight = (1.0 + kx_grid.square() + ky_grid.square()).pow(float(s))

    power = diff_hat.real.square() + diff_hat.imag.square()
    weighted_power = power * weight.unsqueeze(0)
    return weighted_power.mean()


class HiddenGradientFlowTrainer2D:
    """
    Trainer for model implementing:
      - predict_step(u_k, f, dt=..., return_latent=True) -> (u_{k+1}, z_k, z_{k+1})
      - decode(z) -> u
      - latent_energy(z, f) -> scalar energy per sample

    The proximal regularizer follows the discrete energy-dissipation inequality
    style used in the paper:

        E(z_{k+1}) + ||z_{k+1} - z_k||^2 / (2 dt) <= E(z_k).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dt: float,
        h_x: float,
        h_y: float,
        lambda_recon: float = 1.0,
        lambda_mono: float = 1.0,
        lambda_prox: float = 1.0,
        lambda_spec: float = 0.0,
        spectral_s: float = 1.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
        lr_step_size: int = 100,
        lr_gamma: float = 0.5,
        max_epochs: int = 200,
        device: str = "cpu",
        output_dir: Optional[str] = None,
        show_epoch_pbar: bool = True,
    ):
        self.model = model.to(device)
        self.dt = float(dt)
        self.h_x = float(h_x)
        self.h_y = float(h_y)
        self.area = self.h_x * self.h_y
        self.lambda_recon = float(lambda_recon)
        self.lambda_mono = float(lambda_mono)
        self.lambda_prox = float(lambda_prox)
        self.lambda_spec = float(lambda_spec)
        self.spectral_s = float(spectral_s)
        self.grad_clip = float(grad_clip)
        self.device = device
        self.output_dir = output_dir
        self.show_epoch_pbar = bool(show_epoch_pbar)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=max(1, int(lr_step_size)),
            gamma=float(lr_gamma),
        )

    def _compute_losses(self, u_k: torch.Tensor, u_k1_data: torch.Tensor, f: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred = self.model.predict_step(u_k, f, dt=self.dt, return_latent=True)
        if not (isinstance(pred, (tuple, list)) and len(pred) == 3):
            raise ValueError("predict_step(..., return_latent=True) must return (u_next, z_k, z_next)")
        u_pred, z_k, z_next = pred

        loss_step = F.mse_loss(u_pred, u_k1_data)
        if self.lambda_spec > 0.0:
            loss_spec = spectral_sobolev_loss_2d(u_pred, u_k1_data, s=self.spectral_s)
        else:
            loss_spec = loss_step.new_zeros(())

        u_recon = self.model.decode(z_k)
        loss_recon = F.mse_loss(u_recon, u_k)

        e_prev = self.model.latent_energy(z_k, f)
        e_next = self.model.latent_energy(z_next, f)
        loss_mono = torch.relu(e_next - e_prev).mean()

        d_z_sq = self.area * torch.sum((z_next - z_k) ** 2, dim=(1, 2, 3))
        # Penalize the full EDI residual directly instead of only the positive part.
        prox_residual = e_next + d_z_sq / (2.0 * self.dt) - e_prev
        loss_prox = torch.mean(prox_residual * prox_residual)

        loss_total = (
            loss_step
            + self.lambda_spec * loss_spec
            + self.lambda_recon * loss_recon
            + self.lambda_mono * loss_mono
            + self.lambda_prox * loss_prox
        )
        return {
            "loss": loss_total,
            "loss_step": loss_step,
            "loss_spec": loss_spec,
            "loss_recon": loss_recon,
            "loss_mono": loss_mono,
            "loss_prox": loss_prox,
        }

    def train_epoch(self, loader: DataLoader, epoch: Optional[int] = None) -> Dict[str, float]:
        self.model.train()
        meters = {
            "loss": AverageMeter(),
            "loss_step": AverageMeter(),
            "loss_spec": AverageMeter(),
            "loss_recon": AverageMeter(),
            "loss_mono": AverageMeter(),
            "loss_prox": AverageMeter(),
        }

        show_pbar = self.show_epoch_pbar and (tqdm is not None)
        pbar = None
        iterable = loader
        if show_pbar:
            total = len(loader) if hasattr(loader, "__len__") else None
            desc = f"Epoch {epoch:03d}" if epoch is not None else "Epoch"
            pbar = tqdm(loader, total=total, desc=desc, leave=False, dynamic_ncols=True)
            iterable = pbar

        for batch_idx, batch in enumerate(iterable, start=1):
            u_k, u_k1, f = _unpack_step_batch(batch)
            u_k = u_k.to(self.device)
            u_k1 = u_k1.to(self.device)
            f = f.to(self.device)

            losses = self._compute_losses(u_k, u_k1, f)

            self.optimizer.zero_grad()
            losses["loss"].backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            bsz = int(u_k.shape[0])
            for k, m in meters.items():
                m.update(losses[k].item(), bsz)
            if pbar is not None and (batch_idx == 1 or batch_idx % 10 == 0):
                pbar.set_postfix(
                    total=f"{meters['loss'].avg:.4f}",
                    step=f"{meters['loss_step'].avg:.4f}",
                )

        if pbar is not None:
            pbar.close()

        return {k: v.avg for k, v in meters.items()}

    def validate(self, step_loader: DataLoader, traj_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        self.model.eval()
        meters = {
            "val_loss": AverageMeter(),
            "val_loss_step": AverageMeter(),
            "val_loss_spec": AverageMeter(),
            "val_loss_recon": AverageMeter(),
            "val_loss_mono": AverageMeter(),
            "val_loss_prox": AverageMeter(),
        }

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
                meters["val_loss_recon"].update(losses["loss_recon"].item(), bsz)
                meters["val_loss_mono"].update(losses["loss_mono"].item(), bsz)
                meters["val_loss_prox"].update(losses["loss_prox"].item(), bsz)

        metrics = {k: v.avg for k, v in meters.items()}

        if traj_loader is not None:
            rollout_meter = AverageMeter()
            with torch.no_grad():
                for batch in traj_loader:
                    u0, f, u_ref = _unpack_traj_batch(batch)
                    u0 = u0.to(self.device)
                    f = f.to(self.device)
                    u_ref = u_ref.to(self.device)

                    n_steps = int(u_ref.shape[1] - 1)
                    u_pred = rollout_model_2d(self.model, u0=u0, f=f, n_steps=n_steps, dt=self.dt)
                    rel = relative_l2_error_2d(u_pred, u_ref, area=self.area)  # (batch, K+1)
                    rollout_err = rel.mean(dim=-1)
                    rollout_meter.update(rollout_err.mean().item(), int(u0.shape[0]))
            metrics["val_rollout_rel_l2"] = rollout_meter.avg

        return metrics

    def _save_checkpoint(self, name: str, epoch: int, metrics: Dict[str, float]) -> None:
        if self.output_dir is None:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, name)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "dt": self.dt,
                "h_x": self.h_x,
                "h_y": self.h_y,
                "lambda_recon": self.lambda_recon,
                "lambda_mono": self.lambda_mono,
                "lambda_prox": self.lambda_prox,
                "lambda_spec": self.lambda_spec,
                "spectral_s": self.spectral_s,
            },
            path,
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

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_step_loader, epoch=epoch)
            history["train"].append({"epoch": epoch, **train_metrics})

            if epoch % eval_interval == 0:
                val_metrics = self.validate(val_step_loader, traj_loader=val_traj_loader)
                history["val"].append({"epoch": epoch, **val_metrics})
                monitor = val_metrics.get("val_rollout_rel_l2", val_metrics["val_loss_step"])
                if monitor < best_metric:
                    best_metric = monitor
                    best_epoch = epoch
                    best_metrics = val_metrics
                    self._save_checkpoint("best_model.pt", epoch, val_metrics)

                print(
                    f"[Epoch {epoch:03d}] "
                    f"train_total={train_metrics['loss']:.6f} "
                    f"train_step={train_metrics['loss_step']:.6f} "
                    f"train_spec={train_metrics['loss_spec']:.6f} "
                    f"train_recon={train_metrics['loss_recon']:.6f} "
                    f"train_mono={train_metrics['loss_mono']:.6f} "
                    f"train_prox={train_metrics['loss_prox']:.6f} "
                    f"val_total={val_metrics['val_loss']:.6f} "
                    f"val_step={val_metrics['val_loss_step']:.6f} "
                    f"val_spec={val_metrics['val_loss_spec']:.6f} "
                    f"val_recon={val_metrics['val_loss_recon']:.6f} "
                    f"val_mono={val_metrics['val_loss_mono']:.6f} "
                    f"val_prox={val_metrics['val_loss_prox']:.6f} "
                    f"val_rollout={val_metrics.get('val_rollout_rel_l2', float('nan')):.6f}"
                )
            else:
                print(
                    f"[Epoch {epoch:03d}] "
                    f"train_total={train_metrics['loss']:.6f} "
                    f"train_step={train_metrics['loss_step']:.6f} "
                    f"train_spec={train_metrics['loss_spec']:.6f} "
                    f"train_recon={train_metrics['loss_recon']:.6f} "
                    f"train_mono={train_metrics['loss_mono']:.6f} "
                    f"train_prox={train_metrics['loss_prox']:.6f}"
                )

            if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                snapshot_metrics = best_metrics if best_metrics is not None else train_metrics
                snapshot_epoch = best_epoch if best_metrics is not None else epoch
                self._save_checkpoint(f"best_model_through_epoch_{epoch:04d}.pt", snapshot_epoch, snapshot_metrics)

            self.scheduler.step()

        final_metrics = history["val"][-1] if history["val"] else history["train"][-1]
        self._save_checkpoint("final_model.pt", epochs, final_metrics)

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, "history.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

        return history
