"""
Training utilities for hidden-space gradient-flow models on 1D Burgers data.
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
    from ..utils import compute_relative_l2_error, rollout_model
except ImportError:
    from grad_flow_l2.utils import compute_relative_l2_error, rollout_model


@dataclass
class AverageMeter:
    val: float = 0.0
    avg: float = 0.0
    total: float = 0.0
    count: int = 0

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.total = 0.0
        self.count = 0

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


class HiddenGradientFlowTrainer:
    """
    Trainer for models implementing:
      - predict_step(u_k, f, dt=..., return_latent=True) -> (u_k1, z_k, z_k1)
      - decode(z) -> u
      - latent_energy(z, f) -> scalar energy per sample
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dt: float,
        h: float,
        lambda_mono: float = 0.0,
        lambda_edi: float = 0.0,
        lambda_recon: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
        max_epochs: int = 200,
        device: str = "cpu",
        output_dir: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.dt = float(dt)
        self.h = float(h)
        self.lambda_mono = float(lambda_mono)
        self.lambda_edi = float(lambda_edi)
        self.lambda_recon = float(lambda_recon)
        self.grad_clip = float(grad_clip)
        self.device = device
        self.output_dir = output_dir

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs,
        )

    def _compute_losses(self, u_k: torch.Tensor, u_k1_data: torch.Tensor, f: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred = self.model.predict_step(u_k, f, dt=self.dt, return_latent=True)
        if not (isinstance(pred, (tuple, list)) and len(pred) == 3):
            raise ValueError("model.predict_step(..., return_latent=True) must return (u_next, z_k, z_next)")

        u_pred, z_k, z_next = pred
        l_step = F.mse_loss(u_pred, u_k1_data)

        # Autoencoding consistency for stabilizing latent training.
        u_recon = self.model.decode(z_k)
        l_recon = F.mse_loss(u_recon, u_k)

        e_prev = self.model.latent_energy(z_k, f)
        e_next = self.model.latent_energy(z_next, f)
        l_mono = torch.relu(e_next - e_prev).mean()

        d_z_sq = self.h * torch.sum((z_next - z_k) ** 2, dim=(1, 2))
        # Penalize the full EDI residual directly instead of only the positive part.
        edi_residual = e_next + d_z_sq / (2.0 * self.dt) - e_prev
        l_edi = torch.mean(edi_residual * edi_residual)

        l_total = l_step + self.lambda_recon * l_recon + self.lambda_mono * l_mono + self.lambda_edi * l_edi
        return {
            "loss": l_total,
            "loss_step": l_step,
            "loss_recon": l_recon,
            "loss_mono": l_mono,
            "loss_edi": l_edi,
        }

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()

        meters = {
            "loss": AverageMeter(),
            "loss_step": AverageMeter(),
            "loss_recon": AverageMeter(),
            "loss_mono": AverageMeter(),
            "loss_edi": AverageMeter(),
        }

        for batch in loader:
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

            batch_size = u_k.shape[0]
            for key, meter in meters.items():
                meter.update(losses[key].item(), batch_size)

        return {k: m.avg for k, m in meters.items()}

    @torch.no_grad()
    def validate(self, step_loader: DataLoader, traj_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        self.model.eval()

        meters = {
            "val_loss": AverageMeter(),
            "val_loss_step": AverageMeter(),
            "val_loss_recon": AverageMeter(),
            "val_loss_mono": AverageMeter(),
            "val_loss_edi": AverageMeter(),
        }

        for batch in step_loader:
            u_k, u_k1, f = _unpack_step_batch(batch)
            u_k = u_k.to(self.device)
            u_k1 = u_k1.to(self.device)
            f = f.to(self.device)

            losses = self._compute_losses(u_k, u_k1, f)
            batch_size = u_k.shape[0]
            meters["val_loss"].update(losses["loss"].item(), batch_size)
            meters["val_loss_step"].update(losses["loss_step"].item(), batch_size)
            meters["val_loss_recon"].update(losses["loss_recon"].item(), batch_size)
            meters["val_loss_mono"].update(losses["loss_mono"].item(), batch_size)
            meters["val_loss_edi"].update(losses["loss_edi"].item(), batch_size)

        metrics = {k: m.avg for k, m in meters.items()}

        if traj_loader is not None:
            rollout_err_meter = AverageMeter()
            for batch in traj_loader:
                u0, f, u_traj_ref = _unpack_traj_batch(batch)
                u0 = u0.to(self.device)
                f = f.to(self.device)
                u_traj_ref = u_traj_ref.to(self.device)

                n_steps = u_traj_ref.shape[1] - 1
                u_traj_pred = rollout_model(self.model, u0=u0, f=f, n_steps=n_steps, dt=self.dt)
                rel = compute_relative_l2_error(u_traj_pred, u_traj_ref, h=self.h)
                rollout_err = rel.mean(dim=-1)
                rollout_err_meter.update(rollout_err.mean().item(), u0.shape[0])

            metrics["val_rollout_rel_l2"] = rollout_err_meter.avg

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
                "h": self.h,
                "lambda_mono": self.lambda_mono,
                "lambda_edi": self.lambda_edi,
                "lambda_recon": self.lambda_recon,
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
    ) -> Dict[str, list]:
        history = {"train": [], "val": []}
        best_metric = float("inf")

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_step_loader)
            history["train"].append({"epoch": epoch, **train_metrics})

            if epoch % eval_interval == 0:
                val_metrics = self.validate(val_step_loader, traj_loader=val_traj_loader)
                history["val"].append({"epoch": epoch, **val_metrics})

                monitor = val_metrics.get("val_rollout_rel_l2", val_metrics["val_loss_step"])
                if monitor < best_metric:
                    best_metric = monitor
                    self._save_checkpoint("best_model.pt", epoch, val_metrics)

                print(
                    f"[Epoch {epoch:03d}] "
                    f"train_total={train_metrics['loss']:.6f} "
                    f"train_step={train_metrics['loss_step']:.6f} "
                    f"train_recon={train_metrics['loss_recon']:.6f} "
                    f"train_mono={train_metrics['loss_mono']:.6f} "
                    f"train_edi={train_metrics['loss_edi']:.6f} "
                    f"val_total={val_metrics['val_loss']:.6f} "
                    f"val_step={val_metrics['val_loss_step']:.6f} "
                    f"val_recon={val_metrics['val_loss_recon']:.6f} "
                    f"val_mono={val_metrics['val_loss_mono']:.6f} "
                    f"val_edi={val_metrics['val_loss_edi']:.6f} "
                    f"val_rollout={val_metrics.get('val_rollout_rel_l2', float('nan')):.6f}"
                )
            else:
                print(
                    f"[Epoch {epoch:03d}] "
                    f"train_total={train_metrics['loss']:.6f} "
                    f"train_step={train_metrics['loss_step']:.6f} "
                    f"train_recon={train_metrics['loss_recon']:.6f} "
                    f"train_mono={train_metrics['loss_mono']:.6f} "
                    f"train_edi={train_metrics['loss_edi']:.6f}"
                )

            self.scheduler.step()

        final_metrics = history["val"][-1] if history["val"] else history["train"][-1]
        self._save_checkpoint("final_model.pt", epochs, final_metrics)

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, "history.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

        return history
