"""
Training entrypoint for the periodic 2D Navier-Stokes latent VAE model.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    from ..heat_data import load_dataset_splits
    from ..latent_flow_VAE import (
        FNOLatentTransition2D,
        PeriodicLatentVAE2D,
        TransitionAmplitudeHead2D,
        VariationalStateEncoder2D,
    )
    from ..latent_markov import StateDecoder2D
    from ..latent_markov_trainer import relative_l2_error_2d
    from ..navier_stokes2d_per_data import (
        build_navier_stokes2d_periodic_step_dataset,
        build_navier_stokes2d_periodic_trajectory_dataset_from_split,
    )
except ImportError:
    from grad_flow_l2.heat_data import load_dataset_splits
    from grad_flow_l2.latent_flow_VAE import (
        FNOLatentTransition2D,
        PeriodicLatentVAE2D,
        TransitionAmplitudeHead2D,
        VariationalStateEncoder2D,
    )
    from grad_flow_l2.latent_markov import StateDecoder2D
    from grad_flow_l2.latent_markov_trainer import relative_l2_error_2d
    from grad_flow_l2.navier_stokes2d_per_data import (
        build_navier_stokes2d_periodic_step_dataset,
        build_navier_stokes2d_periodic_trajectory_dataset_from_split,
    )


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


def kl_diag_gaussians(
    q_mu: torch.Tensor,
    q_logvar: torch.Tensor,
    p_mu: torch.Tensor,
    p_logvar_scalar: torch.Tensor,
) -> torch.Tensor:
    """
    Mean KL(q || p) for q diagonal in latent coordinates and p with one
    scalar diagonal variance per sample.
    """
    if q_mu.shape != q_logvar.shape or q_mu.shape != p_mu.shape:
        raise ValueError("q_mu, q_logvar, and p_mu must have identical shapes")
    if p_logvar_scalar.dim() != 1 or p_logvar_scalar.shape[0] != q_mu.shape[0]:
        raise ValueError("p_logvar_scalar must have shape (batch,)")

    q_logvar = torch.clamp(q_logvar, min=-12.0, max=8.0)
    p_logvar = torch.clamp(p_logvar_scalar, min=-12.0, max=8.0).view(-1, 1, 1, 1)
    q_var = torch.exp(q_logvar)
    p_var = torch.exp(p_logvar)
    kl = 0.5 * (p_logvar - q_logvar + (q_var + (q_mu - p_mu).square()) / p_var - 1.0)
    return kl.mean()


def rollout_vae_mean(
    model: PeriodicLatentVAE2D,
    u0: torch.Tensor,
    f: torch.Tensor,
    n_steps: int,
    dt: float,
    delta_clip: Optional[float] = None,
    state_clip: Optional[float] = None,
) -> torch.Tensor:
    squeeze = False
    if u0.dim() == 2:
        u0 = u0.unsqueeze(0)
        f = f.unsqueeze(0)
        squeeze = True

    states = [u0]
    u = u0
    for _ in range(n_steps):
        u_tilde = model.rollout_step(u, f, dt=dt)
        delta = u_tilde - u
        if delta_clip is not None and float(delta_clip) > 0.0:
            delta = torch.clamp(delta, min=-float(delta_clip), max=float(delta_clip))
        u_next = u + delta
        finite = torch.isfinite(u_next).flatten(1).all(dim=1)
        u_next = torch.where(finite[:, None, None], u_next, u)
        if state_clip is not None and float(state_clip) > 0.0:
            u_next = torch.clamp(u_next, min=-float(state_clip), max=float(state_clip))
        u = u_next
        states.append(u)

    traj = torch.stack(states, dim=1)
    if squeeze:
        return traj.squeeze(0)
    return traj


class PeriodicLatentVAETrainer2D:
    def __init__(
        self,
        model: PeriodicLatentVAE2D,
        dt: float,
        h_x: float,
        h_y: float,
        beta_kl: float = 1e-4,
        lambda_rec: float = 1.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
        spectral_var_floor: float = 1e-2,
        alpha_min: float = 1e-4,
        alpha_max: float = 0.5,
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
        self.beta_kl = float(beta_kl)
        self.lambda_rec = float(lambda_rec)
        self.grad_clip = float(grad_clip)
        self.spectral_var_floor = float(spectral_var_floor)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        if self.alpha_min < 0.0 or self.alpha_max <= self.alpha_min:
            raise ValueError("alpha bounds must satisfy 0 <= alpha_min < alpha_max")
        self.device = device
        self.output_dir = output_dir
        self.show_epoch_pbar = bool(show_epoch_pbar)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=max(1, int(lr_step_size)),
            gamma=float(lr_gamma),
        )

    def _filtered_noise_for_training(self, shape, device, dtype) -> torch.Tensor:
        xi = torch.randn(shape, device=device, dtype=dtype)
        xi_hat = torch.fft.rfft2(xi, dim=(-2, -1), norm="ortho")
        c_sqrt = self.model._spectral_filter(device=device, dtype=dtype)
        filt = torch.sqrt(c_sqrt.square() + self.spectral_var_floor)
        return torch.fft.irfft2(
            xi_hat * filt.view(1, 1, self.model.n_x, self.model.n_y // 2 + 1),
            s=(self.model.n_x, self.model.n_y),
            dim=(-2, -1),
            norm="ortho",
        )

    def _compute_losses(
        self,
        u_k: torch.Tensor,
        u_k1_data: torch.Tensor,
        f: torch.Tensor,
        sample: bool = True,
    ) -> Dict[str, torch.Tensor]:
        mu_q, logvar_q = self.model.encode_stats(u_k)
        z_k = self.model.sample_posterior(mu_q, logvar_q) if sample else mu_q

        mu_p, prior_logvar_scalar = self.model.prior_stats(z_k, f, dt=self.dt)
        alpha = torch.exp(0.5 * prior_logvar_scalar)
        if sample:
            noise = self._filtered_noise_for_training(mu_p.shape, device=mu_p.device, dtype=mu_p.dtype)
            z_next = mu_p + alpha.view(-1, 1, 1, 1) * noise
        else:
            z_next = mu_p
        
        u_pred = self.model.decode(z_next)
        loss_step = F.mse_loss(u_pred, u_k1_data)

        u_recon = self.model.decode(z_k)
        loss_recon = F.mse_loss(u_recon, u_k)

        mu_q_next, logvar_q_next = self.model.encode_stats(u_k1_data)
        prior_logvar_scalar = torch.log(alpha.square() + 1e-12)
        loss_kl = kl_diag_gaussians(mu_q_next, logvar_q_next, mu_p, prior_logvar_scalar)

        loss_total = loss_step + self.beta_kl * loss_kl + self.lambda_rec * loss_recon
        return {
            "loss": loss_total,
            "loss_step": loss_step,
            "loss_kl": loss_kl,
            "loss_recon": loss_recon,
            "alpha_mean": alpha.mean().detach(),
        }

    def train_epoch(self, loader: DataLoader, epoch: Optional[int] = None) -> Dict[str, float]:
        self.model.train()
        meters = {name: AverageMeter() for name in ("loss", "loss_step", "loss_kl", "loss_recon", "alpha_mean")}

        show_pbar = self.show_epoch_pbar and (tqdm is not None)
        iterable = loader
        pbar = None
        if show_pbar:
            desc = f"Epoch {epoch:03d}" if epoch is not None else "Epoch"
            pbar = tqdm(loader, total=len(loader), desc=desc, leave=False, dynamic_ncols=True)
            iterable = pbar

        for batch_idx, batch in enumerate(iterable, start=1):
            u_k, u_k1, f = _unpack_step_batch(batch)
            u_k = u_k.to(self.device)
            u_k1 = u_k1.to(self.device)
            f = f.to(self.device)

            losses = self._compute_losses(u_k, u_k1, f, sample=True)
            if not torch.isfinite(losses["loss"]):
                diagnostics = ", ".join(
                    f"{name}={float(value.detach().cpu())}"
                    for name, value in losses.items()
                    if value.numel() == 1
                )
                raise FloatingPointError(
                    f"Non-finite VAE training loss at epoch={epoch}, batch={batch_idx}: {diagnostics}"
                )
            self.optimizer.zero_grad()
            losses["loss"].backward()
            if self.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip, error_if_nonfinite=True)
            self.optimizer.step()

            bsz = int(u_k.shape[0])
            for k, m in meters.items():
                m.update(losses[k].item(), bsz)
            if pbar is not None and (batch_idx == 1 or batch_idx % 10 == 0):
                pbar.set_postfix(total=f"{meters['loss'].avg:.4f}", step=f"{meters['loss_step'].avg:.4f}")

        if pbar is not None:
            pbar.close()
        return {k: v.avg for k, v in meters.items()}

    @torch.no_grad()
    def validate(self, step_loader: DataLoader, traj_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        self.model.eval()
        meters = {
            name: AverageMeter()
            for name in ("val_loss", "val_loss_step", "val_loss_kl", "val_loss_recon", "val_alpha_mean")
        }

        for batch in step_loader:
            u_k, u_k1, f = _unpack_step_batch(batch)
            u_k = u_k.to(self.device)
            u_k1 = u_k1.to(self.device)
            f = f.to(self.device)

            losses = self._compute_losses(u_k, u_k1, f, sample=False)
            bsz = int(u_k.shape[0])
            meters["val_loss"].update(losses["loss"].item(), bsz)
            meters["val_loss_step"].update(losses["loss_step"].item(), bsz)
            meters["val_loss_kl"].update(losses["loss_kl"].item(), bsz)
            meters["val_loss_recon"].update(losses["loss_recon"].item(), bsz)
            meters["val_alpha_mean"].update(losses["alpha_mean"].item(), bsz)

        metrics = {k: v.avg for k, v in meters.items()}

        if traj_loader is not None:
            rollout_rel_meter = AverageMeter()
            rollout_mse_meter = AverageMeter()
            for batch in traj_loader:
                u0, f, u_ref = _unpack_traj_batch(batch)
                u0 = u0.to(self.device)
                f = f.to(self.device)
                u_ref = u_ref.to(self.device)

                n_steps = int(u_ref.shape[1] - 1)
                u_pred = rollout_vae_mean(self.model, u0=u0, f=f, n_steps=n_steps, dt=self.dt)
                rel = relative_l2_error_2d(u_pred, u_ref, area=self.area)
                rollout_rel_meter.update(rel.mean(dim=-1).mean().item(), int(u0.shape[0]))
                rollout_mse_meter.update(F.mse_loss(u_pred, u_ref).item(), int(u0.shape[0]))
            metrics["val_rollout_rel_l2"] = rollout_rel_meter.avg
            metrics["val_rollout_mse"] = rollout_mse_meter.avg

        return metrics

    def _save_checkpoint(
        self,
        name: str,
        epoch: int,
        metrics: Dict[str, float],
        model_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        optimizer_state_dict: Optional[Dict[str, object]] = None,
    ) -> None:
        if self.output_dir is None:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_state_dict if model_state_dict is not None else self.model.state_dict(),
                "optimizer_state_dict": (
                    optimizer_state_dict if optimizer_state_dict is not None else self.optimizer.state_dict()
                ),
                "metrics": metrics,
                "dt": self.dt,
                "h_x": self.h_x,
                "h_y": self.h_y,
                "beta_kl": self.beta_kl,
                "lambda_rec": self.lambda_rec,
                "spectral_var_floor": self.spectral_var_floor,
                "alpha_min": self.alpha_min,
                "alpha_max": self.alpha_max,
                "lr": self.optimizer.param_groups[0]["lr"],
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
        best_model_state: Optional[Dict[str, torch.Tensor]] = None
        best_optimizer_state: Optional[Dict[str, object]] = None
        checkpoint_interval = int(checkpoint_interval)

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_step_loader, epoch=epoch)
            history["train"].append({"epoch": epoch, **train_metrics})

            latest_metrics: Dict[str, float] = train_metrics
            if epoch % eval_interval == 0:
                val_metrics = self.validate(val_step_loader, traj_loader=val_traj_loader)
                history["val"].append({"epoch": epoch, **val_metrics})
                latest_metrics = val_metrics
                monitor = val_metrics.get("val_rollout_rel_l2", val_metrics["val_loss_step"])
                if monitor < best_metric:
                    best_metric = monitor
                    best_epoch = epoch
                    best_metrics = val_metrics
                    best_model_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    best_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
                    self._save_checkpoint("best_model.pt", epoch, val_metrics)

                print(
                    f"[Epoch {epoch:03d}] "
                    f"train_total={train_metrics['loss']:.6f} "
                    f"train_step={train_metrics['loss_step']:.6f} "
                    f"train_kl={train_metrics['loss_kl']:.6f} "
                    f"train_recon={train_metrics['loss_recon']:.6f} "
                    f"train_alpha={train_metrics['alpha_mean']:.6f} "
                    f"val_total={val_metrics['val_loss']:.6f} "
                    f"val_step={val_metrics['val_loss_step']:.6f} "
                    f"val_kl={val_metrics['val_loss_kl']:.6f} "
                    f"val_recon={val_metrics['val_loss_recon']:.6f} "
                    f"val_alpha={val_metrics['val_alpha_mean']:.6f} "
                    f"val_rollout={val_metrics.get('val_rollout_rel_l2', float('nan')):.6f}"
                )
            else:
                print(
                    f"[Epoch {epoch:03d}] "
                    f"train_total={train_metrics['loss']:.6f} "
                    f"train_step={train_metrics['loss_step']:.6f} "
                    f"train_kl={train_metrics['loss_kl']:.6f} "
                    f"train_recon={train_metrics['loss_recon']:.6f} "
                    f"train_alpha={train_metrics['alpha_mean']:.6f}"
                )

            if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                if best_model_state is not None and best_metrics is not None:
                    self._save_checkpoint(
                        f"best_model_through_epoch_{epoch:04d}.pt",
                        best_epoch,
                        best_metrics,
                        model_state_dict=best_model_state,
                        optimizer_state_dict=best_optimizer_state,
                    )
                else:
                    self._save_checkpoint(f"best_model_through_epoch_{epoch:04d}.pt", epoch, latest_metrics)

            self.scheduler.step()

        final_metrics = history["val"][-1] if history["val"] else history["train"][-1]
        self._save_checkpoint("final_model.pt", epochs, final_metrics)

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, "history.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

        return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train latent VAE model on periodic 2D Navier-Stokes data")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to cached periodic dataset (.pt)")
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--enc-blocks", type=int, default=4)
    parser.add_argument("--dec-blocks", type=int, default=4)
    parser.add_argument("--fno-width", type=int, default=None)
    parser.add_argument("--fno-layers", type=int, default=6)
    parser.add_argument("--fno-modes-x", type=int, default=16)
    parser.add_argument("--fno-modes-y", type=int, default=16)
    parser.add_argument("--disable-fno-grid", action="store_true")
    parser.add_argument("--use-dt-channel", action="store_true")
    parser.add_argument("--disable-forcing-channel", action="store_true")
    parser.add_argument("--disable-u-grad-feature", action="store_true")
    parser.add_argument("--amp-head-hidden", type=int, default=32)

    parser.add_argument("--beta-kl", type=float, default=1e-4)
    parser.add_argument("--lambda-rec", type=float, default=1.0)
    parser.add_argument("--noise-corr-length", type=float, default=1.0)
    parser.add_argument("--noise-decay-s", type=float, default=2.0)
    parser.add_argument(
        "--spectral-var-floor",
        type=float,
        default=1e-2,
        help="Jitter added to the spectral transition variance used by the training sampler.",
    )
    parser.add_argument("--alpha-min", type=float, default=1e-4)
    parser.add_argument("--alpha-max", type=float, default=0.5)
    parser.add_argument("--alpha-init", type=float, default=0.075)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=25,
        help="Save checkpoint_epoch_XXXX.pt every N epochs. Use <=0 to disable periodic snapshots.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-step-size", type=int, default=100, help="Halve/decay LR every N epochs.")
    parser.add_argument("--lr-gamma", type=float, default=0.5, help="Multiplicative LR decay for StepLR.")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-epoch-pbar", action="store_true", help="Disable per-epoch batch progress bar.")
    parser.add_argument("--output-dir", type=str, default="grad_flow_l2/ns2d_per/outputs_vae")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _build_model(n_x: int, n_y: int, dt: float, args: argparse.Namespace) -> PeriodicLatentVAE2D:
    boundary_condition = "periodic"
    use_forcing_channel = not args.disable_forcing_channel
    fno_width = args.hidden_channels if args.fno_width is None else args.fno_width

    encoder = VariationalStateEncoder2D(
        n_x=n_x,
        n_y=n_y,
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
        n_blocks=args.enc_blocks,
        use_grad_features=not args.disable_u_grad_feature,
        boundary_condition=boundary_condition,
    )
    decoder = StateDecoder2D(
        n_x=n_x,
        n_y=n_y,
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
        n_blocks=args.dec_blocks,
        boundary_condition=boundary_condition,
    )
    transition = FNOLatentTransition2D(
        n_x=n_x,
        n_y=n_y,
        latent_channels=args.latent_channels,
        width=fno_width,
        n_layers=args.fno_layers,
        modes_x=args.fno_modes_x,
        modes_y=args.fno_modes_y,
        use_forcing_channel=use_forcing_channel,
        use_dt_channel=args.use_dt_channel,
        use_grid_features=not args.disable_fno_grid,
        default_dt=dt,
        boundary_condition=boundary_condition,
    )
    alpha_min = float(getattr(args, "alpha_min", 1e-4))
    alpha_max = float(getattr(args, "alpha_max", 0.5))
    alpha_init = float(getattr(args, "alpha_init", 0.075))
    if not alpha_min < alpha_init < alpha_max:
        raise ValueError("alpha_init must satisfy alpha_min < alpha_init < alpha_max")
    alpha_init_unit = (alpha_init - alpha_min) / (alpha_max - alpha_min)
    alpha_init_logit = float(np.log(alpha_init_unit / (1.0 - alpha_init_unit)))
    amplitude_head = TransitionAmplitudeHead2D(
        n_x=n_x,
        n_y=n_y,
        latent_channels=args.latent_channels,
        hidden_channels=args.amp_head_hidden,
        use_forcing_channel=use_forcing_channel,
        boundary_condition=boundary_condition,
        alpha_init_logit=alpha_init_logit,
    )
    return PeriodicLatentVAE2D(
        encoder=encoder,
        decoder=decoder,
        transition=transition,
        amplitude_head=amplitude_head,
        noise_corr_length=args.noise_corr_length,
        noise_decay_s=args.noise_decay_s,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )


def _assert_dry_run_shapes(model: PeriodicLatentVAE2D, loader: DataLoader, device: str, dt: float) -> None:
    model.eval()
    u_k, _, f = _unpack_step_batch(next(iter(loader)))
    u_k = u_k.to(device)
    f = f.to(device)
    with torch.no_grad():
        mu_q, logvar_q = model.encode_stats(u_k)
        z_k = model.sample_posterior(mu_q, logvar_q)
        u_recon = model.decode(z_k)
        z_next, mu_p, alpha = model.sample_prior(z_k, f, dt=dt)
        u_next = model.rollout_step(u_k, f, dt=dt)

    bsz = int(u_k.shape[0])
    expected_z = (bsz, model.latent_channels, model.n_x, model.n_y)
    expected_u = (bsz, model.n_x, model.n_y)
    assert tuple(mu_q.shape) == expected_z
    assert tuple(logvar_q.shape) == expected_z
    assert tuple(z_k.shape) == expected_z
    assert tuple(mu_p.shape) == expected_z
    assert tuple(z_next.shape) == expected_z
    assert tuple(u_recon.shape) == expected_u
    assert tuple(u_next.shape) == expected_u
    assert tuple(alpha.shape) == (bsz,)
    assert torch.all(alpha > 0.0)
    print(
        "Dry run shapes OK: "
        f"posterior={tuple(mu_q.shape)}, recon={tuple(u_recon.shape)}, "
        f"prior_mean={tuple(mu_p.shape)}, alpha={tuple(alpha.shape)}"
    )


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed, seed_cuda=not args.cpu)
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
    splits = load_dataset_splits(args.dataset_path, map_location="cpu")
    train_split = splits["train"]
    val_split = splits["val"]
    test_split = splits["test"]

    total_train = int(train_split["u0"].shape[0])
    total_val = int(val_split["u0"].shape[0])
    total_test = int(test_split["u0"].shape[0])
    if (total_train, total_val, total_test) != (args.n_train, args.n_val, args.n_test):
        raise ValueError(
            "Dataset split sizes do not match CLI arguments: "
            f"dataset=({total_train},{total_val},{total_test}) vs "
            f"args=({args.n_train},{args.n_val},{args.n_test})."
        )

    meta = splits.get("meta", {})
    n_x = int(train_split["u0"].shape[1])
    n_y = int(train_split["u0"].shape[2])
    n_steps = int(train_split["u_traj"].shape[1] - 1)
    t_final = float(meta.get("t_final", 1.0))
    t_start = float(meta.get("stored_t_start", meta.get("warmup_time", 0.0)))
    stored_horizon = float(meta.get("stored_time_horizon", t_final - t_start))
    h_x = 1.0 / float(n_x)
    h_y = 1.0 / float(n_y)
    dt = float(meta.get("record_dt", stored_horizon / float(n_steps)))

    print(f"Device: {device}")
    print(f"Loaded dataset: {args.dataset_path}")
    print(
        f"Grid from data: n_x={n_x}, n_y={n_y}, n_steps={n_steps}, "
        f"stored_time=[{t_start:.6f},{t_start + dt * n_steps:.6f}], dt={dt:.6f}"
    )

    train_step_ds = build_navier_stokes2d_periodic_step_dataset(train_split)
    val_step_ds = build_navier_stokes2d_periodic_step_dataset(val_split)
    test_step_ds = build_navier_stokes2d_periodic_step_dataset(test_split)
    val_traj_ds = build_navier_stokes2d_periodic_trajectory_dataset_from_split(val_split)
    test_traj_ds = build_navier_stokes2d_periodic_trajectory_dataset_from_split(test_split)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    train_step_loader = DataLoader(
        train_step_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    val_step_loader = DataLoader(
        val_step_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
    )
    test_step_loader = DataLoader(
        test_step_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
    )
    val_traj_loader = DataLoader(
        val_traj_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
    )
    test_traj_loader = DataLoader(
        test_traj_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
    )

    model = _build_model(n_x=n_x, n_y=n_y, dt=dt, args=args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    trainer = PeriodicLatentVAETrainer2D(
        model=model,
        dt=dt,
        h_x=h_x,
        h_y=h_y,
        beta_kl=args.beta_kl,
        lambda_rec=args.lambda_rec,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        spectral_var_floor=args.spectral_var_floor,
        max_epochs=args.epochs,
        device=device,
        output_dir=run_dir,
        show_epoch_pbar=not args.no_epoch_pbar,
    )

    if args.dry_run:
        _assert_dry_run_shapes(model, val_step_loader, device=device, dt=dt)
        val_metrics = trainer.validate(val_step_loader, traj_loader=val_traj_loader)
        print("Dry run val metrics:", val_metrics)
        test_metrics = trainer.validate(test_step_loader, traj_loader=test_traj_loader)
        print("Dry run test metrics:", test_metrics)
        return

    print(
        f"Training config: epochs={args.epochs}, lr={args.lr}, "
        f"lr_step_size={args.lr_step_size}, lr_gamma={args.lr_gamma}, "
        f"beta_kl={args.beta_kl}, lambda_rec={args.lambda_rec}, "
        f"fno_width={args.hidden_channels if args.fno_width is None else args.fno_width}, "
        f"fno_layers={args.fno_layers}, fno_modes=({args.fno_modes_x},{args.fno_modes_y}), "
        f"noise_corr_length={args.noise_corr_length}, noise_decay_s={args.noise_decay_s}, "
        f"spectral_var_floor={args.spectral_var_floor}, "
        f"epoch_pbar={not args.no_epoch_pbar}, output={run_dir}"
    )
    history = trainer.fit(
        train_step_loader=train_step_loader,
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
    test_metrics = trainer.validate(test_step_loader, traj_loader=test_traj_loader)
    print("Test metrics:", test_metrics)
    print(f"Saved training artifacts to: {run_dir}")


if __name__ == "__main__":
    main(parse_args())
