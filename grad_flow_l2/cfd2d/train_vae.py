"""Training entrypoint for periodic 2D compressible NS latent VAE models.

The VAE trainer (LatentVAETrainer2D) is implemented inline here to avoid
coupling to ns2d_per. It shares utilities with latent_markov_trainer_mc.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    from ..cfd2d.cfd_data import STATE_CHANNELS, build_cfd2d_step_dataset, build_cfd2d_trajectory_dataset_from_split
    from ..heat_data import load_dataset_splits
    from ..latent_flow_VAE_mc import (
        FNOLatentTransition2D,
        PeriodicLatentVAE2D,
        TransitionAmplitudeHead2D,
        VariationalStateEncoder2D,
    )
    from ..latent_markov_mc import StateDecoder2D
    from ..latent_markov_trainer_mc import (
        AverageMeter,
        _unpack_step_batch,
        _unpack_traj_batch,
        channel_weighted_mse,
        relative_l2_error_2d,
    )
except ImportError:
    from grad_flow_l2.cfd2d.cfd_data import STATE_CHANNELS, build_cfd2d_step_dataset, build_cfd2d_trajectory_dataset_from_split
    from grad_flow_l2.heat_data import load_dataset_splits
    from grad_flow_l2.latent_flow_VAE_mc import (
        FNOLatentTransition2D,
        PeriodicLatentVAE2D,
        TransitionAmplitudeHead2D,
        VariationalStateEncoder2D,
    )
    from grad_flow_l2.latent_markov_mc import StateDecoder2D
    from grad_flow_l2.latent_markov_trainer_mc import (
        AverageMeter,
        _unpack_step_batch,
        _unpack_traj_batch,
        channel_weighted_mse,
        relative_l2_error_2d,
    )


# ─── KL helper ────────────────────────────────────────────────────────────────

def _kl_diag_gaussians(
    q_mu: torch.Tensor, q_logvar: torch.Tensor,
    p_mu: torch.Tensor, p_logvar_scalar: torch.Tensor,
) -> torch.Tensor:
    """Mean KL(q||p) with diagonal q and scalar-variance p per sample."""
    q_logvar = q_logvar.clamp(-12.0, 8.0)
    p_logvar = p_logvar_scalar.clamp(-12.0, 8.0).view(-1, 1, 1, 1)
    q_var, p_var = q_logvar.exp(), p_logvar.exp()
    return 0.5 * (p_logvar - q_logvar + (q_var + (q_mu - p_mu).square()) / p_var - 1.0).mean()


# ─── VAE rollout (mean / deterministic) ───────────────────────────────────────

@torch.no_grad()
def _rollout_vae_mean(
    model: PeriodicLatentVAE2D,
    u0: torch.Tensor,
    f: torch.Tensor,
    n_steps: int,
    dt: float,
    delta_clip: Optional[float] = None,
) -> torch.Tensor:
    """Deterministic-mean rollout using model.rollout_step at each step."""
    states = [u0]
    u = u0
    for _ in range(n_steps):
        u_next = model.rollout_step(u, f, dt=dt)
        delta = u_next - u
        if delta_clip is not None and delta_clip > 0.0:
            delta = delta.clamp(-delta_clip, delta_clip)
        u_next = u + delta
        finite = torch.isfinite(u_next).flatten(1).all(dim=1)
        u_next = torch.where(finite.reshape(-1, *([1] * (u_next.dim() - 1))), u_next, u)
        u = u_next
        states.append(u)
    return torch.stack(states, dim=1)   # (batch, n_steps+1, C, H, W)


# ─── Trainer ──────────────────────────────────────────────────────────────────

class LatentVAETrainer2D:
    """
    ELBO trainer for PeriodicLatentVAE2D on cfd2d (4-channel primitive state).

    Loss = loss_step + beta_kl * loss_kl + lambda_rec * loss_recon
    """

    def __init__(
        self,
        model: PeriodicLatentVAE2D,
        dt: float,
        h_x: float,
        h_y: float,
        beta_kl: float = 1e-2,
        lambda_rec: float = 1.0,
        channel_weights=None,            # (C,) tensor or None → uniform
        spectral_var_floor: float = 1e-2,
        rollout_delta_clip: Optional[float] = 1.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
        lr_step_size: int = 100,
        lr_gamma: float = 0.5,
        max_epochs: Optional[int] = None,
        device: str = "cpu",
        output_dir: Optional[str] = None,
        show_epoch_pbar: bool = True,
    ):
        self.model = model.to(device)
        self.dt = float(dt)
        self.h_x = float(h_x)
        self.h_y = float(h_y)
        self.area = h_x * h_y
        self.beta_kl = float(beta_kl)
        self.lambda_rec = float(lambda_rec)
        self.spectral_var_floor = float(spectral_var_floor)
        self.rollout_delta_clip = None if rollout_delta_clip is None else float(rollout_delta_clip)
        self.grad_clip = float(grad_clip)
        self.device = device
        self.output_dir = output_dir
        self.show_epoch_pbar = bool(show_epoch_pbar)
        self.channel_weights = (
            None if channel_weights is None
            else torch.as_tensor(channel_weights, dtype=torch.float32)
        )
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=max(1, int(lr_step_size)), gamma=float(lr_gamma)
        )

    @property
    def delta_clip(self) -> Optional[float]:
        return self.rollout_delta_clip

    @delta_clip.setter
    def delta_clip(self, v: Optional[float]) -> None:
        self.rollout_delta_clip = None if v is None else float(v)

    def _weights(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        if self.channel_weights is None:
            return None
        return self.channel_weights.to(device=tensor.device, dtype=tensor.dtype)

    def _bounded_alpha(self, raw: torch.Tensor) -> torch.Tensor:
        return 1e-4 + (0.5 - 1e-4) * torch.sigmoid(raw)

    def _filtered_noise(self, shape, device, dtype) -> torch.Tensor:
        xi = torch.randn(shape, device=device, dtype=dtype)
        xi_hat = torch.fft.rfft2(xi, dim=(-2, -1), norm="ortho")
        c_sqrt = self.model._spectral_filter(device=device, dtype=dtype)
        filt = (c_sqrt.square() + self.spectral_var_floor).sqrt()
        return torch.fft.irfft2(
            xi_hat * filt.view(1, 1, self.model.n_x, self.model.n_y // 2 + 1),
            s=(self.model.n_x, self.model.n_y), dim=(-2, -1), norm="ortho",
        )

    def _compute_losses(
        self, u_k: torch.Tensor, u_k1: torch.Tensor, f: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        w = self._weights(u_k)

        # Posterior encode
        mu_q, logvar_q = self.model.encode_stats(u_k)
        z_k = self.model.sample_posterior(mu_q, logvar_q)

        # Prior / transition
        mu_p, prior_logvar_scalar = self.model.prior_stats(z_k, f, dt=self.dt)
        alpha = self._bounded_alpha(torch.exp(0.5 * prior_logvar_scalar))
        noise = self._filtered_noise(mu_p.shape, device=mu_p.device, dtype=mu_p.dtype)
        z_next = mu_p + alpha.view(-1, 1, 1, 1) * noise

        # Step prediction loss
        u_pred = self.model.decode(z_next)
        loss_step = channel_weighted_mse(u_pred, u_k1, w)

        # Reconstruction loss
        u_recon = self.model.decode(z_k)
        loss_recon = channel_weighted_mse(u_recon, u_k, w)

        # KL loss: posterior vs prior at u_{k+1}
        mu_q_next, logvar_q_next = self.model.encode_stats(u_k1)
        loss_kl = _kl_diag_gaussians(
            mu_q_next, logvar_q_next, mu_p,
            torch.log(alpha.square() + 1e-12),
        )

        loss = loss_step + self.beta_kl * loss_kl + self.lambda_rec * loss_recon
        return {"loss": loss, "loss_step": loss_step, "loss_kl": loss_kl,
                "loss_recon": loss_recon, "alpha_mean": alpha.mean().detach()}

    def train_epoch(self, loader: DataLoader, epoch: Optional[int] = None) -> Dict[str, float]:
        self.model.train()
        keys = ("loss", "loss_step", "loss_kl", "loss_recon", "alpha_mean")
        meters = {k: AverageMeter() for k in keys}
        show_pbar = self.show_epoch_pbar and tqdm is not None
        iterable = loader
        pbar = None
        if show_pbar:
            desc = f"Epoch {epoch:03d}" if epoch is not None else "Epoch"
            pbar = tqdm(loader, total=len(loader), desc=desc, leave=False, dynamic_ncols=True)
            iterable = pbar
        for batch_idx, batch in enumerate(iterable, start=1):
            u_k, u_k1, f = _unpack_step_batch(batch)
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
            if pbar is not None and (batch_idx == 1 or batch_idx % 10 == 0):
                pbar.set_postfix(total=f"{meters['loss'].avg:.4f}",
                                 step=f"{meters['loss_step'].avg:.4f}",
                                 kl=f"{meters['loss_kl'].avg:.4f}")
        if pbar is not None:
            pbar.close()
        return {k: v.avg for k, v in meters.items()}

    @torch.no_grad()
    def validate(
        self, step_loader: DataLoader, traj_loader: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        self.model.eval()
        keys = ("val_loss", "val_loss_step", "val_loss_kl", "val_loss_recon", "val_alpha_mean")
        meters = {k: AverageMeter() for k in keys}
        for batch in step_loader:
            u_k, u_k1, f = _unpack_step_batch(batch)
            u_k, u_k1, f = u_k.to(self.device), u_k1.to(self.device), f.to(self.device)
            losses = self._compute_losses(u_k, u_k1, f)
            bsz = int(u_k.shape[0])
            meters["val_loss"].update(losses["loss"].item(), bsz)
            meters["val_loss_step"].update(losses["loss_step"].item(), bsz)
            meters["val_loss_kl"].update(losses["loss_kl"].item(), bsz)
            meters["val_loss_recon"].update(losses["loss_recon"].item(), bsz)
            meters["val_alpha_mean"].update(losses["alpha_mean"].item(), bsz)
        metrics = {k: v.avg for k, v in meters.items()}

        if traj_loader is not None:
            rollout_meter = AverageMeter()
            ch_meters: Optional[List[AverageMeter]] = None
            for batch in traj_loader:
                u0, f, u_ref = _unpack_traj_batch(batch)
                u0, f, u_ref = u0.to(self.device), f.to(self.device), u_ref.to(self.device)
                u_pred = _rollout_vae_mean(self.model, u0, f,
                                           n_steps=int(u_ref.shape[1] - 1),
                                           dt=self.dt,
                                           delta_clip=self.rollout_delta_clip)
                rel = relative_l2_error_2d(u_pred, u_ref, area=self.area)  # (B, T+1, C)
                bsz = int(u0.shape[0])
                if rel.dim() == 3:
                    if ch_meters is None:
                        ch_meters = [AverageMeter() for _ in range(rel.shape[2])]
                    for c in range(rel.shape[2]):
                        ch_meters[c].update(rel[:, :, c].mean().item(), bsz)
                    rollout_meter.update(rel.mean(dim=2).mean().item(), bsz)
                else:
                    rollout_meter.update(rel.mean(dim=1).mean().item(), bsz)
            metrics["val_rollout_rel_l2"] = rollout_meter.avg
            if ch_meters is not None:
                for c, m in enumerate(ch_meters):
                    metrics[f"val_rollout_rel_l2_ch{c}"] = m.avg
        return metrics

    def _save_checkpoint(self, name: str, epoch: int, metrics: Dict[str, float],
                          state_dict=None) -> None:
        if self.output_dir is None:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": state_dict if state_dict is not None else self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "dt": self.dt, "h_x": self.h_x, "h_y": self.h_y,
            "beta_kl": self.beta_kl, "lambda_rec": self.lambda_rec,
            "rollout_delta_clip": self.rollout_delta_clip,
            "channel_weights": (None if self.channel_weights is None
                                 else self.channel_weights.detach().cpu()),
        }, os.path.join(self.output_dir, name))

    def fit(
        self,
        train_step_loader: DataLoader,
        val_step_loader: DataLoader,
        val_traj_loader: Optional[DataLoader] = None,
        epochs: int = 200,
        eval_interval: int = 1,
        checkpoint_interval: int = 25,
    ) -> Dict[str, list]:
        history: Dict[str, list] = {"train": [], "val": []}
        best_metric = float("inf")
        best_epoch = 0
        best_metrics: Optional[Dict[str, float]] = None
        best_state_dict = None
        for epoch in range(1, epochs + 1):
            train_m = self.train_epoch(train_step_loader, epoch=epoch)
            history["train"].append({"epoch": epoch, **train_m})
            if epoch % eval_interval == 0:
                val_m = self.validate(val_step_loader, traj_loader=val_traj_loader)
                history["val"].append({"epoch": epoch, **val_m})
                monitor = val_m.get("val_rollout_rel_l2", val_m["val_loss_step"])
                if monitor < best_metric:
                    best_metric = monitor
                    best_epoch = epoch
                    best_metrics = val_m
                    best_state_dict = {k: v.cpu().clone()
                                       for k, v in self.model.state_dict().items()}
                    self._save_checkpoint("best_model.pt", epoch, val_m)
                print(
                    f"[Epoch {epoch:03d}] "
                    f"train_total={train_m['loss']:.6f} "
                    f"train_step={train_m['loss_step']:.6f} "
                    f"train_kl={train_m['loss_kl']:.6f} "
                    f"train_recon={train_m['loss_recon']:.6f} "
                    f"alpha={train_m['alpha_mean']:.4f} "
                    f"val_total={val_m['val_loss']:.6f} "
                    f"val_step={val_m['val_loss_step']:.6f} "
                    f"val_kl={val_m['val_loss_kl']:.6f} "
                    f"val_recon={val_m['val_loss_recon']:.6f} "
                    f"val_rollout={val_m.get('val_rollout_rel_l2', float('nan')):.6f}"
                )
            else:
                print(
                    f"[Epoch {epoch:03d}] "
                    f"train_total={train_m['loss']:.6f} "
                    f"train_step={train_m['loss_step']:.6f} "
                    f"train_kl={train_m['loss_kl']:.6f} "
                    f"train_recon={train_m['loss_recon']:.6f} "
                    f"alpha={train_m['alpha_mean']:.4f}"
                )
            # Snapshot the best model seen so far at each checkpoint interval
            if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                snap_sd = best_state_dict if best_state_dict is not None else None
                snap_m  = best_metrics   if best_metrics  is not None else train_m
                snap_ep = best_epoch     if best_epoch > 0            else epoch
                self._save_checkpoint(f"best_model_through_epoch_{epoch:04d}.pt",
                                       snap_ep, snap_m, state_dict=snap_sd)
            self.scheduler.step()

        final_m = history["val"][-1] if history["val"] else history["train"][-1]
        self._save_checkpoint("final_model.pt", epochs, final_m)
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, "history.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        return history


# ─── Model builder ────────────────────────────────────────────────────────────

def set_seed(seed: int, seed_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda:
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train latent VAE model on periodic 2D compressible NS data")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--n-train", type=int, default=1200)
    parser.add_argument("--n-val",   type=int, default=300)
    parser.add_argument("--n-test",  type=int, default=0)
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--hidden-channels",  type=int,   default=64)
    parser.add_argument("--latent-channels",  type=int,   default=16)
    parser.add_argument("--enc-blocks",       type=int,   default=4)
    parser.add_argument("--dec-blocks",       type=int,   default=4)
    parser.add_argument("--fno-width",        type=int,   default=None)
    parser.add_argument("--fno-layers",       type=int,   default=6)
    parser.add_argument("--fno-modes-x",      type=int,   default=16)
    parser.add_argument("--fno-modes-y",      type=int,   default=16)
    parser.add_argument("--disable-fno-grid", action="store_true")
    parser.add_argument("--use-dt-channel",          action="store_true")
    parser.add_argument("--disable-forcing-channel", action="store_true")
    parser.add_argument("--disable-u-grad-feature",  action="store_true")
    parser.add_argument("--amp-head-hidden",  type=int,   default=32)

    parser.add_argument("--beta-kl",            type=float, default=1e-2)
    parser.add_argument("--lambda-rec",         type=float, default=1.0)
    parser.add_argument("--noise-corr-length",  type=float, default=1.0)
    parser.add_argument("--noise-decay-s",      type=float, default=2.0)
    parser.add_argument("--spectral-var-floor", type=float, default=1e-2)
    parser.add_argument("--rollout-delta-clip", type=float, default=1.0)

    parser.add_argument("--epochs",              type=int,   default=200)
    parser.add_argument("--eval-interval",       type=int,   default=1)
    parser.add_argument("--checkpoint-interval", type=int,   default=25)
    parser.add_argument("--lr",                  type=float, default=1e-4)
    parser.add_argument("--lr-step-size",        type=int,   default=100)
    parser.add_argument("--lr-gamma",            type=float, default=0.5)
    parser.add_argument("--weight-decay",        type=float, default=1e-5)
    parser.add_argument("--grad-clip",           type=float, default=1.0)

    parser.add_argument("--channel-weights", type=float, nargs=4, default=None,
                        metavar=("W_RHO", "W_VX", "W_VY", "W_P"),
                        help="Per-channel loss weights. Default: 1/Var from training data.")
    parser.add_argument("--seed",           type=int,  default=42)
    parser.add_argument("--cpu",            action="store_true")
    parser.add_argument("--no-epoch-pbar",  action="store_true")
    parser.add_argument("--output-dir",     type=str,  default="grad_flow_l2/cfd2d/outputs_vae")
    parser.add_argument("--dry-run",        action="store_true")
    return parser.parse_args()


def _build_model(n_x: int, n_y: int, dt: float,
                 args: argparse.Namespace) -> PeriodicLatentVAE2D:
    bc = "periodic"
    use_forcing = not args.disable_forcing_channel
    fno_width = args.hidden_channels if args.fno_width is None else args.fno_width
    encoder = VariationalStateEncoder2D(
        n_x=n_x, n_y=n_y,
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
        n_blocks=args.enc_blocks,
        use_grad_features=not args.disable_u_grad_feature,
        boundary_condition=bc,
        state_channels=STATE_CHANNELS,
    )
    decoder = StateDecoder2D(
        n_x=n_x, n_y=n_y,
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
        n_blocks=args.dec_blocks,
        boundary_condition=bc,
        state_channels=STATE_CHANNELS,
    )
    transition = FNOLatentTransition2D(
        n_x=n_x, n_y=n_y,
        latent_channels=args.latent_channels,
        width=fno_width,
        n_layers=args.fno_layers,
        modes_x=args.fno_modes_x,
        modes_y=args.fno_modes_y,
        use_forcing_channel=use_forcing,
        use_dt_channel=args.use_dt_channel,
        use_grid_features=not args.disable_fno_grid,
        default_dt=dt,
        boundary_condition=bc,
    )
    amplitude_head = TransitionAmplitudeHead2D(
        n_x=n_x, n_y=n_y,
        latent_channels=args.latent_channels,
        hidden_channels=args.amp_head_hidden,
        use_forcing_channel=use_forcing,
        boundary_condition=bc,
    )
    return PeriodicLatentVAE2D(
        encoder=encoder, decoder=decoder,
        transition=transition, amplitude_head=amplitude_head,
        noise_corr_length=args.noise_corr_length,
        noise_decay_s=args.noise_decay_s,
    )


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed, seed_cuda=not args.cpu)
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    splits = load_dataset_splits(args.dataset_path, map_location="cpu")
    train_split = splits["train"]
    val_split   = splits["val"]
    test_split  = splits["test"]
    sizes = (int(train_split["u0"].shape[0]), int(val_split["u0"].shape[0]), int(test_split["u0"].shape[0]))
    if sizes != (args.n_train, args.n_val, args.n_test):
        raise ValueError(f"Dataset split sizes {sizes} do not match args {(args.n_train, args.n_val, args.n_test)}")

    meta    = splits.get("meta", {})
    n_x     = int(train_split["u0"].shape[-2])
    n_y     = int(train_split["u0"].shape[-1])
    n_steps = int(train_split["u_traj"].shape[1] - 1)
    t_final = float(meta.get("t_final", float(n_steps)))
    dt = t_final / float(n_steps)

    if args.channel_weights is not None:
        channel_weights = torch.tensor(args.channel_weights, dtype=torch.float32)
    else:
        u_flat = train_split["u_traj"].permute(2, 0, 1, 3, 4).reshape(STATE_CHANNELS, -1)
        channel_weights = 1.0 / u_flat.var(dim=1).clamp(min=1e-8)
    channel_weights = channel_weights / channel_weights.mean()

    print(f"Device: {device}")
    print(f"Loaded dataset: {args.dataset_path}")
    print(f"Grid: ({n_x},{n_y}), state_channels={STATE_CHANNELS}, steps={n_steps}, dt={dt:.6f}")
    print(f"Channel weights (rho,vx,vy,p): {channel_weights.tolist()}")

    train_step_loader = DataLoader(build_cfd2d_step_dataset(train_split), batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    val_step_loader   = DataLoader(build_cfd2d_step_dataset(val_split),   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_step_loader  = DataLoader(build_cfd2d_step_dataset(test_split),  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_traj_loader   = DataLoader(build_cfd2d_trajectory_dataset_from_split(val_split),  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_traj_loader  = DataLoader(build_cfd2d_trajectory_dataset_from_split(test_split), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = _build_model(n_x=n_x, n_y=n_y, dt=dt, args=args).to(device)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    args_dict = vars(args)
    args_dict["channel_weights_used"] = channel_weights.tolist()
    with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=2)

    trainer = LatentVAETrainer2D(
        model=model, dt=dt,
        h_x=1.0 / float(n_x), h_y=1.0 / float(n_y),
        beta_kl=args.beta_kl,
        lambda_rec=args.lambda_rec,
        channel_weights=channel_weights,
        spectral_var_floor=args.spectral_var_floor,
        rollout_delta_clip=args.rollout_delta_clip if args.rollout_delta_clip > 0 else None,
        lr=args.lr, lr_step_size=args.lr_step_size, lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        max_epochs=args.epochs,
        device=device, output_dir=run_dir,
        show_epoch_pbar=not args.no_epoch_pbar,
    )

    if args.dry_run:
        print("Dry run val metrics:",  trainer.validate(val_step_loader,  traj_loader=val_traj_loader))
        if args.n_test > 0:
            print("Dry run test metrics:", trainer.validate(test_step_loader, traj_loader=test_traj_loader))
        else:
            print("Skipping dry-run test metrics: test split is empty.")
        return

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
    if args.n_test > 0:
        print("Test metrics:", trainer.validate(test_step_loader, traj_loader=test_traj_loader))
    else:
        print("Skipping test metrics: test split is empty.")
    print(f"Saved training artifacts to: {run_dir}")


if __name__ == "__main__":
    main(parse_args())
