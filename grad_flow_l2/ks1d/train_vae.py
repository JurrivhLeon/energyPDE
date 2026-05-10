"""
Training entrypoint for the unforced 1D Kuramoto-Sivashinsky latent VAE model.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
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
    from ..heat_data import build_step_dataset, build_trajectory_dataset_from_split, load_dataset_splits
    from ..latent_flow_VAE import (
        FNOLatentTransition1D,
        LatentVAE1D,
        StateDecoder1D,
        TransitionAmplitudeHead1D,
        VariationalStateEncoder1D,
    )
    from ..utils import compute_relative_l2_error
except ImportError:
    from grad_flow_l2.heat_data import build_step_dataset, build_trajectory_dataset_from_split, load_dataset_splits
    from grad_flow_l2.latent_flow_VAE import (
        FNOLatentTransition1D,
        LatentVAE1D,
        StateDecoder1D,
        TransitionAmplitudeHead1D,
        VariationalStateEncoder1D,
    )
    from grad_flow_l2.utils import compute_relative_l2_error


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
    p_logvar = torch.clamp(p_logvar_scalar, min=-12.0, max=8.0).view(-1, 1, 1)
    q_var = torch.exp(q_logvar)
    p_var = torch.exp(p_logvar)
    kl = 0.5 * (p_logvar - q_logvar + (q_var + (q_mu - p_mu).square()) / p_var - 1.0)
    return kl.mean()


def rollout_vae_mean(
    model: LatentVAE1D,
    u0: torch.Tensor,
    f: torch.Tensor,
    n_steps: int,
    dt: float,
    delta_clip: Optional[float] = None,
    state_clip: Optional[float] = None,
) -> torch.Tensor:
    squeeze = False
    if u0.dim() == 1:
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
        finite = torch.isfinite(u_next).all(dim=1)
        u_next = torch.where(finite[:, None], u_next, u)
        if state_clip is not None and float(state_clip) > 0.0:
            u_next = torch.clamp(u_next, min=-float(state_clip), max=float(state_clip))
        u = u_next
        states.append(u)

    traj = torch.stack(states, dim=1)
    if squeeze:
        return traj.squeeze(0)
    return traj


class KSLatentVAETrainer1D:
    def __init__(
        self,
        model: LatentVAE1D,
        dt: float,
        h: float,
        beta_kl: float = 1e-4,
        lambda_rec: float = 1.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
        spectral_var_floor: float = 1e-2,
        lr_step_size: int = 100,
        lr_gamma: float = 0.5,
        max_epochs: int = 200,
        device: str = "cpu",
        output_dir: Optional[str] = None,
        show_epoch_pbar: bool = True,
    ):
        self.model = model.to(device)
        self.dt = float(dt)
        self.h = float(h)
        self.beta_kl = float(beta_kl)
        self.lambda_rec = float(lambda_rec)
        self.grad_clip = float(grad_clip)
        self.spectral_var_floor = float(spectral_var_floor)
        self.device = device
        self.output_dir = output_dir
        self.show_epoch_pbar = bool(show_epoch_pbar)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=max(1, int(lr_step_size)),
            gamma=float(lr_gamma),
        )

    @staticmethod
    def _bounded_alpha(raw_alpha: torch.Tensor) -> torch.Tensor:
        alpha_min = 1e-4
        alpha_max = 0.50
        return alpha_min + (alpha_max - alpha_min) * torch.sigmoid(raw_alpha)

    def _filtered_noise_for_training(self, shape, device, dtype) -> torch.Tensor:
        xi = torch.randn(shape, device=device, dtype=dtype)
        xi_hat = torch.fft.rfft(xi, dim=-1, norm="ortho")
        c_sqrt = self.model._spectral_filter(device=device, dtype=dtype)
        filt = torch.sqrt(c_sqrt.square() + self.spectral_var_floor)
        return torch.fft.irfft(xi_hat * filt.view(1, 1, -1), n=self.model.n_x, dim=-1, norm="ortho")

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
        raw_alpha = torch.exp(0.5 * prior_logvar_scalar)
        alpha = self._bounded_alpha(raw_alpha)
        if sample:
            noise = self._filtered_noise_for_training(mu_p.shape, device=mu_p.device, dtype=mu_p.dtype)
            z_next = mu_p + alpha.view(-1, 1, 1) * noise
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
                rel = compute_relative_l2_error(u_pred, u_ref, h=self.h)
                rollout_rel_meter.update(rel.mean(dim=-1).mean().item(), int(u0.shape[0]))
                rollout_mse_meter.update(F.mse_loss(u_pred, u_ref).item(), int(u0.shape[0]))
            metrics["val_rollout_rel_l2"] = rollout_rel_meter.avg
            metrics["val_rollout_mse"] = rollout_mse_meter.avg

        return metrics

    def _save_checkpoint(self, name: str, epoch: int, metrics: Dict[str, float]) -> None:
        if self.output_dir is None:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "dt": self.dt,
                "h": self.h,
                "beta_kl": self.beta_kl,
                "lambda_rec": self.lambda_rec,
                "spectral_var_floor": self.spectral_var_floor,
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
        best_checkpoint_interval: int = 25,
    ) -> Dict[str, list]:
        history = {"train": [], "val": []}
        best_metric = float("inf")
        checkpoint_interval = int(checkpoint_interval)
        best_checkpoint_interval = int(best_checkpoint_interval)

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
                self._save_checkpoint(f"checkpoint_epoch_{epoch:04d}.pt", epoch, latest_metrics)

            if (
                best_checkpoint_interval > 0
                and epoch % best_checkpoint_interval == 0
                and self.output_dir is not None
            ):
                best_path = os.path.join(self.output_dir, "best_model.pt")
                if os.path.exists(best_path):
                    shutil.copy2(best_path, os.path.join(self.output_dir, f"best_model_epoch_{epoch:04d}.pt"))

            self.scheduler.step()

        final_metrics = history["val"][-1] if history["val"] else history["train"][-1]
        self._save_checkpoint("final_model.pt", epochs, final_metrics)

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, "history.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

        return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train latent VAE model on unforced 1D Kuramoto-Sivashinsky data")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="grad_flow_l2/ks1d/datasets/ks_periodic_L32pi_snx1024_nx256_dt1_solverdt0p01.pt",
        help="Path to cached KS dataset (.pt)",
    )
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--enc-blocks", type=int, default=4)
    parser.add_argument("--dec-blocks", type=int, default=4)
    parser.add_argument("--fno-width", type=int, default=None)
    parser.add_argument("--fno-layers", type=int, default=6)
    parser.add_argument("--fno-modes", type=int, default=16)
    parser.add_argument("--disable-fno-grid", action="store_true")
    parser.add_argument("--use-dt-channel", action="store_true")
    parser.add_argument(
        "--disable-forcing-channel",
        action="store_true",
        default=True,
        help="Compatibility flag; KS VAE training always disables the forcing channel.",
    )
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

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=25,
        help="Save checkpoint_epoch_XXXX.pt every N epochs. Use <=0 to disable periodic snapshots.",
    )
    parser.add_argument(
        "--best-checkpoint-interval",
        type=int,
        default=25,
        help="Save best_model_epoch_XXXX.pt snapshots of the best-so-far model every N epochs. Use <=0 to disable.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-step-size", type=int, default=50, help="Decay LR every N epochs.")
    parser.add_argument("--lr-gamma", type=float, default=0.5, help="Multiplicative LR decay for StepLR.")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-epoch-pbar", action="store_true", help="Disable per-epoch batch progress bar.")
    parser.add_argument("--output-dir", type=str, default="grad_flow_l2/ks1d/outputs_vae")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _build_model(n_x: int, dt: float, boundary_condition: str, args: argparse.Namespace) -> LatentVAE1D:
    use_forcing_channel = False
    fno_width = args.hidden_channels if args.fno_width is None else args.fno_width

    encoder = VariationalStateEncoder1D(
        n_x=n_x,
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
        n_blocks=args.enc_blocks,
        use_grad_features=not args.disable_u_grad_feature,
        boundary_condition=boundary_condition,
    )
    decoder = StateDecoder1D(
        n_x=n_x,
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
        n_blocks=args.dec_blocks,
        boundary_condition=boundary_condition,
    )
    transition = FNOLatentTransition1D(
        n_x=n_x,
        latent_channels=args.latent_channels,
        width=fno_width,
        n_layers=args.fno_layers,
        modes=args.fno_modes,
        use_forcing_channel=use_forcing_channel,
        use_dt_channel=args.use_dt_channel,
        use_grid_features=not args.disable_fno_grid,
        default_dt=dt,
    )
    amplitude_head = TransitionAmplitudeHead1D(
        n_x=n_x,
        latent_channels=args.latent_channels,
        hidden_channels=args.amp_head_hidden,
        use_forcing_channel=use_forcing_channel,
        boundary_condition=boundary_condition,
    )
    return LatentVAE1D(
        encoder=encoder,
        decoder=decoder,
        transition=transition,
        amplitude_head=amplitude_head,
        noise_corr_length=args.noise_corr_length,
        noise_decay_s=args.noise_decay_s,
    )


def _assert_dry_run_shapes(model: LatentVAE1D, loader: DataLoader, device: str, dt: float) -> None:
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
    expected_z = (bsz, model.latent_channels, model.n_x)
    expected_u = (bsz, model.n_x)
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
    n_steps = int(train_split["u_traj"].shape[1] - 1)
    t_final = float(meta.get("t_final", 1.0))
    boundary_condition = str(meta.get("boundary_condition", "periodic" if meta.get("periodic", False) else "dirichlet"))
    domain_length = float(meta.get("domain_length", 1.0))
    h_default = domain_length / float(n_x) if boundary_condition == "periodic" else 1.0 / float(n_x + 1)
    h = float(meta.get("h", h_default))
    dt = t_final / float(n_steps)

    print(f"Device: {device}")
    print(f"Loaded dataset: {args.dataset_path}")
    print(
        f"Grid from data: n_x={n_x}, n_steps={n_steps}, t_final={t_final:.6f}, "
        f"h={h:.6f}, dt={dt:.6f}, boundary_condition={boundary_condition}"
    )

    train_step_ds = build_step_dataset(train_split)
    val_step_ds = build_step_dataset(val_split)
    test_step_ds = build_step_dataset(test_split)
    val_traj_ds = build_trajectory_dataset_from_split(val_split)
    test_traj_ds = build_trajectory_dataset_from_split(test_split)

    train_step_loader = DataLoader(train_step_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_step_loader = DataLoader(val_step_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_step_loader = DataLoader(test_step_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_traj_loader = DataLoader(val_traj_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_traj_loader = DataLoader(test_traj_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = _build_model(n_x=n_x, dt=dt, boundary_condition=boundary_condition, args=args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    trainer = KSLatentVAETrainer1D(
        model=model,
        dt=dt,
        h=h,
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
            f"fno_layers={args.fno_layers}, fno_modes={args.fno_modes}, "
            f"use_forcing_channel=False, "
        f"noise_corr_length={args.noise_corr_length}, noise_decay_s={args.noise_decay_s}, "
        f"spectral_var_floor={args.spectral_var_floor}, "
        f"checkpoint_interval={args.checkpoint_interval}, "
        f"best_checkpoint_interval={args.best_checkpoint_interval}, "
        f"epoch_pbar={not args.no_epoch_pbar}, output={run_dir}"
    )
    history = trainer.fit(
        train_step_loader=train_step_loader,
        val_step_loader=val_step_loader,
        val_traj_loader=val_traj_loader,
        epochs=args.epochs,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        best_checkpoint_interval=args.best_checkpoint_interval,
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
