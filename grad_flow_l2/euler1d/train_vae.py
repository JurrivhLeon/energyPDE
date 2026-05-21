"""Train probabilistic latent VAE Markov models on 1D Euler data."""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    from ..heat_data import load_dataset_splits
    from ..latent_flow_VAE_mc import (
        FNOLatentTransition1D,
        LatentVAE1D,
        StateDecoder1D,
        TransitionAmplitudeHead1D,
        VariationalStateEncoder1D,
    )
    from .common import (
        STATE_CHANNELS,
        channel_weighted_mse,
        channel_weights_from_split,
        relative_l2_error_1d,
        resolve_device,
        rollout_vae_mean_1d,
    )
    from .euler_data import (
        build_euler1d_step_dataset,
        build_euler1d_trajectory_dataset_from_split,
    )
except ImportError:
    from grad_flow_l2.heat_data import load_dataset_splits
    from grad_flow_l2.latent_flow_VAE_mc import (
        FNOLatentTransition1D,
        LatentVAE1D,
        StateDecoder1D,
        TransitionAmplitudeHead1D,
        VariationalStateEncoder1D,
    )
    from grad_flow_l2.euler1d.common import (
        STATE_CHANNELS,
        channel_weighted_mse,
        channel_weights_from_split,
        relative_l2_error_1d,
        resolve_device,
        rollout_vae_mean_1d,
    )
    from grad_flow_l2.euler1d.euler_data import (
        build_euler1d_step_dataset,
        build_euler1d_trajectory_dataset_from_split,
    )


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value, n):
        self.total += float(value) * int(n)
        self.count += int(n)
        self.avg = self.total / max(1, self.count)


def set_seed(seed: int, seed_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda:
        torch.cuda.manual_seed_all(seed)


def _kl_diag_gaussians(q_mu, q_logvar, p_mu, p_logvar_scalar):
    q_logvar = q_logvar.clamp(-12.0, 8.0)
    p_logvar = p_logvar_scalar.clamp(-12.0, 8.0).view(-1, 1, 1)
    q_var, p_var = q_logvar.exp(), p_logvar.exp()
    return (
        0.5
        * (p_logvar - q_logvar + (q_var + (q_mu - p_mu).square()) / p_var - 1.0).mean()
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train latent VAE model on 1D Euler data")
    p.add_argument("--dataset-path", type=str, required=True)
    p.add_argument("--n-train", type=int, default=1200)
    p.add_argument("--n-val", type=int, default=300)
    p.add_argument("--n-test", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--hidden-channels", type=int, default=64)
    p.add_argument("--latent-channels", type=int, default=16)
    p.add_argument("--enc-blocks", type=int, default=4)
    p.add_argument("--dec-blocks", type=int, default=4)
    p.add_argument("--fno-width", type=int, default=None)
    p.add_argument("--fno-layers", type=int, default=6)
    p.add_argument("--fno-modes", type=int, default=64)
    p.add_argument("--disable-fno-grid", action="store_true")
    p.add_argument("--use-dt-channel", action="store_true")
    p.add_argument("--disable-forcing-channel", action="store_true")
    p.add_argument("--disable-u-grad-feature", action="store_true")
    p.add_argument("--amp-head-hidden", type=int, default=32)
    p.add_argument("--noise-corr-length", type=float, default=1.0)
    p.add_argument("--noise-decay-s", type=float, default=2.0)
    p.add_argument("--beta-kl", type=float, default=1e-2)
    p.add_argument("--lambda-rec", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--eval-interval", type=int, default=1)
    p.add_argument("--checkpoint-interval", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-step-size", type=int, default=100)
    p.add_argument("--lr-gamma", type=float, default=0.5)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--rollout-delta-clip", type=float, default=5.0)
    p.add_argument(
        "--channel-weights",
        type=float,
        nargs=3,
        default=None,
        metavar=("W_RHO", "W_U", "W_P"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-epoch-pbar", action="store_true")
    p.add_argument("--output-dir", type=str, default="grad_flow_l2/euler1d/outputs_vae")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _build_model(
    n_x: int, dt: float, boundary_condition: str, args: argparse.Namespace
) -> LatentVAE1D:
    use_forcing = not args.disable_forcing_channel
    width = args.hidden_channels if args.fno_width is None else args.fno_width
    encoder = VariationalStateEncoder1D(
        n_x=n_x,
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
        n_blocks=args.enc_blocks,
        use_grad_features=not args.disable_u_grad_feature,
        boundary_condition=boundary_condition,
        state_channels=STATE_CHANNELS,
    )
    decoder = StateDecoder1D(
        n_x=n_x,
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
        n_blocks=args.dec_blocks,
        boundary_condition=boundary_condition,
        state_channels=STATE_CHANNELS,
    )
    transition = FNOLatentTransition1D(
        n_x=n_x,
        latent_channels=args.latent_channels,
        width=width,
        n_layers=args.fno_layers,
        modes=args.fno_modes,
        use_forcing_channel=use_forcing,
        use_dt_channel=args.use_dt_channel,
        use_grid_features=not args.disable_fno_grid,
        default_dt=dt,
    )
    amp = TransitionAmplitudeHead1D(
        n_x=n_x,
        latent_channels=args.latent_channels,
        hidden_channels=args.amp_head_hidden,
        use_forcing_channel=use_forcing,
        boundary_condition=boundary_condition,
    )
    return LatentVAE1D(
        encoder=encoder,
        decoder=decoder,
        transition=transition,
        amplitude_head=amp,
        noise_corr_length=args.noise_corr_length,
        noise_decay_s=args.noise_decay_s,
    )


class EulerLatentVAETrainer:
    def __init__(
        self,
        model,
        dt,
        h,
        channel_weights,
        beta_kl=1e-2,
        lambda_rec=1.0,
        lr=1e-4,
        lr_step_size=100,
        lr_gamma=0.5,
        weight_decay=1e-5,
        grad_clip=1.0,
        rollout_delta_clip: Optional[float] = 2.0,
        device="cpu",
        output_dir=None,
        show_epoch_pbar=True,
    ):
        self.model = model.to(device)
        self.dt = float(dt)
        self.h = float(h)
        self.channel_weights = torch.as_tensor(channel_weights, dtype=torch.float32)
        self.beta_kl = float(beta_kl)
        self.lambda_rec = float(lambda_rec)
        self.grad_clip = float(grad_clip)
        self.rollout_delta_clip = (
            None if rollout_delta_clip is None else float(rollout_delta_clip)
        )
        self.device = device
        self.output_dir = output_dir
        self.show_epoch_pbar = bool(show_epoch_pbar)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=max(1, int(lr_step_size)), gamma=float(lr_gamma)
        )

    def _weights(self, tensor):
        return self.channel_weights.to(device=tensor.device, dtype=tensor.dtype)

    def _compute_losses(self, u_k, u_k1, f):
        w = self._weights(u_k)
        mu_q, logvar_q = self.model.encode_stats(u_k)
        z_k = self.model.sample_posterior(mu_q, logvar_q)
        z_next, mu_p, alpha = self.model.sample_prior(z_k, f, dt=self.dt)
        u_pred = self.model.decode(z_next)
        loss_step = channel_weighted_mse(u_pred, u_k1, w)
        loss_recon = channel_weighted_mse(self.model.decode(z_k), u_k, w)
        mu_q_next, logvar_q_next = self.model.encode_stats(u_k1)
        loss_kl = _kl_diag_gaussians(
            mu_q_next, logvar_q_next, mu_p, torch.log(alpha.square() + 1e-12)
        )
        loss = loss_step + self.beta_kl * loss_kl + self.lambda_rec * loss_recon
        return {
            "loss": loss,
            "loss_step": loss_step,
            "loss_kl": loss_kl,
            "loss_recon": loss_recon,
            "alpha_mean": alpha.mean().detach(),
        }

    def train_epoch(self, loader, epoch=None):
        self.model.train()
        meters = {
            k: AverageMeter()
            for k in ("loss", "loss_step", "loss_kl", "loss_recon", "alpha_mean")
        }
        iterable = loader
        pbar = None
        if self.show_epoch_pbar and tqdm is not None:
            pbar = tqdm(
                loader,
                total=len(loader),
                desc=f"Epoch {epoch:03d}" if epoch else "Epoch",
                leave=False,
                dynamic_ncols=True,
            )
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
                pbar.set_postfix(
                    total=f"{meters['loss'].avg:.4f}", kl=f"{meters['loss_kl'].avg:.4f}"
                )
        if pbar is not None:
            pbar.close()
        return {k: v.avg for k, v in meters.items()}

    @torch.no_grad()
    def validate(self, step_loader, traj_loader=None):
        self.model.eval()
        meters = {
            k: AverageMeter()
            for k in (
                "val_loss",
                "val_loss_step",
                "val_loss_kl",
                "val_loss_recon",
                "val_alpha_mean",
            )
        }
        for u_k, u_k1, f in step_loader:
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
            rollout = AverageMeter()
            ch = [AverageMeter() for _ in range(STATE_CHANNELS)]
            for batch in traj_loader:
                u0, f, ref = (
                    batch["u0"].to(self.device),
                    batch["f"].to(self.device),
                    batch["u_traj"].to(self.device),
                )
                pred = rollout_vae_mean_1d(
                    self.model,
                    u0,
                    f,
                    n_steps=int(ref.shape[1] - 1),
                    dt=self.dt,
                    delta_clip=self.rollout_delta_clip,
                )
                rel = relative_l2_error_1d(pred, ref, h=self.h)
                bsz = int(u0.shape[0])
                rollout.update(rel.mean(dim=2).mean().item(), bsz)
                for c in range(STATE_CHANNELS):
                    ch[c].update(rel[:, :, c].mean().item(), bsz)
            metrics["val_rollout_rel_l2"] = rollout.avg
            for c, m in enumerate(ch):
                metrics[f"val_rollout_rel_l2_ch{c}"] = m.avg
        return metrics

    def _save_checkpoint(self, name, epoch, metrics, state_dict=None):
        if self.output_dir is None:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": (
                    self.model.state_dict() if state_dict is None else state_dict
                ),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "dt": self.dt,
                "h": self.h,
                "beta_kl": self.beta_kl,
                "lambda_rec": self.lambda_rec,
                "rollout_delta_clip": self.rollout_delta_clip,
                "channel_weights": self.channel_weights.detach().cpu(),
            },
            os.path.join(self.output_dir, name),
        )

    def fit(
        self,
        train_step_loader,
        val_step_loader,
        val_traj_loader=None,
        epochs=200,
        eval_interval=1,
        checkpoint_interval=25,
    ):
        hist = {"train": [], "val": []}
        best = float("inf")
        best_state = None
        best_metrics = None
        best_epoch = 0
        for epoch in range(1, int(epochs) + 1):
            tr = self.train_epoch(train_step_loader, epoch=epoch)
            hist["train"].append({"epoch": epoch, **tr})
            if epoch % int(eval_interval) == 0:
                val = self.validate(val_step_loader, val_traj_loader)
                hist["val"].append({"epoch": epoch, **val})
                monitor = val.get("val_rollout_rel_l2", val["val_loss_step"])
                if monitor < best:
                    best = monitor
                    best_epoch = epoch
                    best_metrics = val
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                    self._save_checkpoint("best_model.pt", epoch, val)
                print(
                    f"[Epoch {epoch:03d}] train_total={tr['loss']:.6f} train_step={tr['loss_step']:.6f} train_kl={tr['loss_kl']:.6f} train_recon={tr['loss_recon']:.6f} alpha={tr['alpha_mean']:.4f} val_total={val['val_loss']:.6f} val_step={val['val_loss_step']:.6f} val_kl={val['val_loss_kl']:.6f} val_recon={val['val_loss_recon']:.6f} val_rollout={val.get('val_rollout_rel_l2',float('nan')):.6f}"
                )
            else:
                print(
                    f"[Epoch {epoch:03d}] train_total={tr['loss']:.6f} train_step={tr['loss_step']:.6f} train_kl={tr['loss_kl']:.6f} train_recon={tr['loss_recon']:.6f} alpha={tr['alpha_mean']:.4f}"
                )
            if checkpoint_interval > 0 and epoch % int(checkpoint_interval) == 0:
                self._save_checkpoint(
                    f"best_model_through_epoch_{epoch:04d}.pt",
                    best_epoch or epoch,
                    best_metrics or tr,
                    state_dict=best_state,
                )
            self.scheduler.step()
        final = hist["val"][-1] if hist["val"] else hist["train"][-1]
        self._save_checkpoint("final_model.pt", int(epochs), final)
        if self.output_dir:
            with open(
                os.path.join(self.output_dir, "history.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(hist, f, indent=2)
        return hist


def main(args):
    set_seed(args.seed, seed_cuda=not args.cpu)
    device = resolve_device(cpu=args.cpu, device=args.device)
    splits = load_dataset_splits(args.dataset_path, map_location="cpu")
    train_split, val_split, test_split = splits["train"], splits["val"], splits["test"]
    sizes = (
        int(train_split["u0"].shape[0]),
        int(val_split["u0"].shape[0]),
        int(test_split["u0"].shape[0]),
    )
    if sizes != (args.n_train, args.n_val, args.n_test):
        raise ValueError(
            f"Dataset split sizes {sizes} do not match args {(args.n_train,args.n_val,args.n_test)}"
        )
    meta = splits.get("meta", {})
    n_x = int(train_split["u0"].shape[-1])
    n_steps = int(train_split["u_traj"].shape[1] - 1)
    t_final = float(meta.get("t_final", float(n_steps)))
    dt = float(meta.get("dataset_dt", t_final / float(n_steps)))
    domain_length = float(meta.get("domain_length", 1.0))
    h = domain_length / float(n_x)
    bc = str(meta.get("boundary_condition", "periodic"))
    channel_weights = channel_weights_from_split(train_split, args.channel_weights)
    print(f"Device: {device}")
    print(f"Loaded dataset: {args.dataset_path}")
    print(
        f"Grid: n_x={n_x}, channels={STATE_CHANNELS}, steps={n_steps}, dt={dt:.6f}, L={domain_length}, bc={bc}"
    )
    print(f"Channel weights (rho,u,p): {channel_weights.tolist()}")
    train_step = DataLoader(
        build_euler1d_step_dataset(train_split),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_step = DataLoader(
        build_euler1d_step_dataset(val_split),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    val_traj = DataLoader(
        build_euler1d_trajectory_dataset_from_split(val_split),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_step = (
        DataLoader(
            build_euler1d_step_dataset(test_split),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        if args.n_test > 0
        else None
    )
    test_traj = (
        DataLoader(
            build_euler1d_trajectory_dataset_from_split(test_split),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        if args.n_test > 0
        else None
    )
    model = _build_model(n_x=n_x, dt=dt, boundary_condition=bc, args=args)
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    run_dir = os.path.join(
        args.output_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(run_dir, exist_ok=True)
    args_dict = vars(args).copy()
    args_dict["channel_weights_used"] = channel_weights.tolist()
    args_dict["domain_length"] = domain_length
    with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=2)
    trainer = EulerLatentVAETrainer(
        model,
        dt=dt,
        h=h,
        channel_weights=channel_weights,
        beta_kl=args.beta_kl,
        lambda_rec=args.lambda_rec,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        rollout_delta_clip=(
            args.rollout_delta_clip if args.rollout_delta_clip > 0 else None
        ),
        device=device,
        output_dir=run_dir,
        show_epoch_pbar=not args.no_epoch_pbar,
    )
    if args.dry_run:
        print("Dry run val metrics:", trainer.validate(val_step, val_traj))
        return
    hist = trainer.fit(
        train_step,
        val_step,
        val_traj,
        args.epochs,
        args.eval_interval,
        args.checkpoint_interval,
    )
    print("Training complete.")
    print("Last train metrics:", hist["train"][-1])
    print("Last val metrics:", hist["val"][-1] if hist["val"] else None)
    if args.n_test > 0:
        print("Test metrics:", trainer.validate(test_step, test_traj))
    print(f"Saved training artifacts to: {run_dir}")


if __name__ == "__main__":
    main(parse_args())
