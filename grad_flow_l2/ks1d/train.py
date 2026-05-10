"""
Train a deterministic FNO latent Markov baseline on unforced 1D KS data.

This is the non-probabilistic counterpart to ``train_vae.py``:
  z_k = E(u_k)
  z_{k+1} = z_k + G_FNO(z_k, f, dt)
  u_{k+1} = D(z_{k+1})
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
    from ..heat_data import build_step_dataset, build_trajectory_dataset_from_split, load_dataset_splits
    from ..latent_markov_1d import LatentMarkovFNO1D, build_latent_markov_fno_1d
    from ..utils import compute_relative_l2_error, rollout_model
except ImportError:
    from grad_flow_l2.heat_data import build_step_dataset, build_trajectory_dataset_from_split, load_dataset_splits
    from grad_flow_l2.latent_markov_1d import LatentMarkovFNO1D, build_latent_markov_fno_1d
    from grad_flow_l2.utils import compute_relative_l2_error, rollout_model


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


class SupervisedLatentMarkovTrainer:
    def __init__(
        self,
        model: LatentMarkovFNO1D,
        dt: float,
        h: float,
        lambda_recon: float = 1.0,
        lr: float = 1e-4,
        lr_step_size: int = 100,
        lr_gamma: float = 0.5,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
        max_epochs: int = 200,
        device: str = "cpu",
        output_dir: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.dt = float(dt)
        self.h = float(h)
        self.lambda_recon = float(lambda_recon)
        self.grad_clip = float(grad_clip)
        self.device = device
        self.output_dir = output_dir
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=max(1, int(lr_step_size)),
            gamma=float(lr_gamma),
        )

    def _compute_losses(self, u_k: torch.Tensor, u_k1_data: torch.Tensor, f: torch.Tensor) -> Dict[str, torch.Tensor]:
        u_pred, z_k, _ = self.model.predict_step(u_k, f, dt=self.dt, return_latent=True)
        loss_step = F.mse_loss(u_pred, u_k1_data)
        loss_recon = F.mse_loss(self.model.decode(z_k), u_k)
        loss = loss_step + self.lambda_recon * loss_recon
        return {"loss": loss, "loss_step": loss_step, "loss_recon": loss_recon}

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        meters = {name: AverageMeter() for name in ("loss", "loss_step", "loss_recon")}
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
            bsz = int(u_k.shape[0])
            for k, meter in meters.items():
                meter.update(losses[k].item(), bsz)
        return {k: v.avg for k, v in meters.items()}

    @torch.no_grad()
    def validate(self, step_loader: DataLoader, traj_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        self.model.eval()
        meters = {name: AverageMeter() for name in ("val_loss", "val_loss_step", "val_loss_recon")}
        for batch in step_loader:
            u_k, u_k1, f = _unpack_step_batch(batch)
            u_k = u_k.to(self.device)
            u_k1 = u_k1.to(self.device)
            f = f.to(self.device)
            losses = self._compute_losses(u_k, u_k1, f)
            bsz = int(u_k.shape[0])
            meters["val_loss"].update(losses["loss"].item(), bsz)
            meters["val_loss_step"].update(losses["loss_step"].item(), bsz)
            meters["val_loss_recon"].update(losses["loss_recon"].item(), bsz)

        metrics = {k: v.avg for k, v in meters.items()}
        if traj_loader is not None:
            rollout_rel = AverageMeter()
            rollout_mse = AverageMeter()
            for batch in traj_loader:
                u0, f, u_ref = _unpack_traj_batch(batch)
                u0 = u0.to(self.device)
                f = f.to(self.device)
                u_ref = u_ref.to(self.device)
                n_steps = int(u_ref.shape[1] - 1)
                u_pred = rollout_model(self.model, u0=u0, f=f, n_steps=n_steps, dt=self.dt)
                rel = compute_relative_l2_error(u_pred, u_ref, h=self.h)
                rollout_rel.update(rel.mean(dim=-1).mean().item(), int(u0.shape[0]))
                rollout_mse.update(F.mse_loss(u_pred, u_ref).item(), int(u0.shape[0]))
            metrics["val_rollout_rel_l2"] = rollout_rel.avg
            metrics["val_rollout_mse"] = rollout_mse.avg
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
                "lambda_recon": self.lambda_recon,
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
        best_checkpoint_interval: int = 25,
    ) -> Dict[str, list]:
        history = {"train": [], "val": []}
        best_metric = float("inf")
        best_checkpoint_interval = int(best_checkpoint_interval)
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
                    f"val_total={val_metrics['val_loss']:.6f} "
                    f"val_step={val_metrics['val_loss_step']:.6f} "
                    f"val_recon={val_metrics['val_loss_recon']:.6f} "
                    f"val_rollout={val_metrics.get('val_rollout_rel_l2', float('nan')):.6f}"
                )
            else:
                print(
                    f"[Epoch {epoch:03d}] "
                    f"train_total={train_metrics['loss']:.6f} "
                    f"train_step={train_metrics['loss_step']:.6f} "
                    f"train_recon={train_metrics['loss_recon']:.6f}"
                )
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
            with open(os.path.join(self.output_dir, "history.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        return history


def set_seed(seed: int, seed_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda:
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train deterministic latent FNO Markov baseline on 1D KS")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="grad_flow_l2/ks1d/datasets/ks_periodic_L32pi_snx1024_nx256_dt1_solverdt0p01.pt",
    )
    parser.add_argument("--n-train", type=int, default=1500)
    parser.add_argument("--n-val", type=int, default=300)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--enc-blocks", type=int, default=4)
    parser.add_argument("--dec-blocks", type=int, default=4)
    parser.add_argument("--fno-width", type=int, default=None)
    parser.add_argument("--fno-layers", type=int, default=6)
    parser.add_argument("--fno-modes", type=int, default=64)
    parser.add_argument("--disable-fno-grid", action="store_true")
    parser.add_argument("--use-dt-channel", action="store_true")
    parser.add_argument(
        "--disable-forcing-channel",
        action="store_true",
        default=True,
        help="Compatibility flag; KS deterministic training always disables the forcing channel.",
    )
    parser.add_argument("--disable-u-grad-feature", action="store_true")

    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=1)
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
    parser.add_argument("--output-dir", type=str, default="grad_flow_l2/ks1d/outputs_sv")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _build_model(n_x: int, dt: float, boundary_condition: str, args: argparse.Namespace) -> LatentMarkovFNO1D:
    return build_latent_markov_fno_1d(
        n_x=n_x,
        dt=dt,
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
        enc_blocks=args.enc_blocks,
        dec_blocks=args.dec_blocks,
        fno_width=args.fno_width,
        fno_layers=args.fno_layers,
        fno_modes=args.fno_modes,
        use_forcing_channel=False,
        use_dt_channel=args.use_dt_channel,
        use_grid_features=not args.disable_fno_grid,
        use_grad_features=not args.disable_u_grad_feature,
        boundary_condition=boundary_condition,
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
    split_sizes = (int(train_split["u0"].shape[0]), int(val_split["u0"].shape[0]), int(test_split["u0"].shape[0]))
    if split_sizes != (args.n_train, args.n_val, args.n_test):
        raise ValueError(f"Dataset split sizes {split_sizes} do not match args {(args.n_train, args.n_val, args.n_test)}")

    meta = splits.get("meta", {})
    n_x = int(train_split["u0"].shape[1])
    n_steps = int(train_split["u_traj"].shape[1] - 1)
    t_final = float(meta.get("t_final", 1.0))
    boundary_condition = str(meta.get("boundary_condition", "periodic" if meta.get("periodic", False) else "dirichlet"))
    domain_length = float(meta.get("domain_length", 1.0))
    h_default = domain_length / float(n_x) if boundary_condition == "periodic" else 1.0 / float(n_x + 1)
    h = float(meta.get("h", h_default))
    dt = float(meta.get("dataset_dt", t_final / float(n_steps)))

    print(f"Device: {device}")
    print(f"Loaded dataset: {args.dataset_path}")
    print(
        f"Grid from data: n_x={n_x}, n_steps={n_steps}, h={h:.8f}, "
        f"dt={dt:.8f}, bc={boundary_condition}, L={domain_length:.8f}, use_forcing_channel=False"
    )

    train_step_loader = DataLoader(
        build_step_dataset(train_split), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_step_loader = DataLoader(
        build_step_dataset(val_split), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_step_loader = DataLoader(
        build_step_dataset(test_split), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    val_traj_loader = DataLoader(
        build_trajectory_dataset_from_split(val_split),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_traj_loader = DataLoader(
        build_trajectory_dataset_from_split(test_split),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = _build_model(n_x=n_x, dt=dt, boundary_condition=boundary_condition, args=args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    trainer = SupervisedLatentMarkovTrainer(
        model=model,
        dt=dt,
        h=h,
        lambda_recon=args.lambda_recon,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        max_epochs=args.epochs,
        device=device,
        output_dir=out_dir,
    )

    if args.dry_run:
        val_metrics = trainer.validate(val_step_loader, traj_loader=val_traj_loader)
        print("Dry run val metrics:", val_metrics)
        test_metrics = trainer.validate(test_step_loader, traj_loader=test_traj_loader)
        print("Dry run test metrics:", test_metrics)
        return

    history = trainer.fit(
        train_step_loader=train_step_loader,
        val_step_loader=val_step_loader,
        val_traj_loader=val_traj_loader,
        epochs=args.epochs,
        eval_interval=args.eval_interval,
        best_checkpoint_interval=args.best_checkpoint_interval,
    )
    print("Training complete.")
    print("Last train metrics:", history["train"][-1])
    if history["val"]:
        print("Last val metrics:", history["val"][-1])
    print("Test metrics:", trainer.validate(test_step_loader, traj_loader=test_traj_loader))
    print(f"Saved training artifacts to: {out_dir}")


if __name__ == "__main__":
    main(parse_args())
