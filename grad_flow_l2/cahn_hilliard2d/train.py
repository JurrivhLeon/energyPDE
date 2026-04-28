"""
Training entrypoint for hidden-space 2D Cahn-Hilliard latent gradient-flow model.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from ..cahn_hilliard2d_data import (
        build_cahn_hilliard2d_step_dataset,
        build_cahn_hilliard2d_trajectory_dataset_from_split,
    )
    from ..heat_data import load_dataset_splits
    from .model import build_cahn_hilliard2d_model
    from .trainer import HiddenGradientFlowTrainer2D
except ImportError:
    from grad_flow_l2.cahn_hilliard2d.trainer import HiddenGradientFlowTrainer2D
    from grad_flow_l2.cahn_hilliard2d_data import (
        build_cahn_hilliard2d_step_dataset,
        build_cahn_hilliard2d_trajectory_dataset_from_split,
    )
    from grad_flow_l2.heat_data import load_dataset_splits
    from grad_flow_l2.cahn_hilliard2d.model import build_cahn_hilliard2d_model


def set_seed(seed: int, seed_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda:
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hidden-space model on 2D Cahn-Hilliard data")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to cached Cahn-Hilliard dataset (.pt)")
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--enc-blocks", type=int, default=4)
    parser.add_argument("--dec-blocks", type=int, default=4)
    parser.add_argument("--prox-blocks", type=int, default=6)
    parser.add_argument("--prox-simulator-type", type=str, default="cnn", choices=["cnn", "fno"])
    parser.add_argument("--fno-modes-x", type=int, default=16)
    parser.add_argument("--fno-modes-y", type=int, default=16)
    parser.add_argument("--disable-fno-grid", action="store_true")
    parser.add_argument("--energy-layers", type=int, default=4)
    parser.add_argument("--use-dt-channel", action="store_true")
    parser.add_argument("--disable-forcing-channel", action="store_true")
    parser.add_argument("--disable-z-grad-feature", action="store_true")
    parser.add_argument("--disable-u-grad-feature", action="store_true")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--lambda-mono", type=float, default=1.0)
    parser.add_argument("--lambda-prox", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-epoch-pbar", action="store_true", help="Disable per-epoch batch progress bar.")
    parser.add_argument("--output-dir", type=str, default="grad_flow_l2/cahn_hilliard2d/outputs")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _build_model(
    n_x: int,
    n_y: int,
    h_x: float,
    h_y: float,
    dt: float,
    args: argparse.Namespace,
) -> HiddenGradientFlowModel2D:
    return build_cahn_hilliard2d_model(n_x=n_x, n_y=n_y, h_x=h_x, h_y=h_y, dt=dt, args=args)


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
    h_x = float(meta.get("h_x", 1.0 / float(n_x + 1)))
    h_y = float(meta.get("h_y", 1.0 / float(n_y + 1)))
    dt = t_final / float(n_steps)

    print(f"Device: {device}")
    print(f"Loaded dataset: {args.dataset_path}")
    print(
        f"Grid from data: n_x={n_x}, n_y={n_y}, n_steps={n_steps}, "
        f"t_final={t_final:.6f}, dt={dt:.6f}, h_x={h_x:.6f}, h_y={h_y:.6f}"
    )

    train_step_ds = build_cahn_hilliard2d_step_dataset(train_split)
    val_step_ds = build_cahn_hilliard2d_step_dataset(val_split)
    test_step_ds = build_cahn_hilliard2d_step_dataset(test_split)

    train_traj_ds = build_cahn_hilliard2d_trajectory_dataset_from_split(train_split)
    val_traj_ds = build_cahn_hilliard2d_trajectory_dataset_from_split(val_split)
    test_traj_ds = build_cahn_hilliard2d_trajectory_dataset_from_split(test_split)

    train_step_loader = DataLoader(
        train_step_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_step_loader = DataLoader(
        val_step_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_step_loader = DataLoader(
        test_step_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    val_traj_loader = DataLoader(
        val_traj_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_traj_loader = DataLoader(
        test_traj_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = _build_model(n_x=n_x, n_y=n_y, h_x=h_x, h_y=h_y, dt=dt, args=args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    trainer = HiddenGradientFlowTrainer2D(
        model=model,
        dt=dt,
        h_x=h_x,
        h_y=h_y,
        lambda_recon=args.lambda_recon,
        lambda_mono=args.lambda_mono,
        lambda_prox=args.lambda_prox,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        max_epochs=args.epochs,
        device=device,
        output_dir=run_dir,
        show_epoch_pbar=not args.no_epoch_pbar,
    )

    if args.dry_run:
        val_metrics = trainer.validate(val_step_loader, traj_loader=val_traj_loader)
        print("Dry run val metrics:", val_metrics)
        test_metrics = trainer.validate(test_step_loader, traj_loader=test_traj_loader)
        print("Dry run test metrics:", test_metrics)
        return

    print(
        f"Training config: epochs={args.epochs}, lr={args.lr}, "
        f"lambda_recon={args.lambda_recon}, lambda_mono={args.lambda_mono}, lambda_prox={args.lambda_prox}, "
        f"prox_type={args.prox_simulator_type}, "
        f"fno_modes=({args.fno_modes_x},{args.fno_modes_y}), "
        f"epoch_pbar={not args.no_epoch_pbar}, "
        f"output={run_dir}"
    )
    history = trainer.fit(
        train_step_loader=train_step_loader,
        val_step_loader=val_step_loader,
        val_traj_loader=val_traj_loader,
        epochs=args.epochs,
        eval_interval=args.eval_interval,
    )
    print("Training complete.")
    print("Last train metrics:", history["train"][-1])
    if history["val"]:
        print("Last val metrics:", history["val"][-1])
    test_metrics = trainer.validate(test_step_loader, traj_loader=test_traj_loader)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main(parse_args())
