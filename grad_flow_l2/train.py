"""
Training entrypoint for grad_flow_l2 heat-equation solver.
"""

from __future__ import annotations

import argparse
import os
import random
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader

try:
    from .data import (
        build_step_dataset,
        build_trajectory_dataset_from_split,
        load_dataset_splits,
    )
    from .generator import EnergyHead1D, GradientFlowModel, ProximalMap1D
    from .trainer import GradientFlowTrainer
except ImportError:
    # Allow running as a script: python grad_flow_l2/train.py
    from data import (
        build_step_dataset,
        build_trajectory_dataset_from_split,
        load_dataset_splits,
    )
    from generator import EnergyHead1D, GradientFlowModel, ProximalMap1D
    from trainer import GradientFlowTrainer


def set_seed(seed: int, seed_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda:
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train L2 gradient-flow heat equation model")

    # Dataset and dataloader.
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/heat_l2_nx100_steps10.pt",
        help="Path to cached precomputed train/val/test dataset file (.pt). Prepare with data.py first.",
    )

    # Model defaults.
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--prox-blocks", type=int, default=6)
    parser.add_argument("--energy-layers", type=int, default=4)
    parser.add_argument("--use-dt-channel", action="store_true")

    # Training defaults.
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-mono", type=float, default=1.0)
    parser.add_argument("--lambda-edi", type=float, default=1.0)

    # Misc.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-dir", type=str, default="grad_flow_l2/outputs")
    parser.add_argument("--dry-run", action="store_true", help="Build everything and run one validation pass only")

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed, seed_cuda=not args.cpu)

    if args.cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(
            f"Dataset not found: {args.dataset_path}. "
            "Run data generation first (e.g., `python -m grad_flow_l2.data --force-regenerate-data`)."
        )
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

    n_x = int(train_split["u0"].shape[1])
    n_steps = int(train_split["u_traj"].shape[1] - 1)
    h = 1.0 / float(n_x + 1)
    dt = 1.0 / float(n_steps)

    print(f"Device: {device}")
    print(f"Loaded dataset: {args.dataset_path}")
    print(f"Grid from data: n_x={n_x}, n_steps={n_steps}, h={h:.6f}, dt={dt:.6f}")

    train_traj_ds = build_trajectory_dataset_from_split(train_split)
    val_traj_ds = build_trajectory_dataset_from_split(val_split)
    test_traj_ds = build_trajectory_dataset_from_split(test_split)

    train_step_ds = build_step_dataset(train_split)
    val_step_ds = build_step_dataset(val_split)
    test_step_ds = build_step_dataset(test_split)

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
    val_traj_loader = DataLoader(
        val_traj_ds,
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
    test_traj_loader = DataLoader(
        test_traj_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Model.
    prox_map = ProximalMap1D(
        n_x=n_x,
        hidden_channels=args.hidden_channels,
        n_blocks=args.prox_blocks,
        use_dt_channel=args.use_dt_channel,
        default_dt=dt,
    )
    energy_head = EnergyHead1D(
        n_x=n_x,
        h=h,
        hidden_channels=args.hidden_channels,
        n_layers=args.energy_layers,
        use_ux_feature=True,
    )
    model = GradientFlowModel(prox_map=prox_map, energy_head=energy_head)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    trainer = GradientFlowTrainer(
        model=model,
        dt=dt,
        h=h,
        lambda_mono=args.lambda_mono,
        lambda_edi=args.lambda_edi,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        max_epochs=args.epochs,
        device=device,
        output_dir=out_dir,
    )

    if args.dry_run:
        metrics = trainer.validate(val_step_loader, traj_loader=val_traj_loader)
        print("Dry run metrics:", metrics)
        test_metrics = trainer.validate(test_step_loader, traj_loader=test_traj_loader)
        print("Dry run test metrics:", test_metrics)
        return

    print(
        f"Training config: epochs={args.epochs}, lr={args.lr}, "
        f"lambda_mono={args.lambda_mono}, lambda_edi={args.lambda_edi}, output={out_dir}"
    )

    history = trainer.fit(
        train_step_loader=train_step_loader,
        val_step_loader=val_step_loader,
        val_traj_loader=val_traj_loader,
        epochs=args.epochs,
        eval_interval=args.eval_interval,
    )

    last_train = history["train"][-1]
    last_val = history["val"][-1] if history["val"] else None
    print("Training complete.")
    print("Last train metrics:", last_train)
    if last_val is not None:
        print("Last val metrics:", last_val)

    test_metrics = trainer.validate(test_step_loader, traj_loader=test_traj_loader)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main(parse_args())
