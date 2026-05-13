"""
Training entrypoint for hidden-space 2D Navier-Stokes gradient-flow model
on the periodic dataset.
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
    from ..heat_data import load_dataset_splits
    from ..navier_stokes2d_per_data import (
        build_navier_stokes2d_periodic_step_dataset,
        build_navier_stokes2d_periodic_trajectory_dataset_from_split,
    )
    from ..latent_markov import (
        FNOProximalStepSimulator2D,
        LatentMarkovModel2D,
        ProximalStepSimulator2D,
        StateDecoder2D,
        StateEncoder2D,
    )
    from ..latent_markov_trainer import LatentMarkovTrainer2D
except ImportError:
    from grad_flow_l2.heat_data import load_dataset_splits
    from grad_flow_l2.navier_stokes2d_per_data import (
        build_navier_stokes2d_periodic_step_dataset,
        build_navier_stokes2d_periodic_trajectory_dataset_from_split,
    )
    from grad_flow_l2.latent_markov import (
        FNOProximalStepSimulator2D,
        LatentMarkovModel2D,
        ProximalStepSimulator2D,
        StateDecoder2D,
        StateEncoder2D,
    )
    from grad_flow_l2.latent_markov_trainer import LatentMarkovTrainer2D


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hidden-space model on periodic 2D Navier-Stokes data")
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to cached periodic navier-stokes dataset (.pt)",
    )
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
    parser.add_argument("--use-dt-channel", action="store_true")
    parser.add_argument("--disable-forcing-channel", action="store_true")
    parser.add_argument("--disable-u-grad-feature", action="store_true")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=25,
        help="Save best_model_through_epoch_XXXX.pt every N epochs. Use <=0 to disable periodic snapshots.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-step-size", type=int, default=100, help="Halve/decay LR every N epochs.")
    parser.add_argument("--lr-gamma", type=float, default=0.5, help="Multiplicative LR decay for StepLR.")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--rollout-delta-clip",
        type=float,
        default=10.0,
        help="L-infinity clip for rollout increments during validation. Use <=0 to disable.",
    )
    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--lambda-spec", type=float, default=1.0)
    parser.add_argument("--spectral-s", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-epoch-pbar", action="store_true", help="Disable per-epoch batch progress bar.")
    parser.add_argument("--output-dir", type=str, default="grad_flow_l2/ns2d_per/outputs")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _build_model(n_x: int, n_y: int, h_x: float, h_y: float, dt: float, args: argparse.Namespace) -> LatentMarkovModel2D:
    boundary_condition = "periodic"
    use_forcing_channel = not args.disable_forcing_channel
    encoder = StateEncoder2D(
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
    if args.prox_simulator_type == "cnn":
        prox_step = ProximalStepSimulator2D(
            n_x=n_x,
            n_y=n_y,
            latent_channels=args.latent_channels,
            hidden_channels=args.hidden_channels,
            n_blocks=args.prox_blocks,
            use_forcing_channel=use_forcing_channel,
            use_dt_channel=args.use_dt_channel,
            default_dt=dt,
            boundary_condition=boundary_condition,
        )
    elif args.prox_simulator_type == "fno":
        prox_step = FNOProximalStepSimulator2D(
            n_x=n_x,
            n_y=n_y,
            latent_channels=args.latent_channels,
            width=args.hidden_channels,
            n_layers=args.prox_blocks,
            modes_x=args.fno_modes_x,
            modes_y=args.fno_modes_y,
            use_forcing_channel=use_forcing_channel,
            use_dt_channel=args.use_dt_channel,
            use_grid_features=not args.disable_fno_grid,
            default_dt=dt,
            boundary_condition=boundary_condition,
        )
    else:
        raise ValueError(f"Unsupported prox-simulator-type: {args.prox_simulator_type}")

    return LatentMarkovModel2D(
        encoder=encoder,
        decoder=decoder,
        transition=prox_step,
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

    train_traj_ds = build_navier_stokes2d_periodic_trajectory_dataset_from_split(train_split)
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

    model = _build_model(n_x=n_x, n_y=n_y, h_x=h_x, h_y=h_y, dt=dt, args=args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    model = model.to(device)

    trainer = LatentMarkovTrainer2D(
        model=model,
        dt=dt,
        h_x=h_x,
        h_y=h_y,
        lambda_recon=args.lambda_recon,
        lambda_spec=args.lambda_spec,
        spectral_s=args.spectral_s,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        rollout_delta_clip=args.rollout_delta_clip,
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
        f"lr_step_size={args.lr_step_size}, lr_gamma={args.lr_gamma}, "
        f"lambda_recon={args.lambda_recon}, lambda_spec={args.lambda_spec}, spectral_s={args.spectral_s}, "
        f"rollout_delta_clip={args.rollout_delta_clip}, "
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
