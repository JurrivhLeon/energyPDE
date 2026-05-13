"""Training entrypoint for periodic 2D FitzHugh-Nagumo hidden-space models."""

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
    from ..fhn2d.fhn_data import STATE_CHANNELS, build_fhn2d_step_dataset, build_fhn2d_trajectory_dataset_from_split
    from ..grad_flow2d_mc import (
        EnergyHead2D,
        FNOProximalStepSimulator2D,
        HiddenGradientFlowModel2D,
        ProximalStepSimulator2D,
        StateDecoder2D,
        StateEncoder2D,
    )
    from ..heat_data import load_dataset_splits
    from ..navier_stokes2d.trainer import HiddenGradientFlowTrainer2D
except ImportError:
    from grad_flow_l2.fhn2d.fhn_data import STATE_CHANNELS, build_fhn2d_step_dataset, build_fhn2d_trajectory_dataset_from_split
    from grad_flow_l2.grad_flow2d_mc import (
        EnergyHead2D,
        FNOProximalStepSimulator2D,
        HiddenGradientFlowModel2D,
        ProximalStepSimulator2D,
        StateDecoder2D,
        StateEncoder2D,
    )
    from grad_flow_l2.heat_data import load_dataset_splits
    from grad_flow_l2.navier_stokes2d.trainer import HiddenGradientFlowTrainer2D


def set_seed(seed: int, seed_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda:
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hidden-space model on periodic 2D FitzHugh-Nagumo data")
    parser.add_argument("--dataset-path", type=str, required=True)
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
    parser.add_argument("--energy-head-type", type=str, default="local", choices=["local", "fno"])
    parser.add_argument("--energy-fno-modes-x", type=int, default=16)
    parser.add_argument("--energy-fno-modes-y", type=int, default=16)
    parser.add_argument("--use-dt-channel", action="store_true")
    parser.add_argument("--disable-forcing-channel", action="store_true")
    parser.add_argument("--disable-z-grad-feature", action="store_true")
    parser.add_argument("--disable-u-grad-feature", action="store_true")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-step-size", type=int, default=100)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--lambda-mono", type=float, default=1.0)
    parser.add_argument("--lambda-prox", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-epoch-pbar", action="store_true")
    parser.add_argument("--output-dir", type=str, default="grad_flow_l2/fhn2d/outputs")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _build_model(n_x: int, n_y: int, h_x: float, h_y: float, dt: float, args: argparse.Namespace) -> HiddenGradientFlowModel2D:
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
        state_channels=STATE_CHANNELS,
    )
    decoder = StateDecoder2D(
        n_x=n_x,
        n_y=n_y,
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
        n_blocks=args.dec_blocks,
        boundary_condition=boundary_condition,
        state_channels=STATE_CHANNELS,
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
    else:
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
    energy_head = EnergyHead2D(
        n_x=n_x,
        n_y=n_y,
        h_x=h_x,
        h_y=h_y,
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
        n_layers=args.energy_layers,
        use_forcing_channel=use_forcing_channel,
        use_grad_norm_feature=not args.disable_z_grad_feature,
        boundary_condition=boundary_condition,
        head_type=args.energy_head_type,
        energy_fno_modes_x=args.energy_fno_modes_x,
        energy_fno_modes_y=args.energy_fno_modes_y,
    )
    return HiddenGradientFlowModel2D(encoder=encoder, decoder=decoder, prox_step=prox_step, energy_head=energy_head)


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed, seed_cuda=not args.cpu)
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    splits = load_dataset_splits(args.dataset_path, map_location="cpu")
    train_split = splits["train"]
    val_split = splits["val"]
    test_split = splits["test"]
    sizes = (int(train_split["u0"].shape[0]), int(val_split["u0"].shape[0]), int(test_split["u0"].shape[0]))
    if sizes != (args.n_train, args.n_val, args.n_test):
        raise ValueError(f"Dataset split sizes {sizes} do not match args {(args.n_train, args.n_val, args.n_test)}")

    meta = splits.get("meta", {})
    n_x = int(train_split["u0"].shape[-2])
    n_y = int(train_split["u0"].shape[-1])
    n_steps = int(train_split["u_traj"].shape[1] - 1)
    t_final = float(meta.get("t_final", float(n_steps)))
    dt = t_final / float(n_steps)
    h_x = 1.0 / float(n_x)
    h_y = 1.0 / float(n_y)

    print(f"Device: {device}")
    print(f"Loaded dataset: {args.dataset_path}")
    print(f"Grid: ({n_x},{n_y}), state_channels={STATE_CHANNELS}, steps={n_steps}, dt={dt:.6f}")

    train_step_loader = DataLoader(build_fhn2d_step_dataset(train_split), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_step_loader = DataLoader(build_fhn2d_step_dataset(val_split), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_step_loader = DataLoader(build_fhn2d_step_dataset(test_split), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_traj_loader = DataLoader(build_fhn2d_trajectory_dataset_from_split(val_split), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_traj_loader = DataLoader(build_fhn2d_trajectory_dataset_from_split(test_split), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = _build_model(n_x=n_x, n_y=n_y, h_x=h_x, h_y=h_y, dt=dt, args=args).to(device)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

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
        lambda_spec=0.0,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        max_epochs=args.epochs,
        device=device,
        output_dir=run_dir,
        show_epoch_pbar=not args.no_epoch_pbar,
    )

    if args.dry_run:
        print("Dry run val metrics:", trainer.validate(val_step_loader, traj_loader=val_traj_loader))
        print("Dry run test metrics:", trainer.validate(test_step_loader, traj_loader=test_traj_loader))
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
    print("Test metrics:", trainer.validate(test_step_loader, traj_loader=test_traj_loader))
    print(f"Saved training artifacts to: {run_dir}")


if __name__ == "__main__":
    main(parse_args())
