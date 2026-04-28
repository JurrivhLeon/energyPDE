"""
CH-2D model construction helpers.

This module centralizes the boundary-condition choice for Cahn-Hilliard.
The CH benchmark uses homogeneous Neumann BCs, so we hard-code that here
instead of relying on the generic model defaults.
"""

from __future__ import annotations

import argparse

try:
    from ..grad_flow2d import (
        EnergyHead2D,
        FNOProximalStepSimulator2D,
        HiddenGradientFlowModel2D,
        ProximalStepSimulator2D,
        StateDecoder2D,
        StateEncoder2D,
    )
except ImportError:
    from grad_flow_l2.grad_flow2d import (
        EnergyHead2D,
        FNOProximalStepSimulator2D,
        HiddenGradientFlowModel2D,
        ProximalStepSimulator2D,
        StateDecoder2D,
        StateEncoder2D,
    )


CH_BOUNDARY_CONDITION = "neumann"


def build_cahn_hilliard2d_model(
    *,
    n_x: int,
    n_y: int,
    h_x: float,
    h_y: float,
    dt: float,
    args: argparse.Namespace,
) -> HiddenGradientFlowModel2D:
    """
    Build the CH-2D hidden-space model with Neumann BCs enforced by construction.
    """
    use_forcing_channel = not args.disable_forcing_channel
    boundary_condition = CH_BOUNDARY_CONDITION

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
    )
    return HiddenGradientFlowModel2D(
        encoder=encoder,
        decoder=decoder,
        prox_step=prox_step,
        energy_head=energy_head,
    )
