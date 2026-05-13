"""Compressible 2D Navier-Stokes data utilities."""

from .cfd_data import (
    STATE_CHANNELS,
    STATE_NAMES_CONSERVED,
    STATE_NAMES_PRIMITIVE,
    CFD2DStepDataset,
    CFD2DTrajectoryTensorDataset,
    build_cfd2d_step_dataset,
    build_cfd2d_trajectory_dataset_from_split,
    conserved_to_primitive,
    generate_cfd2d_dataset_splits,
    primitive_to_conserved,
    sample_cfd2d_initial_conditions,
    solve_cfd2d_trajectory,
)

__all__ = [
    "STATE_CHANNELS",
    "STATE_NAMES_CONSERVED",
    "STATE_NAMES_PRIMITIVE",
    "CFD2DStepDataset",
    "CFD2DTrajectoryTensorDataset",
    "build_cfd2d_step_dataset",
    "build_cfd2d_trajectory_dataset_from_split",
    "conserved_to_primitive",
    "generate_cfd2d_dataset_splits",
    "primitive_to_conserved",
    "sample_cfd2d_initial_conditions",
    "solve_cfd2d_trajectory",
]
