"""
Trainer alias for 2D latent gradient-flow models.

For Cahn-Hilliard experiments we reuse the same hidden-space trainer
implementation used in the Navier-Stokes benchmark.
"""

from __future__ import annotations

try:
    from ..navier_stokes2d.trainer import HiddenGradientFlowTrainer2D
except ImportError:
    from grad_flow_l2.navier_stokes2d.trainer import HiddenGradientFlowTrainer2D

__all__ = ["HiddenGradientFlowTrainer2D"]

