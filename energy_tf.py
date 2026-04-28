"""Deprecated transformer-energy-head entrypoint.

The transformer-based latent energy head was removed from the active
Navier-Stokes code path. Keep this module import-safe so stale references
fail with a clear error message instead of an ImportError from a missing
package module.
"""

from __future__ import annotations


class _RemovedTransformerComponent:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Transformer energy-head components were removed from this code base. "
            "Use grad_flow_l2.grad_flow2d.EnergyHead2D with head_type='fno' "
            "for the current spectral/local energy model."
        )


TransformerEnergyHead2D = _RemovedTransformerComponent
TransformerEnergyBlock2D = _RemovedTransformerComponent
RopeMultiheadAttention2D = _RemovedTransformerComponent

__all__ = [
    "TransformerEnergyHead2D",
    "TransformerEnergyBlock2D",
    "RopeMultiheadAttention2D",
]


def _smoke_test() -> None:
    raise RuntimeError(
        "TransformerEnergyHead2D is deprecated. The active energy head lives in "
        "grad_flow_l2.grad_flow2d.EnergyHead2D."
    )


if __name__ == "__main__":
    _smoke_test()
