# grad_flow_l2

Gradient-flow based learning for 1D parabolic PDEs (heat equation) in L2 space.

## Scripts

- `data.py`: generate cached train/val/test datasets and sample visualizations
- `train.py`: train a model from a prepared dataset
- `eval.py`: evaluate one-step/test rollout errors and create prediction-vs-reference plots
- `generator.py`: model definitions (`ProximalMap1D`, `EnergyHead1D`, `GradientFlowModel`)
- `trainer.py`: training loop and losses
- `utils.py`: finite-difference solver and helper utilities

## Quick Start

Generate data:

```bash
python -m grad_flow_l2.data --help
```

Train:

```bash
python -m grad_flow_l2.train --help
```

Evaluate:

```bash
python -m grad_flow_l2.eval --help
```
