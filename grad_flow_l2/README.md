# grad_flow_l2

Gradient-flow based learning for 1D parabolic PDEs (heat equation) in L2 space.

Now also includes 2D benchmarks:
- 2D Navier-Stokes (existing latent gradient-flow setup)
- 2D Cahn-Hilliard with homogeneous Neumann BC (new)

Periodic Navier-Stokes data generation is available via:
- `python -m grad_flow_l2.navier_stokes2d_per_data --help`
- `python -m grad_flow_l2.navier_stokes2d_per_solver --help`
- `python -m grad_flow_l2.ns2d_per.train --help`
- `python -m grad_flow_l2.ns2d_per.eval --help`
- `python -m grad_flow_l2.ns2d_per.eval_edi --help`

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

## 2D Cahn-Hilliard Benchmark

Generate data:

```bash
python -m grad_flow_l2.cahn_hilliard2d_data --help
```

Train:

```bash
python -m grad_flow_l2.cahn_hilliard2d.train --help
```

Evaluate:

```bash
python -m grad_flow_l2.cahn_hilliard2d.eval --help
```

Energy-head-only evaluation:

```bash
python -m grad_flow_l2.cahn_hilliard2d.eval_energy_head --help
```
