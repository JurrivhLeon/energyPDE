# 1D Viscous Burgers Experiment Report

## 1. Equation Form
We trained and evaluated on the **1D viscous Burgers equation** in conservative form:


$$\partial_t u + \partial_x\left(\frac{u^2}{2}\right) = \nu\,\partial_{xx}u,
\quad x\in[0,1],\ t\in[0,1],\ \nu=0.01.$$

Boundary/forcing assumptions used in this run:
- Homogeneous Dirichlet boundary conditions: $u(0,t)=u(1,t)=0$
- Zero external forcing term

## 2. Data and Training Setup
### Dataset
- File: `grad_flow_l2/datasets/burgers_l2_nu0p01_nx100_steps10.pt`
- Spatial grid: `n_x = 100` interior points
- Time discretization: `n_steps = 10` over `[0,1]` so `dt = 0.1` and trajectory length = 11
- Splits: train/val/test = `3000 / 500 / 500`

### Model
Hidden-space gradient-flow model (`HiddenGradientFlowModel1D`):
- Hidden channels: `64`
- Latent channels: `16`
- Encoder blocks: `4`
- Decoder blocks: `4`
- Latent dynamics blocks: `6`
- Latent energy layers: `4`

### Optimization
- Run directory: `grad_flow_l2/burgers/outputs/run_burgers_hom`
- Epochs: `200`
- Batch size: `256`
- Optimizer: `AdamW`
- Learning rate: `1e-4`
- Weight decay: `1e-5`
- Gradient clip: `1.0`
- Loss weights: `lambda_mono = 1.0`, `lambda_edi = 1.0`, `lambda_recon = 1.0`

## 3. Training Results
From `history.json`:
- Final train total loss (epoch 200): `9.0304e-06`
- Final val total loss (epoch 200): `9.8481e-06`
- Best val rollout relative L2: `5.7933e-03` (epoch 198)

### Loss Curves
![Train/Val Loss Curves](outputs/run_burgers_hom/train_val_loss_curves.png)

## 4. Test Metrics
Evaluation used checkpoint: `best_model.pt`.

- One-step test MSE: **`8.9251e-06`**
- Rollout mean relative L2 across all time steps: **`5.5856e-03`**
- Final-time (`t=1.0`, step 10) rollout MSE: **`4.1192e-05`**
- Final-time (`t=1.0`, step 10) rollout relative L2: **`8.9185e-03`**

### Rollout Error Curve
![Test Rollout Curve](outputs/run_burgers_hom/eval/test_rollout_error_curve.png)

## 5. Sample Visualizations
The following are reference/prediction comparisons with snapshot overlays.

### Sample 0
![Sample 0](outputs/run_burgers_hom/eval/test_sample_comparisons/sample_0000_comparison.png)

### Sample 166
![Sample 166](outputs/run_burgers_hom/eval/test_sample_comparisons/sample_0166_comparison.png)

### Sample 332
![Sample 332](outputs/run_burgers_hom/eval/test_sample_comparisons/sample_0332_comparison.png)

### Sample 443
![Sample 443](outputs/run_burgers_hom/eval/test_sample_comparisons/sample_0443_comparison.png)

### Sample 499
![Sample 499](outputs/run_burgers_hom/eval/test_sample_comparisons/sample_0499_comparison.png)

## 6. Summary
The hidden-space gradient-flow model fit the Burgers trajectory operator well on this dataset (`nu=0.01`, `dt=0.1`), reaching low one-step error and stable multi-step rollout error. The final-time rollout degradation is moderate and visible in the error curve, but trajectory-level predictions remain accurate across representative test samples.

---

## 7. Forced Burgers (New Run)
This section reports the newer run with nonzero sampled forcing:
- Run directory: `grad_flow_l2/burgers/outputs/run_burgers_forced`
- Dataset: `grad_flow_l2/datasets/burgers_forced_l2_nu0p01_nx100_steps10_n6000.pt`
- Eval output: `grad_flow_l2/burgers/outputs/run_burgers_forced/eval_test10`

### 7.1 Equation Form
$$
\partial_t u + \partial_x\left(\frac{u^2}{2}\right) = \nu\,\partial_{xx}u + g(x),
\quad \nu=0.01,\ x\in[0,1],\ t\in[0,1].
$$

with homogeneous Dirichlet BCs and sampled static forcing $g(x)$ per sample.

### 7.2 Training Setting
From `run_burgers_forced/args.json`:
- Splits: `4000 / 1000 / 1000`
- `n_x = 100`, `n_steps = 10` (`dt=0.1`)
- Model: hidden channels 64, latent channels 16, encoder/decoder/latent blocks = 4/4/6
- Optimization: AdamW, `lr=1e-4`, `weight_decay=1e-5`, batch size 256, 200 epochs
- Loss weights: `lambda_recon=1.0`, `lambda_mono=1.0`, `lambda_edi=1.0`

### 7.3 Forced-Case Results
From `history.json` and `eval_test10`:
- One-step test MSE: **`4.0936e-06`**
- Mean rollout relative L2 over all steps: **`3.7418e-03`**
- Final-time (`t=1.0`, step 10) rollout MSE: **`3.1334e-06`**
- Final-time (`t=1.0`, step 10) rollout relative L2: **`3.9415e-03`**
- Best val rollout relative L2: `3.7551e-03` (epoch 198)

### 7.4 Forced-Case Curves
![Forced Train/Val Loss Curves](outputs/run_burgers_forced/train_val_loss_curves.png)

![Forced Test Rollout Curve](outputs/run_burgers_forced/eval_test10/test_rollout_error_curve.png)

### 7.5 Forced-Case Sample Visualizations
The eval run generated 10 comparison samples:
- `sample_0000`, `sample_0111`, `sample_0222`, `sample_0333`, `sample_0444`
- `sample_0555`, `sample_0666`, `sample_0777`, `sample_0888`, `sample_0999`

Examples:

![Forced Sample 0](outputs/run_burgers_forced/eval_test10/test_sample_comparisons/sample_0000_comparison.png)

![Forced Sample 111](outputs/run_burgers_forced/eval_test10/test_sample_comparisons/sample_0111_comparison.png)

![Forced Sample 222](outputs/run_burgers_forced/eval_test10/test_sample_comparisons/sample_0222_comparison.png)

![Forced Sample 333](outputs/run_burgers_forced/eval_test10/test_sample_comparisons/sample_0333_comparison.png)

![Forced Sample 444](outputs/run_burgers_forced/eval_test10/test_sample_comparisons/sample_0444_comparison.png)

![Forced Sample 555](outputs/run_burgers_forced/eval_test10/test_sample_comparisons/sample_0555_comparison.png)

![Forced Sample 666](outputs/run_burgers_forced/eval_test10/test_sample_comparisons/sample_0666_comparison.png)

![Forced Sample 777](outputs/run_burgers_forced/eval_test10/test_sample_comparisons/sample_0777_comparison.png)

![Forced Sample 888](outputs/run_burgers_forced/eval_test10/test_sample_comparisons/sample_0888_comparison.png)

![Forced Sample 999](outputs/run_burgers_forced/eval_test10/test_sample_comparisons/sample_0999_comparison.png)

## 8. OOD Test: Longer Horizon $t\in[0,2]$
We evaluated the same forced-model checkpoint (`run_burgers_forced/best_model.pt`) on an out-of-distribution dataset with a longer time horizon:
- Dataset: `grad_flow_l2/datasets/burgers_forced_ood_t0to2_nu0p01_nx100_steps20_test1000.pt`
- Grid/time: `n_x=100`, `n_steps=20`, `t_final=2.0`, so `dt=0.1`
- Split evaluated: test (1000 samples)

### 8.1 OOD Metrics
- One-step test MSE: **`2.9527e-06`**
- Mean rollout relative L2 over all steps (`0..20`): **`4.2596e-03`**
- Final-time (`t=2.0`, step 20) rollout MSE: **`8.2523e-06`**
- Final-time (`t=2.0`, step 20) rollout relative L2: **`5.0667e-03`**

Compared with the in-distribution forced test (`t\in[0,1]`), the long-horizon final relative error increases (from about `3.94e-03` to `5.07e-03`), while remaining stable and low.

### 8.2 OOD Curves and Samples
![OOD Test Rollout Curve](outputs/run_burgers_forced/eval_ood_t0to2/test_rollout_error_curve.png)

OOD sample comparison plots were generated for 10 test samples in:
- `outputs/run_burgers_forced/eval_ood_t0to2/test_sample_comparisons/`
