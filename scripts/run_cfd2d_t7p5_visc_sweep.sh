#!/usr/bin/env bash
set -euo pipefail

cd /scratch/jl1344/energyPDE

PY=/hpc/home/jl1344/anaconda3/bin/python
LOG=grad_flow_l2/cfd2d/datasets/logs/cfd2d_t7p5_visc_sweep.log
mkdir -p grad_flow_l2/cfd2d/datasets/logs

COMMON=(
  --n-x 64
  --n-y 64
  --solver-factor 4
  --record-dt -1
  --gamma 1.6666666666666667
  --prandtl 0.72
  --cfl 0.45
  --rho0 1.0
  --rho-amp 0.1
  --mach-min 0.1
  --mach-max 0.1
  --p-amp 0.05
  --ic-type grf
  --n-modes-min 4
  --n-modes-max 16
  --k-max 4
  --grf-ls-min 0.05
  --grf-ls-max 0.15
  --chunk-size 8
  --device cuda:3
)

: > "$LOG"
date >> "$LOG"

"$PY" -m grad_flow_l2.cfd2d.cfd_data "${COMMON[@]}" \
  --n-steps 10 --t-final 1.0 --mu 1e-8 --zeta 1e-8 \
  --n-train 1200 --n-val 300 --n-test 0 --seed 42 \
  --dataset-path grad_flow_l2/cfd2d/datasets/cfd2d_train1500_visc1e-8_M0p1.pt \
  --settings-path grad_flow_l2/cfd2d/datasets/cfd2d_train1500_visc1e-8_M0p1_settings.json \
  >> "$LOG" 2>&1
date >> "$LOG"

"$PY" -m grad_flow_l2.cfd2d.cfd_data "${COMMON[@]}" \
  --n-steps 75 --t-final 7.5 --mu 1e-8 --zeta 1e-8 \
  --n-train 0 --n-val 0 --n-test 200 --seed 2042 \
  --dataset-path grad_flow_l2/cfd2d/datasets/cfd2d_ood200_t7p5_visc1e-8_M0p1.pt \
  --settings-path grad_flow_l2/cfd2d/datasets/cfd2d_ood200_t7p5_visc1e-8_M0p1_settings.json \
  >> "$LOG" 2>&1
date >> "$LOG"

"$PY" -m grad_flow_l2.cfd2d.cfd_data "${COMMON[@]}" \
  --n-steps 10 --t-final 1.0 --mu 1e-2 --zeta 1e-2 \
  --n-train 1200 --n-val 300 --n-test 0 --seed 42 \
  --dataset-path grad_flow_l2/cfd2d/datasets/cfd2d_train1500_visc1e-2_M0p1.pt \
  --settings-path grad_flow_l2/cfd2d/datasets/cfd2d_train1500_visc1e-2_M0p1_settings.json \
  >> "$LOG" 2>&1
date >> "$LOG"

"$PY" -m grad_flow_l2.cfd2d.cfd_data "${COMMON[@]}" \
  --n-steps 75 --t-final 7.5 --mu 1e-2 --zeta 1e-2 \
  --n-train 0 --n-val 0 --n-test 200 --seed 2042 \
  --dataset-path grad_flow_l2/cfd2d/datasets/cfd2d_ood200_t7p5_visc1e-2_M0p1.pt \
  --settings-path grad_flow_l2/cfd2d/datasets/cfd2d_ood200_t7p5_visc1e-2_M0p1_settings.json \
  >> "$LOG" 2>&1
date >> "$LOG"
