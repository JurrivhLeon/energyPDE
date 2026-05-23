#!/usr/bin/env bash
set -euo pipefail

cd /scratch/jl1344/energyPDE

PY=/hpc/home/jl1344/anaconda3/bin/python
mkdir -p grad_flow_l2/euler1d/logs

COMMON=(
  --solver-n-x 8192
  --solver-dt 0.001
  --gamma 1.4
  --cfl 0.45
  --rho0 1.0
  --rho-amp 0.15
  --p0 1.0
  --p-amp 0.1
  --mach-min 0.05
  --mach-max 0.35
  --max-modes 5
  --decay 2.0
  --solve-batch-size 64
  --max-substeps 300000
  --device cuda:3
)

run_dataset() {
  local L="$1"
  local n_x="$2"
  local n_steps="$3"
  local t_final="$4"
  local n_train="$5"
  local n_val="$6"
  local n_test="$7"
  local seed="$8"
  local dataset_path="$9"
  local log_path="${10}"

  : > "$log_path"
  date >> "$log_path"
  "$PY" -m grad_flow_l2.euler1d.euler_data "${COMMON[@]}" \
    --domain-length "$L" \
    --n-x "$n_x" \
    --n-steps "$n_steps" \
    --t-final "$t_final" \
    --n-train "$n_train" \
    --n-val "$n_val" \
    --n-test "$n_test" \
    --seed "$seed" \
    --dataset-path "$dataset_path" \
    >> "$log_path" 2>&1
  date >> "$log_path"
}

run_dataset 5.0 512 10 1.0 1200 300 0 123 \
  grad_flow_l2/euler1d/datasets/euler1d_L5_n512_solver8192_dt0p1_t1_train1200_val300.pt \
  grad_flow_l2/euler1d/logs/euler1d_L5_trainval_n512_solver8192_dt0p1_t1.log

run_dataset 5.0 512 100 10.0 0 0 200 456 \
  grad_flow_l2/euler1d/datasets/euler1d_L5_n512_solver8192_dt0p1_t10_ood_test200.pt \
  grad_flow_l2/euler1d/logs/euler1d_L5_ood_n512_solver8192_dt0p1_t10.log

run_dataset 10.0 1024 10 1.0 1200 300 0 123 \
  grad_flow_l2/euler1d/datasets/euler1d_L10_n1024_solver8192_dt0p1_t1_train1200_val300.pt \
  grad_flow_l2/euler1d/logs/euler1d_L10_trainval_n1024_solver8192_dt0p1_t1.log

run_dataset 10.0 1024 100 10.0 0 0 200 456 \
  grad_flow_l2/euler1d/datasets/euler1d_L10_n1024_solver8192_dt0p1_t10_ood_test200.pt \
  grad_flow_l2/euler1d/logs/euler1d_L10_ood_n1024_solver8192_dt0p1_t10.log
