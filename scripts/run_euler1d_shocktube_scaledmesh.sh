#!/usr/bin/env bash
set -euo pipefail

cd /scratch/jl1344/energyPDE
mkdir -p grad_flow_l2/euler1d/datasets grad_flow_l2/euler1d/logs

run_dataset() {
  local domain_length="$1"
  local label="$2"
  local n_x="$3"
  local device="$4"
  local split="$5"
  local n_steps="$6"
  local t_final="$7"
  local n_train="$8"
  local n_val="$9"
  local n_test="${10}"
  local seed="${11}"
  local output="${12}"
  local log="${13}"

  echo "[$(date +%F_%T)] Starting shocktube ${label} n${n_x} ${split} on ${device}"
  /hpc/home/jl1344/anaconda3/bin/python -m grad_flow_l2.euler1d.euler_data \
    --dataset-path "${output}" \
    --n-x "${n_x}" \
    --solver-n-x "${n_x}" \
    --n-steps "${n_steps}" \
    --t-final "${t_final}" \
    --domain-length "${domain_length}" \
    --solver-dt 0.001 \
    --gamma 1.4 \
    --cfl 0.45 \
    --ic-type shocktube \
    --boundary-condition outflow \
    --x0-middle-fraction 0.2 \
    --solve-batch-size 64 \
    --max-substeps 300000 \
    --n-train "${n_train}" \
    --n-val "${n_val}" \
    --n-test "${n_test}" \
    --seed "${seed}" \
    --device "${device}" 2>&1 | tee "${log}"
  echo "[$(date +%F_%T)] Finished shocktube ${label} n${n_x} ${split}"
}

run_pair() {
  local domain_length="$1"
  local label="$2"
  local n_x="$3"
  local device="$4"

  run_dataset "${domain_length}" "${label}" "${n_x}" "${device}" trainval 15 1.5 1200 300 0 123 \
    "grad_flow_l2/euler1d/datasets/euler1d_shocktube_${label}_n${n_x}_solver${n_x}_dt0p1_t1p5_train1200_val300.pt" \
    "grad_flow_l2/euler1d/logs/euler1d_shocktube_${label}_trainval_n${n_x}_solver${n_x}_dt0p1_t1p5.log"

  run_dataset "${domain_length}" "${label}" "${n_x}" "${device}" ood 100 10.0 0 0 200 456 \
    "grad_flow_l2/euler1d/datasets/euler1d_shocktube_${label}_n${n_x}_solver${n_x}_dt0p1_t10_ood_test200.pt" \
    "grad_flow_l2/euler1d/logs/euler1d_shocktube_${label}_ood_n${n_x}_solver${n_x}_dt0p1_t10.log"
}

run_pair 2.5 L2p5 512 cuda:0 &
pid_l2p5=$!
run_pair 5.0 L5 1024 cuda:1 &
pid_l5=$!
run_pair 10.0 L10 2048 cuda:2 &
pid_l10=$!

wait "${pid_l2p5}" "${pid_l5}" "${pid_l10}"
echo "[$(date +%F_%T)] Shocktube scaled-mesh Euler 1D datasets completed."
