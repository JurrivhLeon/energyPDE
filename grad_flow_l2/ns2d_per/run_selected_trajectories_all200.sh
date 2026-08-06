#!/usr/bin/env bash
set -euo pipefail

cd /scratch/jl1344/energyPDE

LOG_DIR="grad_flow_l2/ns2d_per/results/selected_trajectory_plots"
mkdir -p "${LOG_DIR}"

INDICES=$(/hpc/home/jl1344/anaconda3/bin/python -c 'print(",".join(map(str, range(200))))')

python3 -m grad_flow_l2.ns2d_per.plot_selected_trajectories \
  --viscosities 4 5 \
  --forcings grf sinusoidal \
  --sample-indices "${INDICES}" \
  2>&1 | tee "${LOG_DIR}/all200_generation.log"
