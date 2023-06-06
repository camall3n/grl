#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

cd ../../
source venv/bin/activate
parallel --eta -u < 'scripts/runs/runs_tiger_grid_mi_pi_obs_space.txt'
