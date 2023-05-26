#!/usr/bin/env bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

cd ../../
source venv/bin/activate
parallel --eta --jobs 1 -u < 'scripts/runs/runs_tiger_grid_mi_pi_obs_space.txt'
