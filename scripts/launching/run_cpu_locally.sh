#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

cd ../../
source venv/bin/activate
parallel --eta -u < 'scripts/runs/runs_tmaze_sweep_junction_pi.txt'
