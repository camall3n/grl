#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

cd ../
source venv/bin/activate
parallel --eta -u < 'scripts/runs/runs_pomdps_mi_pi_q_l2.txt'
