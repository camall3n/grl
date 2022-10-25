#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

cd ../
source venv/bin/activate
parallel -u < 'scripts/runs/runs_tmaze_mi_pi.txt'
