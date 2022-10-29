#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

cd ../
source venv/bin/activate
parallel -u < 'scripts/runs/runs_slippery_tmaze_two_thirds_up_mem_grad.txt'
