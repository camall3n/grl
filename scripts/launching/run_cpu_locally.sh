#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

N_JOBS=8

cd ../../
source venv/bin/activate

#TO_RUN=$(sed -n "101,800p" scripts/runs/runs_rnn_reruns_sweep_mc.txt)

parallel --eta -u --jobs $N_JOBS < 'scripts/runs/runs_popgym_sweep_mc_test.txt'
#parallel --eta -u < "$TO_RUN"
