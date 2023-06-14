#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

cd ../../
source venv/bin/activate

#TO_RUN=$(sed -n "101,800p" scripts/runs/runs_rnn_reruns_sweep_td.txt)

parallel --eta -u < 'scripts/runs/runs_rnn_reruns_sweep_td.txt'
#parallel --eta -u < "$TO_RUN"
