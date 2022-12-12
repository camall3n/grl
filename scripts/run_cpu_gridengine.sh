#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

cd ../
source venv/bin/activate

TO_RUN=$(sed -n "${SGE_TASK_ID}p" scripts/runs/runs_pomdps_mi_pi_extra_it.txt)
#eval $TO_RUN
echo $TO_RUN
