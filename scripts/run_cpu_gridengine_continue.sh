#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

cd ../
source venv/bin/activate

NUM_LINES=$1
IFS=',' read -r -a INDICES <<< "$NUM_LINES"

TO_RUN=$(sed -n "${INDICES[SGE_TASK_ID - 1]}p" scripts/runs/runs_pomdps_mi_pi_extra_it.txt)
echo "INDEX: ${INDICES[SGE_TASK_ID - 1]}, TASK_INDEX: ${SGE_TASK_ID}"
echo "Running script: ${TO_RUN}"
eval $TO_RUN
