#!/usr/bin/env bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2

cd ../
source venv/bin/activate
parallel --eta --jobs 4 -u < 'scripts/runs/runs_pomdps_mi_pi.txt'
