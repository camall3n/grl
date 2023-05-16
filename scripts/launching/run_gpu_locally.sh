#!/usr/bin/env bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

cd ../../
source venv/bin/activate
parallel --eta --jobs 1 -u < 'scripts/runs/runs_final_hallway_analytical.txt'
