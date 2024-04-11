#!/usr/bin/env bash
#export XLA_PYTHON_CLIENT_MEM_FRACTION=0.22

cd ../../
source venv/bin/activate
#parallel --eta --jobs 4 -u < 'scripts/runs/runs_memoryless_hallway_analytical.txt'
parallel --eta --jobs 1 -u < 'scripts/runs/runs_batch_run_pg.txt'
