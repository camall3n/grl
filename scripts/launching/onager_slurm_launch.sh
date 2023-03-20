cd ../../

onager launch \
    --backend slurm \
    --jobname tmaze_sweep_eps_lambda1 \
    --tasks-per-node 4 \
    --mem 3 \
    --cpus 5 \
    --duration 0-03:00:00 \
    --venv venv \
