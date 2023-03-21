cd ../../

onager launch \
    --backend slurm \
    --jobname tmaze2_sweep_eps \
    --tasks-per-node 4 \
    --mem 3 \
    --cpus 5 \
    --duration 0-01:00:00 \
    --venv venv \
