cd ../../

onager launch \
    --backend slurm \
    --jobname tmaze_sweep_junction_pi_leaky \
    --tasks-per-node 2 \
    --mem 2 \
    --cpus 4 \
    --duration 0-01:00:00 \
    --venv venv \
