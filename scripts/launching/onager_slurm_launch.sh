cd ../../

onager launch \
    --backend slurm \
    --jobname tmaze_sweep_junction_pi_uniform \
    --mem 1 \
    --cpus 2 \
    --duration 0-01:00:00 \
    --venv venv \
