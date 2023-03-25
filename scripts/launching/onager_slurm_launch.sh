cd ../../

onager launch \
    --backend slurm \
    --jobname sample_based_tmaze_sweep_eps \
    --mem 3 \
    --cpus 3 \
    --duration 1-00:00:00 \
    --venv venv \
#    --tasks-per-node 4 \
