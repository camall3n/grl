cd ../../

onager launch \
    --backend slurm \
    --jobname tiger_grid_mi_pi_obs_space \
    --mem 12 \
    --cpus 1 \
    --duration 0-06:00:00 \
    --venv venv \
    --gpus 1 \
    -q 3090-gcondo

#    --tasks-per-node 4 \
