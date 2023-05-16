cd ../../

onager launch \
    --backend slurm \
    --jobname final_hallway_analytical \
    --mem 12 \
    --cpus 1 \
    --duration 0-03:00:00 \
    --venv venv \
    --gpus 1 \
    -q 3090-gcondo

#    --tasks-per-node 4 \
