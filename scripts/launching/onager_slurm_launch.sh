cd ../../

onager launch \
    --backend slurm \
    --jobname rnn_rocksample_sweep_td \
    --mem 7 \
    --cpus 1 \
    --duration 0-9:00:00 \
    --venv venv \
#    --tasks-per-node 4 \
