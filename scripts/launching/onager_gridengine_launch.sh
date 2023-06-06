cd ../../

onager launch \
    --backend gridengine \
    --jobname tmaze_sweep_eps_leaky \
    --tasks-per-node 4 \
    --mem 3 \
    --cpus 5 \
    --duration 0-03:00:00 \
    --venv venv \
    -q '*@@mblade12'\
