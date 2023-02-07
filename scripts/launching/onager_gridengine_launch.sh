cd ../../

onager launch \
    --backend gridengine \
    --jobname tmaze_sweep_junction_q_l2 \
    --mem 1 \
    --cpus 2 \
    --duration 0-03:00:00 \
    --venv venv \
    -q '*@@mblade12'\
