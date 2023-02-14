cd ../../

onager launch \
    --backend gridengine \
    --jobname tmaze_sweep_alpha \
    --mem 1 \
    --cpus 2 \
    --duration 0-03:00:00 \
    --venv venv \
    -q '*@@mblade12'\
