cd ../../

onager launch \
    --backend gridengine \
    --jobname all_pomdps_mi_pi_flip_count \
    --mem 1 \
    --cpus 2 \
    --duration 0-03:00:00 \
    --venv venv \
    -q '*@@mblade12'\
