cd ../../

onager launch \
    --backend gridengine \
    --jobname pomdps_mi_pi_q_abs_extra_it \
    --mem 1 \
    --cpus 8 \
    --duration 3-00:00:00 \
    --venv venv \
    -q '*@@mblade12'\
