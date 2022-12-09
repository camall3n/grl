#!/usr/bin/env bash

QUEUE=day
NUM_LINES=160
AVAIL_MEM=1G

mkdir -p ~/logs/outputs/
mkdir -p ~/logs/errors/

qsub -t 1-$NUM_LINES \
     -l vf=$AVAIL_MEM \
     -l $QUEUE \
     -m abe \
     -cwd \
     -o ~/logs/outputs/ \
     -e ~/logs/errors/ \
     -q '*@@mblade12' \
     ./run_cpu_gridengine.sh