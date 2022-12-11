#!/usr/bin/env bash

QUEUE=inf
NUM_CPUS=2
#NUM_LINES=1-160
NUM_LINES=11,12,13,14,15,16,17,18,19,20,71,72,73,74,75,76,77,78,79,80,91,92,93,94,95,96
AVAIL_MEM=1G

mkdir -p ~/logs/outputs/
mkdir -p ~/logs/errors/

qsub -t $NUM_LINES \
     -l vf=$AVAIL_MEM \
     -l $QUEUE \
     -m abe \
     -cwd \
     -o ~/logs/outputs/ \
     -e ~/logs/errors/ \
     -q '*@@mblade12' \
     -pe smp $NUM_CPUS \
     ./run_cpu_gridengine.sh