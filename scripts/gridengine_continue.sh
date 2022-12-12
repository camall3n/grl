#!/usr/bin/env bash

QUEUE=inf
NUM_CPUS=2
NUM_LINES=11,12,13,14,15,16,17,18,19,20,71,72,73,74,75,76,77,78,79,80,91,92,93,94,95,96
IFS=',' read -r -a INDICES <<< "$NUM_LINES"
AVAIL_MEM=1G

mkdir -p ~/logs/outputs/
mkdir -p ~/logs/errors/

#echo "length: ${#INDICES[@]}"
#for element in "${INDICES[@]}"
#do
#    echo "$element"
#done
qsub -t 1-${#INDICES[@]} \
     -l vf=$AVAIL_MEM \
     -l $QUEUE \
     -cwd \
     -o ~/logs/outputs/ \
     -e ~/logs/errors/ \
     -q '*@@mblade12' \
     -m abe \
     -pe smp $NUM_CPUS \
     ./run_cpu_gridengine_continue.sh $NUM_LINES