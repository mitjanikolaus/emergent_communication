#!/bin/bash

module purge

module load cpuarch/amd

module load python
conda activate emergent_communication

set -x

echo "Running on $(hostname)"
echo "$@"

CMD="python -u train.py "$@
eval $CMD
