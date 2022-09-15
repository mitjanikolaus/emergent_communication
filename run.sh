#!/bin/bash

set -e

source activate emergent_communication

echo "Running on $(hostname)"
echo "$@"

CMD="python -u train.py "$@
eval $CMD
