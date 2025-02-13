#!/usr/bin/env bash

CONFIG=$1
GPUS=$2

PORT=${PORT:-29501}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
nohup python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py --config $CONFIG --launcher pytorch ${@:3} > log.txt 2>&1 &
