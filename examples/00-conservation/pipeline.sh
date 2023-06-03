#!/bin/bash

if [ $# -lt 1 ]; then
    echo "usage: $0 [ acc | cons ]"
    exit 1
fi

step="$1"

if [ "$step" == "acc" ]; then
    ../scripts/evaluate_mlm_per_region.py
elif [ "$step" == "cons" ]; then
    ../scripts/fetch_embed_cons.py
else
    echo "unknown step: $step, must be one of 'acc' and 'cons'"
    exit 1
fi
