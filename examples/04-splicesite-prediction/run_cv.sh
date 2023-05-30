#!/bin/bash

model="../../models/SpliceBERT.510nt/"

prefix="finetune_splicebert_on_spliceator"
for group in donor acceptor; do
    run_name="${prefix}_${pos_data}-${neg_data}_${group}_cv"
    test -e ${run_name}.log && continue
    ./train_splicebert_cv.py \
        -lr 0.00001 \
        -m ${model} \
        -p ./data/spliceator/Training_data/Positive/GS/POS_${group}_600.csv \
        -n ./data/spliceator/Training_data/Negative/GS/GS_1/NEG_600_${group}.csv \
        -o ${run_name} &> ${run_name}.log
done