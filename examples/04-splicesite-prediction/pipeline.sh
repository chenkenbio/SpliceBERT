#!/bin/bash

test -d ./output || mkdir ./output
source ../scripts/config.py

model=$SPLICEBERT_510
prefix="finetune_splicebert_on_spliceator"
batch_size=16


for group in "donor" "acceptor"; do
    run_name="./output/${prefix}_GS-GS_1_${group}_cv"
    test -e ${run_name}.log && continue
    ./train_splicebert_cv.py \
        -lr 0.00001 \
        -m ${model} \
        -b ${batch_size} \
        -p ../data/spliceator/Training_data/Positive/GS/POS_${group}_600.csv \
        -n ../data/spliceator/Training_data/Negative/GS/GS_1/NEG_600_${group}.csv \
        -o ${run_name} &> ${run_name}.log
done


for ss_type in "donor" "acceptor"; do
    for fold in `seq 0 9`; do
        weight="./output/${prefix}_GS-GS_1_${ss_type}_cv/fold${fold}/checkpoint.pt.best_model.pt"
        if [ -e $weight ]; then
            echo "run model: $weight"
        else
            echo "missing $weight, skip"
            continue
        fi

        for species in Danio Fly Worm Thaliana; do
            outdir="./output/${prefix}_GS-GS_1_${ss_type}_cv/fold${fold}/benchmark_${species}"
            mkdir -p $outdir
            ./predict_by_splicebert.py \
                -w $weight \
                -p "../data/spliceator/Benchmarks/${species}/SA_sequences_${ss_type}_400_Final_3.positive.txt" \
                -n "../data/spliceator/Benchmarks/${species}/SA_sequences_${ss_type}_400_Final_3.negative.txt" \
                -o $outdir &> ${outdir}.log
        done
        wait 
    done
done

