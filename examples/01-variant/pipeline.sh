#!/bin/bash

source ../scripts/config.py
test -d output || mkdir output

## generate in silico mutagenesis data
./insilico_mutagenesis_in_transcripts.py &> ./output/insilico_mutation_in_transcripts.pkl.log

## run SpliceBERT/DNABERT on VexSeq/MFASS data
for mtype in vert human; do
    for db in "vexseq" "mfass"; do
        if [ "$mtype" = "vert" ]; then
            output=./output/${db}.SpliceBERT.output.pkl
            model_path=$SPLICEBERT_510
        else
            output=./output/${db}.SpliceBERT-human.output.pkl
            model_path=$SPLICEBERT_HUMAN
        fi
        log=${output}.log
        test -e $log && echo "skip $output" && continue
        ../scripts/analyze_vexseq_mfass.py \
            -d $db \
            -m $model_path \
            -o $output &> $log
    done
done

for k in 3 4 5 6; do
    for db in "vexseq" "mfass"; do
        output=./output/${db}.DNABERT${k}.output.pkl
        log=${output}.log
        test -e $log && echo "skip $output" && continue
        ../scripts/analyze_vexseq_mfass.py \
            -d $db \
            -m "$DNABERT_PREFIX/$k-new-12w-0" \
            -o $output &> $log
    done
done


