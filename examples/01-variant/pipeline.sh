#!/bin/bash

source ../scripts/config.py
test -d output || mkdir output

## generate in silico mutagenesis data
if [ -e ./output/insilico_mutation_in_transcripts.pkl ]; then
    echo "found insilico_mutation_in_transcripts.pkl, skip"
else
    ../scripts/insilico_mutagenesis_in_transcripts.py &> ./output/insilico_mutation_in_transcripts.pkl.log && echo "finished: insilico_mutation_in_transcripts.pkl"
fi

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
        test -e $output && echo "found $output, skip" && continue
        ../scripts/analyze_vexseq_mfass.py \
            -d $db \
            -m $model_path \
            -o $output &> $log && echo "finished: $output"
    done
done

for k in 3 4 5 6; do
    for db in "vexseq" "mfass"; do
        output=./output/${db}.DNABERT${k}.output.pkl
        log=${output}.log
        test -e $output && echo "found $output, skip" && continue
        ../scripts/analyze_vexseq_mfass.py \
            -d $db \
            -m "$DNABERT_PREFIX/$k-new-12w-0" \
            -o $output &> $log && echo "finished: $output"
    done
done


