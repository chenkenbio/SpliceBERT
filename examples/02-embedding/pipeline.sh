#!/bin/bash

source ../scripts/config.py

test -d ./output || mkdir ./output

ss_bed="../data/hg19.ss-motif.for_umap.bed.gz"
sse_bed="../data/K562.SSE.hg38.for_visualization.bed.gz"

if [ ! -e $ss_bed ] || [ ! -e $sse_bed ]; then
    test -e $ss_bed || echo "Missing file $ss_bed"
    test -e $sse_bed || echo "Missing file $sse_bed"
fi


## Figure 3A
bed=$ss_bed
genome="$hg19"
skip=""

## SpliceBERT
../scripts/fetch_embedding.py \
    $bed $skip \
    -m $SPLICEBERT_510 \
    -g $genome -o ./output/$(basename $bed .bed.gz).SpliceBERT &> ./output/$(basename $bed .bed.gz).SpliceBERT.log

## SpliceBERT-human
../scripts/fetch_embedding.py \
    $bed \
    -m $SPLICEBERT_HUMAN \
    --skip-donor-acceptor-umap \
    -g $genome -o ./output/$(basename $bed .bed.gz).SpliceBERT-human &> ./output/$(basename $bed .bed.gz).SpliceBERT-human.log

## run DNABERT
for k in 3 4 5 6; do
../scripts/fetch_embedding.py \
    $bed \
    --skip-donor-acceptor-umap \
    -m $DNABERT_PREFIX/$k-new-12w-0 \
    -g $genome -o ./output/$(basename $bed .bed.gz).DNABERT$k &> ./output/$(basename $bed .bed.gz).DNABERT$k.log
done

## run onehot
../scripts/fetch_embedding.py \
    $bed \
    -m onehot \
    -g $genome -o ./output/$(basename $bed .bed.gz).onehot &> ./output/$(basename $bed .bed.gz).onehot.log



## Figure 3B
bed=$sse_bed
genome="$hg38"
skip="--skip-donor-acceptor-umap"

## SpliceBERT
../scripts/fetch_embedding.py \
    $bed $skip \
    -m $SPLICEBERT_510 \
    -g $genome -o ./output/$(basename $bed .bed.gz).SpliceBERT &> ./output/$(basename $bed .bed.gz).SpliceBERT.log

## SpliceBERT-human
../scripts/fetch_embedding.py \
    $bed \
    -m $SPLICEBERT_HUMAN \
    --skip-donor-acceptor-umap \
    -g $genome -o ./output/$(basename $bed .bed.gz).SpliceBERT-human &> ./output/$(basename $bed .bed.gz).SpliceBERT-human.log

## run DNABERT
for k in 3 4 5 6; do
../scripts/fetch_embedding.py \
    $bed \
    --skip-donor-acceptor-umap \
    -m $DNABERT_PREFIX/$k-new-12w-0 \
    -g $genome -o ./output/$(basename $bed .bed.gz).DNABERT$k &> ./output/$(basename $bed .bed.gz).DNABERT$k.log
done

## run onehot
../scripts/fetch_embedding.py \
    $bed \
    -m onehot \
    -g $genome -o ./output/$(basename $bed .bed.gz).onehot &> ./output/$(basename $bed .bed.gz).onehot.log



# usage: fetch_embedding.py [-h] [--skip-donor-acceptor-umap] -m MODEL -g GENOME
#                           -o OUTPUT
#                           bed
# 
# positional arguments:
#   bed                   bed file
# 
# optional arguments:
#   -h, --help            show this help message and exit
#   --skip-donor-acceptor-umap
#                         skip layer umap (default: False)
#   -m MODEL, --model MODEL
#                         model path (default: None)
#   -g GENOME, --genome GENOME
#                         genome path (default: None)
#   -o OUTPUT, --output OUTPUT
#                         output file (default: None)
