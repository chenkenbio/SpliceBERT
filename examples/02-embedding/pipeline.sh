#!/bin/bash

source ../scripts/config.py

test -d ./output || mkdir ./output

# Figure 3A
bed="../data/hg19.ss-motif.for_umap.bed.gz"
genome="$hg19"
skip=""

## SpliceBERT
../scripts/fetch_embedding.py \
    $bed $skip \
    -m $SPLICEBERT \
    -g $genome -o ./output/$(basename $bed .bed.gz).SpliceBERT 2> ./output/$(basename $bed .bed.gz).SpliceBERT.log

## SpliceBERT-human
../scripts/fetch_embedding.py \
    $bed \
    -m $SPLICEBERT_HUMAN \
    --skip-donor-acceptor-umap \
    -g $genome -o ./output/$(basename $bed .bed.gz).SpliceBERT-human 2> ./output/$(basename $bed .bed.gz).SpliceBERT-human.log

## run DNABERT
for k in 3 4 5 6; do
../scripts/fetch_embedding.py \
    $bed \
    --skip-donor-acceptor-umap \
    -m $DNABERT_PREFIX/$k-new-12w-0 \
    -g $genome -o ./output/$(basename $bed .bed.gz).DNABERT$k 2> ./output/$(basename $bed .bed.gz).DNABERT$k.log
done

## run onehot
../scripts/fetch_embedding.py \
    $bed \
    -m onehot \
    -g $genome -o ./output/$(basename $bed .bed.gz).onehot 2> ./output/$(basename $bed .bed.gz).onehot.log



# Figure 3B
bed="../data/K562.SSE.hg38.for_visualization.bed.gz"
genome="$hg38"
skip="--skip-donor-acceptor-umap"

## SpliceBERT
../scripts/fetch_embedding.py \
    $bed $skip \
    -m $SPLICEBERT \
    -g $genome -o ./output/$(basename $bed .bed.gz).SpliceBERT 2> ./output/$(basename $bed .bed.gz).SpliceBERT.log

## SpliceBERT-human
../scripts/fetch_embedding.py \
    $bed \
    -m $SPLICEBERT_HUMAN \
    --skip-donor-acceptor-umap \
    -g $genome -o ./output/$(basename $bed .bed.gz).SpliceBERT-human 2> ./output/$(basename $bed .bed.gz).SpliceBERT-human.log

## run DNABERT
for k in 3 4 5 6; do
../scripts/fetch_embedding.py \
    $bed \
    --skip-donor-acceptor-umap \
    -m $DNABERT_PREFIX/$k-new-12w-0 \
    -g $genome -o ./output/$(basename $bed .bed.gz).DNABERT$k 2> ./output/$(basename $bed .bed.gz).DNABERT$k.log
done

## run onehot
../scripts/fetch_embedding.py \
    $bed \
    -m onehot \
    -g $genome -o ./output/$(basename $bed .bed.gz).onehot 2> ./output/$(basename $bed .bed.gz).onehot.log



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
