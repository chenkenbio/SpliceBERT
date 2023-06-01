#!/bin/bash

source ../scripts/config.py

test -d ./output || mkdir ./output

n=1000
# for bedpe in ./hg19.*${n}.bedpe; do
for bedpe in ../data/hg19.*${n}_per_group.bedpe.gz; do
    echo $bedpe
    bn=`basename $bedpe .bedpe.gz`
    ../scripts/fetch_attention_bedpe.py \
        $bedpe \
        -m $SPLICEBERT \
        -g $hg19 \
        -o ./output/${bn}.att.v2.pkl &> ./output/${bn}.att.v2.pkl.log
done

bedpe="../data/gencode.v41lift37.MANE.intron.for_attention.bedpe.gz"
bn=`basename $bedpe .bedpe.gz`
../scripts/fetch_attention_bedpe.py \
    $bedpe \
    -m $SPLICEBERT \
    -g $hg19 \
    -o ./output/${bn}.att.v2.pkl &> ./output/${bn}.att.v2.pkl.log

