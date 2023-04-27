#!/bin/bash

source ../scripts/config.py

# n=1000
# for bedpe in ./hg19.*${n}_per_group.bedpe.gz; do
#     echo $bedpe
# # for bedpe in ./hg19.*${n}.bedpe; do
#     bn=`basename $bedpe .bedpe`
#     run_script.sh -n 1 ../scripts/fetch_attention_bedpe.py $bedpe -m $SPLICEBERT -g $hg19 -b 4 -o ${bn}.att.v2.pkl &> ${bn}.att.v2.pkl.log
# done

bn=gencode.v41lift37.MANE.intron.for_attention
run_script.sh -n 1 ../scripts/fetch_attention_bedpe.py ./gencode.v41lift37.MANE.intron.for_attention.bedpe -m $SPLICEBERT -g $hg19 -b 4 -o ${bn}.att.v2.pkl &> ${bn}.att.v2.pkl.log

