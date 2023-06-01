#!/bin/bash

working_dir=$(realpath `dirname $0`)

echo "working_dir: $working_dir"

## prepare input files
# wget -o $working_dir/data/hg19.fa.gz -c https://hgdownload.cse.ucsc.edu/goldenpath/hg19/bigZips/hg19.fa.gz


cd $working_dir/scripts
python3 ./setup.py build_ext --inplace

