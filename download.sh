#!/bin/bash


echo "Downloading the data to ./examples/ ..."
wget -c -O ./examples/data.tar.gz https://zenodo.org/record/7995778/files/data.tar.gz?download=1 && cd examples && tar -xzvf data.tar.gz && cd .. && echo "Done"

echo "Downloading the model weights ..."
wget -c -O models.tar.gz https://zenodo.org/record/7995778/files/models.tar.gz?download=1 && tar -xzvf models.tar.gz && echo "Done"


## check dnabert
echo "Preparing the DNABERT weights ..."
test -d ./models/dnabert || mkdir -p ./models/dnabert
cd ./models/dnabert
for k in 3 4 5 6; do
    if [ ! -e "${k}-new-12w-0" ]; then
        if [ -e "${k}-new-12w-0.zip" ]; then
            unzip "${k}-new-12w-0.zip" && echo "unzip: ${k}-new-12w-0.zip -> ${k}-new-12w-0"
        else
            echo "NOTE: Users should manually download the weights of DNABERT${k} from https://github.com/jerryji1993/DNABERT and decompress it to ./models/dnabert/"
        fi
    fi
done
