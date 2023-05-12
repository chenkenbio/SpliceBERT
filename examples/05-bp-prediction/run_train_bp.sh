#!/bin/bash

source ../scripts/config.py

run_script.sh -n 1 ../scripts/finetune_for_bp_prediction.py \
    -m $SPLICEBERT_510 \
    -o train_mercer_bp &> train_mercer_bp.log

cat ./train_mercer_bp/fold*/test_results.txt > train_mercer_bp.all_prediction.txt # combine predictions in test folds

# usage: finetune_for_bp_prediction.py [-h] [--dnabert] [--no-pretrain]
#                                      [--freeze-bert FREEZE_BERT] [-n N_FOLD]
#                                      -m MODEL_PATH [-b BATCH_SIZE] [-lr LR]
#                                      [--patience PATIENCE]
#                                      [--num-workers NUM_WORKERS] [-d DEVICE]
#                                      -o OUTDIR [--seed SEED]
# 
# optional arguments:
#   -h, --help            show this help message and exit
#   --dnabert
#   --no-pretrain
#   --freeze-bert FREEZE_BERT
#   -n N_FOLD, --n-fold N_FOLD
#   -m MODEL_PATH, --model-path MODEL_PATH
#   -b BATCH_SIZE, --batch-size BATCH_SIZE
#                         batch size (default: 8)
#   -lr LR                learning rate (default: 5e-05)
#   --patience PATIENCE   patience in early stopping (default: 10)
#   --num-workers NUM_WORKERS
#                         num_workers in dataloader (default: 8)
#   -d DEVICE, --device DEVICE
#                         device (default: None)
#   -o OUTDIR, --outdir OUTDIR
#                         output directory (default: None)
#   --seed SEED
