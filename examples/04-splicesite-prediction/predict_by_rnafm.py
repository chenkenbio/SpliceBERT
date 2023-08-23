#!/usr/bin/env python3

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.append("../scripts")
from utils import make_directory, make_logger, get_run_info
from config import SPLICEBERT_510
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.cuda.amp import autocast
import spliceator_data
sys.path.append("../../related/RNA-FM/")
import fm
from train_rnafm_cv import RnafmForSequenceClassification


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-w', "--weight", required=True)
    p.add_argument('-p', "--positive", required=True)
    p.add_argument('-n', "--negative", required=True)
    p.add_argument('-o', "--outdir", required=True)
    p.add_argument("-b", "--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", 8)))
    p.add_argument("--debug", action="store_true", help="debug mode")
    return p


if __name__ == "__main__":
    args = get_args().parse_args()

    args.outdir = make_directory(args.outdir)
    logger = make_logger(filename=os.path.join(args.outdir, "train.log"), level="DEBUG" if args.debug else "INFO")
    logger.info(get_run_info(argv=sys.argv, args=args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = spliceator_data.SpliceatorDataset(
        positive=args.positive,
        negative=args.negative,
        tokenizer=None,
        max_len=400,
        rnafm_mode=True
    )

    train_loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=min(args.batch_size, os.cpu_count()),
        collate_fn=ds.collate_fn,
    )

    rnafm, alphabet = fm.pretrained.rna_fm_t12()
    batch_converter = alphabet.get_batch_converter()
    model = RnafmForSequenceClassification(rnafm, num_labels=1).to(device)
    model.load_state_dict(torch.load(args.weight, map_location="cpu")["model"])
    model.eval()

    ## train
    pbar = tqdm(train_loader, total=len(train_loader))
    with torch.no_grad():
        all_scores, all_labels = list(), list()
        for it, (ids, mask, label) in enumerate(pbar):
            ids = batch_converter(ids)[-1]
            ids = ids.to(device)
            with autocast():
                logits = model.forward(ids).logits.squeeze(1)
                all_scores.append(logits.detach().cpu().numpy())
                all_labels.append(label.detach().cpu().numpy())
    
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    pickle.dump((all_labels, all_scores), open("{}/results.pkl".format(args.outdir), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    auc_score = roc_auc_score(all_labels, all_scores)
    ap_score = average_precision_score(all_labels, all_scores)
    f1 = f1_score(all_labels, all_scores > 0.5)

    logger.info("AUC/AUPR/F1: {:.4f} {:.4f} {:.4f}".format(auc_score, ap_score, f1))

