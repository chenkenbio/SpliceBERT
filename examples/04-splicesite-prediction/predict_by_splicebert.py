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

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-w', "--weight", required=True)
    p.add_argument('-p', "--positive", required=True)
    p.add_argument('-n', "--negative", required=True)
    p.add_argument('-o', "--outdir", required=True)
    p.add_argument("-b", "--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", 8)))
    p.add_argument("--debug", action="store_true", help="debug mode")
    return p

@torch.no_grad()
@autocast()
def test_model(model: AutoModelForSequenceClassification, loader: DataLoader):
    r"""
    Return:
    auc : float
    f1 : float
    pred : list
    true : list
    """
    model.eval()
    pred, true = list(), list()
    for it, (ids, mask, label) in enumerate(tqdm(loader, desc="predicting", total=len(loader))):
        ids = ids.to(device)
        mask = mask.to(device)
        # score = torch.softmax(model.forward(ids, attention_mask=mask).logits, dim=1)[:, 1].detach().cpu().numpy()
        score = torch.sigmoid(model.forward(ids, attention_mask=mask).logits.squeeze(1)).detach().cpu().numpy()
        del ids
        label = label.numpy()
        pred.append(score.astype(np.float16))
        true.append(label.astype(np.float16))
    pred = np.concatenate(pred)
    true = np.concatenate(true)
    auc_list = roc_auc_score(true.T, pred.T)
    f1 = f1_score(true.T, pred.T > 0.5)
    return auc_list, f1, pred, true

if __name__ == "__main__":
    args = get_args().parse_args()

    args.outdir = make_directory(args.outdir)
    logger = make_logger(filename=os.path.join(args.outdir, "train.log"), level="DEBUG" if args.debug else "INFO")
    logger.info(get_run_info(argv=sys.argv, args=args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(SPLICEBERT_510)

    ds = spliceator_data.SpliceatorDataset(
        positive=args.positive,
        negative=args.negative,
        tokenizer=tokenizer,
        max_len=400
    )

    train_loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=min(args.batch_size, os.cpu_count()),
    )

    model = AutoModelForSequenceClassification.from_pretrained(SPLICEBERT_510, num_labels=1).to(device)
    model.load_state_dict(torch.load(args.weight, map_location="cpu")["model"])
    model.eval()

    ## train
    pbar = tqdm(train_loader, total=len(train_loader))
    with torch.no_grad():
        all_scores, all_labels = list(), list()
        for it, (ids, mask, label) in enumerate(pbar):
            ids, mask = ids.to(device), mask.to(device)
            with autocast():
                logits = model.forward(ids, attention_mask=mask).logits.squeeze(1)
                all_scores.append(logits.detach().cpu().numpy())
                all_labels.append(label.detach().cpu().numpy())
    
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    pickle.dump((all_labels, all_scores), open("{}/results.pkl".format(args.outdir), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    auc_score = roc_auc_score(all_labels, all_scores)
    ap_score = average_precision_score(all_labels, all_scores)
    f1 = f1_score(all_labels, all_scores > 0.5)

    logger.info("AUC/AUPR/F1: {:.4f} {:.4f} {:.4f}".format(auc_score, ap_score, f1))

