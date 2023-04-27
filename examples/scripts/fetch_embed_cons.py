#!/usr/bin/env python3
"""
Author: Ken Chen
Email: chenkenbio@gmail.com
Date: 2022-11-24
"""

import os
import sys
import json
import pickle
import gzip
from glob import glob
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
from importlib import reload
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
new_rc_params = {'text.usetex': False, 'svg.fonttype': 'none' }
plt.rcParams.update(new_rc_params)

import h5py
import pyBigWig
import time
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from transformers import BertForMaskedLM
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from scipy.stats import pearsonr, spearmanr
from biock.pytorch import set_seed
from biock.plot._plot import hide_spines
import scanpy as sc
import anndata as ad
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse

import analysis_embedding

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--num-seqs", type=int, default=1000, help="number of randomly selected sequences")
    p.add_argument('-o', "--output", default="./embed_cons.pkl")
    p.add_argument('--seed', type=int, default=2020)
    return p

SPLICEBERT = {
    "human":"/home/chenken/Documents/github/SpliceBERT/models/SpliceBERT-human.510nt",
    "vertebrate": "/home/chenken/Documents/github/SpliceBERT/models/SpliceBERT.510nt",
}



if __name__ == "__main__":
    args = get_args().parse_args()

    set_seed(args.seed)

    ds = analysis_embedding.GenomicRegionData()
    subset = np.random.choice(np.arange(len(ds)), size=args.num_seqs, replace=False)
    loader = DataLoader(Subset(ds, indices=subset), batch_size=8, num_workers=8, shuffle=False)


    splicebert = BertForMaskedLM.from_pretrained(SPLICEBERT["vertebrate"])
    human_model = BertForMaskedLM.from_pretrained(SPLICEBERT["human"])
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    device = torch.device("cuda")
    splicebert = splicebert.to(device)
    human_model = human_model.to(device)
    splicebert.eval()
    human_model.eval()


    vert_scores = {'ids': list(), 'h': list(), 'cons': list()}
    human_scores = {'ids': list(), 'h': list(), 'cons': list()}

    splicebert_results = {
        "ids_extend": list(),
        "splicebert": list(),
        "splicebert-human": list(),
        "phastcons": list(),
        "phylop": list(),
        "annotation": list()
    }

    with torch.no_grad(), autocast():
        for it, (ids, masked_ids, ids_extend, label, phastcons, phylop, annotation) in enumerate(tqdm(loader)):
            splicebert_results["ids_extend"].append(ids_extend[:, 1:-1].numpy())
            splicebert_results["phastcons"].append(phastcons.numpy())
            splicebert_results["phylop"].append(phylop.numpy())
            splicebert_results["annotation"].append(annotation.numpy())

            ids = ids.to(device)
            h1 = splicebert.forward(ids).hidden_states[-1].detach().cpu()[:, 1:-1].numpy()
            h2 = human_model.forward(ids).hidden_states[-1].detach().cpu()[:, 1:-1].numpy()
            del ids
            splicebert_results["splicebert"].append(h1)
            splicebert_results["splicebert-human"].append(h2)

    for k, v in splicebert_results.items():
        splicebert_results[k] = np.concatenate(v)
    pickle.dump(splicebert_results, open(args.output, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    #         dim = vert_states.shape[-1]
    #         vert_states = vert_states.numpy().reshape(-1, dim)
    #         human_states = human_states.numpy().reshape(-1, dim)
    #         cons = cons.numpy().reshape(-1)
    #         keep = np.where(~np.isnan(cons))[0]
    #         cons = cons[keep]
    #         vert_states = vert_states[keep]
    #         human_states = human_states[keep]
    #         vert_scores['h'].append(vert_states)
    #         human_scores['h'].append(human_states)
    #         vert_scores['cons'].append(cons)
    #         human_scores['cons'].append(cons)



    # vert_scores['cons'] = np.concatenate(vert_scores['cons'])
    # vert_scores['h'] = np.concatenate(vert_scores['h'])
    # human_scores['cons'] = np.concatenate(human_scores['cons'])
    # human_scores['h'] = np.concatenate(human_scores['h'])


    # pickle.dump((vert_scores, human_scores), open(args.output, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)







