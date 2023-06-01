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
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from importlib import reload
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
new_rc_params = {'text.usetex': False, 'svg.fonttype': 'none' }
plt.rcParams.update(new_rc_params)

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from transformers import AutoModelForMaskedLM
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from utils import set_seed
from config import SPLICEBERT_510, SPLICEBERT_HUMAN

import analysis_embedding


SPLICEBERT = {
    "human": SPLICEBERT_HUMAN,
    "vertebrate": SPLICEBERT_510,
}


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--num-seqs", type=int, default=100000, help="number of randomly selected sequences")
    p.add_argument("--metrics", type=str, choices=("acc", "balanced-acc"), default="balanced-acc", help="metrics to evaluate, acc or balanced_acc")
    p.add_argument("-o", "--output", type=str, default="pred_in_region.pkl")
    p.add_argument('--seed', type=int, default=2020)
    return p


if __name__ == "__main__":
    args = get_args().parse_args()
    set_seed(args.seed)

    ds = analysis_embedding.GenomicRegionData()
    subset = np.random.choice(np.arange(len(ds)), size=args.num_seqs, replace=False)
    loader = DataLoader(Subset(ds, indices=subset), batch_size=8, num_workers=8, shuffle=False)

    splicebert = AutoModelForMaskedLM.from_pretrained(SPLICEBERT["vertebrate"])
    human_model = AutoModelForMaskedLM.from_pretrained(SPLICEBERT["human"])

    device = torch.device("cuda")
    splicebert = splicebert.to(device)
    human_model = human_model.to(device)
    splicebert.eval()
    human_model.eval()


    all_label = list()
    all_anno = list()
    all_vert_logits = list()
    all_human_logits = list()
    all_repeat = list()


    with torch.no_grad(), autocast():
        for it, (ids, masked_ids, _, label, phastcons, phylop, annotation, repeat) in enumerate(tqdm(loader)):
            # ids: (B, S)
            # label: (B, S)
            # annotation: (B, S, 12)
            dim = annotation.shape[-1]
            annotation = annotation.reshape(-1, dim)
            # print(torch.unique(repeat, return_counts=True), file=sys.stderr)
            repeat = repeat.reshape(-1)

            label = label[:, 1:-1].reshape(-1)

            masked_ids = masked_ids.to(device)
            out = splicebert.forward(masked_ids)
            vert_logits = out.logits.detach().cpu()[:, 1:-1].reshape(label.shape[0], -1)
            # vert_logits = torch.argmax(vert_logits, dim=-1).reshape(-1) # (B, S)
            del out

            out = human_model.forward(masked_ids)
            human_logits = out.logits.detach().cpu()[:, 1:-1].reshape(label.shape[0], -1)
            del out


            assert annotation.shape[0] == label.shape[0], "annotation and label should have the same length, but got {} and {}".format(annotation.shape, label.shape)

            keep = torch.where(label != -100)[0]
            all_anno.append(annotation[keep])
            all_label.append(label[keep])
            all_vert_logits.append(vert_logits[keep])
            all_human_logits.append(human_logits[keep])
            all_repeat.append(repeat[keep])
            # print(torch.unique(repeat[keep], return_counts=True), file=sys.stderr)

    all_label = torch.cat(all_label).long()
    all_vert_logits = torch.cat(all_vert_logits).float()
    all_human_logits = torch.cat(all_human_logits).float()
    all_anno = np.concatenate(all_anno)
    all_repeat = torch.cat(all_repeat).numpy()
    print(np.unique(all_repeat, return_counts=True), file=sys.stderr)
    print(all_label.shape, all_vert_logits.shape, all_human_logits.shape, all_anno.shape, all_repeat.shape, file=sys.stderr)

    pickle.dump((all_label, all_vert_logits, all_human_logits, all_anno, all_repeat), open(args.output, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


    name_list = sorted([(n, idx) for n, idx in ds.name2idx.items()], key=lambda x:x[1])


    acc_fun = accuracy_score if args.metrics == "acc" else balanced_accuracy_score
    print("#region\tn\tacc_vert\tacc_human\tece_vert\tece_human")
    print('\t'.join([
        "all", '.',
        "{:.3f}".format(acc_fun(all_label.numpy(), torch.argmax(all_vert_logits, dim=-1).numpy())),
        "{:.3f}".format(acc_fun(all_label.numpy(), torch.argmax(all_human_logits, dim=-1).numpy())),
        "{:.3f}".format(2**F.cross_entropy(all_vert_logits, all_label).item()),
        "{:.3f}".format(2**F.cross_entropy(all_human_logits, all_label).item()),
    ]))
    print()
    print("#region\tn\tacc\tacc(repeat)\tacc(non-repeat)")
    for i in range(all_anno.shape[1]):
        k = np.where(all_anno[:, i] == 1)[0]
        # print("{}(n={})\t{:.3f}\t{:.3f}".format(name_list[i][0], len(k), 2**F.cross_entropy(all_vert_logits[k], all_label[k]).item(), 2**F.cross_entropy(all_human_logits[k], all_label[k]).item()))
        k_repeat = np.where((all_repeat == 1) & (all_anno[:, i] == 1))[0]
        k_nonrepeat = np.where((all_repeat == 0) & (all_anno[:, i] == 1))[0]
        print('\t'.join([
            str(name_list[i][0]), str(len(k)) + '/' + str(len(k_repeat)) + '/' + str(len(k_nonrepeat)),
            "{:.3f}".format(acc_fun(all_label[k].numpy(), torch.argmax(all_vert_logits[k], dim=-1).numpy())),
            "{:.3f}".format(acc_fun(all_label[k_repeat].numpy(), torch.argmax(all_vert_logits[k_repeat], dim=-1).numpy())),
            "{:.3f}".format(acc_fun(all_label[k_nonrepeat].numpy(), torch.argmax(all_vert_logits[k_nonrepeat], dim=-1).numpy())),

            # "{:.3f}".format(2**F.cross_entropy(all_vert_logits[k], all_label[k]).item()),
            # "{:.3f}".format(2**F.cross_entropy(all_human_logits[k], all_label[k]).item()),
        ]))
