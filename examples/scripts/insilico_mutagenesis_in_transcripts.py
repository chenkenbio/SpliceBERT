#!/usr/bin/env python
"""
Author: Ken Chen
Email: chenkenbio@gmail.com
"""


import os
import sys
import json
import pickle
import gzip
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
from importlib import reload
from collections import defaultdict, OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.cuda.amp import autocast
import insilico_mutagenesis_data
from config import SPLICEBERT_510


model = AutoModelForMaskedLM.from_pretrained(SPLICEBERT_510)
tokenizer = AutoTokenizer.from_pretrained(SPLICEBERT_510)
model = model.eval()


def kl_divergence(ref, alt, eps=1e-6):
    if isinstance(ref, np.ndarray):
        ref = ref.astype(np.float64)
        alt = alt.astype(np.float64)
    assert ref.max() <= 1 and ref.max() >= 0
    assert alt.max() <= 1 and alt.max() >= 0
    if isinstance(ref, torch.Tensor):
        ref = torch.clip(ref, min=eps, max=1 - eps)
        alt = torch.clip(alt, min=eps, max=1 - eps)
        kl = (alt * torch.log(alt / ref)).sum(axis=-1) 
    else:
        ref = np.clip(ref, a_min=eps, a_max=1 - eps)
        alt = np.clip(alt, a_min=eps, a_max=1 - eps)
        kl = (alt * np.log(alt / ref)).sum(axis=-1) 
    return kl

device = torch.device("cuda")
model = model.to(device)

ds = insilico_mutagenesis_data.InSilicoMutagenesisData.load_dataset(fraction=0.1)

max_sample = 20000
np.random.seed(2023)
inds = np.random.permutation(np.arange(len(ds)))[:max_sample]
batch_size = int(os.environ.get("BATCH_SIZE", 16))
loader = DataLoader(
    Subset(ds, indices=inds), 
    batch_size=batch_size, 
    num_workers=min(batch_size, os.cpu_count()), 
    collate_fn=ds.collate_fn, 
    shuffle=True)


all_kls = list()
all_chroms = list()
all_positions = list()
all_in_exon = list()
all_e2d = list()
all_e2a = list()
all_i2d = list()
all_i2a = list()
all_phastcons = list()
all_phylop = list()

with torch.no_grad(), autocast():
    for it, (ref_ids, alt_ids, mask_ids, chrom, p, in_exon, e2d, e2a, i2d, i2a, phastcons, phylop) in enumerate(tqdm(loader, total=min(len(loader), max_sample))):
        if it >= max_sample:
            break
        ref_prob = torch.softmax(model.forward(ref_ids.to(device)).logits, dim=-1)[:, 1:-1].detach()
        alt_prob = torch.softmax(model.forward(alt_ids.to(device)).logits, dim=-1)[:, 1:-1].detach()
        kl = kl_divergence(ref_prob, alt_prob).reshape(chrom.shape[0], 3, -1).max(dim=1)[0].cpu().numpy().astype(np.float16)
        all_kls.append(kl)
        all_chroms.append(chrom)
        all_positions.append(p.astype(np.int32))
        all_in_exon.append(in_exon.astype(np.int8))
        all_e2d.append(e2d.astype(np.int32))
        all_e2a.append(e2a.astype(np.int32))
        all_i2d.append(i2d.astype(np.int32))
        all_i2a.append(i2a.astype(np.int32))
        all_phastcons.append(phastcons.astype(np.float16))
        all_phylop.append(phylop.astype(np.float16))
    
all_kls = np.concatenate(all_kls)
all_chroms = np.concatenate(all_chroms)
all_positions = np.concatenate(all_positions)
all_in_exon = np.concatenate(all_in_exon)
all_e2d = np.concatenate(all_e2d)
all_e2a = np.concatenate(all_e2a)
all_i2d = np.concatenate(all_i2d)
all_i2a = np.concatenate(all_i2a)
all_phastcons = np.concatenate(all_phastcons)
all_phylop = np.concatenate(all_phylop)

pickle.dump((all_kls, all_chroms, all_positions, all_in_exon, all_e2d, all_e2a, all_i2d, all_i2a, all_phastcons, all_phylop), open("./output/insilico_mutation_in_transcripts.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)



