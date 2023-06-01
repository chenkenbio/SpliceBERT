#!/usr/bin/env python3
"""
Author: Ken Chen
Email: chenkenbio@gmail.com
Date: 2022-11-29
"""

import argparse
import pickle
import os
import gzip
import sys
import numpy as np
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import h5py
from transformers import AutoTokenizer, BertTokenizer
from utils import get_reverse_strand, load_fasta, encode_sequence, GenePred
from config import hg19 as HG19_FASTA
from config import SPLICEBERT, rand200_transcripts, hg19_genepred, hg19_phastcons, hg19_phylop

import logging
logger = logging.getLogger(__name__)

CHROM2INT = {
    "chr1": 0, "chr2": 1, "chr3": 2, "chr4": 3,
    "chr5": 4, "chr6": 5, "chr7": 6, "chr8": 7,
    "chr9": 8, "chr10": 9, "chr11": 10, "chr12": 11, 
    "chr13": 12, "chr14": 13, "chr15": 14, "chr16": 15, 
    "chr17": 16, "chr18": 17, "chr19": 18, "chr20": 19,
    "chr21": 20, "chr22": 21, "chrX": 22
}
INT2CHROM = {v:k for k, v in CHROM2INT.items()}

class InSilicoMutagenesisData(Dataset):
    def __init__(self, bed, genepred, seq_len, k: int, tokenizer: BertTokenizer, genome, phastcons, phylop, fraction:float=None, seed=2023) -> None:
        self.seed = seed
        self.bed = bed
        self.fraction = fraction
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.k = k
        self.tokenizer = tokenizer
        self.samples = list()
        self.names = list()
        self.labels = list()
        self.genome = load_fasta(genome)
        self.phastcons = h5py.File(phastcons, 'r')
        self.phylop = h5py.File(phylop, 'r')
        self.genepred = GenePred(genepred)
        self.process()
    
    @classmethod
    def load_dataset(self, fraction: float=None):
        return self(
            bed=rand200_transcripts,
            genepred=hg19_genepred,
            seq_len=510,
            k=1,
            tokenizer=AutoTokenizer.from_pretrained(SPLICEBERT),
            genome=HG19_FASTA,
            phastcons=hg19_phastcons,
            phylop=hg19_phylop,
            fraction=fraction
        )
    
    def process(self):
        if self.fraction is None:
            processed = self.bed + ".insilico-mutagenesis.pkl.gz"
        else:
            processed = self.bed + ".insilico-mutagenesis.fraction.{}.pkl.gz".format(self.fraction)
        if os.path.exists(processed):
            self.samples = pickle.load(gzip.open(processed, 'rb'))
        else:
            np.random.seed(self.seed)
            skip = {
                "n": 0,
                "nan_cons": 0,
                "distant_exon": 0,
                "distant_intron": 0,
            }
            with gzip.open(self.bed, 'rt') as infile:
                for l in tqdm(infile, desc=os.path.basename(self.bed)):
                    chrom, start, end, name, _, strand = l.strip('\n').split('\t')[:6]
                    start, end = int(start), int(end)
                    tx_id, gene_name = name.split('|')[0:2]
                    t = self.genepred.tx_info[tx_id]
                    if len(t.intron) == 0:
                        continue
                    phastcons = self.phastcons[chrom][start:end]
                    seq = np.asarray(encode_sequence(self.genome[chrom][start:end]))
                    
                    for intervals, region in [(t.exon, 1), (t.intron, 0)]:# 1 for exon, 0 for intron
                        for it, (left, right) in enumerate(intervals):
                            for p in range(max(left, t.tx_start + self.seq_len //2), min(right, t.tx_end - self.seq_len//2)):
                                p0 = p - t.tx_start
                                ref = seq[p0]
                                if np.isnan(phastcons[p0]) or ref == 0:
                                    if ref == 0:
                                        skip['nan_cons'] += 1
                                    else:
                                        skip['nan_cons'] += 1
                                    continue
                                if region == 1:
                                    e2a, e2d = p - left + 1, right - p
                                    if self.fraction is None and e2a > 200 and e2d > 200:
                                        skip["distant_exon"] += 1
                                        continue
                                    elif self.fraction is not None and np.random.rand() > self.fraction:
                                        continue
                                    if it == 0:
                                        e2a = -100
                                    elif it == len(t.exon) - 1:
                                        e2d = -100
                                    if strand == '-':
                                        e2d, e2a = e2a, e2d
                                    i2d, i2a = -100, -100
                                    for alt in range(1, 5):
                                        if alt != ref:
                                            self.samples.append((chrom, p, ref, alt, strand, region, e2d, e2a, i2d, i2a))
                                else:
                                    i2d, i2a = p - left, right - p - 1
                                    if self.fraction is not None and np.random.rand() > self.fraction:
                                        continue
                                    elif self.fraction is None and i2a > 200 and i2d > 200:
                                        skip["distant_intron"] += 1
                                        continue
                                    if strand == '-':
                                        i2a, i2d = i2d, i2a

                                    e2a, e2d = -100, -100
                                    for alt in range(1, 5):
                                        if alt != ref:
                                            self.samples.append((chrom, p, ref, alt, strand, region, e2d, e2a, i2d, i2a))
            logger.warning("{}".format(skip))
            self.samples = np.asarray(self.samples)
            pickle.dump(self.samples, gzip.open(processed, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        
    def __len__(self):
        return len(self.samples) // 3
    
    def get_single(self, index):
        chrom, p, ref, alt, strand, in_exon, e2d, e2a, i2d, i2a = self.samples[index]
        p = int(p)
        ref, alt = int(ref), int(alt)
        e2d, e2a, i2d, i2a = int(e2d), int(e2a), int(i2d), int(i2a)
        if strand == '+':
            start = p - self.seq_len // 2
            end = start + self.seq_len
            loc = self.seq_len//2
        else:
            end = p + 1 + self.seq_len // 2
            start = end - self.seq_len
            loc = self.seq_len//2 - 1
        ref_ids = np.asarray(encode_sequence(self.genome[chrom][start:end]))
        alt_ids = ref_ids.copy()
        assert ref_ids[loc] == ref, "{}".format((ref_ids[loc], loc, ref))
        alt_ids[loc] = alt
        if strand == '-':
            ref_ids = get_reverse_strand(ref_ids, integer=True)
            alt_ids = get_reverse_strand(alt_ids, integer=True)
            ref = get_reverse_strand(ref, integer=True)
            alt = get_reverse_strand(alt, integer=True)
        ref_ids += 5
        alt_ids += 5
        ref_ids = np.concatenate((
            np.asarray([self.tokenizer.cls_token_id]),
            ref_ids,
            np.asarray([self.tokenizer.sep_token_id])
        ))
        alt_ids = np.concatenate((
            np.asarray([self.tokenizer.cls_token_id]),
            alt_ids,
            np.asarray([self.tokenizer.sep_token_id])
        ))

        mask_ids = ref_ids.copy()
        mask_ids[self.seq_len//2 + 1] = self.tokenizer.mask_token_id
        
        return (
            torch.as_tensor(ref_ids), torch.as_tensor(alt_ids), torch.as_tensor(mask_ids),
            chrom, p, in_exon, e2d, e2a, i2d, i2a
        )
    
    def __getitem__(self, index):
        all_ref_ids, all_alt_ids, all_mask_ids = list(), list(), list()
        chrom, p, _, _, _, in_exon, e2d, e2a, i2d, i2a = self.samples[index * 3]
        p = int(p)
        phastcons = self.phastcons[chrom][p]
        phylop = self.phylop[chrom][p]
        for i in range(3):
            ref_ids, alt_ids, mask_ids, chrom, p, in_exon, e2d, e2a, i2d, i2a = self.get_single(3 * index + i)
            all_ref_ids.append(ref_ids)
            all_alt_ids.append(alt_ids)
            all_mask_ids.append(mask_ids)
        return (
            torch.stack(all_ref_ids).long(),
            torch.stack(all_alt_ids).long(),
            torch.stack(all_mask_ids).long(),
            chrom, p, in_exon, e2d, e2a, i2d, i2a, phastcons, phylop
        )

    
    def collate_fn(self, inputs):
        ref_ids, alt_ids, mask_ids, chrom, p, in_exon, e2d, e2a, i2d, i2a, phastcons, phylop = map(list, zip(*inputs))
        ref_ids = torch.concat(ref_ids).long()
        alt_ids = torch.concat(alt_ids).long()
        mask_ids = torch.concat(mask_ids).long()
        chrom = np.asarray(chrom)
        p = np.asarray(p)
        in_exon = np.asarray(in_exon)
        e2d = np.asarray(e2d)
        e2a = np.asarray(e2a)
        i2d = np.asarray(i2d)
        i2a = np.asarray(i2a)
        phastcons = np.asarray(phastcons)
        phylop = np.asarray(phylop)
        return ref_ids, alt_ids, mask_ids, chrom, p, in_exon, e2d, e2a, i2d, i2a, phastcons, phylop
