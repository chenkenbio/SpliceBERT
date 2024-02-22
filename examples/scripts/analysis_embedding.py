#!/usr/bin/env python3
"""
Author: Ken Chen
Email: chenkenbio@gmail.com
Date: 2022-11-24
"""

import json
import sys
import gzip
import numpy as np
from tqdm import tqdm
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
import h5py
from transformers import AutoTokenizer
from selene_custom.genomic_features import GenomicFeatures
from genome import EasyGenome
from config import SPLICEBERT, hg19_phylop, hg19_phastcons, hg19_regions, hg19_transcript, hg19

def encode_dnabert(seq: str, k: int):
    seq_new = list()
    N = len(seq)
    seq = "N" * (k//2) + seq.upper() + "N" * k
    for i in range(k//2, N + k//2):
        seq_new.append(seq[i-k//2:i-k//2+k])
    return ' '.join(seq_new)

def mask_bases(input_ids: Tensor, vocab_size, num_special_tokens: int, mask_token_id: int, mlm_rate=0.15, edge_size: int=0, rand_rate=0.1, unchange_rate=0.1, mask_left: int=0, mask_right: int=0):
    assert input_ids.ndim == 1
    input_ids = input_ids.clone()

    label = -100 * torch.ones(input_ids.size(), dtype=torch.long)
    
    valid = input_ids >= num_special_tokens
    prob = torch.rand(input_ids.size()) * valid
    keep = (prob <= mlm_rate) & valid
    label[keep] = input_ids[keep].clone()
    if edge_size > 0:
        label[:edge_size] = -100
        label[-edge_size:] = -100
    prob *= (prob <= mlm_rate) # set other to 0
    prob /= mlm_rate

    mask_inds = torch.where(prob > (rand_rate + unchange_rate))[0]
    input_ids[mask_inds] = mask_token_id
    if mask_left + mask_right > 0:
        for i in mask_inds:
            for j in range(i-mask_left, i+mask_right+1):
                if j >= 0 and j < input_ids.size(0):
                    input_ids[j] = mask_token_id

    shuffle_inds = torch.where((prob > unchange_rate) & (prob <= (rand_rate + unchange_rate)))[0]
    input_ids[shuffle_inds] = torch.randint(low=num_special_tokens, high=vocab_size, size=shuffle_inds.size()) # skip 'N'
    if mask_left + mask_right > 0:
        for i in shuffle_inds:
            for j in range(i - mask_left, i + mask_right + 1):
                if j >= 0 and j < input_ids.size(0):
                    if j != i:
                        input_ids[j] = np.random.randint(low=num_special_tokens, high=vocab_size) # skip 'N'


    return input_ids, label

class GenomicRegionData(Dataset):
    def __init__(self, 
            bed=hg19_transcript, 
            bin_size=510, 
            region=hg19_regions, 
            name_idx=hg19_regions.replace("bed.gz", "names.json"),
            phastcons=hg19_phastcons,
            phylop=hg19_phylop,
            tokenizer: AutoTokenizer=AutoTokenizer.from_pretrained(SPLICEBERT),
            genome=hg19,
            dnabert_k: int=None
        ) -> None:
        super().__init__()
        self.bed = bed
        self.dnabert_k = dnabert_k
        self.bin_size = bin_size
        self.name2idx = json.load(open(name_idx))
        self.annotation = GenomicFeatures(
            input_path=region,
            features=name_idx,
            binary=True,
            nt_level=True,
        )

        self.tokenizer = tokenizer
        self.samples = list()
        self.genome = EasyGenome(genome)
        self.phastcons = h5py.File(phastcons, 'r')
        self.phylop = h5py.File(phylop, 'r')
        self.cls = np.asarray([self.tokenizer.cls_token_id], dtype=np.int8)
        self.sep = np.asarray([self.tokenizer.sep_token_id], dtype=np.int8)
        self.process()
    
    def process(self):
        txs = dict()
        with gzip.open(self.bed, 'rt') as infile:
            for l in tqdm(infile, desc="Loading bed: {}".format(self.bed)):
                chrom, start, end, name, _, strand = l.strip('\n').split('\t')[:6]
                if chrom == "chrY" or chrom == "chrM":
                    continue
                start, end = int(start), int(end)
                _, _, gene_name, gene_type, _ = name.split('|')
                if gene_name in txs and end - start < txs[gene_name][2] - txs[gene_name][1]:
                    continue
                txs[gene_name] = (chrom, start, end, strand, gene_type)
        for gene_name in txs:
            chrom, start, end, strand, gene_type = txs[gene_name]
            for p in range(start - 100, end, self.bin_size):
                self.samples.append((chrom, p, p + self.bin_size, strand))
        self.samples = np.asarray(self.samples)
    
    def __getitem__(self, index):
        chrom, start, end, strand = self.samples[index]
        start, end = int(start), int(end)
        annotation = self.annotation.get_feature_data(chrom, start, end).astype(np.int8)
        phastcons = self.phastcons[chrom][start:end]
        phylop = self.phylop[chrom][start:end]
        if strand == '-':
            annotation = annotation[::-1].copy()
            phastcons = phastcons[::-1].copy()
            phylop = phylop[::-1].copy()
        # ids = self.genome.fetch_sequence(chrom, start, end, reverse=strand == '-', padding=-105) + 5
        # ids_extend = self.genome.fetch_sequence(chrom, start - self.bin_size, end + self.bin_size, reverse=strand == '-', padding=-105) + 5 # for one-hot encoding of position
        # ids = np.concatenate((self.cls, ids, self.sep))
        # ids_extend = np.concatenate((self.cls, ids_extend, self.sep))
        seq = self.genome.fetch_sequence(chrom, start, end, reverse=strand == '-', padding='N')
        seq_extend = self.genome.fetch_sequence(chrom, start - self.bin_size, end + self.bin_size, reverse=strand == '-', padding='N')
        is_repeat = torch.from_numpy(np.isin(list(seq), ['a', 't', 'c', 'g']).astype(np.int8))
        # if torch.sum(is_repeat) > 0:
            # print(chrom, start, end, strand, is_repeat.sum().item(), flush=True)
        if self.dnabert_k:
            ids = self.tokenizer.encode(encode_dnabert(seq, self.dnabert_k))
            ids_extend = self.tokenizer.encode(encode_dnabert(seq_extend, self.dnabert_k))
        else:
            ids = self.tokenizer.encode(' '.join(seq.upper()))
            ids_extend = self.tokenizer.encode(' '.join(seq_extend.upper()))
        ids = torch.as_tensor(ids).long()
        ids_extend = torch.as_tensor(ids_extend).long()
        masked_ids, label = mask_bases(
            ids.clone(), 
            vocab_size=self.tokenizer.vocab_size, 
            num_special_tokens=5,
            mask_token_id=self.tokenizer.mask_token_id,
            mask_left=(self.dnabert_k - 1 - self.dnabert_k//2) if self.dnabert_k else 0,
            mask_right=(self.dnabert_k//2) if self.dnabert_k else 0,
        )
        phastcons = torch.as_tensor(phastcons)
        phylop = torch.as_tensor(phylop)
        annotation = torch.as_tensor(annotation)
        return ids, masked_ids, ids_extend, label, phastcons, phylop, annotation, is_repeat
    
    def __len__(self):
        return len(self.samples)
    