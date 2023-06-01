#!/usr/bin/env python3

import pickle
import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import BertTokenizer, BertTokenizerFast
from torch.nn.utils.rnn import pad_sequence
import logging
logger = logging.getLogger(__name__)


def seq_to_dnabert_kmers(seq, k: int):
    kmers = list()
    for i in range(0, len(seq) - k + 1):
        kmers.append(seq[i:i+k])
    return ' '.join(kmers)

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("bed")
    # p.add_argument('--seed', type=int, default=2020)
    return p

VOCAB_FILES = {
    '1': "/home/chenken/Documents/dnalm-with-pcl/vocab/dna_vocab_1bp.txt",
    "3": "/home/chenken/Documents/dnalm-with-pcl/vocab/dna_vocab_3bp.txt"
}

class SpliceatorDataset(Dataset):
    def __init__(self, positive, negative, tokenizer: BertTokenizer, max_len: int, dnabert_k: int=None, shift=0, reverse=False):
        super().__init__()
        self.shift = shift
        self.reverse = reverse
        self.max_len = max_len
        self.positive = positive if isinstance(positive, list) else [positive]
        self.negative = negative if isinstance(negative, list) else [negative]
        self.tokenizer = tokenizer
        self.labels = list()
        self.groups = list()
        self.sequences = list()
        self.k = dnabert_k
        self.process()
    
    def process(self):
        for label, files in [[1, self.positive], [0, self.negative]]:
            for fn in files:
                bn = os.path.basename(fn)
                with open(fn) as infile:
                    for l in infile:
                        if l.startswith("ID_uniprot"):
                            continue
                        fields = l.strip().split(';')
                        if len(fields[1]) < 100:
                            seq = fields[2]
                        else:
                            seq = fields[1]
                        # assert len(seq) == 600, "{}".format((len(seq), fn, fields))
                        skip_left = (len(seq) - self.max_len) // 2 # + np.random.randint(-10, 11)
                        if self.shift > 0:
                            skip_left += np.random.randint(-self.shift, self.shift + 1)
                        seq = seq[skip_left:skip_left + self.max_len]
                        self.sequences.append(seq)
                        self.groups.append(fields[0].split('_')[-1])
                        self.labels.append(label)
                        # self.samples.append((bn, label, fields[0]))
        self.labels = np.array(self.labels)
        self.groups = np.array(self.groups)
        self.sequences = np.array(self.sequences)
    
    def __getitem__(self, index):
        seq = self.sequences[index]
        label = int(self.labels[index])
        if self.k is None:
            input_ids = torch.tensor(self.tokenizer.encode(' '.join(list(seq.upper()))))
        else:
            input_ids = torch.tensor(self.tokenizer.encode(seq_to_dnabert_kmers(seq.upper(), k=self.k)))
        mask = torch.ones_like(input_ids)
        return input_ids, mask, label
    
    def __len__(self):
        return len(self.sequences)
    
    def collate_fn(self, inputs):
        ids, mask, label = map(list, zip(*inputs))
        ids = pad_sequence(ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        mask = pad_sequence(mask, batch_first=True)
        label = torch.tensor(label)
        return ids, mask, label
    

    # def collate_fn(self, inputs):
    #     input_ids, labels = map(list, zip(*inputs))
    #     return torch.cat(input_ids, dim=0), torch.cat(labels, dim=0)

if __name__ == "__main__":
    args = get_args().parse_args()

    last_cluster = None
    with open(args.bed) as infile:
        for l in infile:
            chrom, start, end, name, _, strand, cluster = l.strip().split()
