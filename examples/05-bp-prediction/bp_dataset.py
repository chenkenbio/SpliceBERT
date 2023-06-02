#!/usr/bin/env python3
"""
Author: Ken Chen
Email: chenkenbio@gmail.com
"""

import argparse
import os
import sys
import gzip
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
#from biock import load_fasta, HG19_FASTA, get_reverse_strand
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
from transformers import BertTokenizer
import logging
logger = logging.getLogger(__name__)
sys.path.append("../scripts")
from utils import load_fasta, get_reverse_strand, fast_seq_to_kmer_list
from config import hg19 as HG19_FASTA
from config import MERCER_DATASET, SPLICEBERT


class BranchPointData(Dataset):
    def __init__(self, 
            dataset=MERCER_DATASET, 
            seq_len=510,
            tokenizer=BertTokenizer.from_pretrained(SPLICEBERT),
            genome=HG19_FASTA,
            dnabert_mode: bool=False,
            no_special_token: bool=False,
        ) -> None:
        super().__init__()
        self.chroms: List[str]=list()
        self.starts: List[int]=list()
        self.ends: List[int]=list()
        self.labels: List[np.ndarray]=list()
        self.strands: List[str]=list()

        self.dataset = dataset
        self.process()
        self.genome = load_fasta(genome)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.dnabert_mode = dnabert_mode
        self.add_special_tokens = not no_special_token
        self.k = np.log2(len(self.tokenizer.vocab) - 5)//2
    
    def __repr__(self) -> str:
        return "{}{}(file={},seq_len={},genome={},size={})".format(self.__class__.__name__, "" if not self.dnabert_mode else "(for DNABERT-{})".format(self.k), self.dataset, self.seq_len, HG19_FASTA, self.__len__())

    def process(self):
        expect_len = 44 - 18 + 1
        skip = 0
        self.raw_index = list()
        with gzip.open(self.dataset, 'rt') as infile:
            for nr, l in enumerate(infile):
                chrom, start, end, name, label, strand = l.strip('\n').split('\t')[:6]
                label = [int(x) for x in label.split(',')]
                if max(label) < 1:
                    skip += 1
                    continue
                start, end = int(start), int(end)
                self.chroms.append(chrom)
                if start - end < expect_len:
                    if strand == '+':
                        label = [-100 for _ in range(expect_len - end + start)] + label
                        start -= (expect_len - end + start)
                    else:
                        label = label + [-100 for _ in range(expect_len - end + start)]
                        end += (expect_len - end + start)
                self.starts.append(start)
                self.ends.append(end)
                self.labels.append(np.asarray(label, dtype=np.int8))
                self.strands.append(strand)
                self.raw_index.append(nr)
        logger.warning("- {} samples were dropped due to no high-confidence bp".format(skip))
        self.chroms = np.asarray(self.chroms)
        self.starts = np.asarray(self.starts)
        self.ends = np.asarray(self.ends)
        self.strands = np.asarray(self.strands)
        self.labels = np.stack(self.labels)
        self.raw_index = np.asarray(self.raw_index)
    
    def __len__(self):
        return len(self.chroms)
    
    def __getitem__(self, index):

        ## +: exon|d     (bp) a|exon
        ## -: exon|a (bp)     d|exon
        chrom = self.chroms[index]
        start = self.starts[index]
        end = self.ends[index]
        strand = self.strands[index]
        label = self.labels[index]
        up = (self.seq_len - end + start) // 2
        down = self.seq_len - end + start - up
        if strand == '+':
            left, right = start - up, end + down
            label = np.concatenate((
                -100 * np.ones(up, dtype=np.int8),
                label,
                -100 * np.ones(down, dtype=np.int8),
            ))
            seq = self.genome[chrom][left:right].upper()
        else:
            left, right = start - down, end + up
            label = np.concatenate((
                -100 * np.ones(down, dtype=np.int8),
                label,
                -100 * np.ones(up, dtype=np.int8),
            ))
            seq = self.genome[chrom][left:right].upper()
            seq = get_reverse_strand(seq)
            label = label[::-1].copy()
        if self.dnabert_mode:
            ids = self.tokenizer.encode(' '.join(fast_seq_to_kmer_list(seq, k=self.k, pad=True)), add_special_tokens=self.add_special_tokens)
        else:
            ids = self.tokenizer.encode(' '.join(list(seq)))
        ids = torch.as_tensor(ids).long()
        label = torch.as_tensor(label).long()
        return ids, label, self.raw_index[index]

