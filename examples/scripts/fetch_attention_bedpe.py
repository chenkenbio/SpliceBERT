#!/usr/bin/env python3
"""
Author: Ken Chen
Email: chenkenbio@gmail.com
Date: 2023-01-10
"""

import argparse
import warnings
from tqdm import tqdm
import os
import sys
import pickle
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForMaskedLM, AutoModelForTokenClassification, BertTokenizer, BertModel
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import load_fasta, get_reverse_strand, auto_open, encode_sequence
import scanpy as sc

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("bedpe", type=str, help="bedpe file")
    p.add_argument("-b", "--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", 8)), help="batch size")
    p.add_argument("-m", "--model", type=str, required=True, help="model path")
    p.add_argument("-g", "--genome", type=str, required=True, help="genome path")
    p.add_argument("-o", "--output", type=str, required=True, help="output file")
    # p.add_argument('--seed', type=int, default=2020)
    return p

def encode_dnabert(seq: str, k: int):
    seq_new = list()
    N = len(seq)
    seq = "N" * (k//2) + seq.upper() + "N" * k
    for i in range(k//2, N + k//2):
        seq_new.append(seq[i-k//2:i-k//2+k])
    return ' '.join(seq_new)

class BEDPEData(Dataset):
    def __init__(self, bed, seq_len, genome, tokenizer=None) -> None:
        super().__init__()
        self.genome = load_fasta(genome)
        self.seq_len = seq_len
        self.bed = bed
        self.tokenizer = tokenizer
        self.samples = list()
        self.process()
    
    def process(self):
        with auto_open(self.bed, 'rt') as f:
            for line in f:
                if line.startswith("#"):
                    continue
                # chrom, start, end, name, name2, strand = line.strip().split('\t')
                # start, end = int(start), int(end)
                # left = (start + end) // 2 - self.seq_len // 2
                # right = left + self.seq_len
                # d = start - left
                # a = end - 1 - left
                # if strand == '-':
                #     d, a = self.seq_len - 1 - a, self.seq_len - 1 - d
                # self.samples.append((chrom, left, right, strand, d, a))
                # self.name.append(name)
                # self.name2.append(name2)
                chrom, p, _, _, q, _, name, _, strand, _ = line.split()
                p, q = int(p), int(q)
                try:
                    tx_region, group, t1, t2, dimer1, dimer2, dist = name.split('|')
                except:
                    tx_region = name.split('|')[0]
                    group = "__unknown__"
                    t1, t2 = ("D", "A") if strand == '+' else ("A", "D")
                    warnings.warn("Invalid name type")

                tx_start, tx_end = tx_region.split(':')[1].split('-')
                tx_start, tx_end = int(tx_start), int(tx_end)
                left = (p + q) //2 - self.seq_len // 2
                right = left + self.seq_len
                if left < tx_start or right > tx_end:
                    continue
                p = p - left
                q = q - left
                if strand == '-':
                    p, q = self.seq_len - 1 - p, self.seq_len - 1 - q
                self.samples.append((chrom, left, right, strand, p, q, t1, t2, group))
        self.samples = np.array(self.samples)
    
    def __getitem__(self, index):
        chrom, left, right, strand, p, q, t1, t2, group = self.samples[index]
        left, right = int(left), int(right)
        p, q = int(p), int(q)
        seq = self.genome[chrom][left:right]
        if strand == '-':
            seq = get_reverse_strand(seq)
        seq = torch.as_tensor(self.tokenizer.encode(' '.join(seq.upper())))
        return seq, p, q, t1, t2, group
    
    def __len__(self):
        return len(self.samples)
    
    def collate_fn(self, batch):
        seq, p, q, t1, t2, group = map(list, zip(*batch))
        seq = torch.stack(seq)
        p = np.asarray(p)
        q = np.asarray(q)
        t1 = np.asarray(t1)
        t2 = np.asarray(t2)
        group = np.asarray(group)
        return seq, p, q, t1, t2, group


if __name__ == "__main__":
    args = get_args().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "onehot":
        tokenizer = None
    else:
        tokenizer = BertTokenizer.from_pretrained(args.model)
        model = BertModel.from_pretrained(args.model, add_pooling_layer=False, output_hidden_states=True).to(device)
    model.eval()
    ds = BEDPEData(args.bedpe, 1000, args.genome, tokenizer)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=ds.collate_fn, num_workers=16)

    p_att = list()
    q_att = list()
    pos1 = list()
    pos2 = list()
    type1 = list()
    type2 = list()
    dist = list()

    for it, (seq, p, q, t1, t2, _) in enumerate(tqdm(loader)):
        seq = seq.to(device)
        # h = torch.stack(model(seq).hidden_states, dim=1).detach().cpu().numpy()
        att = model(seq, output_attentions=True).attentions
        att = torch.stack(att, dim=1).detach().max(dim=2)[0][:, :, 1:-1, 1:-1] # (B, layer, S, S)
        for k in range(att.shape[0]):
            p_att.append(att[k, :, p[k]-1:p[k]+2].cpu().numpy())
            pos1.append(p[k])
            q_att.append(att[k, :, q[k]-1:q[k]+2].cpu().numpy())
            pos2.append(q[k])
            type1.append(t1[k])
            type2.append(t2[k])
            dist.append(abs(p[k] - q[k]))
    p_att = np.stack(p_att, axis=0).astype(np.float16)
    q_att = np.stack(q_att, axis=0).astype(np.float16)
    pos1 = np.asarray(pos1).astype(np.int16)
    pos2 = np.asarray(pos2).astype(np.int16)
    type1 = np.asarray(type1)
    type2 = np.asarray(type2)
    dist = np.asarray(dist).astype(np.int16)
    pickle.dump(
        (p_att, q_att, pos1, pos2, type1, type2, dist), 
        open(args.output, 'wb'), 
        protocol=pickle.HIGHEST_PROTOCOL
    )
            

    
    # pickle.dump((embedding, ds.name, ds.name2), open(args.output, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
