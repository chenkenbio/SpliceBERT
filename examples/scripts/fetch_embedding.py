#!/usr/bin/env python3
"""
Author: Ken Chen
Email: chenkenbio@gmail.com
"""

import argparse
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
from transformers import AutoTokenizer, AutoModel
import scanpy as sc
from utils import load_fasta, get_reverse_strand, encode_sequence, auto_open, get_run_info

ONEHOT = np.concatenate((
    np.zeros((1, 4)),
    np.eye(4),
)).astype(np.int8)

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("bed", type=str, help="bed file")
    p.add_argument("--skip-per-layer-umap", action="store_true", help="skip per layer umap")
    p.add_argument("--skip-donor-acceptor-umap", action="store_true", help="skip layer umap")
    p.add_argument("-m", "--model", type=str, required=True, help="model path")
    p.add_argument("-g", "--genome", type=str, required=True, help="genome path")
    p.add_argument("-o", "--output", type=str, required=True, help="output file")
    p.add_argument('--seed', type=int, default=2023)
    return p

def encode_dnabert(seq: str, k: int):
    seq_new = list()
    N = len(seq)
    seq = "N" * (k//2) + seq.upper() + "N" * k
    for i in range(k//2, N + k//2):
        seq_new.append(seq[i-k//2:i-k//2+k])
    return ' '.join(seq_new)

class FiexedBedData(Dataset):
    def __init__(self, bed, seq_len, genome, tokenizer=None, dnabert=None) -> None:
        super().__init__()
        self.dnabert = dnabert
        if dnabert is not None:
            assert dnabert in {3, 4, 5, 6}
        self.genome = load_fasta(genome)
        self.seq_len = seq_len
        self.bed = bed
        self.tokenizer = tokenizer
        self.samples = list()
        self.name = list()
        self.name2 = list()
        self.process()
    
    def process(self):
        with auto_open(self.bed, 'rt') as f:
            for line in f:
                chrom, start, end, name, name2, strand = line.strip().split('\t')
                start, end = int(start), int(end)
                left = (start + end) // 2 - self.seq_len // 2
                right = left + self.seq_len
                i = start - left
                j = end - left
                if strand == '-':
                    i, j = self.seq_len - 1 - j, self.seq_len - 1 - i
                self.samples.append((chrom, left, right, strand, i, j))
                self.name.append(name)
                self.name2.append(name2)
        self.samples = np.array(self.samples)
        self.name = np.array(self.name)
        self.name2 = np.array(self.name2)
    
    def __getitem__(self, index):
        chrom, left, right, strand, i, j = self.samples[index]
        left, right, i, j = int(left), int(right), int(i), int(j)
        seq = self.genome[chrom][left:right]
        if strand == '-':
            seq = get_reverse_strand(seq)
        if self.tokenizer is None:
            seq = torch.from_numpy(ONEHOT[encode_sequence(seq)])
        elif self.tokenizer == "seq":
            pass
        else:
            if self.dnabert is None:
                seq = torch.as_tensor(self.tokenizer.encode(' '.join(seq.upper())))
            else:
                seq = torch.as_tensor(self.tokenizer.encode(encode_dnabert(seq.upper(), self.dnabert)))
        return seq, i, j, self.name[index], self.name2[index]
    
    def __len__(self):
        return len(self.samples)
    
    def collate_fn(self, batch):
        seq, i, j, name, name2 = map(list, zip(*batch))
        if isinstance(seq[0], str):
            seq = [(str(i), x) for i, x in enumerate(seq)]
        else:
            seq = torch.stack(seq)
        i = np.asarray(i)
        j = np.asarray(j)
        name = np.asarray(name)
        name2 = np.asarray(name2)
        return seq, i, j, name, name2


if __name__ == "__main__":
    args = get_args().parse_args()
    np.random.seed(args.seed)
    print(get_run_info(sys.argv, args=args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k = None
    if args.model == "onehot":
        tokenizer = None
        k = None
    elif args.model.startswith("rnafm"):
        sys.path.append("../..//related/RNA-FM")
        import fm
        model, alphabet = fm.pretrained.rna_fm_t12()
        batch_converter = alphabet.get_batch_converter()
        model.eval()  # disables dropout for deterministic results
        model = model.to(device)
        tokenizer = "seq"
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model, add_pooling_layer=False, output_hidden_states=True).to(device)
        k = np.log2(tokenizer.vocab_size - 5)//2
        if k > 2:
            k = int(round(k))
        else:
            k = None
    ds = FiexedBedData(args.bed, 510, args.genome, tokenizer, dnabert=k)
    batch_size = int(os.environ.get("BATCH_SIZE", 8))
    loader = DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=ds.collate_fn, 
        num_workers=min(os.cpu_count(), batch_size)
    )

    embedding = list()
    for it, (seq, i, j, name, name2) in enumerate(tqdm(loader)):
        if args.model == "onehot":
            embedding.append(seq.numpy()) # (B, S, 4)
        else:
            if tokenizer == "seq":
                batch_labels, batch_strs, batch_tokens = batch_converter(seq)
                batch_tokens = batch_tokens.to(device)
                h = model(batch_tokens, repr_layers=range(0, 13))["representations"]
                h = [h[i].detach() for i in h]
            else:
                seq = seq.to(device)
                # h = torch.stack(model(seq).hidden_states, dim=1).detach().cpu().numpy()
                h = model(seq).hidden_states
            h = torch.stack(h, dim=1).detach()
            del seq
            tmp_embed = list()
            for k in range(h.shape[0]):
                tmp_embed.append(h[k, :, i[k]+1:j[k]+1, :])
            embedding.append(torch.stack(tmp_embed, dim=0).detach().cpu().numpy().astype(np.float16))
    
    # pickle.dump((embedding, ds.name, ds.name2), open(args.output, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    embedding = np.concatenate(embedding, axis=0)

    print("embedding shape:", embedding.shape, file=sys.stderr)

    if args.model == "onehot":
        # embedding: (N, S, 4)
        length = embedding.shape[1]
        for flanking in tqdm([10, 20, 50, 100, 200], desc="UMAP"):
            embed = embedding[:, length//2 - 1 - flanking:length//2 + 1 + flanking, :].reshape(embedding.shape[0], -1) # (B, 2*flanking+2, 4) -> (B, 4*(2*flanking+2))
            adata = sc.AnnData(embed)
            adata.obs['name'] = ds.name
            adata.obs['name2'] = ds.name2
            adata.obs["label"] = [x.split("|")[-1] for x in ds.name]
            if embed.shape[1] > 128:
                sc.pp.pca(adata, n_comps=128, random_state=0)
                sc.pp.neighbors(adata, use_rep='X_pca')
            else:
                sc.pp.neighbors(adata, use_rep='X')
            sc.tl.umap(adata, random_state=0)
            sc.tl.leiden(adata)
            adata.write_h5ad(f"{args.output}.flanking{flanking}.h5ad")
    else:
        if args.skip_donor_acceptor_umap:
            layers = [embedding.shape[1] - 1]
        else:
            layers = range(embedding.shape[1])
        for h in tqdm(layers, desc="UMAP"):
            adata = sc.AnnData(embedding[:, h, :, :].reshape(embedding.shape[0], -1))
            adata.obs['name'] = ds.name
            adata.obs['name2'] = ds.name2
            adata.obs["label"] = [x.split("|")[-1] for x in ds.name]
            sc.pp.pca(adata, n_comps=128, random_state=0)
            sc.pp.neighbors(adata, use_rep='X_pca')
            sc.tl.umap(adata, random_state=0)
            sc.tl.leiden(adata)
            adata.write_h5ad(f"{args.output}.L{h}.h5ad")
            if args.skip_donor_acceptor_umap:
                continue
            is_gt = np.asarray([x.split('|')[-1].startswith("GT") for x in adata.obs["label"]])
            is_ag = np.asarray([x.split('|')[-1].startswith("AG") for x in adata.obs["label"]])
            gt_adata = adata[is_gt]
            ag_adata = adata[is_ag]
            sc.pp.pca(gt_adata, n_comps=128, random_state=0)
            sc.pp.neighbors(gt_adata, use_rep='X_pca')
            sc.tl.umap(gt_adata, random_state=0)
            sc.tl.leiden(gt_adata)
            sc.pp.pca(ag_adata, n_comps=128, random_state=0)
            sc.pp.neighbors(ag_adata, use_rep='X_pca')
            sc.tl.umap(ag_adata, random_state=0)
            sc.tl.leiden(ag_adata)
            gt_adata.write_h5ad(f"{args.output}.L{h}.GT.h5ad")
            ag_adata.write_h5ad(f"{args.output}.L{h}.AG.h5ad")
