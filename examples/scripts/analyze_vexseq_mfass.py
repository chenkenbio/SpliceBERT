#!/usr/bin/env python3
"""
Author: Ken Chen
Email: chenkenbio@gmail.com
"""

import argparse
import os
import sys
from tqdm import tqdm
import numpy as np
import h5py
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict
from torch.cuda.amp import autocast
from transformers import AutoModelForMaskedLM, AutoTokenizer

import misc_for_variant_analysis
from misc_for_variant_analysis import INT2CHROM
from utils import get_run_info, make_logger
from config import hg19_phastcons as HG19_PHASTCONS
from config import hg19_phylop as HG19_PHYLOP

def to_rank(x):
    raw_idx = [t[1] for t in sorted(zip(x, np.arange(len(x))), key=lambda ar:ar[0], reverse=False)]
    new_score = [t[1] for t in sorted(zip(raw_idx, np.arange(1, 1 + len(raw_idx))/len(raw_idx)), key=lambda ar:ar[0])]
    return np.asarray(new_score)

def normalize(x):
    xmin, xmax = x.min(), x.max()
    return (x - xmin) / (xmax - xmin)

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-m', "--model-path", type=str, required=True)
    p.add_argument("-o", required=True)
    p.add_argument("-d", "--dataset", choices=("vexseq", "mfass"), required=True)
    # p.add_argument('--seed', type=int, default=2020)
    return p


if __name__ == "__main__":
    args = get_args().parse_args()

    logger = make_logger()
    logger.info(get_run_info(sys.argv, args))

    hg19_phastcons = h5py.File(HG19_PHASTCONS, 'r')
    hg19_phylop = h5py.File(HG19_PHYLOP, 'r')

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dnabert_mode = False
    if tokenizer.vocab_size > 10:
        k = int(np.log2(tokenizer.vocab_size - 5) // 2)
        dnabert_mode = True
    else:
        k = 1
    logger.info("- Tokenizer: k={}\n{}".format(k, tokenizer))

    if args.dataset == "vexseq":
        ds = misc_for_variant_analysis.ExonCenterVariant.load_vexseq(tokenizer=tokenizer, k=k)
    else:
        ds = misc_for_variant_analysis.ExonCenterVariant.load_mfass(tokenizer=tokenizer, k=k)
    loader = DataLoader(ds, batch_size=16, num_workers=16)

    device = torch.device("cuda")

    model = AutoModelForMaskedLM.from_pretrained(args.model_path)
    model = model.to(device)

    all_wt_logits = list()
    all_mt_logits = list()
    all_phastcons = list()
    all_phylop = list()
    all_positions = list()
    all_pos = list()
    all_labels = list()
    all_d1, all_d2 = list(), list()
    all_a1, all_a2 = list(), list()
    all_ref, all_alt = list(), list()
    all_masked_prob = list()
    all_distance = list()
    d_kl = list()
    a_kl = list()
    kl_distance = list()
    flanking = 100
    all_neighbor = list()
    all_prob_scores = list()

    with torch.no_grad(), autocast():
        model.eval()
        for ids_wt, ids_mt, pos, d_start, d_end, a_start, a_end, label, chrom, position in tqdm(loader):
            all_distance.append(torch.minimum(abs(pos - d_start), abs(pos - (a_end - 1))).numpy())
            out_wt = model(ids_wt.to(device))
            out_mt = model(ids_mt.to(device))
            masked_ids = list()
            for i in range(ids_wt.shape[0]):
                ids = ids_wt[i].clone()
                ids[pos[i] + 1] = ds.tokenizer.mask_token_id
                masked_ids.append(ids)
            masked_ids = torch.stack(masked_ids)
            masked_logits = torch.softmax(model(masked_ids.to(device)).logits, dim=-1).detach().cpu().numpy()
            for i in range(ids_wt.shape[0]):
                all_ref.append(ids_wt[i, pos[i] + 1].item())
                all_alt.append(ids_mt[i, pos[i] + 1].item())
                ref = all_ref[-1]
                alt = all_alt[-1]
                if not dnabert_mode:
                    all_masked_prob.append(masked_logits[i, pos[i] + 1])
                all_prob_scores.append(masked_logits[i, pos[i] + 1][alt] - masked_logits[i, pos[i] + 1][ref])
                

            wt_logits = torch.softmax(out_wt.logits.detach()[:, 1:-1, 6:], dim=-1).cpu()
            mt_logits = torch.softmax(out_mt.logits.detach()[:, 1:-1, 6:], dim=-1).cpu()

            if not dnabert_mode:
                all_wt_logits.append(wt_logits.numpy())
                all_mt_logits.append(mt_logits.numpy())
            else:
                # DNABERT token logits (64d-4096d) for the whole sequence is too large to save
                kl_div = misc_for_variant_analysis.kl_divergence(wt_logits, mt_logits).numpy()
                for i in range(kl_div.shape[0]):
                    d_kl.append(kl_div[i, d_start[i]:d_start[i]+2].mean())
                    a_kl.append(kl_div[i, a_start[i]:a_start[i]+2].mean())
                    neighbor = kl_div[i, pos[i]-flanking:pos[i]+flanking + 1]
                    kl_distance.append(neighbor[:flanking][::-1]/2 + neighbor[101:]/2)
                    # all_neighbor.append(neighbor)
            chrom = [INT2CHROM[c.item()] for c in chrom]
            all_phastcons.append(np.asarray([hg19_phastcons[c][p] for c, p in zip(chrom, position.numpy())], dtype=np.float16))
            all_phylop.append(np.asarray([hg19_phylop[c][p] for c, p in zip(chrom, position.numpy())], dtype=np.float16))
            all_pos.append(pos.numpy())
            all_labels.append(label.numpy())
            all_d1.append(d_start.numpy())
            all_d2.append(d_end.numpy())
            all_a1.append(a_start.numpy())
            all_a2.append(a_end.numpy())



    all_distance = np.concatenate(all_distance)
    logger.info("Distance: {}".format(all_distance.shape))
    all_phastcons = np.concatenate(all_phastcons)
    logger.info("Phastcons: {}".format(all_phastcons.shape))
    all_phylop = np.concatenate(all_phylop)
    logger.info("Phylop: {}".format(all_phylop.shape))
    all_pos = np.concatenate(all_pos)
    logger.info("Pos: {}".format(all_pos.shape))
    all_labels = np.concatenate(all_labels)
    logger.info("Labels: {}".format(all_labels.shape))
    all_d1 = np.concatenate(all_d1)
    logger.info("D1: {}".format(all_d1.shape))
    all_d2 = np.concatenate(all_d2)
    logger.info("D2: {}".format(all_d2.shape))
    all_a1 = np.concatenate(all_a1)
    logger.info("A1: {}".format(all_a1.shape))
    all_a2 = np.concatenate(all_a2)
    logger.info("A2: {}".format(all_a2.shape))
    if len(all_masked_prob) > 0:
        all_masked_prob = np.stack(all_masked_prob)
        logger.info("Masked prob: {}".format(all_masked_prob.shape))

    if not dnabert_mode:
        all_prob_scores = all_masked_prob[np.arange(len(all_masked_prob)), all_alt] - all_masked_prob[np.arange(len(all_masked_prob)), all_ref]
        # all_masked_prob = np.stack(all_masked_prob)
        all_wt_logits = np.concatenate(all_wt_logits)
        all_mt_logits = np.concatenate(all_mt_logits)
        all_kl = misc_for_variant_analysis.kl_divergence(all_wt_logits, all_mt_logits)
        neighboring = list()
        d_kl = list()
        a_kl = list()
        logit_change = list()
        for i in range(all_kl.shape[0]):
            neighboring.append(all_kl[i][all_pos[i] - flanking:all_pos[i] + flanking + 1])
            d_kl.append(all_kl[i][all_d1[i]:all_d1[i] + 2].mean(axis=0))
            a_kl.append(all_kl[i][all_a1[i] + 0:all_a1[i] + 2].mean(axis=0))
        neighboring = np.stack(neighboring)
        d_kl = np.asarray(d_kl)
        a_kl = np.asarray(a_kl)
        kl_distance = (neighboring[:, :100][:, ::-1] + neighboring[:, 101:]) / 2
    else:
        d_kl = np.asarray(d_kl)
        a_kl = np.asarray(a_kl)
        # kl_distance = list()
        kl_distance = np.stack(kl_distance)
        all_kl = list()
        # neighboring = np.stack(all_neighbor)
        neighboring = list()
        all_prob_scores = np.asarray(all_prob_scores)

    output = OrderedDict(
        wt_logits=all_wt_logits,
        mt_logits=all_mt_logits,
        all_kl=all_kl,
        masked_probs=all_masked_prob,
        prob_change_scores=all_prob_scores,
        donor_starts=all_d1, 
        donor_ends=all_d2,
        acceptor_starts=all_a1, 
        acceptor_ends=all_a2,
        donor_kl=d_kl,
        acceptor_kl=a_kl,
        flanking_kl=neighboring,
        kl_distance=kl_distance,
        phastcons_scores=all_phastcons,
        phylop_scores=all_phylop,
        ref_psi=ds.refpsi,
        delta_psi=ds.dpsi,
        is_exonic=ds.is_exonic,
        distance=all_distance,
        labels=ds.labels,
        dataset=str(ds)
    )



    pickle.dump(output, open(args.o, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    
