#!/usr/bin/env python3
"""
Author: Ken Chen
Email: chenkenbio@gmail.com
"""

import os
import torch
import warnings
import numpy as np
import gzip
from tqdm import tqdm
from collections import defaultdict
from transformers import BertForMaskedLM, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp.autocast_mode import autocast

from utils import get_reverse_strand, load_fasta, distance, fast_seq_to_kmer_list
from config import hg19 as HG19_FASTA
from config import VEXSEQ, MFASS


    
CHROM2INT = {
    "chr1": 0,
    "chr2": 1, 
    "chr3": 2,
    "chr4": 3,
    "chr5": 4,
    "chr6": 5,
    "chr7": 6,
    "chr8": 7,
    "chr9": 8,
    "chr10": 9,
    "chr11": 10,
    "chr12": 11,
    "chr13": 12,
    "chr14": 13,
    "chr15": 14,
    "chr16": 15,
    "chr17": 16,
    "chr18": 17,
    "chr19": 18,
    "chr20": 19,
    "chr21": 20,
    "chr22": 21,
    "chrX": 22,
    "chrY": 23
}
INT2CHROM = {v:k for k, v in CHROM2INT.items()}

def kl_divergence(ref, alt):
    return (alt * np.log(alt / ref)).sum(axis=-1)


class ExonCenterVariant(Dataset):
    def __init__(self, vcf, size: int, genome: str, tokenizer: BertTokenizer, k=1, exon_zero_start=True):
        super().__init__()
        self.k = k
        self.exon_zero_start = exon_zero_start
        self.size = size
        self.genome = load_fasta(genome, cache=True)
        self.samples = list()
        self.tokenizer = tokenizer
        self.distances = list()
        self.mutation_ids = list()
        self.groups = list()
        with gzip.open(vcf, 'rt') as infile:
            for l in infile:
                if l.startswith('#'):
                    continue
                chrom, pos, name, ref, alt = l.strip().split('\t')[:5]
                if ref == '.' or alt == '.':
                    warnings.warn("unsupported allele '.'")
                    continue
                if len(ref) > 1 or len(alt) > 1:
                    warnings.warn("skip indels")
                    continue
                pos = int(pos) - 1
                mut_id, _, e_start, e_end, strand, ref_psi, dpsi, label = name.split('|')
                if not self.exon_zero_start:
                    e_start = int(e_start) - 1
                self.mutation_ids.append(mut_id)
                label = 1 if label == "True" else 0
                self.groups.append(mut_id.split('_')[0])
                self.samples.append((
                    chrom, pos, ref, alt, e_start, e_end, strand, ref_psi, dpsi, label
                ))
                self.distances.append(
                    min(
                        distance(int(e_start) - 2, int(e_start), pos, pos + 1),
                        distance(int(e_end), int(e_end) + 2, pos, pos + 1)
                    )
                )
        self.groups = np.asarray(self.groups)
        self.samples = np.asarray(self.samples)
        self.distances = np.asarray(self.distances)
        self.labels = np.asarray([int(x[-1]) for x in self.samples])
        self.refpsi = np.asarray([float(x[-3]) for x in self.samples])
        self.dpsi = np.asarray([float(x[-2]) for x in self.samples])

        self.is_exonic = list()
        for x in self.samples:
            _, pos, _, _, e1, e2 = x[:6]
            pos, e1, e2 = int(pos), int(e1), int(e2)
            if pos >= e1 and pos < e2:
                self.is_exonic.append(1)
            else:
                self.is_exonic.append(0)
        self.is_exonic = np.asarray(self.is_exonic)

        self.repr_string = "{}(vcf={},genome={}, seq_size={},vocab_size/k={}/{})".format(
            self.__class__.__name__,
            vcf,
            genome,
            size,
            self.tokenizer.vocab_size,
            k
        )
    
    def __repr__(self) -> str:
        return self.repr_string
    
    def __getitem__(self, index, get_seq: bool=False):
        chrom, position, ref, alt, e_start, e_end, strand, _, _, label = self.samples[index]
        position, e_start, e_end, label = int(position), int(e_start), int(e_end), int(label)
        left = (e_start + e_end) // 2 - self.size // 2
        seq_wt = self.genome[chrom][left:left + self.size].upper()
        e_start, e_end = e_start - left, e_end - left
        d_start, d_end = e_end, e_end + 2
        a_start, a_end = e_start - 2, e_start

        pos = position - left
        assert seq_wt[pos] == ref, "{}".format((index))
        seq_wt = list(seq_wt)
        seq_mt = seq_wt.copy()
        seq_mt[pos] = alt
        if strand == '-':
            seq_wt = get_reverse_strand(seq_wt, join=False)
            seq_mt = get_reverse_strand(seq_mt, join=False)

            e_start, e_end = len(seq_wt) - e_end, len(seq_wt) - e_start
            d_start, d_end = e_end, e_end + 2
            a_start, a_end = e_start - 2, e_start
            pos = len(seq_wt) - 1 - pos
        if self.k == 1:
            ids_wt = self.tokenizer.encode(' '.join(seq_wt))
            ids_mt = self.tokenizer.encode(' '.join(seq_mt))
        else:
            seq_wt = ' '.join(fast_seq_to_kmer_list(''.join(seq_wt), k=self.k, pad=True))
            seq_mt = ' '.join(fast_seq_to_kmer_list(''.join(seq_mt), k=self.k, pad=True))
            ids_wt = self.tokenizer.encode(seq_wt)
            ids_mt = self.tokenizer.encode(seq_mt)
        ids_wt = torch.as_tensor(ids_wt).long()
        ids_mt = torch.as_tensor(ids_mt).long()
        if get_seq:
            return ids_wt, ids_mt, pos, d_start, d_end, a_start, a_end, int(label), seq_wt, seq_mt
        else:
            return ids_wt, ids_mt, pos, d_start, d_end, a_start, a_end, int(label), CHROM2INT[chrom], position
    def __len__(self):
        return len(self.samples)
    
    def check_dataset(self):
        d_counts = defaultdict(int)
        a_counts = defaultdict(int)
        noncanonical = list()
        unmatched = list()
        for idx in tqdm(range(len(self.samples))):
            ids_wt, ids_mt, pos, d_start, d_end, a_start, a_end, label, seq_wt, seq_mt = self.__getitem__(idx, get_seq=True)
            d_counts["{}{}".format(*ids_wt[d_start + 1:d_end + 1].numpy())] += 1
            d_counts[''.join(seq_wt[d_start:d_end])] += 1
            a_counts["{}{}".format(*ids_wt[a_start + 1:a_end + 1].numpy())] += 1
            a_counts[''.join(seq_wt[a_start:a_end])] += 1
            if seq_wt[d_start:d_end] != "GT" or seq_wt[a_start:a_end] != "AG":
                noncanonical.append((idx, self.mutation_ids[idx], seq_wt[d_start:d_end], seq_wt[a_start:a_end], self.samples[idx]))
            if ids_wt[pos + 1] == ids_mt[pos + 1] or seq_wt[pos] == seq_mt[pos]:
                unmatched.append(idx)
        return dict(d_counts=d_counts, a_counts=a_counts, unmatched=unmatched, noncanonical=noncanonical)
    
    @classmethod
    def load_mfass(self, tokenizer, size=510, k=1):
        return self(vcf=MFASS, size=size, genome=HG19_FASTA, tokenizer=tokenizer, k=k)
        
    @classmethod
    def load_vexseq(self, tokenizer, size=510, k=1):
        return self(vcf=VEXSEQ, size=size, genome=HG19_FASTA, tokenizer=tokenizer, k=k, exon_zero_start=False)
        


@autocast()
@torch.no_grad()
def evaluate_mutation(model: BertForMaskedLM, ds: Dataset, batch_size: int=16, num_workers=8):
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    all_wt_logits = list()
    all_mt_logits = list()
    all_positions = list()
    all_labels = list()
    all_donors = list()
    all_acceptors = list()
    device = next(model.parameters()).device
    for ids_wt, ids_mt, pos, d_start, d_end, a_start, a_end, label in tqdm(loader):
        ids_wt, ids_mt = ids_wt.to(device), ids_mt.to(device)
        logits_wt = model.forward(ids_wt).logits[:, 1:-1, 5:].detach()
        logits_mt = model.forward(ids_mt).logits[:, 1:-1, 5:].detach()
        logits_wt = torch.softmax(logits_wt, dim=-1).cpu().numpy()
        logits_mt = torch.softmax(logits_mt, dim=-1).cpu().numpy()
        all_wt_logits.append(logits_wt)
        all_mt_logits.append(logits_mt)
        all_labels.append(label.numpy())
        all_positions.append(pos.numpy())
        all_donors.append(np.asarray([(a, b) for a, b in zip(d_start.numpy(), d_end.numpy())]))
        all_acceptors.append(np.asarray([(a, b) for a, b in zip(a_start.numpy(), a_end.numpy())]))
    all_wt_logits = np.concatenate(all_wt_logits)
    all_mt_logits = np.concatenate(all_mt_logits)
    all_positions = np.concatenate(all_positions)
    all_donors = np.concatenate(all_donors)
    all_acceptors = np.concatenate(all_acceptors)
    all_labels = np.concatenate(all_labels)
    return all_wt_logits, all_mt_logits, all_positions, all_donors, all_acceptors, all_labels


# def get_args():
#     p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     p.add_argument("--model-path", default="../pretrained/mlm_on_tx", required=True)
#     p.add_argument("--mask-ss", action="store_true")
#     p.add_argument("--mask-mut", action="store_true")
#     p.add_argument('-o', "--outdir", required=True)
#     # p.add_argument('--seed', type=int, default=2020)
#     return p


if __name__ == "__main__":
    pass

#     args = get_args().parse_args()
#     args.outdir = make_directory(args.outdir)
#     logger = make_logger(filename=os.path.join(args.outdir, "train.log"))
#     logger.info(get_run_info(argv=sys.argv, args=args))

#     ds = ExonCenterVariant(
#         vcf="../data/mfass/MFASS_mmsplice.vcf",
#         size=510,
#         genome=HG19_FASTA,
#         # mask_mut=args.mask_mut,
#         # mask_ss=args.mask_ss
#     )
#     pickle.dump(ds.labels, open("{}/labels.pkl".format(args.outdir), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

#     NN2INT = {
#         'N': 0,
#         'A': 1,
#         'C': 2,
#         'G': 3,
#         'T': 4
#     }

#     fn_logits = "{}/logits.pt".format(args.outdir)
#     if os.path.exists(fn_logits):
#         logger.info("- use {}".format(fn_logits))
#         all_wt_logits, all_mt_logits, all_positions, all_donors, all_acceptors, all_labels = pickle.load(open(fn_logits, 'rb'))
#     else:
#         device = torch.device("cuda")
#         bert = BertForMaskedLM.from_pretrained(args.model_path).to(device)

#         all_wt_logits, all_mt_logits, all_positions, all_donors, all_acceptors, all_labels = evaluate_mutation(bert.to(device), ds, batch_size=24, num_workers=12)

#         pickle.dump(
#             (all_wt_logits, all_mt_logits, all_positions, all_donors, all_acceptors, all_labels), 
#             open(fn_logits, 'wb'), 
#             protocol=pickle.HIGHEST_PROTOCOL
#         )

#     if args.mask_mut:
#         all_mut_scores = list()
#         for idx in tqdm(range(all_wt_logits.shape[0])):
#             ref, alt = ds.samples[idx][2:4]
#             p = all_positions[idx]
#             ar = all_wt_logits[idx, p]
#             mut_score = min(max(ar[NN2INT[alt]], 1E-6), 1 - 1E-6) / min(max(ar[NN2INT[ref]], 1E-6), 1 - 1E-6)
#             all_mut_scores.append(mut_score)
#         all_mut_scores = np.asarray(all_mut_scores)
#         pickle.dump(all_mut_scores, open("{}/all_mut_scores.pkl".format(args.outdir), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
#     else:
#         all_ss_scores = list()
#         for idx in tqdm(range(all_wt_logits.shape[0])):
#             d1, d2 = all_donors[idx]
#             a1, a2 = all_acceptors[idx]
#             ss_score = max(
#                 kl_divergence(all_wt_logits[idx, d1], all_mt_logits[idx, d1]) + \
#                     kl_divergence(all_wt_logits[idx, d1 + 1], all_mt_logits[idx, d1 + 1]), 
#                 kl_divergence(all_wt_logits[idx, a1], all_mt_logits[idx, a1]) + \
#                     kl_divergence(all_wt_logits[idx, a1 + 1], all_mt_logits[idx, a1 + 1])
#             )
#             all_ss_scores.append(ss_score)
#         all_ss_scores = np.asarray(all_ss_scores)
#         if args.mask_ss:
#             pickle.dump(all_ss_scores, open("{}/all_ss_scores.mask_ss.pkl".format(args.outdir), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
#         else:
#             pickle.dump(all_ss_scores, open("{}/all_ss_scores.pkl".format(args.outdir), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


# #         average_precision_score(all_labels, all_scores), roc_auc_score(all_labels, all_scores)
# # 
# # 
# # 
# # 
# #         average_precision_score(all_labels, all_ss_scores_mask_ss), roc_auc_score(all_labels, all_ss_scores_mask_ss)
# # 
# # 
# # 
# # 
# #         p, r, _ = precision_recall_curve(all_labels, all_scores)
# #         f, t, _ = roc_curve(all_labels, all_scores)
# # 
# # 
# # 
# # 
# #         ax = plt.subplot(121)
# #         ax.plot(r, p)
# #         ax = plt.subplot(122)
# #         ax.plot(f, t)
# # 
# # 
# # 
# # 
# #         all_dpsi = np.asarray([float(x[-2]) for x in ds.samples])
# #         all_refpsi = np.asarray([float(x[-3]) for x in ds.samples])
# # 
# # 
# # 
# # 
# #         average_precision_score(np.abs(all_dpsi) > 0.4, all_scores),     average_precision_score(all_labels, all_scores),     average_precision_score(all_dpsi < 0, all_scores)
# # 
# # 
# # 
# # 
# #         import pyBigWig
# # 
# # 
# # 
# # 
# #         bw.values('chrX', 48837917, 48837917+1)
# # 
# # 
# # 
# # 
# #         phast46_scores = list()
# #         bw = pyBigWig.open("/bigdat1/pub/UCSC/hg19/vertebrate.phastCons46way.bw")
# #         for s in tqdm(ds.samples):
# #             chrom, position = s[:2]
# #             phast46_scores.append(
# #                 bw.values(chrom, int(position), int(position) + 1)[0]
# #             )
# #         bw.close()
# # 
# #         phast100_scores = list()
# #         bw = pyBigWig.open("/bigdat1/pub/UCSC/hg19/hg19.100way.phastCons.bw")
# #         for s in tqdm(ds.samples):
# #             chrom, position = s[:2]
# #             phast100_scores.append(
# #                 bw.values(chrom, int(position), int(position) + 1)[0]
# #             )
# #         bw.close()
# # 
# # 
# # 
# # 
# #         # phast46_scores = np.asarray(phast46_scores)
# #         phast100_scores = np.asarray(phast100_scores)
# # 
# # 
# # 
# # 
# #         average_precision_score(all_labels, phast46_scores), roc_auc_score(all_labels, phast46_scores)
# # 
# # 
# # 
# # 
# #         average_precision_score(all_labels, phast100_scores), roc_auc_score(all_labels, phast100_scores)
# # 
# # 
# # 
# # 
# #         torch.save(phast46_scores, "phast46_scores.pt")
# #         torch.save(phast100_scores, "phast100_scores.pt")
# # 
# # 
# # 
# # 
# #         phast100_scores = torch.load("./phast100_scores.pt")
# # 
# # 
# # 
# # 
# #         phast100_scores
# # 
# # 
# # 
# # 
# #         X = np.stack([phast100_scores, all_scores, ds.distances]).T
# # 
# # 
# # 
# # 
# #         lr.fit(X, y=ds.labels)
# # 
# # 
# # 
# # 
# #         y = lr.predict_log_proba(X).T[1]
# # 
# # 
# # 
# # 
# #         average_precision_score(ds.labels, all_scores), roc_auc_score(ds.labels, all_scores)
# # 
# # 
# # 
# # 
# #         average_precision_score(ds.labels, y), roc_auc_score(ds.labels, y)
# # 
# # 
# # 
# # 
# #         average_precision_score(ds.labels, y), roc_auc_score(ds.labels, y)
# # 
# # 
# # 
# # 
# # 
# # 
# # 