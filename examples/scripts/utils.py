#!/usr/bin/env python3
"""
Author: Ken Chen
Email: chenkenbio@gmail.com
"""

import argparse
import os
import sys
import numpy as np
import gzip
import json
import warnings
import pickle
import time
import shutil
import random
import torch
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
from io import TextIOWrapper
from _encode_sequence import _encode_sequence
from _sequence_to_kmer_list import _fast_seq_to_kmer_list

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import logging
logger = logging.getLogger(__name__)

def scientific_notation(x, decimal: int=3):
    template = "{:." + str(decimal) + "e}"
    number, exp = template.format(x).split('e')
    exp = int(exp)
    return r"$%s\times 10^{%d}$" % (number, exp)

def auto_open(input: Union[str, TextIOWrapper], mode='rt') -> TextIOWrapper:
    if isinstance(input, str):
        if input == '-':
            return sys.stdin
        elif input.endswith(".gz") or input.endswith(".bgz"):
            return gzip.open(input, mode=mode)
        else:
            return open(input, mode=mode)
    elif isinstance(input, TextIOWrapper):
        return input
    else:
        raise IOError("Unknown input type {}".format(type(input)))

def load_fasta(fn: str, no_chr: bool = False, ordered: bool = False, cache: bool = True, gencode_style: bool = False) -> Dict[str, str]:
    r"""
    load fasta as sequence dict
    Input
    -----
    fn : path to fasta file
    ordered : False - dict, True - OrderedDict
    gencode_style : seq_id|xxxxxx

    Return
    -------
    seq_dict : Dict[str, str] or OrderedDict[str, str]
    """
    # if fn == "GRCh38":
    #     fn = HG38_FASTA
    #     logger.warning("- using {}".format(fn))
    #     cache = True
    # elif fn == "GRCh37":
    #     fn = HG19_FASTA
    #     logger.warning("- using {}".format(fn))
    #     cache = True

    if ordered:
        fasta = OrderedDict()
    else:
        fasta = dict()
    name, seq = None, list()
    if cache:
        if no_chr:
            cache = fn + \
                (".gencode.nochr.cache.pkl" if gencode_style else ".nochr.cache.pkl")
        else:
            cache = fn + \
                (".gencode.cache.pkl" if gencode_style else ".cache.pkl")
    else:
        cache = None
    if cache is not None and os.path.exists(cache):
        # logger.info("- load processed genome: {}".format(cache))
        logger.warning("- load processed genome: {}".format(cache))
        fasta = pickle.load(open(cache, 'rb'))
    else:
        with auto_open(fn) as infile:
            for l in infile:
                if l.startswith('>'):
                    if name is not None:
                        # print("{}\n{}".format(name, ''.join(seq)))
                        if no_chr:
                            name = name.replace("chr", '')
                        fasta[name] = ''.join(seq)
                    if gencode_style:
                        name = l.strip().lstrip('>').split('|')[0]
                    else:
                        name = l.strip().lstrip('>').split()[0]
                    seq = list()
                else:
                    seq.append(l.strip())
        fasta[name] = ''.join(seq)
        if cache is not None:
            try:
                pickle.dump(fasta, open(cache, 'wb'),
                            protocol=pickle.HIGHEST_PROTOCOL)
            except IOError as err:
                warnings.warn("{}".format(err))
    return fasta

def encode_sequence(seq: str, token_dict: Dict[str, int]=None) -> List[int]:
    if token_dict is None:
        ids = _encode_sequence(seq)
    else:
        ids = _encode_sequence(seq, token_dict)
    return ids

NN_COMPLEMENT = {
    'A': 'T', 'a': 't',
    'C': 'G', 'c': 'g',
    'G': 'C', 'g': 'c',
    'T': 'A', 't': 'a',
    'R': 'Y', 'r': 'y',
    'Y': 'R', 'y': 'r',
    'S': 'S', 's': 's',
    'W': 'W', 'w': 'w',
    'K': 'M', 'k': 'm',
    'M': 'K', 'm': 'k',
    'B': 'V', 'b': 'v',
    'D': 'H', 'd': 'h',
    'H': 'D', 'h': 'd',
    'V': 'B', 'v': 'b',
    'N': 'N', 'n': 'n'
}
NN_COMPLEMENT_INT = np.array([0, 4, 3, 2, 1]) # 0: N, 1: A, 2: C, 3: G, 4: T

def get_reverse_strand(seq, join: bool = True, integer: bool = False):
    is_iter = True
    if not hasattr(seq, "__iter__"):
        is_iter = False
        seq = [seq]
    if integer:
        seq = NN_COMPLEMENT_INT[seq][::-1].copy()
    else:
        if join:
            seq = ''.join([NN_COMPLEMENT.get(n, n) for n in seq[::-1]])
        else:
            seq = [NN_COMPLEMENT.get(n, n) for n in seq[::-1]]
    if not is_iter:
        seq = seq[0]
    return seq


def set_spines(ax: Axes, left=True, right=False, top=False, bottom=True):
    ax.spines["left"].set_visible(left)
    ax.spines["right"].set_visible(right)
    ax.spines["top"].set_visible(top)
    ax.spines["bottom"].set_visible(bottom)


def get_fontsize_rc_params(fontsize=8, titlesize=None, labelsize=None, ticklabelsize=None, legendfontsize=None, figuretitlesize=None):
    font_rc_params = dict()
    font_rc_params["font.size"] = fontsize
    font_rc_params["axes.titlesize"] = fontsize if titlesize is None else titlesize
    font_rc_params["axes.labelsize"] = fontsize if labelsize is None else fontsize
    font_rc_params["xtick.labelsize"] = fontsize if ticklabelsize is None else fontsize
    font_rc_params["ytick.labelsize"] = fontsize if ticklabelsize is None else fontsize
    font_rc_params["legend.fontsize"] = fontsize if legendfontsize is None else fontsize
    font_rc_params["figure.titlesize"] = fontsize if figuretitlesize is None else fontsize
    return font_rc_params

def get_figure_size(width: float, height: float=None, page_width: Literal["A4", "USLetter"]=8.3) -> Tuple[float, float]:
    r"""
    Input:
    ---
    width: float, ratio of the width of the figure to the page width
    height: float, ratio of the height of the figure to the page width
    Return:
    ---
    (width, height): float
    """
    if page_width == "A4":
        page_width = 8.3
    elif page_width == "USLetter":
        page_width = 8.5
    else:
        assert isinstance(page_width, float), "page_width must be either 'A4' or 'USLetter' or a float (inch)."
    if height is None:
        height = width
    return width * page_width, height * page_width


def get_run_info(argv: List[str], args: argparse.Namespace=None, **kwargs) -> str:
    s = list()
    s.append("")
    s.append("##time: {}".format(time.asctime()))
    s.append("##cwd: {}".format(os.getcwd()))
    s.append("##cmd: {}".format(' '.join(argv)))
    if args is not None:
        s.append("##args: {}".format(args))
    for k, v in kwargs.items():
        s.append("##{}: {}".format(k, v))
    return '\n'.join(s)



def make_logger(
        title: Optional[str]="", 
        filename: Optional[str]=None, 
        level: Literal["INFO", "DEBUG"]="INFO", 
        mode: Literal['w', 'a']='w',
        trace: bool=True, 
        **kwargs):
    if isinstance(level, str):
        level = getattr(logging, level)
    logger = logging.getLogger(title)
    logger.setLevel(level)
    sh = logging.StreamHandler()
    sh.setLevel(level)

    if trace is True or ("show_line" in kwargs and kwargs["show_line"] is True):
        formatter = logging.Formatter(
                '%(levelname)s(%(asctime)s) [%(filename)s:%(lineno)d]:%(message)s', datefmt='%Y%m%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(levelname)s(%(asctime)s):%(message)s', datefmt='%Y%m%d %H:%M:%S'
        )
    # formatter = logging.Formatter(
    #     '%(message)s\t%(levelname)s(%(asctime)s)', datefmt='%Y%m%d %H:%M:%S'
    # )

    sh.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(sh)

    if filename is not None:
        if os.path.exists(filename):
            i = 0
            # suffix = time.strftime("%Y%m%d-%H%M%S", time.localtime(os.path.getmtime(filename)))
            suffix = ".{}.log".format(i)
            while os.path.exists(filename + suffix):
                i += 1
                suffix = ".{}.log".format(i)
                # suffix = "{}_1".format(suffix)
            shutil.move(filename, filename + suffix)
            warnings.warn("log {} exists, moved to {}".format(filename, filename + suffix))
        fh = logging.FileHandler(filename=filename, mode=mode)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def overlap_length(x1, x2, y1, y2):
    """ [x1, x2), [y1, y2) """
    length = 0
    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
    if x2 <= y1:
        length = x2 - y1
    elif x1 <= y2:
        length = min(x2, y2) - max(x1, y1)
    else:
        length = y2 - x1
    return length

def distance(x1, x2, y1, y2, nonoverlap=False):
    """ interval distance """
    d = overlap_length(x1, x2, y1, y2)
    if nonoverlap and d < 0:
        warnings.warn("[{}, {}) overlaps with [{}, {})".format(x1, x2, y1, y2))
    return max(-d, 0)

def make_directory(in_dir):
    if os.path.isfile(in_dir):
        warnings.warn("{} is a regular file".format(in_dir))
        return None
    outdir = in_dir.rstrip('/')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    return outdir


def fast_seq_to_kmer_list(text: str, k=1, pad: bool=False, shift: int=None):
    if shift is None:
        shift = 1
    if k == 1:
        assert shift == 1
        return ' '.join(list(text))
    else:
        if pad:
            return _fast_seq_to_kmer_list(text, k, k//2, k, shift)
        else:
            return _fast_seq_to_kmer_list(text, k, 0, k, shift)

class Transcript(object):

    def __init__(self, tx_id, gene_id, chrom, tx_start, tx_end, strand, cds_start=None, cds_end=None, buildver=None, gene_name=None, **kwargs) -> None:
        self.tx_id = tx_id
        self.gene_id = gene_id
        self.chrom = chrom
        self.tx_start = tx_start
        self.tx_end = tx_end
        self.strand = strand
        self.cds_start = cds_start
        self.cds_end = cds_end
        self.buildver = buildver
        self.gene_name = gene_name
        self.gene_type = kwargs.get("gene_type", None)
        self._cds: List[Tuple[int, int]]=list()
        self._utr5: List[Tuple[int, int]]=list()
        self._utr3: List[Tuple[int, int]]=list()
        for k, v in kwargs.items():
            self.__dict__[k] = v

    def __repr__(self) -> str:
        repr = list()
        for k, v in [("tx_id", self.tx_id), ("gene_id", self.gene_id), ("gene_name",  self.gene_name), ("chrom", self.chrom), ("tx_start", self.tx_start), ("tx_end", self.tx_end), ("strand", self.strand)]:
            if v is not None:
                repr.append("{}={}".format(k, v))
        return "Transcript({})".format(",".join(repr))
    
    @property
    def intron_starts(self):
        if not hasattr(self, "_intron_starts"):
            self._intron_starts = self.exon_ends[0:-1]
        return self._intron_starts
    @property
    def intron_ends(self):
        if not hasattr(self, "_intron_ends"):
            self._intron_ends = self.exon_starts[1:]
        return self._intron_ends

    @property
    def intron(self):
        return list(zip(self.intron_starts, self.intron_ends))

    @property
    def exon(self):
        return list(zip(self.exon_starts, self.exon_ends))
    
    def _get_utr_cds(self):
        assert self.cds_start is not None and self.cds_end is not None, "the start/end of CDS is unknown"
        if len(self._utr5) == 0 and len(self._cds) == 0 and len(self._utr3) == 0:
            utr_left, utr_right = list(), list()
            for e1, e2 in zip(self.exon_starts, self.exon_ends):
                left, right = max(e1, self.cds_start), min(e2, self.cds_end)
                if left < right:
                    self._cds.append((left, right))
                if e1 < self.cds_start:
                    utr_left.append((e1, min(e2, self.cds_start)))
                if e2 > self.cds_end:
                    utr_right.append((max(self.cds_end, e1), e2))
            if self.strand == '+':
                self._utr5 = utr_left
                self._utr3 = utr_right
            else:
                self._utr5 = utr_right
                self._utr3 = utr_left
    
    def check(self):
        # check cds&utr
        exon_size, cds_size, utr_size = 0, 0, 0
        if self.strand == '+':
            for x, y in self.utr5:
                utr_size += y - x
                assert x < y, "{}".format(((x, y), self.cds_start, self.cds_end))
                assert y <= self.cds_start, "{}".format(((x, y), self.cds_start, self.cds_end))
            for x, y in self.utr3:
                utr_size += y - x
                assert x < y, "{}".format(((x, y), self.cds_start, self.cds_end))
                assert x >= self.cds_end, "{}".format(((x, y), self.cds_start, self.cds_end))
        else:
            for x, y in self.utr5:
                utr_size += y - x
                assert x < y, "{}".format(((x, y), self.cds_start, self.cds_end))
                assert x >= self.cds_end, "{}".format(((x, y), self.cds_start, self.cds_end))
            for x, y in self.utr3:
                utr_size += y - x
                assert x < y, "{}".format(((x, y), self.cds_start, self.cds_end))
                assert y <= self.cds_start, "{}".format(((x, y), self.cds_start, self.cds_end))
        for x, y in self.cds:
            assert x >= self.cds_start, "{}".format(((x, y), self.cds_start, self.cds_end))
            assert y <= self.cds_end, "{}".format(((x, y), self.cds_start, self.cds_end))
            cds_size += y - x
        exon_size = sum([y - x for x, y in zip(self.exon_starts, self.exon_ends)])
        assert exon_size == cds_size + utr_size, "{}".format((exon_size, utr_size, cds_size))
        intervals = sorted(self.utr5 + self.utr3 + self.cds + self.intron, key=lambda x:x[0])
        for i in range(len(intervals) - 1):
            assert intervals[i][1] == intervals[i +  1][0], ';'.join(["({},{})".format(x, y) for x, y in intervals])

    @property
    def cds(self):
        if len(self._cds) == 0:
            self._get_utr_cds()
        return self._cds

    @property
    def utr5(self):
        if len(self._utr5) == 0:
            self._get_utr_cds()
        return self._utr5

    @property
    def utr3(self):
        if len(self._utr3) == 0:
            self._get_utr_cds()
        return self._utr3

    def get_function_region(self, function_region: Literal["exon", "intron", "ss-exon", "ss-intron-1", "ss-intron-2"], name: List[str]='.', name_sep: str='|', numbering: bool=False) -> List[Tuple[str, int, int, str, str, str]]:
        r"""
        Arguments:
        ---
        function_region: 
            exon: 
            intron:
            ss-exon: splice sites in exons (1nt)
            ss-intron-1: splice sites in introns (1nt)
            ss-intron-2: splice sites in introns (2nt)
        name : List[str] feature names to be shown in 'name'
        name_sep : str
        numbering : bool : whether add exon/intron number

        Return:
        ---
        bed_list : (chrom, start, end, name, ., strand)
        """
        if type(name) is str and name != '.':
            name = [name]
        name_prefix = list()
        for k in name:
            if k == '.':
                name_prefix = ""
            elif self.__dict__[k] is None:
                name_prefix.append('NaN')
            else:
                name_prefix.append(self.__dict__[k])
        name_prefix = name_sep.join(name_prefix)

        intervals = list()
        if function_region == "exon":
            for i, (r1, r2) in enumerate(zip(self.exon_starts, self.exon_ends)):
                if numbering:
                    if self.strand == '-':
                        exon_id = "|EXON{}".format(len(self.exon_starts) - i)
                    else:
                        exon_id = "|EXON{}".format(i + 1)
                    iname = exon_id
                else:
                    iname = ""
                intervals.append((r1, r2, "exon{}".format(iname)))
        elif function_region == "intron":
            for i, (r1, r2) in enumerate(zip(self.intron_starts, self.intron_ends)):
                intron_id = ""
                if numbering:
                    if self.strand == '-':
                        intron_id = "|INT{}".format(len(self.intron_starts) - i)
                    else:
                        intron_id = "|INT{}".format(i + 1)
                intervals.append((r1, r2, "intron{}".format(intron_id)))
        elif function_region == "ss-exon":
            for i, (d, a) in enumerate(zip(self.intron_starts, self.intron_ends)):
                if self.strand == '-':
                    intervals.append((d - 1, d, "3'SS"))
                    intervals.append((a, a + 1, "5'SS"))
                else:
                    intervals.append((d - 1, d, "5'SS"))
                    intervals.append((a, a + 1, "3'SS"))
        elif function_region == "ss-intron-1":
            for i, (d, a) in enumerate(zip(self.intron_starts, self.intron_ends)):
                if self.strand == '-':
                    intervals.append((d, d + 1, "acceptor"))
                    intervals.append((a - 1, a, "donor"))
                else:
                    intervals.append((d, d + 1, "donor"))
                    intervals.append((a - 1, a, "acceptor"))
        elif function_region == "ss-intron-2":
            for i, (d, a) in enumerate(zip(self.intron_starts, self.intron_ends)):
                if self.strand == '-':
                    intervals.append((d, d + 2, "acceptor"))
                    intervals.append((a - 2, a, "donor"))
                else:
                    intervals.append((d, d + 2, "donor"))
                    intervals.append((a - 2, a, "acceptor"))
        else:
            raise NotImplementedError("unknown function_region: {}".format(function_region))

        bed_list = list()

        if name_prefix != '':
            name_prefix = name_prefix + name_sep

        for l, r, rname in intervals:
            bed_list.append((self.chrom, l, r, "{}{}".format(name_prefix, rname), '.', self.strand))

        return bed_list

   
class GenePredRecord(Transcript):
    def __init__(self, line: str) -> None:
        r"""
        line: gene record line
        """
        tx_id, chrom, strand, tx_start, tx_end, cds_start, cds_end, num_exon, exon_starts, exon_ends, _, gene_id, _, _, _ = line.strip('\n').split('\t')
        super().__init__(tx_id=tx_id, gene_id=gene_id, chrom=chrom, tx_start=tx_start, tx_end=tx_end, strand=strand, cds_start=cds_start, cds_end=cds_end, buildver=None, gene_name=None)
        self.tx_id = tx_id
        self.gene_id = gene_id
        self.chrom = chrom
        self.strand = strand
        self.tx_start, self.tx_end = int(tx_start), int(tx_end)
        self.cds_start, self.cds_end = int(cds_start), int(cds_end)
        self.exon_starts = [int(x) for x in exon_starts.strip(',').split(',')]
        self.exon_ends = [int(x) for x in exon_ends.strip(',').split(',')]
        self.exon_num = len(self.exon_starts)
        self.gene_type = None 


class GenePred(object):
    def __init__(self, genepred: str, additional_tx_info=None) -> None:
        self.genepred = genepred
        self.tx_info: Dict[str, Transcript] = dict()
        self.additional_tx_info = additional_tx_info
        self.tag = None
        self.process()
    
    def __repr__(self) -> str:
        return "GenePred({}, n={})".format(self.genepred, len(self.tx_info))
    
    def process(self, ):
        if self.additional_tx_info is not None:
            additional_tx_info = json.load(open(self.additional_tx_info))
        else:
            additional_tx_info = dict()
        with auto_open(self.genepred) as infile:
            for l in infile:
                tx = GenePredRecord(l)
                if tx.tx_id in additional_tx_info:
                    gene_type = additional_tx_info[tx.tx_id].get("gene_type", "__unknown__")
                    gene_name = additional_tx_info[tx.tx_id].get("gene_name", "__unknown__")
                    tag = additional_tx_info[tx.tx_id].get("tag", "__unknown__")
                else:
                    gene_type = "__unknown__"
                    gene_name = "__unknown__"
                    tag = "__unknown__"
                tx.gene_type = gene_type
                tx.gene_name = gene_name
                tx.tag = tag
                self.tx_info[tx.tx_id] = tx
    
    def keys(self):
        return self.tx_info.keys()
    

def count_items(ar: List, sort_counts: bool=False, reverse: bool=True, fraction: bool=False):
    ar = np.asarray(ar)
    if sort_counts:
        results = sorted(zip(*np.unique(ar, return_counts=True)), key=lambda x:x[1], reverse=reverse)
    else:
        results = list(zip(*np.unique(ar, return_counts=True)))
    if fraction:
        total = len(ar)
        results = [list(x) + [round(x[1] / total, 3)] for x in results]
    return results

def set_seed(seed: int, force_deterministic: bool=False):
    # if float(torch.version.cuda) >= 10.2:
    #     os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if force_deterministic:
        logger.warning("torch.use_deterministic_algorithms was set to True!")
        torch.use_deterministic_algorithms(True)


def pr_curve(labels, scores):
    r"""
    An implementation of precision-recall curve, which is different from sklearn.metrics.precision_recall_curve in dealing with equal scores.
    """
    new_labels, new_scores = list(zip(*sorted(zip(labels, scores), key=lambda x:x[1], reverse=True)))
    precision, recall, thresholds = list(), list(), list()
    num_p = sum(new_labels)
    tp, fp = 0, 0
    for i in range(len(new_labels)):
        if new_labels[i] == 1:
            tp += 1
        else:
            fp += 1
        if i == len(new_labels) - 1 or new_scores[i + 1] != new_scores[i]:
            precision.append(tp/(tp + fp))
            recall.append(tp / num_p)
            thresholds.append(new_scores[i])
    return np.asarray(precision), np.asarray(recall), np.asarray(thresholds)


def boxplot_with_scatter(x, size=None, ax=None, max_sample=None, scatter_kwargs=dict(), **kwargs):
    r"""
    Not fully tested yet, just for quick plot
    """
    if ax is None:
        ax = plt.subplot()

    if "sym" not in kwargs:
        kwargs["sym"] = ''
    bb = ax.boxplot(x=x, **kwargs)
    # bb2 = ax.boxplot(x=x2, **kwargs)

    if not hasattr(x[0], "__iter__"):
        x = [x]

    vertical = kwargs.get("vert", True)
    scatter_color = scatter_kwargs.get("c", None)
    align_anchors = ax.get_xticks() if vertical else ax.get_yticks()
    for i, ind in enumerate(align_anchors):
        if type(scatter_color) is not str:
            if hasattr(scatter_color, "__iter__"):
                c = scatter_color[i]
            else:
                c = None
        else:
            c = scatter_color
        scatter_kwargs['c'] = c

        ar = np.asarray(x[i])
        if max_sample is not None and len(ar) > max_sample:
            ar = np.random.permutation(ar)[:max_sample]

        if vertical:
            try:
                left, right = bb["boxes"][i].get_data()[0][:2]
            except AttributeError:
                left, right = bb["caps"][i].get_data()[i]
            xs = np.random.randn(len(ar))
            xmin, xmax = xs.min(), xs.max()
            xs =  (2 * (xs - xmin) / (xmax - xmin) - 1) * (right - left) / 2 + ind
            scatter = ax.scatter(x=xs, y=ar, s=size, **scatter_kwargs)
        else:
            try:
                left, right = bb["boxes"][i].get_data()[0][:2]
            except AttributeError:
                left, right = bb["caps"][i].get_data()[i]

            ys = np.random.randn(len(ar))
            ymin, ymax = ys.min(), ys.max()
            ys =  (2 * (ys - ymin) / (ymax - ymin) - 1) * (right - left) / 2 + ind
            scatter = ax.scatter(x=ar, y=ys, s=size, **scatter_kwargs)