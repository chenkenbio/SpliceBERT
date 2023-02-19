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
import warnings
import pickle
from tqdm import tqdm
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
from io import TextIOWrapper
from _encode_sequence import _encode_sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import logging
logger = logging.getLogger(__name__)

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