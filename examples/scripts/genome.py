#!/usr/bin/env python3
"""
Author: Ken Chen
Email: chenkenbio@gmail.com
Date: 2022-12-09
"""

import argparse
import os

import sys
import warnings
import h5py
import numpy as np
from utils import load_fasta, get_reverse_strand, encode_sequence
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union, TextIO


_TYPE_FASTA = 0
_TYPE_FASTA_GZ = 1
_TYPE_HDF5 = 2

def guess_filetype(fn: str) -> int:
    if fn.endswith(".fa") or fn.endswith(".fasta"):
        ftype = _TYPE_FASTA
    elif fn.endswith("fa.gz") or fn.endswith("fna.gz"):
        ftype = _TYPE_FASTA_GZ
    elif fn.endswith(".h5") or fn.endswith(".hdf5"):
        ftype = _TYPE_HDF5
    else:
        raise TypeError(f"Unknown file type: {fn}, expect .fa, .fasta, .fa.gz, .fna.gz, .h5, .hdf5")
    return ftype


class EasyGenome(object):
    def __init__(self, genome) -> None:
        r"""
        Args:
            genome: genome file path in fasta/hdf5 format.
        """
        self.genome_path = genome
        self.ft = guess_filetype(genome)
        if self.ft == _TYPE_FASTA or self.ft == _TYPE_FASTA_GZ:
            self.genome = load_fasta(genome)
            self.default_padding = 'N'
            self.is_integer = False
        elif self.ft == _TYPE_HDF5:
            self.genome = h5py.File(genome, "r")
            self.default_padding = 0
            self.is_integer = True
        else:
            raise ValueError(f"Unknown file type: {genome}")
    
    def __repr__(self) -> str:
        return "EasyGenome(path: {}, #chroms: {})".format(self.genome_path, len(self.genome))
    
    def fetch_sequence(self, chrom: str, start: int, end: int, reverse: bool=False, no_padding: bool=False, padding: Optional[Union[str, int]]=None, left: int=None, right: int=None) -> Union[np.ndarray, str]:
        r"""
        chrom (str): chromosome name
        start (int): start position, 0-based
        end (int): end position, 0-based
        reverse (bool): reverse the sequence
        no_padding (bool): if True, no padding will be added
        padding (int/str): padding character, default is 'N' for fasta, 0 for hdf5
        left (int): left border, default is 0
        right (int): right border, default is length of the chromosome
        """
        assert start < end, f"start {start} should be smaller than end {end}"
        if padding is None:
            padding = self.default_padding
        
        left = 0 if left is None else left
        right = len(self.genome[chrom]) if right is None else right
        if start >= right or end <= left:
            # warnings.warn(f"chrom {chrom} start {start} end {end} is out of range [{left}, {right})")
            start, end = right, right + end - start

        left_pad, right_pad = 0, 0
        if not no_padding:
            if start < left:
                left_pad = left - start
                start = left
            if end > right:
                right_pad = end - right
                end = right
        else:
            start = max(start, left)
            end = min(end, right)
        if left_pad + right_pad == 0:
            no_padding = True

        if self.ft == _TYPE_FASTA or self.ft == _TYPE_FASTA_GZ:
            seq = self.genome[chrom][start:end]
            if not no_padding:
                seq = padding * left_pad + seq + padding * right_pad
        else:
            seq = self.genome[chrom][start:end]
            if not no_padding:
                seq = np.pad(seq, (left_pad, right_pad), constant_values=padding)

        if reverse:
            seq = get_reverse_strand(seq, integer=self.is_integer)
        return seq
    
    @classmethod
    def fasta2hdf5(cls, fasta, hdf5):
        r"""
        fasta: fasta file path
        hdf5: hdf5 file path (output)
        """
        genome = load_fasta(fasta)
        with h5py.File(hdf5, "w") as f:
            for chrom, seq in genome.items():
                seq = np.asarray(encode_sequence(seq), dtype=np.int8)
                f.create_dataset(chrom, data=seq, dtype=np.int8)
        f.close()
        