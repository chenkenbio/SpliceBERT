#!/usr/bin/env python3

from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(["./_encode_sequence.pyx", "_sequence_to_kmer_list.pyx", "./selene_custom/_genomic_features.pyx"]),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
