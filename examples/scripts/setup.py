#!/usr/bin/env python3
from setuptools import setup
from Cython.Build import cythonize

setup(
    # name="sequqnce_to_list",
    # ext_modules=cythonize("./_sequence_to_kmer_list.pyx", "./_encode_sequence.pyx"),
    ext_modules=cythonize("./_encode_sequence.pyx"),
    zip_safe=False,
)
