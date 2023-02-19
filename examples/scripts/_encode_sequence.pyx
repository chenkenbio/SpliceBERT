# cython: language_level=3

# Author: Ken Chen (chenkenbio@gmail.com)

cimport cython

NN_DICT = {
    'N': 0, 'n': 0,
    'A': 1, 'a': 1,
    'C': 2, 'c': 2,
    'G': 3, 'g': 3,
    'T': 4, 't': 4,
    'U': 4, 'u': 4,
}

@cython.boundscheck(False)
@cython.wraparound(False) 
def _encode_sequence(str seq, dict nn_dict=NN_DICT):
    cdef list ids = list()
    cdef int seq_len = len(seq)

    for i in range(seq_len):
        ids.append(nn_dict[seq[i]])
    return ids

## compile: python setup.py build_ext --inplace
