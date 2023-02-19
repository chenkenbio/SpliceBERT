# cython: language_level=3

cimport cython


@cython.boundscheck(False)
@cython.wraparound(False) 
def _fast_seq_to_kmer_list(
        str seq,
        int k,
        int left_pad,
        int right_pad,
        int shift
    ):
    cdef int pad = k // 2
    cdef int end = left_pad + len(seq)
    cdef int start
    cdef list kmers = list()
    cdef str pad_seq = ''.join(['N' * left_pad, seq, 'N' * right_pad])

    for i in range(left_pad, end, shift):
        start = i - left_pad 
        kmers.append(pad_seq[start:start + k])
    return kmers

## compile: python compile_cython.py build_ext --inplace
