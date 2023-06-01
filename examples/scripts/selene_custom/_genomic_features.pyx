# cython: language_level=3
import numpy as np

cimport cython
cimport numpy as np

ctypedef np.int_t DTYPE_t
ctypedef np.float32_t FDTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False) 
def _fast_get_feature_data(int start,
                           int end,
                           np.ndarray[FDTYPE_t, ndim=1] thresholds,
                           dict feature_index_dict,
                           rows,
                           int nt_level,
                           str strand,
                           int binary,
                           FDTYPE_t default_value):
    cdef int n_features = len(feature_index_dict)
    cdef int query_length = end - start
    cdef int feature_start, feature_end, index_start, index_end, index_feat
    # cdef np.ndarray[FDTYPE_t, ndim=2] encoding = np.zeros(
    #     (query_length, n_features), dtype=np.float32)
    cdef np.ndarray[FDTYPE_t, ndim=2] encoding = np.full(
        (query_length, n_features), default_value, dtype=np.float32)

    cdef np.ndarray[DTYPE_t, ndim=1] targets = np.zeros(n_features, dtype=np.int)
    cdef list row
    cdef float value

    if rows is None:
        return np.zeros((n_features,))
    
    for row in rows:
        if strand != '.' and strand != row[5]:
            continue
        feature_start = int(row[1])
        feature_end = int(row[2])
        index_start = max(0, feature_start - start)
        index_end = min(feature_end - start, query_length)
        index_feat = feature_index_dict[row[3]]
        if binary > 0:
            value = 1.0
        else:
            assert len(row) > 6, "signalValue (the 7th column) is not available in bed file"
            value = float(row[6])
        if index_start == index_end:
            index_end += 1
        encoding[index_start:index_end, index_feat] = value

    if nt_level > 0:
        return encoding
    else:
        thresholds = (thresholds * query_length - 1).clip(min=0)
        targets = (np.sum(encoding, axis=0) > thresholds.astype(int)).astype(int)
        return targets

