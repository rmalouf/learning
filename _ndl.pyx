import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
def rwUpdate(np.ndarray[np.float_t, ndim=2] W, np.ndarray[np.int_t, ndim=1] D, np.ndarray[np.int_t, ndim=1] O, float Alpha, float Beta, float Lambda):

    cdef int cues = len(D)
    cdef int outcomes = len(O)

    cdef unsigned int i
    cdef unsigned int j

    cdef np.ndarray[np.float_t, ndim=1] Vtotal = np.zeros((outcomes), dtype=np.float)
    for i in range(cues):
        if D[i] == 1:
            for j in range(outcomes):
                Vtotal[j] += W[i,j]

    for i in range(outcomes):
        if O[i] == 1:
            Vtotal[i] = Alpha * Beta * (Lambda - Vtotal[i])
        else:
            Vtotal[i] = Alpha * Beta * (0.0 - Vtotal[i])

    for i in range(cues):
        if D[i] == 1:
            for j in range(outcomes):
                W[i,j] += Vtotal[j]
