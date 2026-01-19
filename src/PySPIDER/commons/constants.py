import numpy as np
from .library import Constant
from .process_library_terms import multi_index_iterator

def Q_constant(n,complexity=4):
    Qobs = Constant(string='Q', rank=4, custom_complexity=complexity, can_commute_indices=True, allow_contractions=False)
    Qdata = np.zeros((n,)*4)
    it = multi_index_iterator(n,rank=4)
    for x in it:
        i,j,k,l = it.multi_index
        if i == j == k == l:
            Qdata[i,i,i,i] = 1-n
        elif (i == j and k == l) or (i == k and j == l) or (i == l and j == k):
            Qdata[i,j,k,l] = 1
    return Qobs, Qdata