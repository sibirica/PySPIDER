import numpy as np

def keep_inds(vector, inds): # set all but inds of vector to 0
    inds = list(inds)
    temp = vector*0
    temp[inds] = vector[inds]
    return temp

def smallest_sv(A, inds=None, value=False):
    if inds is None:
        U, Sigma, V = np.linalg.svd(A, full_matrices=True)
    else:
        inds = list(inds)
        all_inds = list(range(A.shape[0]))
        U, Sigma, V = np.linalg.svd(A[:, inds], full_matrices=True)
        #print("SMALLEST_SV", A[np.ix_(inds, inds)], Sigma)
        #print("INDS", inds)
    V = V.transpose()  # since numpy SVD returns the transpose
    if value:
        return Sigma[-1] # smallest singular value
    else:
        return V[:, -1] # smallest singular vector

def smallest_eig(A, inds=None, value=False):
    if inds is None:
        lambdas, vs = np.linalg.eig(A)
    else:
        lambdas, vs = np.linalg.eig(A[np.ix_(inds, inds)])
    #idx = lambdas.argsort()[::-1]
    #vs = vs[:,idx]
    idx = np.argmin(lambdas)
    if value:
        return lambdas[idx]
    else:
        return vs[:, idx]

#inhomogeneous regression is UNTESTED
#def solve_ATA(A, inds, inhomog_col=None):
#    w = A.shape[1]
#    x = np.zeros(shape=(w,))
#    if inhomog_col is None:
#        x[np.ix_(inds)] = smallest_sv(A[np.ix_(inds, inds)]) # work on submatrix with inds
#    else: # note that [A b]^T[A b] = [A^TA A^Tb; ... ...]
#        inds_minus_b = inds.copy()
#        inds_minus_b.remove(inhomog_col)
#        ATA = A[np.ix_(inds_minus_b, inds_minus_b)]
#        ATb = A[np.ix_(inds_minus_b, [inhomog_col])]
#        #x, _, _, _ = np.linalg.lstsq(ATA, ATb, rcond=None)
#        x[np.ix_(inds_minus_b)] = np.linalg.solve(ATA, ATb)
#        x[inhomog_col] = -1 # put back in the -1 coefficient for b
#    return x
#
#def solve(A, inds, inhomog_col=None):
#    h, w = A.shape
#    x = np.zeros(shape=(w,))
#    if inhomog_col is None:
#        x[np.ix_(inds)] = smallest_sv(A[np.ix_(range(h), inds)]) # work on submatrix with inds
#    else: # note that [A b]^T[A b] = [A^TA A^Tb; ... ...]
#        inds_minus_b = inds.copy()
#        inds_minus_b.remove(inhomog_col)
#        ATA = A[np.ix_(range(h), inds_minus_b)]
#        ATb = A[np.ix_(range(h), [inhomog_col])]
#        x[np.ix_(inds_minus_b)], _, _, _ = np.linalg.lstsq(ATA, ATb, rcond=None)
#        x[inhomog_col] = -1 # put back in the -1 coefficient for b
#    return x

def solve_ATA(A, inds, inhomog_col=None): # A here is A^TA
    w = A.shape[1]
    x = np.zeros(shape=(w,))
    inds = list(inds)
    if inhomog_col is None:
        x[inds] = smallest_eig(A[np.ix_(inds, inds)]) # work on submatrix with inds
    else: # note that [A b]^T[A b] = [A^TA A^Tb; ... ...]
        inds_minus_b = inds.copy()
        inds_minus_b.remove(inhomog_col)
        ATA = A[np.ix_(inds_minus_b, inds_minus_b)]
        ATb = A[np.ix_(inds_minus_b, [inhomog_col])]
        #x, _, _, _ = np.linalg.lstsq(ATA, ATb, rcond=None)
        x[inds_minus_b] = np.linalg.solve(ATA, ATb)[:, 0]
        x[inhomog_col] = -1 # put back in the -1 coefficient for b
    return x

def solve(A, inds, inhomog_col=None):
    w = A.shape[1]
    x = np.zeros(shape=(w,))
    inds = list(sorted(inds))
    if inhomog_col is None:
        x[inds] = smallest_sv(A[:, inds]) # work on submatrix with inds
    else: # note that [A b]^T[A b] = [A^TA A^Tb; ... ...]
        inds_minus_b = inds.copy()
        inds_minus_b.remove(inhomog_col)
        A_submx = A[:, inds_minus_b]
        b = A[:, inhomog_col]
        x[inds_minus_b], _, _, _ = np.linalg.lstsq(A_submx, b, rcond=None)
        x[inhomog_col] = -1 # put back in the -1 coefficient for b
    return x