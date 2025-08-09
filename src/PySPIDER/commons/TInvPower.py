import numpy as np
from .sr_utils import *

### forced_col (for inhomogeneous regression) is UNTESTED
def TInvPower(
    A, k, x0=None, mu0=None, tol=1e-12, exact=True, fixed_mu=True, 
    max_iter=50, verbose=False, forced_col=None
):
    w = A.shape[0]
    if x0 is None:
        x = smallest_eig(A)
    else:
        x = x0
    if mu0 is None:
        mu = x.T @ A @ x
    else:
        mu = mu0
    if verbose:
        print("non-sparse mu: ", mu)
        print("non-sparse x: ", x)
    update_size = np.inf 
    it = 0
    x /= np.linalg.norm(x)
    while update_size>tol and it<max_iter:
        try:
            y = np.linalg.solve(A - mu * np.eye(w), x) # Rayleigh iteration
            # y = np.linalg.solve(A, x) # pure inverse iteration
        except:
            #print("Exiting early due to singular matrix")
            return x, mu, it
        # indices of largest absolute entries
        inds = np.argpartition(np.abs(y), -k)[-k:]
        # for inhomogeneous regression
        if forced_col is not None and forced_col not in inds:
            inds = list(inds[1:]) # drop the last entry
            inds.append(forced_col)
        inds = sorted(inds) # this should make life much easier
        if exact: # solve subsystem with inds exactly
            y = solve_ATA(A, inds, forced_col)
        else: # hard threshold without finding exact solution
            y = keep_inds(y, inds)
        y /= np.linalg.norm(y)
        if not fixed_mu:
            mu = y.T @ A @ y # comment out to fix mu
        update_size = (min(np.linalg.norm(y - x), np.linalg.norm(y + x)) / 
                       np.linalg.norm(y))
        if verbose:
            print("x:", x, "y:", y, "mu:", mu, "update_size:", update_size)
            #print("mu:", mu)
        x = y
        it += 1
    return x, mu, it

                                             

    