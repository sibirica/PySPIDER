import numpy as np
import os
import sys
import subprocess
from timeit import default_timer as timer
from PySPIDER.commons.TInvPower import *
#from PySPIDER.commons.Kaczmarz import *

def sparse_reg(theta, threshold='AIC', brute_force=True, delta=1e-10, epsilon=1e-2, gamma=2,
               verbose=False, n_terms=-1, char_sizes=None, row_norms=None, valid_single=None, avoid=None, subinds=None, anchor_norm=None, method="stepwise", max_k=10, start_k=20, inhomog=False, inhomog_col=None):
    # compute sparse regression on Theta * xi = 0
    # Theta: matrix of integrated terms
    # char_sizes: vector of characteristic term sizes (per column)
    # row_norms: desired norm of each row
    # valid_single: vector of 1s/0s (valid single term model/not)
    # avoid: coefficient vectors to be orthogonal to
    # and a lot more not described above
    # NEW ARGUMENTS: start_k - used in method='hybrid', power method at k=start_k and then stepwise reduction the rest of the way
    # inhomog - for inhomogeneous regression, paired with inhomog_col: which term to use as b.
    # note: only brute force stepwise method implemented for inhomogeneous regression

    if avoid is None:
        avoid = []
    theta = np.copy(theta)  # avoid bugs where array is modified in place
    if row_norms is not None:
        for row in range(len(row_norms)):
            # rownm = np.linalg.norm(Theta[row, :])
            # if rownm != 0:
            #    Theta[row, :] *= (row_norms[row]/rownm)
            theta[row, :] *= row_norms[row]
    if char_sizes is not None:  # do this here: char_sizes are indexed by full column set
        char_sizes = np.array(char_sizes)
        # char_sizes /= np.max(char_sizes)
        for term in range(len(char_sizes)):
            theta[:, term] = theta[:, term] / char_sizes[term]  # renormalize by characteristic size
    # do this exactly here: when we divide by Thetanm later, we work with the normalized columns
    if anchor_norm is None:
        thetanm = np.linalg.norm(theta)
    else:
        thetanm = anchor_norm
    if verbose:
        np.set_printoptions(precision=3)
        print('Thetanm:', thetanm)

    if subinds is not None:
        if not subinds:  # no inds allowed at all
            return None, np.inf, None, np.inf
        theta = theta[:, subinds]
        if char_sizes is not None:
            char_sizes = np.array(char_sizes)
            char_sizes = char_sizes[subinds]
        if valid_single is not None:
            valid_single = np.array(valid_single)
            valid_single = valid_single[subinds]
    m = 100 * theta.shape[0]
    for xi in avoid:
        theta = np.vstack([theta, m * np.transpose(xi)])  # Acts as a constraint - weights should be orthogonal to Xi

    h, w = theta.shape
    if anchor_norm is None:
        thetanm /= np.sqrt(w)  # scale norm of Theta by square root of # columns to fix scaling of Theta@Xi vs Thetanm
    beta = w / h  # aspect ratio

    if valid_single is None:
        valid_single = np.ones(shape=(w, 1))

    if inhomog:
        b = theta[:, inhomog_col]
        theta_wo_b = np.hstack([theta[:, :inhomog_col], theta[:, inhomog_col+1:]])
        xi_unaug, _, _, _ = np.linalg.lstsq(theta_wo_b, b, rcond=None)
        xi = np.hstack([xi_unaug[:inhomog_col], [-1], xi_unaug[inhomog_col:]])
    else:
        u, sigma, v = np.linalg.svd(theta, full_matrices=True)
        v = v.transpose()  # since numpy SVD returns the transpose
        xi = v[:, -1]
    lambd = np.linalg.norm(theta @ xi) / thetanm
    if verbose:
        # print("sigma:", sigma)
        # Sigmas = sigma[sigma[:]>0]
        #print("v:", v)
        #print("scores:", np.log(sigma))  # np.log(Sigmas)/np.min(Sigmas)
        pass
    if verbose:
        print('lambda:', lambd)
    # find best one-term model as well
    nrm = np.zeros(w)
    for term in range(w):
        nrm[term] = np.linalg.norm(theta[:, term]) / (thetanm * valid_single[term])
        lambda1, best_term = min(nrm), np.argmin(nrm)
        if verbose:
            print(f'nrm[{term}]:', nrm[term])
    if w == 1:  # no regression to run
        # noinspection PyUnboundLocalVariable
        return [1], np.inf, best_term, lambda1
    
    if method == "discrete": # (does not work)
        # BUGS: COEFFICIENTS AREN'T ALWAYS IN THE CORRECT ORDER (hopefully fixed) 
        # Additionally if Sigma entries are too large Mosek can crash (hopefully fixed)
        # and imports are very slow (batching helps).
        # Most importantly, the optimization program doesn't do at all what we want.
        max_k = min(max_k, w) # max_k can't be bigger than w
        xis = np.zeros(shape=(max_k, w))
        lambdas = np.zeros(max_k)
        #sigma_in = -theta.T @ theta
        #sigma_in = theta.T @ theta
        sigma_in = np.max(sigma[:])**2*np.eye(w)-theta.T @ theta
        
        #print(theta.shape, sigma.shape)
        save_loc = 'temp/sigma.csv'
        if not os.path.exists('temp'): # make temp directory if it does not exist yet
            os.makedirs('temp')
        with open(save_loc, 'w') as save_file:
             np.savetxt(save_file, sigma_in, delimiter=",")
        batch = True
        if batch: # run all k's together to reduce amount of time Julia wastes on imports
            load_template = 'temp/output_@.csv'
            xis, ubs, lbs = batch_discrete_sr(max_k, save_loc, load_template)
            for i in range(max_k):
                k = max_k - i
                xi = xis[i]
                if verbose:
                    print("k:", k, "xi:", xi, "dual bound:", ubs[i], "primal bound:", lbs[i])
                lambdas[i] = np.linalg.norm(theta @ xi) / thetanm
        else: 
            for i in range(max_k):
                k = max_k - i
                load_loc = f'temp/output_{k}.csv'
                xi, ub, lb = discrete_sr(k, save_loc, load_loc)
                if verbose:
                    print("k:", k, "xi:", xi, "dual bound:", ub, "primal bound:", lb)
                xis[i] = xi
                lambdas[i] = np.linalg.norm(theta @ xi) / thetanm
        margins = lambdas[1:]/lambdas[:-1]
    elif method == "power":
        xi = None
        max_k = min(max_k, w) # max_k can't be bigger than w
        xis = np.zeros(shape=(max_k, w))
        lambdas = np.zeros(max_k)
        for i in range(max_k):
            k = max_k - i
            sigma_in = theta.T @ theta 
            xi, mu, it = TInvPower(sigma_in, k, x0=xi, mu0=0, verbose=False)
            xis[i] = xi
            lambdas[i] = np.linalg.norm(theta @ xi) / thetanm
            if verbose:
                print("k:", k, "xi:", xi, "lambda:", lambdas[i])
        margins = lambdas[1:]/lambdas[:-1]
    if method == "stepwise": # initialization for stepwise method
        xis = np.zeros(shape=(w, w))  # record coefficients
        smallinds = np.zeros(w)
        if inhomog:
            smallinds[inhomog_col] = 1
        margins = np.zeros(w)  # increases in residual per time step
        lambdas = np.zeros(w)
        lambdas[0] = lambd
        iters = w-1
    elif method == "hybrid": # alternate initialization for hybrid power method
        xi = None
        max_k = min(start_k, w) # max_k can't be bigger than w
        xis = np.zeros(shape=(max_k, w))
        lambdas = np.zeros(max_k)
        sigma_in = theta.T @ theta 
        xi, mu, it = TInvPower(sigma_in, max_k, x0=xi, mu0=0, verbose=False)
        xis[0] = xi
        lambdas[0] = np.linalg.norm(theta @ xi) / thetanm
        smallinds = (xis[0]==0)
        if verbose:
            print("initial k:", max_k, "xi:", xi, "lambda:", lambdas[0])
        iters = max_k-1
        margins = np.zeros(max_k)
        method = "stepwise" # switch to rest of stepwise iteration
    if method == "stepwise": # run stepwise iteration
        flag = False
        for i in range(iters):
            xis[i] = xi
            if brute_force:
                # product of the coefficient and characteristic size of library function
                res_inc = np.ones(shape=(w, 1)) * np.inf
            product = np.zeros(shape=(w, 1))
            for p_ind in range(w):
                if brute_force:
                    if smallinds[p_ind] == 0:
                        # Try dropping each term
                        smallinds_copy = np.copy(smallinds)
                        smallinds_copy[p_ind] = 1
                        xi_copy = np.copy(xi)
                        xi_copy[p_ind] = 0
                        if inhomog:
                            subtheta = theta[:, smallinds_copy == 0]
                            xi_copy, _, _, _ = np.linalg.lstsq(subtheta, b, rcond=None)
                            res_inc[p_ind] = np.linalg.norm(b - subtheta @ xi_copy) / thetanm / lambd
                        else:
                            _, _, v = np.linalg.svd(theta[:, smallinds_copy == 0], full_matrices=True)
                            v = v.transpose()
                            xi_copy[smallinds_copy == 0] = v[:, -1]
                            # noinspection PyUnboundLocalVariable
                            res_inc[p_ind] = np.linalg.norm(theta @ xi_copy) / thetanm / lambd
                else:
                    col = theta[:, p_ind]
                    # project out other columns
                    for q_ind in range(w):
                        if (p_ind != q_ind) and smallinds[q_ind] == 0:
                            other_col = theta[:, q_ind]
                            col = col - np.dot(col, other_col) / np.linalg.norm(other_col) ** 2 * other_col
                    # product[p_ind] = np.linalg.norm(xi[p_ind]*col)/np.linalg.norm(Theta)
                    product[p_ind] = np.linalg.norm(xi[p_ind] * col)
            if brute_force:
                #if inhomog: # prevent the required column from being dropped
                #    res_inc[inhomog_col] = np.inf
                y, ii = min(res_inc), np.argmin(res_inc)
                if verbose:
                    print("y:", y, "ii:", ii)
                margins[i] = y
                if verbose:
                    # print('res_inc:', res_inc)
                    if threshold != "multiplicative":
                        print("i", i, "lambda", lambd)  # for pareto plot
                if (y <= gamma) or (threshold != "threshold") or (lambd <= delta) or (n_terms > 1):
                    smallinds[ii] = 1
                    xi[ii] = 0
                    if inhomog:
                        subtheta = theta[:, smallinds == 0]
                        xi[smallinds == 0], _, _, _ = np.linalg.lstsq(subtheta, b, rcond=None)
                        xi[inhomog_col] = -1
                    else:
                        _, _, v = np.linalg.svd(theta[:, smallinds == 0], full_matrices=True)
                        v = v.transpose()
                        xi[smallinds == 0] = v[:, -1]
                    lambd = np.linalg.norm(theta @ xi) / thetanm
                    lambdas[i + 1] = lambd
                    if sum(smallinds == 0) == 1:
                        margins[-1] = np.inf
                        if inhomog: # the normal last iteration is never run since one term cannot be kept
                            lambdas[-1] = np.linalg.norm(b) / thetanm
                        break
                else:
                    if verbose:
                        print("y:", y, "ii:", ii)
                    lambdas[i + 1] = margins[i] * lambd
                    break
            else:
                #if inhomog: # prevent the required column from being dropped
                #    product[inhomog_col] = np.inf
                product[smallinds == 1] = np.inf
                y, ii = min(product), np.argmin(product)
                if verbose:
                    print("y:", y, "ii:", ii)
                smallinds[ii] = 1
                if sum(smallinds == 0) == 1:
                    break
                # if verbose:
                # print("prod:", product.transpose())
                xi_old = xi
                xi[smallinds == 1] = 0  # set negligible terms to 0
                _, _, v = np.linalg.svd(theta[:, smallinds == 0], full_matrices=True)
                v = v.transpose()
                xi[smallinds == 0] = v[:, -1]
                lambda_old = lambd
                lambd = np.linalg.norm(theta @ xi) / thetanm
                lambdas[i + 1] = lambd
                margin = lambd / lambda_old
                if verbose:
                    print("lambda:", lambd, " margin:", margin)
                margins[i] = margin
                if (margin > gamma) and (lambd > delta) and (threshold == "threshold") and (n_terms == -1):
                    print("ii:", ii)
                    xi = xi_old
                    print("xi:", xi)
                    break
    xis[-1] = xi
    if threshold == "AIC":
        aics = [AIC(lambd, max_k - i, h) for i, lambd in enumerate(lambdas)]
        opt_i = np.argmin(aics)
        if verbose:
            print('AICS:', aics)
            print('optimal i', opt_i)
        xi = xis[opt_i]
        lambd = lambdas[opt_i]
    elif threshold == "BIC":
        bics = [BIC(lambd, max_k - i, h) for i, lambd in enumerate(lambdas)]
        opt_i = np.argmin(bics)
        if verbose:
            print('BICS:', bics)
            print('optimal i', opt_i)
        xi = xis[opt_i]
        lambd = lambdas[opt_i]
    elif threshold == "pareto":
        y_mar, i_mar = max(margins), np.argmax(margins)
        #if n_terms > 1:
        #    i_mar = sum(margins > 0) - n_terms
        if verbose:
            print("margins:", margins)
            print("y_mar:", y_mar, "i_mar:", i_mar)
        i_mar = max(np.argmax(lambdas > delta) - 1, i_mar)
        stopping_point = i_mar - 1
        xi = xis[i_mar]  # stopping_point
        lambd = np.linalg.norm(theta @ xi) / thetanm
    elif threshold == "multiplicative":
        i_sm = np.argmax((lambdas > epsilon * lambda1) & (lambdas > delta)) - 1
        xi = xis[i_sm]  # stopping_point
        lambd = np.linalg.norm(theta @ xi) / thetanm
    else: #if threshold == 'threshold'
        #if n_terms > 1:  # Don't think this line does anything functionally but I also don't really use this
        #    i_mar = sum(margins > 0) - n_terms
        lambdas[0] = lambdas[1]  # FIXME DUCT TAPE since I don't know what's going on (basically first lambda is big)
        gt_delta = (lambdas > delta)
        large_margin = (margins > gamma)
        if not any(gt_delta) or not any(large_margin): # didn't trip the criteria while sparsifying
            i_mar = -1 # select sparsest term
        else: # select first term which tripped the criteria
            i_mar = max(np.argmax(gt_delta) - 1, np.argmax(large_margin))
        if verbose:
            print(lambdas > delta)
            print(margins > gamma)
            print("i_mar:", i_mar)
        xi = xis[i_mar]  # stopping_point
        if inhomog: # might give wrong results for a 2-term library(?) but that won't happen in practice
            lambd = lambdas[i_mar]
        else:
            lambd = np.linalg.norm(theta @ xi) / thetanm
    # n_terms logic done directly here now
    if n_terms > 1:
        i_mar = w - n_terms
        xi = xis[i_mar]  # stopping_point
        if inhomog: # might give wrong results for a 2-term library(?) but that won't happen in practice
            lambd = lambdas[i_mar]
        else:
            lambd = np.linalg.norm(theta @ xi) / thetanm
    if verbose:
        print("xis:", xis)
    # now compare single term and sparsified model
    if verbose:
        print("lambda:", lambd, "lambda1:", lambda1)
        print("lambdas:", lambdas)
    # print("Xi1:", xi)
    if char_sizes is not None:
        xi = xi / char_sizes  # renormalize by char. size
        # print("Xi2:", xi)
        # nm = np.linalg.norm(xi)
        # # divide errors by the norm to make errors consistent
        # lambd /= nm
        # lambda1 /= nm
    # divide errors by square root of number of rows to make errors consistent
    # lambd /= h**0.5
    # lambda1 /= h**0.5
    if -min(xi) > max(xi):  # ensure vectors are "positive"
        xi = -xi
    xi = xi / max(xi)  # make largest coeff 1
    # make residuals relative to original norm(Theta)*norm(xi)
    #nm = np.linalg.norm(xi)
    # lambd /= (nm*thetanm)
    # lambd /= thetanm
    # lambda1 /= thetanm

    # noinspection PyUnboundLocalVariable
    return xi, lambd, best_term, lambda1

def regress(Theta, col_numbers, col_weights, normalization=None):  # regression on a fixed set of terms
    h, w = Theta.shape
    if normalization is None:
        thetanm = np.linalg.norm(Theta[:, 0])
    else:
        thetanm = normalization
    #col_weights = np.linalg.norm(Theta, axis=0)
    #print(h, w, col_weights.shape)
    Theta_copy = Theta.copy()
    for term in range(len(col_weights)):
            Theta_copy[:, term] = Theta_copy[:, term] / col_weights[term]
    # fix scaling w/ respect to number of columns
    #thetanm /= np.sqrt(w)
    smallinds = np.ones(shape=(w,))
    xi = np.zeros(shape=(w,))
    smallinds[np.array(col_numbers)] = 0
    _, _, v = np.linalg.svd(Theta_copy[:, smallinds == 0], full_matrices=True)
    v = v.transpose()
    xi[smallinds == 0] = v[:, -1]
    
    nms = np.zeros(shape=(len(xi),1))
    for i in range(len(xi)):
        nms[i] = np.linalg.norm(Theta_copy[:, i] * xi[i])
    thetanm = np.max(nms) # relative residual is max ||Q_ic_i||
    lambd = np.linalg.norm(Theta_copy @ xi)/thetanm # changed to relative residual
    xi = xi / col_weights
    if -min(xi) > max(xi):  # ensure vectors are "positive"
        xi = -xi
    xi = xi / max(xi)  # make largest coeff 1

    return xi, lambd

# taken with minor modifications from Bertsekas paper code
def AIC(lambd, k, m, add_correction=True):
    rss = lambd ** 2
    aic = 2 * k + m * np.log(rss / m)
    if add_correction:
        correction_term = (2 * (k + 1) * (k + 2)) / max(m - k - 2, 1)  # In case k == m
        aic += correction_term
    return aic

def BIC(lambd, k, m):
    rss = lambd ** 2
    bic = np.log(m) * k + m * np.log(rss / m)
    return bic

# call Julia to compute Xi
def discrete_sr(k, save_loc, load_loc):
    # run Julia script via interface
    #os.system(f'julia interface.jl {k} {save_loc} {load_loc}')
    #try:
    #    subprocess.check_call(f'julia interface.jl {k} "{save_loc}" "{load_loc}"')
    #except subprocess.CalledProcessError as e:
    #    print(e.output)
    ### apparently Popen.communicate is a more standard way to pipe io than saving to file
    #with open('julia_path.config', 'r') as fj: # format: absolute path of interface.jl in quotes
    #    path = fj.readline()
    start = timer()
    path = '"../Julia/ScalableSPCA.jl/interface.jl"'
    _run_command(f'julia -q -J../Julia/Sysimage.so {path} {k} "{save_loc}" "{load_loc}"') # see below

    # load csv from load_loc
    with open(load_loc, 'r') as f:
        line1 = f.readline()
        ublb = line1.split(',')
        ub = float(ublb[0])
        lb = float(ublb[1])
        line2 = f.readline()
        xi_split = [float(string.replace("\n", "").strip()) for string in line2.split(',')]
        xi = np.array(xi_split)
        
    time = timer() - start
    print(f"[ran in {time:.2f} s]")
    return xi, ub, lb

def batch_discrete_sr(max_k, save_loc, load_template):
    start = timer()
    path = '"../Julia/ScalableSPCA.jl/batch_interface.jl"'
    _run_command(f'julia -q -J../Julia/Sysimage.so {path} {max_k} "{save_loc}" "{load_template}"') # see below

    xis = []
    ubs = []
    lbs = []
    for i in range(max_k):
        k = max_k-i
        load_loc = load_template.replace("@", str(k))
        # load csv from load_loc
        with open(load_loc, 'r') as f:
            line1 = f.readline()
            ublb = line1.split(',')
            ub = float(ublb[0])
            lb = float(ublb[1])
            line2 = f.readline()
            xi_split = [float(string.replace("\n", "").strip()) for string in line2.split(',')]
            xi = np.array(xi_split)
        xis.append(xi)
        ubs.append(ub)
        lbs.append(lb)
        
    time = timer() - start
    print(f"[ran in {time:.2f} s]")
    return xis, ubs, lbs

def _run_command(command):
    print("Running command: {}".format(command))
    with open("error_log.txt", "w") as f:
        try:
            subprocess.check_call(command, stderr=f)
            df = subprocess.Popen(command, stdout=subprocess.PIPE)
            output, err = df.communicate()
        except subprocess.CalledProcessError as e:
            print("===================")
            print(e.stderr)
            print("===================")
            raise e       