import numpy as np
from itertools import combinations
import copy

import random # for test-train sampling

from PySPIDER.commons.TInvPower import TInvPower
from PySPIDER.commons.sr_utils import *

# new approach: for modularity, pass objects handling separate steps of the regression with their own params

# note: in the future it might be cleaner to implement things like method, iter_direction, etc. as enums
# also, not a good idea to use backward-forward=false with combinatorial start as it will stop at max_k

# rich regression output object
class RegressionResult(object):
    def __init__(self, xi, lambd, best_term, lambda1, all_xis=None, all_lambdas=None, 
                 lambda1_test=None, lambda_test = None, all_lambdas_test=None, sublibrary=None):
        self.xi = xi
        self.lambd = lambd
        self.lambda_test = lambda_test
        self.best_term = best_term
        self.lambda1 = lambda1
        self.lambda1_test = lambda1_test
        self.all_xis = all_xis # reminder: row i of these arrays corresponds to i+1 term model
        self.all_lambdas = all_lambdas
        self.all_lambdas_test = all_lambdas_test
        self.sublibrary = sublibrary # list of actual terms corresponding to this sublibrary - can also be set later

class Scaler(object): # pre- and postprocessing by scaling/nondimensionalizing data
    def __init__(self, sub_inds, char_sizes, row_norms=None, unit_rows=False, train_fraction=1): # note: char_sizes should always be set
        self.full_w = len(char_sizes)
        self.sub_inds = sub_inds if sub_inds is not None else list(range(self.full_w)) # default is keeping all indices
        self.w = len(self.sub_inds)
        # useful if we want to reuse a Scaler except with different sub_inds
        #self.full_cs = np.array(char_sizes) if char_sizes is not None else np.ones(shape=(self.w,))
        #self.char_sizes = self.full_cs[self.sub_inds]
        self.char_sizes = np.array(char_sizes)
        self.row_norms = np.array(row_norms) if row_norms is not None else None
        self.unit_rows = unit_rows
        
        self.train_fraction = train_fraction # test-train split fraction
        
    def reset_inds(self, sub_inds): # change sub_inds
        self.sub_inds = sub_inds
        self.w = len(self.sub_inds)
        #self.char_sizes = self.full_cs[self.sub_inds]
    
    def index(self, col): # find the full indexing to sub_inds conversion of a given column number
        return self.sub_inds.index(col) if col is not None else None
    
    def norm_col(self, theta, col): # compute norm of given column after nondimensionalization
        #idx_col = self.index(col)
        #return np.linalg.norm(theta[:, idx_col])/self.full_cs[idx_col]
        return np.linalg.norm(theta[:, col])/self.full_cs[col]
    
    # def scale_theta(self, theta): # rescale theta and select columns from subinds
    #     theta = np.copy(theta)  # avoid bugs where array is modified in place
    #     theta = theta[:, self.sub_inds]
    #     for term in range(len(self.char_sizes)):
    #         theta[:, term] /= self.char_sizes[term]  # renormalize by characteristic size
    #     if self.row_norms is not None:
    #         for row in range(len(self.row_norms)):
    #             theta[row, :] *= self.row_norms[row]
    #     elif self.unit_rows: # normalize rows to ~unit entries
    #         h, w = theta.shape
    #         for row in range(h):
    #             theta[row, :] /= (np.linalg.norm(theta[row, :])/np.sqrt(w))
    #     return theta

    def scale_theta(self, theta): # rescale theta and select columns from subinds
        # note that this function now runs before sampling sub_inds columns in theta
        theta = np.copy(theta)  # avoid bugs where array is modified in place
        for term in range(len(self.char_sizes)):
            theta[:, term] /= self.char_sizes[term]  # renormalize by characteristic size
        if self.row_norms is not None:
            for row in range(len(self.row_norms)):
                theta[row, :] *= self.row_norms[row]
        elif self.unit_rows: # normalize rows to ~unit entries
            h, w = theta.shape
            for row in range(h):
                theta[row, :] /= (np.linalg.norm(theta[row, :])/np.sqrt(w))
        return theta
    
    def select_cols(self, theta):
        return theta[:, self.sub_inds]

    def train_test_split(self, theta):
        h = theta.shape[0]
        n_train = int(h*self.train_fraction)
        train_indices = random.sample(range(h), n_train)
        test_indices = list(set(range(h)) - set(train_indices))
        #theta_train = theta[:n_train, :]
        #theta_test = theta[n_train:, :]
        theta_train = theta[train_indices, :]
        theta_test = theta[test_indices, :]
        return theta_train, theta_test

    def scale_model(self, model): # rescale given xi vector
        model = np.copy(model)  # avoid bugs where array is modified in place
        model = model[self.sub_inds]
        model *= self.char_sizes[self.sub_inds]  # renormalize by characteristic size
        return model
        
    def postprocess_multi_term(self, xi, lambd, test_train_ratio=None, lambda_test=None, verbose=False): # Xi postprocessing
        full_xi = np.zeros(shape=(self.full_w,))
        #print('old xi', xi)
        xi = xi / self.char_sizes[self.sub_inds]  # renormalize by char. size
        #print('new xi', xi)
        if -min(xi) > max(xi):  # ensure vectors are "positive"
            xi = -xi
        xi = xi / max(xi)  # make largest coeff 1
        for i, c in enumerate(xi): # construct full_xi
            full_xi[self.sub_inds[i]] = c
        if verbose:
            print("final xi", full_xi)
            print("final lambda:", lambd)
        return full_xi, lambd, lambda_test/np.sqrt(test_train_ratio) if lambda_test else None 
        #/norm # already normalized in multi-term regression
        
    def postprocess_single_term(self, best_term, lambda1, test_train_ratio=None, lambda1_test=None, verbose=False):
        if verbose:
            print("final lambda1:", lambda1)
        return self.sub_inds[best_term], lambda1, lambda1_test/np.sqrt(test_train_ratio) if lambda1_test else None
    
    def __repr__(self):
        return f"Scaler(sub_inds={self.sub_inds}, char_sizes={self.char_sizes}, row_norms={self.row_norms})"

class Initializer(object): # selecting initial guess
    def __init__(self, method, start_k=None):
        self.method = method
        if start_k is not None:
            self.start_k = start_k
        else:
            self.start_k = 2 if method == 'combinatorial' else 10
        self.inhomog = None
        self.inhomog_col = None #note: should be set to converted index in sublibrary
    
    def check_start_k(self, w):
        self.start_k = min(w, self.start_k)
        return self.start_k
    
    def prepare_inhomog(self, inhomog, inhomog_col, scaler):
        self.inhomog = inhomog
        self.inhomog_col = scaler.index(inhomog_col)
    
    def make_model(self, theta, verbose):
        if self.method == 'combinatorial':
            # note - can also be used to initialize iteration from full library using start_k>=w
            w = theta.shape[1]
            inds = list(range(w))
            best_lambd = np.inf
            combos = combinations(inds, self.start_k) if self.start_k<len(inds) else (inds,)
            for combo in combos: # return all combinations of start_k terms
                if self.inhomog and self.inhomog_col not in combo:
                    continue
                xi_try = solve(theta, combo, self.inhomog_col)
                lambd = np.linalg.norm(theta @ xi_try)
                if lambd<best_lambd:
                    best_lambd = lambd
                    xi = xi_try
            lambd = best_lambd
            iter_direction = "backward"
        elif self.method == 'power':
            sigma_in = theta.T @ theta 
            xi, mu, it = TInvPower(sigma_in, self.start_k, mu0=None, verbose=False, forced_col=self.inhomog_col)
            if verbose:
                print("mu:", mu, ", # of iterations: ", it)
            lambd = np.linalg.norm(theta @ xi) # always return absolute residual until postprocessing
            iter_direction = "forward"
        if self.inhomog: # normalization with b term = -1
            lambd /= np.abs(xi[self.inhomog_col])
            xi /= -xi[self.inhomog_col] # normalize it to -1
        if verbose:
            print("initial xi:", xi)
            print("initial lambda:", lambd)
        return xi, lambd, iter_direction
    
    def __repr__(self):
        return f"Initializer(method={self.method}, start_k={self.start_k})"

class ModelIterator(object): # selecting next iterate and enforcing stopping condition
    def __init__(self, max_k, backward_forward=True, max_passes=10, use_best_solution=True): #threshold, brute_force=True
        self.max_k = max_k # do not try models with more than max_k terms
        #self.threshold = threshold # threshold object
        self.backward_forward = backward_forward
        #self.brute_force = brute_force # sadly non-brute force seems to drop key terms in dominant balance
        self.inhomog = None
        self.inhomog_col = None # note: should be set to converted index in sublibrary

        # state variables
        self.k = None # current number of terms
        self.direction = None # forward: dropping terms; backward: adding terms
        self.terms = None # indices of non-zero terms
        #self.has_reversed = False # True if backward-forward iteration has already turned around
        self.state = None # most recently saved set of xis
        #self.stopped_early = False # may want to set True if early stopping so selection is clear
        self.max_passes = max_passes # to prevent infinite loop from ever occuring
        self.passes = 0

        self.use_best_solution = use_best_solution
        # best xi/lambda pairs found so far (and associated test errors)
        self.best_state = None
        self.best_lambdas = None
        self.best_test_lambdas = None
        
    def __repr__(self):
        return f"ModelIterator(max_k={self.max_k}, backward_forward={self.backward_forward}, max_passes={self.max_passes}, inhomog={self.inhomog}, inhomog_col={self.inhomog_col}, k={self.k}, direction={self.direction}, terms={self.terms}, passes={self.passes})"
        
        
    def reset(self, k, max_k, direction): # reset state variables
        self.k = k
        self.max_k = max_k
        # potentially override direction
        if k==max_k:
            self.direction = 'forward'
        elif k==1:
            self.direction = 'backward'
        else:
            self.direction = direction
        self.passes = 0
        self.terms = None
        self.state = None
        self.best_state = None
        self.best_lambdas = None
        self.best_test_lambdas = None # don't forget
              
    #def set_k(self, k):
    #    self.k = k
    #    
    #def set_direction(self, direction):
    #    self.direction = direction
        
    def set_terms(self, inds, verbose):
        self.terms = list(inds)
        if verbose:
            print("Terms:", self.terms)
        if len(self.terms)<self.k:
            self.k = len(self.terms)
            if self.k == 1: # actually only one term left
                self.direction = 'backward'
        
    def prepare_inhomog(self, inhomog, inhomog_col, scaler):
        self.inhomog = inhomog
        self.inhomog_col = scaler.index(inhomog_col)
    
    def save_state(self, xis, lambdas, test_lambdas, verbose): # set the internal state & update best solutions found so far
        self.state = np.copy(xis)
        if self.best_state is None or not self.use_best_solution: # no values have been set yet or we are just saving last state
            if verbose:
                print("Setting best state for the first time...")
            self.best_state = np.copy(xis)
            self.best_lambdas = np.copy(lambdas)
            self.best_test_lambdas = np.copy(test_lambdas)
        else: # model_iterator returns the best solution up to this point
            for i in range(xis.shape[0]): # update each row individually
                if self.best_lambdas[i]>lambdas[i]: # better solution has been found for this sparsity
                    if verbose:
                        print(f"Best lambda for k={i+1}: {self.best_lambdas[i]} -> {lambdas[i]}")
                        #print(f"Best xi for k={i}: {self.best_state[i]} -> {xis[i]}")
                    self.best_state[i, :] = xis[i]
                    self.best_lambdas[i] = lambdas[i]
                    self.best_test_lambdas[i] = test_lambdas[i]
        
    def other_terms(self, w): # return range(w)\terms
        return [i for i in range(w) if i not in self.terms]
        
    def get_next(self, theta, xi, verbose, represent=None):
        if represent is None:
            represent = lambda x:x
            
        if verbose:
            print(f"Direction: {self.direction}; k: {self.k}")
        if self.direction == "forward":
            ind = self.drop(theta, xi, verbose, represent) # choose a term to drop
            self.terms.remove(ind)
            self.k -= 1
            if self.k==1:
                #self.has_reversed=True
                self.direction = "backward"
                if verbose:
                    print("One term left! Direction may be reversed.")
        else:
            ind = self.pick(theta, xi, verbose, represent) # choose a term to pick up
            self.terms.append(ind)
            self.k += 1
            if self.k==self.max_k:
                #self.has_reversed=True
                self.direction = "forward"
                if verbose:
                    print("max_k terms left! Direction may be reversed")
        xi = solve(theta, self.terms, self.inhomog_col)
        lambd = np.linalg.norm(theta @ xi)
        if verbose:
            print("xi:", xi)
            print("lambda:", lambd)
            print("terms:", [represent(term) for term in self.terms])
        return xi, lambd, self.k
    
    def drop(self, theta, xi, verbose, represent):
        s = np.zeros(shape=(len(self.terms), 1)) # term with lowest score will be dropped
        for i, ind in enumerate(self.terms):
            if ind == self.inhomog_col:
                s[i] = np.inf # do not remove
            else:
                # if self.brute_force: # check all possible removals
                terms_copy = self.terms.copy()
                terms_copy.remove(ind)
                if self.inhomog:
                    xi = solve(theta, terms_copy, self.inhomog_col)
                    s[i] = np.linalg.norm(theta @ xi)/np.abs(xi[self.inhomog_col])
                else:
                    s[i] = smallest_sv(theta, terms_copy, value=True)
                # else: # use heuristic of term norm attributed to this column only
                #     col = theta[:, ind]
                #     for j, other_ind in enumerate(self.terms):
                #         # project out other columns
                #         if i != j:
                #             other_col = theta[:, other_ind]
                #             col -= np.dot(col, other_col) / np.linalg.norm(other_col)**2 * other_col
                #     s[i] = np.linalg.norm(xi[ind] * col)
        best = self.terms[np.argmin(s)]
        if verbose:
            print("Scores of terms to remove:", [(represent(i), float(j)) for i, j in zip(self.terms, s)])
            print("Removing term:", represent(best))
        return best 
        
    def pick(self, theta, xi, verbose, represent):
        w = theta.shape[1]
        s = np.zeros(shape=(w-len(self.terms), 1)) # term with lowest score will be dropped
        other_terms = self.other_terms(w)
        #if not self.brute_force:
        #    residual_col = theta @ xi
        for i, ind in enumerate(other_terms):
            # if self.brute_force: # check all possible removals
            terms_copy = self.terms.copy()
            terms_copy.append(ind)
            if self.inhomog:
                xi = solve(theta, terms_copy, self.inhomog_col)
                s[i] = np.linalg.norm(theta @ xi)/np.abs(xi[self.inhomog_col])
            else:
                s[i] = smallest_sv(theta, terms_copy, value=True)
            # else: # use heuristic of projection of residual onto this column
            #     col = theta[:, ind]
            #     proj = residual_col - np.dot(residual_col, col) / np.linalg.norm(col)**2 * col
            #     s[i] = -np.linalg.norm(proj) # guess for xi[col] not yet available 
        best = other_terms[np.argmin(s)]
        if verbose:
            print("Scores of terms to add:", [(represent(i), float(j)) for i, j in zip(other_terms, s)])
            print("Adding term:", represent(best))
        return best
    
    def check_exit(self, xis, lambdas, test_lambdas, verbose):
        if self.k!=1 and self.k!=self.max_k: # we haven't finished iterating to one of the ends of the pass range
            return False # it is possible to add early stopping conditions on lambdas here
        new_terms = np.abs(np.sign(xis))
        old_terms = np.abs(np.sign(self.state))
        differences = new_terms-old_terms
        if verbose:
            print("Changed terms:", differences)
        self.save_state(xis, lambdas, test_lambdas, verbose) # update iterator's saved solutions
        if self.backward_forward and np.any(differences): # entries in xis have changed via iteration
            self.passes += 1
            if self.passes>=self.max_passes:
                if verbose:
                    print(f"Max number of passes reached ({self.max_passes}), exiting.")
                return True
            if verbose:
                #print("Current xis:", xis)
                #print("Saved xis:", self.state)
                print(f"{self.passes} passes, continuing backward-forward iteration...")
            return False
        if verbose:
            print("Iteration completed.")
        return True # otherwise we are done
        # for instance suppose xis did not change during b or f iteration. when it was saved it was optimal going in the other direction, and currently is optimal in the other direction, so no update will occur.

class Residual(object): # residual computation
    # residual_type "absolute" is always 1
    # residual_type "fixed_column" is computed based on a fixed column of theta
    # residual_type "matrix_relative" is relative to norm of theta[:, sub_inds] after preprocessing
    # residual_type "hybrid" uses "matrix_relative" for regression and then theta[:, support] for reporting
    # residual_type "dominant_balance" is relative to norm of largest term in initial dominant balance
    def __init__(self, residual_type, anchor_col=None):
        self.residual_type = residual_type
        if residual_type == "absolute":
            self.norm = 1
        self.anchor_col = anchor_col
    #    self.indexed_col = None
    #def set_anchor(self, scaler):
    #    self.indexed_col = scaler.index(self.anchor_col)
    def set_norm(self, value):
        self.norm = value
        
    def __repr__(self):
        return f"Residual(type={self.residual_type}, norm={self.norm}, anchor_col={self.anchor_col})"
        
class Threshold(object):
    def __init__(self, threshold_type, delta=1e-10, gamma=1.5, epsilon=1e-2, ic=None, n_terms=None):
        self.type = threshold_type
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon
        self.information_criterion = ic
        self.n_terms = n_terms
        
    def __repr__(self):
        return f"Threshold(type={self.type}, delta={self.delta}, gamma={self.gamma}, epsilon={self.epsilon}, ic={self.information_criterion}, n_terms={self.n_terms})"
    
    def select_model(self, lambdas, theta, lambda1, verbose):
        if self.n_terms is not None:
            return min(theta.shape[1]-1, self.n_terms-1)
        if self.type == 'jump': # check when lambda>delta and jump in lambda>gamma
            jumps = lambdas[:-1]/lambdas[1:]
            i = len(lambdas)-1
            while (jumps[i-1]<self.gamma or lambdas[i-1]<self.delta) and i>0:
                i -= 1
            if verbose:
                print("Jumps:", jumps)
                print('optimal i', i)
            return i
        elif self.type == 'information': # minimize information criterion
            h = theta.shape[0]
            ics = [self.information_criterion(lambd, i+1, h) for i, lambd in enumerate(lambdas)]
            opt_i = np.argmin(ics)
            if verbose:
                print(f'{self.information_criterion.__name__}s: {ics}')
                print('optimal i', opt_i)
            return opt_i
        elif self.type == 'multiplicative': # check when lambda<epsilon*lambda1
            i = 0
            if verbose:
                print("Ratios:", lambdas/lambda1)
            while lambdas[i]/lambda1>self.epsilon and i<len(lambdas):
                i += 1
            return i
        elif self.type == 'pareto': # select model before largest residual increase - don't use this one
            jumps = lambdas[1:]/lambdas[:-1]
            biggest_jump = np.argmax(jumps)
            if verbose:
                print("jumps:", jumps)
                print("biggest jump:", biggest_jump, "to", biggest_jump+1)
            return biggest_jump+1
        
def sparse_reg_bf(theta, scaler, initializer, residual, model_iterator, threshold, inhomog=False, inhomog_col=None, full_regression=False, term_names=None, verbose=False, seed=1):
    # compute sparse regression on Theta * xi = 0
    # theta: matrix of integrated terms
    # threshold: model selection criterion
    # brute_force: true if next model iteration found by brute force
    # char_sizes: vector of characteristic term sizes (per column)
    # row_norms: desired norm of each row
    # verbose: flag for rich output
    # n_terms: fix the number of terms selected
    # and a lot more not described above
    # NEW ARGUMENTS: start_k - used in method='hybrid', power method at k=start_k and then stepwise reduction the rest of the way
    # inhomog - for inhomogeneous regression, paired with inhomog_col: which term to use as b.
    # max_k: max number of terms to keep in model
    # full_regression: True if searching for dense solution
    # term_names: name of terms corresponding to columns
    # seed: random seed to use for test-train split

    random.seed(seed)
    
    # use copies of all mutable objects
    scaler = copy.copy(scaler)
    initializer = copy.copy(initializer)
    model_iterator = copy.copy(model_iterator)

    #verbose = False # to suppress output only from here

    # set relative residual normalization (if not dominant balance residual)
    np.set_printoptions(precision=3)
    if residual.residual_type == "fixed_column":
        residual.set_norm(scaler.norm_col(theta, residual.anchor_col))
        if verbose:
            print('Residual normalization:', residual.norm)

    
    ### PREPROCESSING
    #print("Initial theta:", theta)
    theta = scaler.scale_theta(theta)
    theta = scaler.select_cols(theta)
    if residual.residual_type in ["matrix_relative", "hybrid"]:
        residual.set_norm(np.linalg.norm(theta)) 
        if verbose:
            print('Residual normalization:', residual.norm)

    #print("Scaled theta:", theta)

    represent = lambda term: term if term_names is None else term_names[term]
    if term_names is not None and verbose:
        print(f"Starting regression with the sublibrary {term_names}")
    
    if scaler.train_fraction < 1:
        theta, theta_test = scaler.train_test_split(theta)
        h_test = theta_test.shape[0]
    else:
        h_test = 0
    h, w = theta.shape
    test_train_ratio = h_test/h
    # note that ||Theta*c|| is invariant under orthogonal transformations! so we add QR decomposition for efficiency+better conditioning
    theta = np.linalg.qr(theta, mode='r')
    theta_test = np.linalg.qr(theta_test, mode='r') if h_test>0 else None
        
    ### CHECK ONE-TERM MODELS
    nrm = np.zeros(w)
    nrm_test = np.zeros(w) if h_test>0 else None
    if verbose:
        print("Checking single-term residuals...")
    for term in range(w):
        nrm[term] = np.linalg.norm(theta[:, term])
        if verbose:
            print(f'nrm[{represent(term)}]:', nrm[term])
    best_term, lambda1 = np.argmin(nrm), min(nrm)
    lambda1_test = np.linalg.norm(theta_test[:, best_term]) if h_test>0 else None
    if residual.residual_type in ["matrix_relative", "hybrid", "fixed_column", "absolute"]: # won't work for dominant balance
        lambda1 = lambda1 / residual.norm
        lambda1_test = lambda1_test / residual.norm if h_test>0 else None

    # HANDLE W=0 (inf), W=1 (one-term model) CASES
    if w == 0:  # no inds allowed at all
        return RegressionResult(xi=None, lambd=np.inf, best_term=None, lambda1=np.inf, sublibrary=term_names)
    if w == 1:  # no regression to run
        norm = residual.norm if hasattr(residual, 'norm') else 1
        best_term, lambda1, lambda1_test = scaler.postprocess_single_term(best_term=best_term, lambda1=lambda1,
                                                                          test_train_ratio=test_train_ratio, lambda1_test=lambda1_test,
                                                                          verbose=verbose)
        return RegressionResult(xi=[1], lambd=np.inf, best_term=best_term, lambda1=lambda1, 
                                lambda1_test=lambda1_test, sublibrary=term_names)
        
    ### INITIAL MODEL
    if full_regression:
        initializer.start_k = w
    k = initializer.check_start_k(w)
    if verbose:
        print("Initializing solution with starting k:", k)
    initializer.prepare_inhomog(inhomog, inhomog_col, scaler)
    xi, lambd, iter_direction = initializer.make_model(theta, verbose)
    
    max_k_for_reset = model_iterator.max_k
    max_k = min(model_iterator.max_k, w)
    k = min(max_k, k)
    if verbose:
        print(f"max_k set to {max_k}")
    xis = np.zeros(shape=(max_k, w))
    lambdas = np.inf*np.ones(shape=(max_k,)) # any lambdas that are not computed are assumed to be very large
    test_lambdas = np.inf*np.ones(shape=(max_k,))
    xis[k-1, :] = xi # we have reversed order of arrays compared to the old SR script - index = n_terms-1
    
    if verbose:
        print("Iterating to find model...")
    if not full_regression:
        ### MODEL SELECTION
        model_iterator.reset(k=k, max_k=max_k, direction=iter_direction)
        model_iterator.prepare_inhomog(inhomog, inhomog_col, scaler)
        w_inds = np.array(range(w))
        model_iterator.set_terms(w_inds[xi!=0], verbose) # NOTE that the indices are wrt sublibrary!
        actual_k = len(model_iterator.terms)
        #print(model_iterator)
        
        if residual.residual_type == "dominant_balance":
            qc_cols = np.zeros(shape=(w,))
            for term in model_iterator.terms:
                qc_cols[term] = np.linalg.norm(theta[:, term]*xi[term])
                if verbose:
                    print(f'qc_col[{term}]:', nrm[term])
            residual.set_norm(np.max(qc_cols))
            if verbose:
                print('Residual normalization:', residual.norm)
        # since it was not properly set before this 
        #print("TESTING: K=", k)
        lambdas[k-1] = lambd/residual.norm
        test_lambdas[k-1] = np.linalg.norm(theta_test @ xi)/residual.norm if h_test>0 else None

        if actual_k<k: # initial support is smaller than the maximum -> the initial guess works for smaller k's as well
            for k_i in range(actual_k, k):
                xis[k_i-1, :] = xis[k-1, :]
                lambdas[k_i-1] = lambdas[k-1]
                test_lambdas[k_i-1] = test_lambdas[k-1]

        model_iterator.save_state(xis, lambdas, test_lambdas, verbose) # save initial state
        exit = False
        while not exit:
            ### ITERATION RULE
            xi, lambd, k = model_iterator.get_next(theta, xi, verbose, represent)
            # update current variables in iteration
            #print("TESTING: K=", k)
            xis[k-1, :] = xi # we have reversed order of arrays compared to old SR - index = n_terms-1
            lambdas[k-1] = lambd/residual.norm # WE NORMALIZE LAMBDA HERE FOR CONSISTENT EXIT CRITERION
            test_lambdas[k-1] = np.linalg.norm(theta_test @ xi)/residual.norm if h_test>0 else None
            if verbose and h_test>0:
                print("new test residual:", test_lambdas[k-1])
            #if verbose:
            #    print("current xis:", xis)

            ### test if final model
            exit = model_iterator.check_exit(xis, lambdas, test_lambdas, verbose)
        if verbose:
            print("all xis:", xis)
            print("all lambdas:", lambdas)
            print("all test lambdas:", test_lambdas)
        #ind = threshold.select_model(lambdas, theta, lambda1, verbose)
        ind = threshold.select_model(model_iterator.best_lambdas, theta, lambda1, verbose) # we now use best solution found so far
    else: # just keep all terms
        ind = -1   
        lambdas[k-1] = lambd/residual.norm
        test_lambdas[k-1] = np.linalg.norm(theta_test @ xi)/residual.norm if h_test>0 else None
        if verbose and h_test>0:
            print("new test residual:", test_lambdas[k-1])
        if residual.residual_type == "dominant_balance":
            qc_cols = np.zeros_like(lambdas)
            for term in range(w):
                qc_cols[term] = np.linalg.norm(theta[:, term]*xi[term])
                if verbose:
                    print(f'qc_col[{term}]:', nrm[term])
            residual.set_norm(np.max(qc_cols))
            if verbose:
                print('Residual normalization:', residual.norm)
    if verbose:
        print("Optimal number of terms:", ind+1 if ind!=-1 else "(all)")
        
    #xi, lambd = xis[ind, :], lambdas[ind]
    #lambda_test = test_lambdas[ind] if h_test>0 else None
    xi, lambd = model_iterator.best_state[ind, :], model_iterator.best_lambdas[ind] # use best solutions!
    lambda_test = model_iterator.best_test_lambdas[ind] if h_test>0 else None

    if residual.residual_type == "hybrid": # switch residual from matrix_relative to hybrid for output
        lambd, lambda1 = lambd*residual.norm, lambda1*residual.norm
        lambda_test, lambda1_test = (lambda_test*residual.norm, lambda1_test*residual.norm) if h_test>0 else (None, None)
        w_inds = np.array(range(w))
        terms = w_inds[xi!=0]
        residual.set_norm(np.linalg.norm(theta[:, terms]))
        lambd, lambda1 = lambd/residual.norm, lambda1/residual.norm
        lambda_test, lambda1_test = (lambda_test/residual.norm, lambda1_test/residual.norm) if h_test>0 else (None, None)
    
    ### POSTPROCESSING
    #print('xi before rescaling:', xi)
    if inhomog: # model is normalized so coeff of b column is 1 instead of unit norm, which impacts the residuals -> we fix that here
        xi_norm = np.linalg.norm(xi)
        lambd, lambda1 = lambd/xi_norm, lambda1/xi_norm
        lambda_test, lambda1_test = (lambda_test/xi.norm, lambda1_test/xi.norm) if h_test>0 else (None, None)
    xi, lambd, lambda_test = scaler.postprocess_multi_term(xi=xi, lambd=lambd, test_train_ratio=test_train_ratio, 
                                                           lambda_test=lambda_test, verbose=verbose)
    #print('xi after rescaling:', xi)
    best_term, lambda1, lambda1_test = scaler.postprocess_single_term(best_term=best_term, lambda1=lambda1, 
                                                                      test_train_ratio=test_train_ratio, lambda1_test=lambda1_test,
                                                                      verbose=verbose)
    
    # Reset max_k
    model_iterator.max_k = max_k_for_reset
    
    #return RegressionResult(xi=xi, lambd=lambd, best_term=best_term, lambda1=lambda1, all_xis=xis, all_lambdas=lambdas,
    #                       lambda_test=lambda_test, all_lambdas_test=test_lambdas, lambda1_test=lambda1_test, sublibrary=term_names)
    return RegressionResult(xi=xi, lambd=lambd, best_term=best_term, lambda1=lambda1, all_xis=model_iterator.best_state,
                            all_lambdas=model_iterator.best_lambdas,
                            lambda_test=lambda_test, all_lambdas_test=model_iterator.best_test_lambdas, 
                            lambda1_test=lambda1_test, sublibrary=term_names)

# evaluate specific model on Q 
# NOTE: use scaler with same column subsampling as any models being compared with & make sure model_vector sparsity matches it
def evaluate_model(theta, model_vector, scaler, residual, verbose=False):
    np.set_printoptions(precision=3)
    if residual.residual_type == "fixed_column":
        residual.set_norm(scaler.norm_col(theta, residual.anchor_col))
        if verbose:
            print('Residual normalization:', residual.norm)

    ### PREPROCESSING
    theta = scaler.scale_theta(theta)   
    model_vector = scaler.scale_model(model_vector)
    h, w = theta.shape

    if residual.residual_type == "matrix_relative":
        residual.set_norm(np.linalg.norm(theta)/theta.shape[1]) 
        if verbose:
            print('Residual normalization:', residual.norm)
    if residual.residual_type == "dominant_balance":
        qc_cols = np.zeros(shape=(w,))
        for term in range(w):
            qc_cols[term] = np.linalg.norm(theta[:, term]*model_vector[term])
            if verbose:
                print(f'qc_col[{term}]:', qc_cols[term])
        residual.set_norm(np.max(qc_cols))
        if verbose:
            print('Residual normalization:', residual.norm)

    model_vector, lambd, _ = scaler.postprocess_multi_term(xi=model_vector, lambd=np.linalg.norm(theta @ model_vector)/residual.norm, verbose=verbose)
    return lambd

def hybrid_residual(theta, xi, scaler, return_xi=False): # compute the "hybrid" residual for an existing model
    # preprocessing
    theta = scaler.scale_theta(theta)   
    theta = scaler.select_cols(theta) 

    # find nonzero inds
    w_inds = np.array(range(theta.shape[1]))
    if xi is None: # allow best choice of xi to be found
        xi = solve(theta, w_inds)
        #print(theta, w_inds, xi)
    else:
        xi = scaler.scale_model(xi)
    terms = w_inds[xi!=0]
    
    norm = np.linalg.norm(theta[:, terms])
    norm = 1 if norm==0 else norm
    xi = xi/np.linalg.norm(xi)
    lambd = np.linalg.norm(theta @ xi)/norm

    #print("xi before postprocessing:", xi)
    #print("lambd before postprocessing:", lambd)
    #print("norm:", norm)

    xi, lambd, _ = scaler.postprocess_multi_term(xi, lambd)

    #print("xi after postprocessing:", xi)
    #print("lambd after postprocessing:", lambd)
    if return_xi:
        #xi, lambd, _ = scaler.postprocess_multi_term(xi, np.linalg.norm(theta @ xi)/norm)
        return lambd, xi[scaler.sub_inds]
    else:
        return lambd

# taken with minor modifications from Bertsimas & Gurnee, 2023
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