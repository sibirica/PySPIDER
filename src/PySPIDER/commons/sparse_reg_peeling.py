"""
Recursive Peeling Regression for PySPIDER

Implements single-step refinement for hierarchical equation discovery.
This module extends sparse_reg_bf to refine existing regression results by:
1. Selecting terms to lock from the input result
2. Using inhomogeneous columns to represent locked terms  
3. Running sparse regression to find corrections/refinements
"""

import numpy as np
import copy
from .sparse_reg_bf import sparse_reg_bf, RegressionResult


def select_terms_to_lock(xi, current_locked):
    """
    Select terms to lock based on what sparse_reg_bf actually chose.
    
    This function simply identifies all non-zero terms in the regression result
    that aren't already locked. This ensures we lock exactly the terms that
    sparse_reg_bf determined to be significant, maintaining complete consistency
    with the original regression logic.

    Parameters
    ----------
    xi : np.ndarray
        Current coefficient vector from sparse_reg_bf (full library indexing)
    current_locked : list
        Indices of already locked terms

    Returns
    -------
    list
        Indices of newly locked terms (not including current_locked)
    """
    # Lock ALL non-zero terms that sparse_reg_bf chose (and aren't already locked)
    newly_locked = [i for i in range(len(xi)) 
                   if xi[i] != 0 and i not in current_locked]
    
    return newly_locked


# NOTE: Stopping criteria are NOT implemented here.
# The peeling algorithm is designed to be single-step and composable.
# Stopping decisions should be made by the caller using:
# 1. Check len(newly_locked) == 0 (no more terms to lock)  
# 2. Use existing Threshold.select_model() logic
# 3. Apply external residual improvement thresholds
# 4. Use ModelIterator.max_passes for iteration limits
#
# This maintains consistency with sparse_reg_bf.py which uses the same approach.




def recursive_peeling_regression(
    theta,                    # Library matrix Q
    scaler,                   # Scaler object
    initializer,              # Initializer object
    residual,                 # Residual object
    model_iterator,           # ModelIterator object
    threshold,                # Threshold object
    initial_result, 
    inhomog,          # Whether to use inhomogeneous columns

    # Standard sparse_reg_bf parameters
    full_regression=False,
    term_names=None,
    verbose=False,
    seed=1
):
    """
    Perform one step of recursive peeling regression refinement.
    
    This function takes an existing RegressionResult and refines it by:
    1. Locking all non-zero terms that sparse_reg_bf chose in the initial result
    2. Setting up inhomogeneous columns for the locked terms
    3. Running sparse regression to find corrections/refinements
    
    Parameters
    ----------
    theta : np.ndarray
        Library matrix Q (domains × terms)
    scaler : Scaler
        Scaler object for preprocessing
    initializer : Initializer  
        Initializer object for initial guess
    residual : Residual
        Residual object for normalization
    model_iterator : ModelIterator
        ModelIterator object for term selection
    threshold : Threshold
        Threshold object for model selection
    initial_result : RegressionResult
        Previous regression result to refine (all non-zero terms will be locked)
    full_regression : bool, optional
        Whether to search for dense solution
    term_names : list, optional
        Names of terms corresponding to columns
    verbose : bool, optional
        Whether to print detailed output
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    RegressionResult
        Refined regression result with peeling-specific metadata:
        - locked_indices: List of locked term indices from this step
    """
    
    if verbose:
        print("=== STARTING PEELING REFINEMENT STEP ===")
        print("Lock strategy: Lock all non-zero terms from sparse_reg_bf")
        print(f"Initial residual: {initial_result.lambd:.2e}")
        print(f"Initial active terms: {np.sum(initial_result.xi != 0)}")
    
    '''class RegressionResult(object):
    def __init__(
        self, xi, lambd, best_term, lambda1, all_xis=None, all_lambdas=None, 
        lambda1_test=None, lambda_test=None, all_lambdas_test=None, 
        sublibrary=None
    ):
        self.xi = xi
        self.lambd = lambd
        self.lambda_test = lambda_test
        self.best_term = best_term
        self.lambda1 = lambda1
        self.lambda1_test = lambda1_test
        # reminder: row i of these arrays corresponds to i+1 term model
        self.all_xis = all_xis
        self.all_lambdas = all_lambdas
        self.all_lambdas_test = all_lambdas_test
        # list of actual terms corresponding to this sublibrary - can also be set later
        self.sublibrary = sublibrary
    '''
    current_locked = getattr(initial_result, 'locked_indices', [])
    newly_locked = select_terms_to_lock(
        initial_result.xi,
        current_locked
    )

    if len(newly_locked) == 0:
        if verbose:
            print("No new terms to lock. Returning unchanged result.")
        # Return the same result but add empty locked_indices if not present
        result_copy = copy.copy(initial_result)
        if not hasattr(result_copy, 'locked_indices'):
            result_copy.locked_indices = current_locked
        return result_copy

    # Update locked indices
    all_locked = current_locked + newly_locked
    
    if verbose:
        print(f"Previously locked terms: {current_locked}")
        print(f"Newly locked terms: {newly_locked}")
        print(f"Total locked terms: {len(all_locked)}")
        # Filter locked indices to only include those within the original library bounds
        valid_locked_verbose = [idx for idx in all_locked if idx < theta.shape[1]]
        print(f"Valid locked terms (within library bounds): {len(valid_locked_verbose)}")
        
        # Print the equation being locked
        if len(valid_locked_verbose) > 0:
            locked_coeffs_preview = initial_result.xi[valid_locked_verbose]
            equation_terms = [f"{coeff:.3f}*T[{idx}]" for idx, coeff in zip(valid_locked_verbose, locked_coeffs_preview) if coeff != 0]
            equation_str = " + ".join(equation_terms) if equation_terms else "0"
            print(f"Locking equation: {equation_str}")
        else:
            print("No valid terms to lock within library bounds")

    # === STEP 2: Set up single RHS column for all locked terms ===
    # Filter locked indices to only include those within the original library bounds
    valid_locked = [idx for idx in all_locked if idx < theta.shape[1]]
    
    if len(valid_locked) == 0:
        # No valid locked terms, just return original result  
        return initial_result
        
    locked_coeffs = initial_result.xi[valid_locked]
    Q_locked = theta[:, valid_locked]
    
    # Get characteristic sizes for locked terms
    locked_char_sizes = scaler.char_sizes[valid_locked]
    
    # Construct RHS by moving locked terms: b = -sum_i c_i a_i
    # For inhomogeneous problem: remaining_terms @ xi = b
    # where b contains the contribution from locked terms
    scaled_terms = []
    for i, (col_idx, coeff) in enumerate(zip(valid_locked, locked_coeffs)):
        # Move locked term to RHS: NO characteristic size scaling here
        # The characteristic size scaling is handled by the sparse regression itself
        scaled_col = coeff * Q_locked[:, i]
        scaled_terms.append(scaled_col)
    
    # Combine all locked terms: b = -sum_i c_i a_i (negative for moving to RHS)
    b = -np.sum(scaled_terms, axis=0) if scaled_terms else np.zeros(theta.shape[0])
    
    # === STEP 3: Augment library with single RHS column ===
    Q_augmented = np.column_stack([theta, b.reshape(-1, 1)])
    rhs_col_idx = theta.shape[1]
    
    if verbose:
        print(f"Added RHS column for {len(all_locked)} locked terms")
        print(f"RHS column index: {rhs_col_idx}")
        print(f"RHS vector norm: {np.linalg.norm(b):.2e}")

    # === STEP 4: Update scaler for augmented problem ===
    # EXACTLY match sparse_reg_bf characteristic size computation
    scaler_aug = copy.copy(scaler)
    
    # Characteristic size of RHS column: use its norm to match other columns
    # This ensures the RHS column has appropriate weight in the regression
    char_size_rhs = np.linalg.norm(b) if np.linalg.norm(b) > 0 else 1.0
    
    # Append RHS characteristic size to the full char_sizes array  
    scaler_aug.char_sizes = np.append(scaler.char_sizes, char_size_rhs)
    scaler_aug.full_w = len(scaler_aug.char_sizes)

    # CRITICAL: Set sub_inds to exclude locked terms from regression
    # Only include: remaining unlocked terms + RHS column
    remaining_indices = [i for i in range(theta.shape[1]) if i not in valid_locked]
    active_indices = remaining_indices + [rhs_col_idx]
    scaler_aug.reset_inds(active_indices)
    
    if verbose:
        print(f"Remaining unlocked terms: {len(remaining_indices)} -> indices: {remaining_indices}")
        print(f"RHS char_size: {char_size_rhs:.2e}")
        print(f"Scaler sub_inds set to: {active_indices} (excludes locked terms {valid_locked})")

    result = sparse_reg_bf(
        Q_augmented,
        scaler=scaler_aug,
        initializer=initializer,
        residual=residual,
        model_iterator=model_iterator,
        threshold=threshold,
        inhomog=inhomog,
        inhomog_col=rhs_col_idx,  # RHS column forced to -1
        full_regression=full_regression,
        term_names=term_names,
        verbose=verbose,
        seed=seed
    )

    # sparse_reg_bf returns coefficients in correct format - no manual extraction needed
    combined_xi = result.xi.copy()  # Already in full library coordinates and proper scale

    # Add back the locked coefficients (these maintain their original scale)  
    combined_xi[valid_locked] += locked_coeffs

    # Update result with combined coefficients
    result.xi = combined_xi

    if verbose:
        print(f"New residual: {result.lambd:.2e}")
        print(f"Total active terms: {np.sum(result.xi != 0)}")
        print("=== PEELING REFINEMENT COMPLETE ===")

    # === STEP 7: Add peeling metadata to existing result ===
    # No need to create a new RegressionResult - just add metadata to the existing one
    result.locked_indices = all_locked

    return result
