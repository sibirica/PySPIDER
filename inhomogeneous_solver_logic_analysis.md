# Logic Analysis: Inhomogeneous Solver for Recursive Peeling

## Current Problem State Analysis

### What Happens in Current `recursive_peeling_regression`:

1. **Input**: 
   - `theta = [X | y]` (shape: samples × 11, where 11 = 10 features + 1 target)
   - `initial_result.xi` (shape: 11, coefficients from homogeneous solver)

2. **Locking Step**:
   ```python
   newly_locked = [0, 1, 2, 3, 10]  # First 4 features + target column
   locked_coeffs = [1.000, 0.100, 0.010, 0.001, -1.000]
   ```

3. **RHS Construction**:
   ```python
   # Move locked terms to RHS: b = -sum(coeff_i * column_i)
   b = -(1.000*X[:,0] + 0.100*X[:,1] + 0.010*X[:,2] + 0.001*X[:,3] + (-1.000)*y)
   b = -(X @ [1,0.1,0.01,0.001] - y)  # This should ≈ 0 if solution is good
   ```

4. **Problem Creation**:
   ```python
   remaining_indices = [4, 5, 6, 7, 8, 9]  # Unlocked feature columns
   Q_augmented = [theta[:, remaining_indices] | b]  # Add b as RHS column
   # Now solving: [X_remaining | b] @ ξ = 0  (STILL HOMOGENEOUS!)
   ```

5. **The Issue**: We're still using homogeneous formulation, but now with a manufactured RHS column

## What Should Happen Instead (Inhomogeneous Approach):

### Mathematical Setup:
```
Original problem: [X | y] @ ξ = 0
Locked terms:     X_locked @ ξ_locked = known_contribution
Remaining:        X_remaining @ ξ_remaining = -known_contribution
```

### Step-by-Step Logic:

1. **Identify Components**:
   ```python
   locked_indices = [0, 1, 2, 3, 10]
   unlocked_indices = [4, 5, 6, 7, 8, 9]
   locked_coeffs = initial_result.xi[locked_indices]
   ```

2. **Extract Matrices**:
   ```python
   X_locked = theta[:, locked_indices]     # Shape: (samples, 5)
   X_remaining = theta[:, unlocked_indices] # Shape: (samples, 6)
   ```

3. **Compute RHS**:
   ```python
   # Known contribution from locked terms
   locked_contribution = X_locked @ locked_coeffs
   # RHS for inhomogeneous system: X_remaining @ ξ = -locked_contribution
   b = -locked_contribution
   ```

4. **Solve Inhomogeneous System**:
   ```python
   # Direct solution: X_remaining @ ξ = b
   # This is standard least squares, not null space finding
   ```

5. **Reconstruct Full Solution**:
   ```python
   full_coeffs = np.zeros(original_library_size)
   full_coeffs[locked_indices] = locked_coeffs
   full_coeffs[unlocked_indices] = ξ_solution
   ```

## Critical Differences from Current Approach:

### 1. Mathematical Formulation
**Current (Wrong)**:
- Still uses homogeneous: `[X_remaining | b] @ ξ = 0`
- Creates artificial null space problem
- Results in extra dimension and -1/+1 artifacts

**Proposed (Correct)**:
- Uses inhomogeneous: `X_remaining @ ξ = b`
- Solves actual least squares problem
- Natural dimension matching

### 2. Coefficient Interpretation
**Current**:
- Output `ξ` has wrong interpretation
- Mixing homogeneous and inhomogeneous coefficients
- Complex reconstruction logic

**Proposed**:
- Output `ξ` directly represents corrections for unlocked terms
- Simple addition: `full_solution = locked + corrections`
- Clean, interpretable results

### 3. Scaling Considerations
**Current**:
- Uses original char_sizes including target column
- Scales augmented matrix with artificial RHS
- Inconsistent scaling between locked and remaining

**Proposed**:
- Use char_sizes only for remaining terms
- Scale remaining matrix and RHS consistently
- Proper scaling isolation

## Implementation Requirements for Inhomogeneous Solver:

### Function Signature:
```python
def sparse_reg_bf_inhomogeneous(
    X,                        # Remaining terms matrix (samples × remaining_terms)
    b,                        # RHS vector (samples,)
    scaler,                   # Scaler for remaining terms only
    initializer, residual, model_iterator, threshold,
    full_regression=False,
    term_names=None,
    verbose=False,
    seed=1
):
    """
    Solve: X @ ξ = b (inhomogeneous least squares with sparsity)
    """
```

### Key Logic Changes:

1. **No Augmentation**: Work directly with X and b, no column stacking
2. **Residual Formulation**: `||X @ ξ - b||` instead of `||[X|b] @ ξ||`
3. **Initialization**: Initialize with least squares solution: `ξ_0 = pinv(X) @ b`
4. **Iteration**: Standard forward/backward selection on `X @ ξ ≈ b`
5. **Output**: Coefficients directly represent corrections for remaining terms

### Validation Checks:

1. **Dimension Consistency**:
   ```python
   assert X.shape[1] == len(scaler.char_sizes), "Scaler size mismatch"
   assert X.shape[0] == len(b), "Sample count mismatch"
   ```

2. **Mathematical Validity**:
   ```python
   # Check that least squares solution exists
   rank_X = np.linalg.matrix_rank(X)
   assert rank_X > 0, "Singular remaining matrix"
   ```

3. **Reconstruction Test**:
   ```python
   # Verify we can reconstruct the RHS
   residual_check = np.linalg.norm(X @ ξ - b)
   assert residual_check < tolerance, "Poor inhomogeneous solution"
   ```

## Potential Pitfalls to Avoid:

### 1. Scaling Mismatch
- **Risk**: Using char_sizes from original augmented library
- **Fix**: Create new char_sizes for remaining terms only

### 2. Index Confusion
- **Risk**: Mixing original library indices with remaining term indices
- **Fix**: Clear mapping between original and reduced spaces

### 3. Residual Calculation
- **Risk**: Using homogeneous residual formula
- **Fix**: Use `||X @ ξ - b||` for inhomogeneous case

### 4. Threshold Application
- **Risk**: Thresholds designed for homogeneous case may not apply
- **Fix**: May need separate threshold logic for inhomogeneous

## Testing Strategy:

### 1. Simple Test Case:
```python
# Known solution test
X = np.random.randn(100, 5)
ξ_true = np.array([1, 0, 2, 0, 0])  # Sparse solution
b = X @ ξ_true + 0.01 * np.random.randn(100)  # Small noise

# Should recover approximately ξ_true
ξ_recovered = sparse_reg_bf_inhomogeneous(X, b, ...)
```

### 2. Peeling Integration Test:
```python
# Test that peeling with new solver gives clean results
result_peeled = recursive_peeling_regression_v2(theta, ..., result)
assert len(result_peeled.xi) == theta.shape[1], "Dimension match"
assert not (result_peeled.xi[-2:] == [-1, 1]).all(), "No artifacts"
```

## Conclusion:

The key insight is that recursive peeling creates **genuine inhomogeneous sub-problems** that should be solved as such, not forced through a homogeneous framework. The inhomogeneous solver should:

1. Accept `X` (remaining terms) and `b` (RHS) directly
2. Solve `X @ ξ = b` with sparsity constraints
3. Return corrections that can be directly added to locked solution
4. Maintain proper dimensions throughout

This approach eliminates the mathematical inconsistency that causes the -1/+1 artifacts and dimension mismatches.
