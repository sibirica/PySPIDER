# Design: Separate Implementations for Homogeneous vs Inhomogeneous Sparse Regression

## Problem Analysis

Currently, we have one `sparse_reg_bf` function that handles both:
1. **Homogeneous problems**: `[X | y] @ ξ = 0` (null space finding)
2. **Inhomogeneous problems**: `X @ ξ = y` (standard regression)

The issue is that `recursive_peeling_regression` creates **inhomogeneous sub-problems** that have different mathematical structure than the original homogeneous setup, leading to:
- Redundant target columns (-1, +1 pattern)
- Coefficient vector dimension mismatches
- Conflicting constraint interpretations

## Proposed Solution Architecture

### 1. Keep Existing `sparse_reg_bf` for Homogeneous Problems
- **Purpose**: Handle `[X | y] @ ξ = 0` formulations
- **Use case**: Initial sparse regression on augmented library
- **Output**: Coefficients where last element ≈ -1 (target coefficient)
- **No changes needed** - this works well currently

### 2. Create New `sparse_reg_bf_inhomogeneous` for Peeling
- **Purpose**: Handle `X_remaining @ ξ = b` formulations from peeling
- **Use case**: Recursive peeling refinement steps
- **Input**: Locked terms moved to RHS, remaining terms to solve
- **Output**: Correction coefficients to add to locked solution

## Key Differences in Implementation

### Mathematical Formulation

**Homogeneous (`sparse_reg_bf`):**
```
Input:  [X | y] (augmented library)
Solve:  [X | y] @ ξ = 0
Output: ξ = [c₁, c₂, ..., cₙ, -1]
```

**Inhomogeneous (`sparse_reg_bf_inhomogeneous`):**
```
Input:  X_remaining, b (RHS from locked terms)
Solve:  X_remaining @ ξ = b
Output: ξ = [δc₁, δc₂, ..., δcₘ] (corrections only)
```

### Coefficient Interpretation

**Homogeneous:**
- Coefficients represent the full solution
- Target column coefficient should be -1
- Recover true coefficients by normalization: `c_true = -ξ[:-1] / ξ[-1]`

**Inhomogeneous:**
- Coefficients represent corrections/refinements
- Add to existing locked solution: `c_new = c_locked + ξ_corrections`
- No target column handling needed

### Scaling and Normalization

**Homogeneous:**
- Characteristic sizes include target column (typically 1.0)
- Scale entire augmented library `[X | y]`
- Maintain homogeneous constraint throughout

**Inhomogeneous:**
- Characteristic sizes only for remaining terms
- Scale remaining library + RHS column separately
- Handle RHS scaling to match remaining terms

## Implementation Strategy

### Phase 1: Create `sparse_reg_bf_inhomogeneous`
```python
def sparse_reg_bf_inhomogeneous(
    X_remaining,              # Remaining unlocked terms
    b,                        # RHS from locked terms
    scaler,                   # Scaler for remaining terms
    initializer, residual, model_iterator, threshold,
    full_regression=False,
    term_names=None,
    verbose=False,
    seed=1
):
    """
    Inhomogeneous sparse regression: X_remaining @ ξ = b
    
    Designed specifically for recursive peeling where:
    - X_remaining contains unlocked library terms
    - b contains contribution from locked terms
    - Output ξ represents corrections to add to locked solution
    """
```

### Phase 2: Modify `recursive_peeling_regression`
Instead of:
```python
# Current: Add RHS column and use homogeneous solver
Q_augmented = np.column_stack([theta, b])
result = sparse_reg_bf(Q_augmented, ...)
```

Use:
```python
# New: Direct inhomogeneous solving
X_remaining = theta[:, unlocked_indices]
result = sparse_reg_bf_inhomogeneous(X_remaining, b, ...)
```

### Phase 3: Clean Coefficient Reconstruction
**Current (problematic):**
```python
# Results in [c₁, c₂, ..., cₙ, -1, +1] with wrong dimensions
combined_xi = result.xi.copy()
combined_xi[valid_locked] += locked_coeffs
```

**New (clean):**
```python
# Results in [c₁, c₂, ..., cₙ] with correct dimensions
combined_xi = np.zeros(original_library_size)
combined_xi[locked_indices] = locked_coeffs
combined_xi[unlocked_indices] += result.xi  # Add corrections
```

## Benefits of This Approach

### 1. Mathematical Clarity
- Homogeneous problems use homogeneous solver
- Inhomogeneous problems use inhomogeneous solver
- No more mixing constraint types

### 2. Dimension Consistency
- Coefficient vectors always match original library size
- No extra RHS columns causing dimension mismatches
- Clean separation of concerns

### 3. Easier Debugging
- Each solver handles one mathematical formulation
- Clear input/output contracts
- Separate testing and validation paths

### 4. Performance Benefits
- No unnecessary column augmentation
- Direct solution of the actual mathematical problem
- Reduced overhead from constraint mixing

## Implementation Details

### Scaler Handling
```python
# Inhomogeneous scaler setup
inhomog_scaler = Scaler(
    sub_inds=None,  # Use all remaining terms
    char_sizes=char_sizes[unlocked_indices]  # Only remaining terms
)
```

### Residual Formulation
```python
# Inhomogeneous residual: ||X_remaining @ ξ - b||
# vs Homogeneous residual: ||(X|y) @ ξ||
```

### Threshold Selection
```python
# May need different threshold strategies for inhomogeneous case
# Since we're solving corrections, not full solutions
```

## Testing Strategy

### 1. Unit Tests
- Test `sparse_reg_bf_inhomogeneous` on simple known cases
- Verify coefficient recovery and dimension consistency
- Compare against manual least squares solutions

### 2. Integration Tests
- Test full recursive peeling pipeline
- Verify no -1/+1 artifacts
- Check coefficient vector dimensions

### 3. Validation Tests
- Compare final solutions against true coefficients
- Test convergence behavior over multiple peeling iterations
- Verify mathematical correctness of constraint handling

## Risk Mitigation

### 1. Backward Compatibility
- Keep existing `sparse_reg_bf` unchanged
- New function is additive, not replacing existing functionality

### 2. Gradual Migration
- Can test new approach alongside old approach
- Easy to revert if issues discovered

### 3. Clear Documentation
- Document when to use each function
- Provide examples and best practices

## Conclusion

This separation addresses the root cause of the homogeneous/inhomogeneous mixing problem by:
1. Using appropriate solvers for appropriate mathematical formulations
2. Maintaining clean coefficient vector dimensions
3. Eliminating redundant constraint handling

The result should be a more robust, debuggable, and mathematically sound recursive peeling implementation.
