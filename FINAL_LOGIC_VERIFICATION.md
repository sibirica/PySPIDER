# FINAL LOGIC VERIFICATION: Inhomogeneous Solver for Recursive Peeling

## STEP 1: Understanding the Current State (What We Have Now)

### Initial Homogeneous Problem Setup:
```
Original data: X (500×10), y (500×1)
Homogeneous formulation: [X | y] @ ξ = 0
Library: theta = [X | y] (500×11)
Solution: ξ = [c₁, c₂, ..., c₁₀, c_target] where c_target ≈ -1
```

### What sparse_reg_bf Returns:
```python
result.xi = [1.000, 0.100, 0.010, 0.001, 0, 0, 0, 0, 0, 0, -1.000]
#           [  feature coefficients (10)     ] [target coefficient]
#           indices: 0,1,2,3,4,5,6,7,8,9,10
```

### Mathematical Verification:
```
[X | y] @ result.xi = X @ result.xi[0:10] + y * result.xi[10]
                    = X @ [1, 0.1, 0.01, 0.001, 0...] + y * (-1)
                    = X @ true_coeffs - y
                    ≈ 0 (since y = X @ true_coeffs)
```
✅ This is mathematically correct.

## STEP 2: Understanding What Happens in Current Recursive Peeling

### Locking Step:
```python
newly_locked = [0, 1, 2, 3, 10]  # Lock first 4 features + target
locked_coeffs = [1.000, 0.100, 0.010, 0.001, -1.000]
remaining_indices = [4, 5, 6, 7, 8, 9]  # Unlocked features only
```

### RHS Construction:
```python
locked_contribution = theta[:, [0,1,2,3,10]] @ locked_coeffs
                    = X[:,0]*1.0 + X[:,1]*0.1 + X[:,2]*0.01 + X[:,3]*0.001 + y*(-1.0)
                    = X[:,[0,1,2,3]] @ [1.0, 0.1, 0.01, 0.001] - y
                    
b = -locked_contribution  # Move to RHS
  = -(X_partial @ coeffs_partial - y)
  = y - X_partial @ coeffs_partial
```

### Current Augmented Problem:
```python
Q_augmented = [X_remaining | b]  # Shape: (500, 7)
# Solve: [X_remaining | b] @ ξ_new = 0  (STILL HOMOGENEOUS!)
```

### The Issue:
We're solving `[X_remaining | b] @ ξ_new = 0`, which means:
```
X_remaining @ ξ_new[0:6] + b * ξ_new[6] = 0
X_remaining @ ξ_new[0:6] = -b * ξ_new[6]
```

But `b` represents the RHS we want to solve for, not another variable!

## STEP 3: What We SHOULD Be Doing (Inhomogeneous Formulation)

### Mathematical Goal:
We want to find corrections for the remaining features such that the overall homogeneous constraint is satisfied.

### Correct Mathematical Setup:
```
Original constraint: [X | y] @ ξ_full = 0
Decompose: X @ ξ_features + y * ξ_target = 0
With locking: X_locked @ ξ_locked + X_remaining @ ξ_remaining + y * ξ_target = 0

Rearrange: X_remaining @ ξ_remaining = -(X_locked @ ξ_locked + y * ξ_target)
                                     = -(locked_contribution)
                                     = b
```

So we should solve: `X_remaining @ ξ_remaining = b`

### Key Insight:
The target column (y) is NOT part of the variables we're solving for in the inhomogeneous step. 
It's part of the constraint that defines the RHS.

## STEP 4: Correct Inhomogeneous Solver Logic

### Input Processing:
```python
def sparse_reg_bf_inhomogeneous(X_remaining, b, scaler, ...):
    """
    Solve: X_remaining @ ξ = b
    
    X_remaining: (samples × n_remaining_features) 
    b: (samples,) - RHS from locked terms
    ξ: (n_remaining_features,) - corrections for remaining features
    """
```

### Key Differences from Current sparse_reg_bf:

1. **No Augmentation**: We don't add b as a column
2. **Different Residual**: `||X @ ξ - b||` instead of `||(X|b) @ ξ||`
3. **Different Initialization**: `ξ₀ = pinv(X) @ b` instead of power method on null space
4. **Different Constraint**: Minimize `||X @ ξ - b||` subject to sparsity

### Reconstruction Logic:
```python
# After solving inhomogeneous system
full_solution = np.zeros(11)  # 10 features + 1 target

# Locked terms (features + target)
full_solution[[0,1,2,3]] = locked_feature_coeffs  # [1.0, 0.1, 0.01, 0.001]
full_solution[10] = locked_target_coeff           # -1.0

# Corrections for remaining features  
full_solution[[4,5,6,7,8,9]] += ξ_remaining      # Add corrections

# Target coefficient stays the same (maintains homogeneous constraint)
```

## STEP 5: Critical Verification Questions

### Q1: Index Management
**Question**: Are we correctly separating feature indices from target index?
**Answer**: YES
- Feature indices: 0,1,2,3,4,5,6,7,8,9 (what we solve for)
- Target index: 10 (constraint, not variable)
- Locked feature indices: 0,1,2,3
- Remaining feature indices: 4,5,6,7,8,9

### Q2: RHS Construction  
**Question**: Is the RHS correctly computed?
**Answer**: YES
```python
b = -(X_locked @ ξ_locked + y * ξ_target)
  = -(X[:,[0,1,2,3]] @ [1.0,0.1,0.01,0.001] + y * (-1.0))
  = -(locked_feature_contribution - y)
```

### Q3: Mathematical Consistency
**Question**: Does solving `X_remaining @ ξ = b` maintain the original homogeneous constraint?
**Answer**: YES
```
After solving: X_remaining @ ξ_remaining = b
Substitute back: X_locked @ ξ_locked + X_remaining @ ξ_remaining + y * ξ_target
                = locked_contribution + b + y * ξ_target
                = locked_contribution - locked_contribution  (since b = -locked_contribution)
                = 0 ✓
```

### Q4: Dimension Consistency
**Question**: Do dimensions work out correctly?
**Answer**: YES
- X_remaining: (500, 6) - 6 remaining features
- b: (500,) - RHS vector
- ξ_remaining: (6,) - corrections for 6 remaining features
- full_solution: (11,) - 10 features + 1 target

## STEP 6: Implementation Checklist

### ✅ Correct Mathematical Formulation:
- Solve `X_remaining @ ξ = b` (inhomogeneous least squares)
- NOT `[X_remaining | b] @ ξ = 0` (homogeneous null space)

### ✅ Correct Input Processing:
- X_remaining: remaining feature columns only (no target column)
- b: RHS from locked terms (includes target contribution)

### ✅ Correct Scaling:
- scaler.char_sizes: only for remaining features
- Scale X_remaining and b consistently

### ✅ Correct Reconstruction:
- full_solution[locked_feature_indices] = locked_feature_coeffs
- full_solution[target_index] = locked_target_coeff  
- full_solution[remaining_feature_indices] += ξ_solution

### ✅ Correct Validation:
- Check: `||X_remaining @ ξ - b||` is small
- Check: Full solution satisfies original homogeneous constraint
- Check: No extra dimensions or artifacts

## STEP 7: Expected Outcome

### What Should Happen:
1. **Correct Dimensions**: `result.xi` has length 11 (10 features + 1 target)
2. **No Artifacts**: No extra -1/+1 coefficients
3. **Better Recovery**: Improved coefficient estimates for smaller terms
4. **Mathematical Consistency**: Solution satisfies original constraint

### Red Flags to Watch For:
1. **Dimension Mismatch**: If result has wrong length
2. **Scaling Issues**: If coefficients have wrong magnitude
3. **Constraint Violation**: If `[X|y] @ result.xi` is not ≈ 0
4. **Index Confusion**: If locked terms change unexpectedly

## CONCLUSION: Ready to Implement?

### ✅ Logic is Sound:
- Mathematical formulation is correct
- Index management is clear
- Dimension consistency verified
- Reconstruction logic is proper

### ✅ Implementation Plan is Clear:
1. Create `sparse_reg_bf_inhomogeneous` function
2. Modify `recursive_peeling_regression` to use it
3. Fix coefficient reconstruction
4. Test and validate

### ✅ Verification Strategy Ready:
- Simple test cases
- Dimension checks  
- Mathematical constraint verification
- Integration testing

**The logic is correct and ready for implementation.**
