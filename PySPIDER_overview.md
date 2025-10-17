### commons/z3base.py
- SymmetryRep (FullRank, Antisymmetric, SymmetricTraceFree): Representation of an irreducible representation (irrep) of the tensor symmetry group used to organize libraries and weights by symmetry.
- VarIndex / LiteralIndex / IndexHole:
  - VarIndex: abstract Einstein indices (free/bound placeholders).
  - LiteralIndex: concrete coordinate indices (e.g., x/y/z/t) for evaluation.
  - IndexHole: placeholder used during canonicalization and mapping.
- EinSumExpr: Base type for symbolic tensor expressions in Einstein notation with traversal/mapping and canonicalization hooks.
- generate_indexings(expr): Formulates and solves an SMT problem to assign consistent free/bound indices that satisfy commutativity and ordering constraints, enumerating canonical indexings to uniquely label terms.

### commons/library.py
- Observable: A measured field (with rank/indices) used as the basic input to model discovery.
- DerivativeOrder: Stores time and spatial derivative orders (including multi-index for spatial).
- LibraryPrime: A derivative applied to a derivand (e.g., an `Observable` or discrete coarse-grained product).
- LibraryTerm: Commutative product of primes with a target rank; supports symmetry checks and canonicalization.
- Equation: Represents an equation as LHS = 0; supports elimination/solving for a given term to express LHS in terms of others.

### commons/process_library_terms.py
- IntegrationDomain: Rectangular subdomain (space × time) defining where inner products are evaluated.
- Weight / TensorWeight / TensorWeightBasis (incl. FactoredTensorWeight): Scalar Legendre-based test functions and tensor bases per irrep; factorization for efficient evaluation.
- LibraryData: Holds the per-irrep library, library matrix G (Q), and scaling info.
- AbstractDataset: Abstract base extended by `continuous/` and `discrete/` variants; contains the dataset and all configuration used for library construction, evaluation, and regression.
- Key methods:
  - make_libraries(): Use `library.py` to create libraries of terms.
  - make_domains(): Randomly place rectangular spatiotemporal domains (with padding options).
  - make_weights(): Build scalar/tensor weight bases per irrep.
  - make_Q / make_Q_parallel(): Assemble library matrices by evaluating all term–weight–domain inner products; parallel version precomputes integration by parts and assigns domains to workers.
  - eval_term() / eval_prime(): Evaluate primes (dataset-specific in subclasses), multiply, and integrate over domains.
  - int_by_parts(): Symbolically integrate by parts across dimensions before evaluation.
  - diff(): Apply finite-difference operators to data.
  - find_scales(), get_char_size(): Compute nondimensionalization statistics and characteristic sizes using length/time scales.

### commons/identify_models.py
- identify_equations(...): For one irrep, sweep complexity, select sublibraries, run regression, form equations, report residuals, and derive implications to exclude terms.
- interleave_identify(...): Alternate across irreps, sharing implications where valid (e.g., reuse FullRank implications).
- make_equation_from_Xi(): Convert regression output into an `Equation` (multi-term) or the best single-term.
- infer_equations(...): Generate derived equations by contractions and by applying time/space derivatives and prime multiplications up to a complexity bound.
- get_all_contractions(), form_equation(), get_primes(): Canonicalization/contract helpers and prime enumeration for implication search.

### commons/sparse_reg_bf.py
- Scaler: Column/row normalization, sublibrary column selection, train/test split, and postprocessing of coefficients/residuals back to full-library coordinates.
- Initializer: Build an initial model (combinatorial or power iteration); handles inhomogeneous setups.
- ModelIterator: Backward–forward term selection with state/best-solution tracking.
- Residual: Residual normalization strategies (absolute, matrix-relative, hybrid, fixed-column, dominant-balance).
- Threshold: Model-selection criteria (jump, information AIC/BIC, multiplicative, pareto, fixed term-count).
- RegressionResult: Chosen model plus per-k histories (xis/lambdas), optional test errors, and sublibrary names.
- sparse_reg_bf(...): End-to-end pipeline—preprocess, single-term check, initialize, iterate, select model, hybrid normalization, postprocess.
- evaluate_model(), hybrid_residual(): Residual of a fixed model; compute hybrid residual and optionally return scaled coefficients.

### continuous/library.py
- generate_terms_to(...): Enumerate continuous terms (derivatives of `Observable`s under bounds), label/canonicalize.

### continuous/process_library_terms.py.SRDataset
- make_domains(): Create domains over a fixed grid.
- eval_prime(): Slice continuous fields and apply finite differences for derivative orders.
- make_libraries(), find_scales(), get_char_size(): Continuous variants using field statistics and length/time scaling.

### discrete/library.py
- CoarseGrainedProduct: Represents ρ[·] applied to products of `Observable`s; enables discrete term construction.
- generate_terms_to(...): Build discrete primes with coarse-graining plus derivative orders; enumerate terms and label.

### discrete/process_library_terms.py.SRDataset
- __post_init__(): Set spacings and kernel scales from physical units and coarse-graining resolution.
- make_libraries(): Partition terms by requested irreps, including anti/STF filters for rank-2.
- make_domains(): Spatial padding and optional time padding on the scaled grid.
- eval_prime(): Coarse grain particle data to a grid using polynomial kernels (KDTree/numba) or periodic variants; optional temporal smoothing; apply derivatives; rescale by ρ.
- make_Q_parallel(): Parallel domain evaluation and collection of per-domain ρ statistics for scaling.
- find_scales(), get_char_size(): Use dataset statistics (including ρ domain std) to compute characteristic sizes (length/time scaled).

### discrete/coarse_grain_utils.py
- gaussian_coarse_grain2d / kd_gaussian_coarse_grain2d: Coarse-grained estimates using Gaussian kernels; KDTree variant restricts to nearby points for speed.
- periodic_* variants: Periodic boundary handling via image replication with the same kernels.
