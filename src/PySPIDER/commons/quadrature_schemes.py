# All implemented quadrature schemes

from typing import Optional

import numpy as np
from numpy.polynomial.chebyshev import (
    chebint, chebpts1, chebpts2, chebval, chebvander, poly2cheb,
)
from scipy.fft import dct


#helper function used quite a bit
def _validate_interval(a: float, b: float) -> None:
    if a > b:
        raise ValueError(f"interval endpoints must satisfy a <= b, got ({a}, {b})")

def mapped_chebyshev_nodes(n: int, a: float, b: float) -> np.ndarray:
    """
    Affinely map the Chebyshev-Lobatto nodes from [-1, 1] onto [a, b].

    Uses `numpy.polynomial.chebyshev.chebpts2` for the reference nodes, then
    applies x -> (a+b)/2 + (b-a)/2 * x and returns them sorted ascending.

    Parameters
    ----------
    n : int
        Number of nodes to be produced.
    a : float
        Left endpoint of the target interval.
    b : float
        Right endpoint of the target interval.

    Returns
    -------
    np.ndarray
        Array of n nodes mapped onto [a, b].

    Examples
    --------

    >>> reference_nodes = mapped_chebyshev_nodes(5, -1.0, 1.0)
    >>> np.round(reference_nodes, 4)
    array([-1.    , -0.7071,  0.    ,  0.7071,  1.    ])
    >>> nodes = mapped_chebyshev_nodes(4, 2.0, 6.0)
    >>> np.round(nodes, 4)
    array([2., 3., 5., 6.])

    """
    _validate_interval(a, b)
    if n < 1:
        raise ValueError("n must be >= 1")
    return (0.5 * (a + b) + 0.5 * (b - a) * chebpts2(n))

def mapped_chebyshev_gauss_nodes(n: int, a: float, b: float) -> np.ndarray:
    """
    Affinely map the Chebyshev-Gauss nodes from [-1, 1] onto [a, b].

    These are the roots of T_n (first-kind / interior nodes). Uses
    `numpy.polynomial.chebyshev.chebpts1` for the reference nodes, then
    applies x -> (a+b)/2 + (b-a)/2 * x and returns them sorted ascending.
    Unlike Lobatto, the endpoints ±1 are not included.

    Parameters
    ----------
    n : int
        Number of nodes to be produced.
    a : float
        Left endpoint of the target interval.
    b : float
        Right endpoint of the target interval.

    Returns
    -------
    np.ndarray
        Array of n nodes mapped onto [a, b].

    Examples
    --------

    >>> nodes = mapped_chebyshev_gauss_nodes(5, 2.0, 6.0)
    >>> np.round(nodes, 4)
    array([2.0979, 2.8244, 4.    , 5.1756, 5.9021])
    >>> reference_nodes = mapped_chebyshev_gauss_nodes(4, -1.0, 1.0)
    >>> np.round(reference_nodes, 4)
    array([-0.9239, -0.3827,  0.3827,  0.9239])

    """
    _validate_interval(a, b)
    if n < 1:
        raise ValueError("n must be >= 1")
    return (0.5 * (a + b) + 0.5 * (b - a) * chebpts1(n))

def truncated_chebyshev_nodes(n: int, a: float, b: float) -> np.ndarray:
    """
    Return the Chebyshev-Lobatto nodes that fall within [a, b].

    The full reference grid is generated with
    `numpy.polynomial.chebyshev.chebpts2` and only the subset lying in
    [a, b] (within a floating point tolerance) is retained, sorted ascending.

    Parameters
    ----------
    n : int
        Number of nodes of the underlying reference grid.
    a : float
        Left endpoint of the retained interval.
    b : float
        Right endpoint of the retained interval.

    Returns
    -------
    np.ndarray
        Array of the reference nodes contained in [a, b].

    Examples
    --------

    >>> parent = mapped_chebyshev_nodes(9, -1.0, 1.0)
    >>> np.round(parent, 4)
    array([-1.    , -0.9239, -0.7071, -0.3827,  0.    ,  0.3827,  0.7071,
            0.9239,  1.    ])
    >>> nodes = truncated_chebyshev_nodes(9, -0.5, 0.5)
    >>> np.round(nodes, 4)
    array([-0.3827,  0.    ,  0.3827])

    """
    _validate_interval(a, b)
    if n < 1:
        raise ValueError("n must be >= 1")
    nodes = chebpts2(n)
    tol = np.finfo(float).eps * max(abs(a), abs(b), 1.0)
    mask = (nodes >= a - tol) & (nodes <= b + tol)
    kept = nodes[mask]
    if kept.size == 0:
        raise ValueError(
            f"No Chebyshev-Lobatto nodes from reference grid n = {n} lie in [{a}, {b}]. "
            "Increase num_nodes or widen the interval."
        )
    return kept

def truncated_chebyshev_gauss_nodes(n: int, a: float, b: float) -> np.ndarray:
    """
    Return the Chebyshev-Gauss nodes that fall within [a, b].

    The full reference grid is generated with
    `numpy.polynomial.chebyshev.chebpts1` (the first-kind / interior-node
    grid on [-1, 1]) and only the subset lying in [a, b] (within a
    floating point tolerance) is retained, sorted ascending.

    Parameters
    ----------
    n : int
        Number of nodes of the underlying reference grid.
    a : float
        Left endpoint of the retained interval.
    b : float
        Right endpoint of the retained interval.

    Returns
    -------
    np.ndarray
        Array of the reference Gauss nodes contained in [a, b].

    Examples
    --------

    >>> parent = mapped_chebyshev_gauss_nodes(8, -1.0, 1.0)
    >>> np.round(parent, 4)
    array([-0.9808, -0.8315, -0.5556, -0.1951,  0.1951,  0.5556,  0.8315,
            0.9808])
    >>> nodes = truncated_chebyshev_gauss_nodes(8, -0.3, 0.3)
    >>> np.round(nodes, 4)
    array([-0.1951,  0.1951])

    """
    _validate_interval(a, b)
    if n < 1:
        raise ValueError("n must be >= 1")
    nodes = chebpts1(n)
    tol = np.finfo(float).eps * max(abs(a), abs(b), 1.0)
    mask = (nodes >= a - tol) & (nodes <= b + tol)
    kept = nodes[mask]
    if kept.size == 0:
        raise ValueError(
            f"No Chebyshev-Gauss nodes from reference grid n = {n} lie in [{a}, {b}]. "
            "Increase num_nodes or widen the interval."
        )
    return kept

def _clenshaw_curtis_from_moments(mu: np.ndarray) -> np.ndarray:
    """
    Compute Clenshaw-Curtis quadrature weights using DCT-I.

    Given moments of the form mu_k = integral_-1^1 w(x) T_k(x) dx, calculates
    quadrature weights using scipy's DCT-I.

    Parameters
    ----------
    mu : np.ndarray
        1D Array where the kth entry equals integral_-1^1 w(x) T_k(x) dx

    Returns
    -------
    np.ndarray
        Clenshaw-Curtis quadrature weights A_0, A_1,..., A_{len(mu)-1} sorted according to node ascending order.
    """
    N = mu.shape[0] - 1
    if N < 1:
        raise ValueError("At least two moments (N>=1) are required")
    a = dct(np.asarray(mu, dtype=float), type=1) / N
    a[0] *= 0.5
    a[-1] *= 0.5
    return a[::-1]

def _ensure_weight_ready(weight) -> None:
    if not getattr(weight, "ready", False) or weight.weight_objs is None:
        weight.make_weight_objs()

def _polynomial_weighted_moments(poly, a: float, b: float, max_degree: int) -> np.ndarray:
    """
    Exact Chebyshev moments of a polynomial weight.

    Returns mu_j = integral_a^b W(x) T_j(x) dx. W is expanded as a
    Chebyshev series and each product T_i T_j is integrated with
    ``_unweighted_moments``.
    """
    if max_degree < 0:
        raise ValueError("max_degree must be >= 0")
    coef = np.asarray(poly.coef, dtype=float).ravel()
    if coef.size == 0 or not np.any(coef):
        return np.zeros(max_degree + 1)
    a_w = poly2cheb(coef)
    I = _unweighted_moments(a, b, a_w.shape[0] - 1 + max_degree)
    mu = np.zeros(max_degree + 1)
    js = np.arange(max_degree + 1)
    for i, ai in enumerate(a_w):
        if ai == 0.0:
            continue
        # T_i T_j = (T_{i+j} + T_{|i-j|}) / 2
        mu += 0.5 * ai * (I[i + js] + I[np.abs(i - js)])
    return mu

def _unweighted_moments(a: float, b: float, max_degree: int) -> np.ndarray:
    """
    Compute the unweighted Chebyshev moments analytically.

    Returns mu_k = integral_a^b T_k(x) dx by integrating the Chebyshev basis
    with `numpy.polynomial.chebyshev.chebint` (antiderivative vanishing at a)
    and evaluating at b.

    Parameters
    ----------
    a : float
        Left endpoint of integration.
    b : float
        Right endpoint of integration.
    max_degree : int
        Largest Chebyshev index k.

    Returns
    -------
    np.ndarray
        Array of moments mu_0, ..., mu_max_degree.
    """
    return chebval(b, chebint(np.eye(max_degree + 1), lbnd=a, axis=0))

#used in integration.py
def _mapped_lobatto_reference_degree(nodes: np.ndarray, a: float, b: float) -> Optional[int]:
    """
    Return N when nodes are a full Lobatto grid affinely mapped onto [a, b].

    Accepts ascending or descending node orderings.
    Returns None when the nodes do not match a full mapped Lobatto grid.
    """
    nodes = np.asarray(nodes, dtype=float)
    if nodes.size < 2:
        return None
    _validate_interval(a, b)
    N = nodes.shape[0] - 1
    expected = mapped_chebyshev_nodes(N + 1, a, b)
    if np.allclose(nodes, expected, rtol=0, atol=1e-12):
        return N
    if np.allclose(nodes, expected[::-1], rtol=0, atol=1e-12):
        return N
    return None

def _mapped_gauss_reference_degree(nodes: np.ndarray, a: float, b: float) -> Optional[int]:
    """
    Return n when nodes are a full Gauss grid affinely mapped onto [a, b].

    Accepts ascending or descending node orderings. n is the number of
    Gauss nodes, not the number of Lobatto intervals.
    Returns None when the nodes do not match a full mapped Gauss grid.
    """
    nodes = np.asarray(nodes, dtype=float)
    if nodes.size < 1:
        return None
    _validate_interval(a, b)
    n = nodes.shape[0]
    expected = mapped_chebyshev_gauss_nodes(n, a, b)
    if np.allclose(nodes, expected, rtol=0, atol=1e-12):
        return n
    if np.allclose(nodes, expected[::-1], rtol=0, atol=1e-12):
        return n
    return None

def _fejer_weights_from_moments(mu: np.ndarray) -> np.ndarray:
    """Chebyshev-Gauss (Fejér-I) node weights from Chebyshev moments via DCT-III.

    DCT-III emits weights for ``x_j = cos(pi (2j+1) / (2n))`` (near 1 ->
    near -1); reverse so they match ascending ``chebpts1`` /
    ``mapped_chebyshev_gauss_nodes``.
    """
    n = mu.shape[0]
    if n < 1:
        raise ValueError("At least one moment is required")
    return (dct(np.asarray(mu, dtype=float), type=3) / n)[::-1]

def clenshaw_curtis_weights(
    num_nodes: int,
    a: float = -1.0,
    b: float = 1.0,
    *,
    weight=None,
    axis: int = 0,
) -> np.ndarray:
    """
    Clenshaw-Curtis weights on the full Lobatto grid mapped to [a, b].

    Returns weights w_j such that sum_j w_j f(x_j) approximates

        integral_a^b w(s(x)) f(x) dx,

    where s maps [a, b] onto [-1, 1]. ``w`` is 1 or one axis of a
    PySPIDER ``Weight``. Uses reference Clenshaw-Curtis weights times
    the affine Jacobian (b-a)/2. Weights are aligned with ascending nodes.

    Parameters
    ----------
    num_nodes : int
        Number of Lobatto nodes; must be at least two.
    a, b : float, optional
        Interval endpoints; default [-1, 1].
    weight : optional
        PySPIDER Weight; moments use that object's attributes for ``axis``.
    axis : int, optional
        Axis of ``weight`` to use.

    Returns
    -------
    np.ndarray
        Array of num_nodes quadrature weights in ascending node order.

    Examples
    --------
    Integrate f(x) = x**2 on [0, 2] without a polynomial weight:

    >>> nodes = mapped_chebyshev_nodes(5, 0.0, 2.0)
    >>> values = nodes**2
    >>> weights = clenshaw_curtis_weights(5, 0.0, 2.0)
    >>> print(f"{np.dot(weights, values):.6f}")
    2.666667

    Include the polynomial weight W(s) = (s**2 - 1)**2, with
    s = x - 1 on [0, 2]. Pass raw function samples: the quadrature
    weights already include W, so do not multiply the samples by it.

    >>> from PySPIDER.commons.process_library_terms import Weight
    >>> weight = Weight(m=[2], q=[0], k=[0], dxs=[1.0])
    >>> weights = clenshaw_curtis_weights(5, 0.0, 2.0, weight=weight)
    >>> print(f"{np.dot(weights, np.ones_like(nodes)):.6f}")
    1.066667
    """
    _validate_interval(a, b)
    if num_nodes < 2:
        raise ValueError("num_nodes must be >= 2")
    mu = _compute_moments(-1.0, 1.0, num_nodes - 1, weight=weight, axis=axis)
    return 0.5 * (b - a) * _clenshaw_curtis_from_moments(mu)

def chebyshev_gauss_weights(
    num_nodes: int,
    a: float = -1.0,
    b: float = 1.0,
    *,
    weight=None,
    axis: int = 0,
) -> np.ndarray:
    """
    Fejér-I weights on the full Gauss grid mapped to [a, b].

    Returns weights w_j such that sum_j w_j f(x_j) approximates

        integral_a^b w(s(x)) f(x) dx,

    where s maps [a, b] onto [-1, 1] and the nodes are the roots of T_n
    (first-kind / interior Chebyshev-Gauss grid). ``w`` is 1 or one axis of a
    PySPIDER ``Weight``. Uses reference DCT-III (Fejér-I) weights times the
    affine Jacobian (b-a)/2. Weights are aligned with ascending nodes.

    Parameters
    ----------
    num_nodes : int
        Number of Gauss nodes n; the grid has n nodes.
    a, b : float, optional
        Interval endpoints; default [-1, 1].
    weight : optional
        PySPIDER Weight; moments use that object's attributes for ``axis``.
    axis : int, optional
        Axis of ``weight`` to use.

    Returns
    -------
    np.ndarray
        Array of n quadrature weights in ascending node order.

    Examples
    --------
    Integrate f(x) = x**2 on [0, 2] using four interior nodes:

    >>> nodes = mapped_chebyshev_gauss_nodes(4, 0.0, 2.0)
    >>> weights = chebyshev_gauss_weights(4, 0.0, 2.0)
    >>> print(f"{np.dot(weights, nodes**2):.6f}")
    2.666667
    """
    _validate_interval(a, b)
    n = num_nodes
    if n < 1:
        raise ValueError("num_nodes must be >= 1")
    mu = _compute_moments(-1.0, 1.0, n - 1, weight=weight, axis=axis)
    return 0.5 * (b - a) * _fejer_weights_from_moments(mu)

#moment matching and relevant helper functions
def _compute_moments(
    a: float,
    b: float,
    max_degree: int,
    *,
    weight=None,
    axis: int = 0,
) -> np.ndarray:
    """
    Compute Chebyshev moments for a chosen weight on [a, b].

    Returns mu_k = integral_a^b w(x) T_k(x) dx. A PySPIDER ``weight``
    is parsed for axis ``axis`` (the 1-D polynomial on that axis,
    including m, q, k, dx). With no weight, w(x) = 1. Both cases are
    evaluated from Chebyshev antiderivatives of the polynomial integrand.

    Parameters
    ----------
    a : float
        Left endpoint of integration.
    b : float
        Right endpoint of integration.
    max_degree : int
        Largest Chebyshev index k.
    weight : optional
        PySPIDER Weight; moments use that object's attributes for ``axis``.
    axis : int, optional
        Axis of ``weight`` to use.

    Returns
    -------
    np.ndarray
        Array of moments mu_0,..., mu_max_degree.
    """
    if weight is not None:
        _ensure_weight_ready(weight)
        return _polynomial_weighted_moments(
            weight.weight_objs[axis], a, b, max_degree
        )
    return _unweighted_moments(a, b, max_degree)

def moment_matched_quad_weights(
    nodes: np.ndarray,
    a: float,
    b: float,
    *,
    weight=None,
    axis: int = 0,
    degree: Optional[int] = None,
) -> np.ndarray:
    """
    Compute quadrature weights for arbitrary nodes by moment matching.

    Solves the (least-squares) moment-matching system V^T w = mu, where V is
    the Chebyshev basis matrix at nodes (`numpy.polynomial.chebyshev.chebvander`)
    and mu are the weighted moments on [a, b].
    
    Parameters
    ----------
    nodes : np.ndarray
        Quadrature nodes in [a, b].
    a : float
        Left endpoint of integration.
    b : float
        Right endpoint of integration.
    weight : optional
        PySPIDER Weight; moments use that object's attributes for ``axis``.
    axis : int, optional
        Axis of ``weight`` to use.
    degree : int, optional
        Polynomial degree to match; defaults to len(nodes) - 1.

    Returns
    -------
    np.ndarray
        Quadrature weights aligned with nodes.

    Notes
    -----
    Coordinates are used as supplied; this routine does not shift or rescale
    them. For a physical interval [a, b], callers should generally map the
    nodes onto [-1, 1] using s = (x - (a+b)/2) / H, where H = (b-a)/2.
    Pass these s coordinates and bounds [-1, 1] to this routine, then multiply
    the weighted sum of the function samples by H to account for dx = H*ds.
    Transform a physical weight consistently:
    W(x) becomes W((a+b)/2 + H*s), whereas a domain-local weight is W(s).

    Callers should select degree and check convergence for their data.

    Examples
    --------
    Match degree two on four irregular nodes. This deliberately uses
    a lower degree than len(nodes)-1:

    >>> nodes = np.array([-1.0, -0.4, 0.2, 1.0])
    >>> weights = moment_matched_quad_weights(nodes, -1.0, 1.0, degree=2)
    >>> print(f"{np.dot(weights, nodes**2):.6f}")
    0.666667

    For physical coordinates, shift and rescale the nodes onto [-1, 1],
    then multiply the weighted sum by the interval half-width.
    Here f(x) = x**2 is integrated on [2, 6]:

    >>> x = np.array([2.0, 2.8, 4.4, 6.0])
    >>> a, b = 2.0, 6.0
    >>> half_width = (b - a) / 2.0
    >>> s = (x - (a + b) / 2.0) / half_width
    >>> weights = moment_matched_quad_weights(s, -1.0, 1.0, degree=2)
    >>> result = half_width * np.dot(weights, x**2)
    >>> print(f"{result:.6f}")
    69.333333
    """
    nodes = np.asarray(nodes, dtype=float)
    _validate_interval(a, b)
    M = len(nodes)
    if M < 2:
        raise ValueError("moment_matched_quad_weights requires at least two nodes (degree >= 1).")

    if degree is None:
        degree = M - 1

    V = chebvander(nodes, degree)
    mu = _compute_moments(a, b, degree, weight=weight, axis=axis)
    return np.linalg.lstsq(V.T, mu, rcond=None)[0]
