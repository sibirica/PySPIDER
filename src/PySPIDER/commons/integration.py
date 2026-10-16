# Interfacing quadrature schemes with PySPIDER

import numpy as np

from .quadrature_schemes import (
    chebyshev_gauss_weights,
    clenshaw_curtis_weights,
    mapped_chebyshev_gauss_nodes,
    mapped_chebyshev_nodes,
    moment_matched_quad_weights,
    truncated_chebyshev_gauss_nodes,
    truncated_chebyshev_nodes,
    _ensure_weight_ready,
    _mapped_gauss_reference_degree,
    _mapped_lobatto_reference_degree,
)

SUPPORTED_SCHEMES = (
    "trapezoidal",
    "truncated-cc-grid",
    "truncated-cg-grid",
    "clenshaw-curtis",
    "chebyshev-gauss",
    "moment-matching",
)


def _interval(opts, default=None):
    spec = opts.get("interval", default)
    return float(spec[0]), float(spec[1])


def _nodes(opts):
    spec = opts.get("nodes")
    if spec is None:
        return None
    return np.asarray(spec, dtype=float)


def _dot(w, arr):
    return np.tensordot(w, arr, axes=(0, 0))


def _prepare_weight(weight, ndim):
    if weight is None:
        return None
    _ensure_weight_ready(weight)
    if len(weight.weight_objs) != ndim:
        raise ValueError(
            f"weight has {len(weight.weight_objs)} axes but array has ndim={ndim}"
        )
    return weight


def _axis_scale(weight, axis):
    if weight is None:
        return 1.0
    return weight.scale if axis == 0 else 1.0


def _trapezoidal(arr, opts, weight=None, axis=0, domain_local_weight=False):
    if weight is not None:
        n = arr.shape[0]
        w = np.asarray(weight.weight_objs[axis].linspace(n)[1], dtype=float)
        arr = arr * w.reshape((n,) + (1,) * (arr.ndim - 1))
    return _axis_scale(weight, axis) * np.trapezoid(arr, axis=0)


def _moment_weights(nodes, a, b, weight, axis, domain_local_weight, *, degree=None):
    """Optionally map a domain-local weight to the quadrature interval.

    Weight derivatives already include the physical half-width in Weight.dxs.
    Shifting and rescaling the nodes from [a, b] to [-1, 1] makes the weight
    polynomial act on the local coordinate s without introducing another
    derivative factor. Multiplying the resulting quadrature weights by the
    interval half-width accounts for dx = H*ds. With domain_local_weight=False,
    the polynomial instead acts on the supplied coordinate x as W(x).
    """
    if domain_local_weight:
        if b <= a:
            raise ValueError("domain-local weights require an interval with a < b")
        half_width = (b - a) / 2
        local_nodes = (nodes - (a + b) / 2) / half_width
        return half_width * moment_matched_quad_weights(
            local_nodes, -1.0, 1.0, weight=weight, axis=axis, degree=degree
        )
    return moment_matched_quad_weights(nodes, a, b, weight=weight, axis=axis, degree=degree)


def _truncated_cc_nodes(n, opts):
    unsupported = set(opts) - {"nodes", "interval", "num_nodes", "degree"}
    if unsupported:
        raise ValueError(f"truncated-cc-grid has unsupported options: {sorted(unsupported)}")
    nodes = _nodes(opts)
    a, b = _interval(opts, (-1.0, 1.0))
    if nodes is None:
        n_parent = opts.get("num_nodes")
        if n_parent is None:
            if not (np.isclose(a, -1.0) and np.isclose(b, 1.0)):
                raise ValueError(
                    f"truncated-cc-grid on [{a}, {b}] needs 'nodes' or 'num_nodes'"
                )
            n_parent = n
        nodes = truncated_chebyshev_nodes(int(n_parent), a, b)
    if nodes.shape[0] != n:
        raise ValueError(
            f"array length {n} does not match {nodes.shape[0]} Chebyshev nodes"
        )
    return nodes, (a, b)


def _truncated_cc(arr, opts, weight=None, axis=0, domain_local_weight=False):
    nodes, (a, b) = _truncated_cc_nodes(arr.shape[0], opts)
    qw = _moment_weights(nodes, a, b, weight, axis, domain_local_weight,
                         degree=opts.get("degree"))
    return _dot(_axis_scale(weight, axis) * qw, arr)


def _truncated_cg_nodes(n, opts):
    nodes = _nodes(opts)
    a, b = _interval(opts, (-1.0, 1.0))
    if nodes is None:
        n_parent = opts.get("num_nodes")
        if n_parent is None:
            if not (np.isclose(a, -1.0) and np.isclose(b, 1.0)):
                raise ValueError(
                    f"truncated-cg-grid on [{a}, {b}] needs 'nodes' or 'num_nodes'"
                )
            n_parent = n
        nodes = truncated_chebyshev_gauss_nodes(int(n_parent), a, b)
    if nodes.shape[0] != n:
        raise ValueError(
            f"array length {n} does not match {nodes.shape[0]} Chebyshev-Gauss nodes"
        )
    return nodes, (a, b)


def _truncated_cg(arr, opts, weight=None, axis=0, domain_local_weight=False):
    nodes, (a, b) = _truncated_cg_nodes(arr.shape[0], opts)
    qw = _moment_weights(nodes, a, b, weight, axis, domain_local_weight,
                         degree=opts.get("degree"))
    return _dot(_axis_scale(weight, axis) * qw, arr)


def _clenshaw_curtis(arr, opts, weight=None, axis=0, domain_local_weight=False):
    n = arr.shape[0]
    a, b = _interval(opts, (-1.0, 1.0))
    nodes = _nodes(opts)
    if nodes is None:
        nodes = mapped_chebyshev_nodes(n, a, b)
    elif _mapped_lobatto_reference_degree(nodes, a, b) is None:
        raise ValueError(
            f"clenshaw-curtis needs the full mapped Lobatto grid on [{a}, {b}]"
        )
    if nodes.shape[0] != n:
        raise ValueError(
            f"array length {n} does not match {nodes.shape[0]} Lobatto nodes"
        )
    qw = clenshaw_curtis_weights(n, a, b, weight=weight, axis=axis)
    # Weights already match mapped_chebyshev_nodes. Flip only if the
    # caller passed descending Lobatto nodes.
    expected_asc = mapped_chebyshev_nodes(n, a, b)
    if np.allclose(nodes, expected_asc[::-1], rtol=0, atol=1e-12):
        qw = qw[::-1]
    return _dot(_axis_scale(weight, axis) * qw, arr)


def _chebyshev_gauss(arr, opts, weight=None, axis=0, domain_local_weight=False):
    n = arr.shape[0]
    a, b = _interval(opts, (-1.0, 1.0))
    nodes = _nodes(opts)
    if nodes is None:
        nodes = mapped_chebyshev_gauss_nodes(n, a, b)
    elif _mapped_gauss_reference_degree(nodes, a, b) is None:
        raise ValueError(
            f"chebyshev-gauss needs the full mapped Gauss grid on [{a}, {b}]"
        )
    if nodes.shape[0] != n:
        raise ValueError(
            f"array length {n} does not match {nodes.shape[0]} Gauss nodes"
        )
    qw = chebyshev_gauss_weights(nodes.shape[0], a, b, weight=weight, axis=axis)
    # Weights already match mapped_chebyshev_gauss_nodes. Flip only if the
    # caller passed descending Gauss nodes.
    expected_asc = mapped_chebyshev_gauss_nodes(nodes.shape[0], a, b)
    if np.allclose(nodes, expected_asc[::-1], rtol=0, atol=1e-12):
        qw = qw[::-1]
    return _dot(_axis_scale(weight, axis) * qw, arr)


def _moment_matching(arr, opts, weight=None, axis=0, domain_local_weight=False):
    nodes = _nodes(opts)
    if nodes is None:
        raise ValueError(
            "moment-matching requires 'nodes' and 'interval' in scheme options; "
            "for subdomain integration use truncated-cc-grid or truncated-cg-grid "
            "(infers nodes from array length) or pass nodes sliced to the subdomain."
        )
    n = arr.shape[0]
    if nodes.shape[0] != n:
        raise ValueError(
            f"moment-matching: array length {n} along the active axis does not "
            f"match {nodes.shape[0]} supplied nodes. Pass nodes sliced to the "
            f"integration subdomain, or use truncated-cc-grid / truncated-cg-grid / "
            f"clenshaw-curtis / chebyshev-gauss (which infer the node set from the "
            f"array length)."
        )
    a, b = _interval(opts)
    qw = _moment_weights(nodes, a, b, weight, axis, domain_local_weight,
                         degree=opts.get("degree"))
    return _dot(_axis_scale(weight, axis) * qw, arr)


_HANDLERS = {
    "truncated-cc-grid": _truncated_cc,
    "truncated-cg-grid": _truncated_cg,
    "clenshaw-curtis": _clenshaw_curtis,
    "chebyshev-gauss": _chebyshev_gauss,
    "moment-matching": _moment_matching,
    "trapezoidal": _trapezoidal,
}


def _quadrature_nodes_and_interval(name, n, opts):
    """Resolve node coordinates for a dataset's physical weight mapping."""
    if name == "truncated-cc-grid":
        return _truncated_cc_nodes(n, opts)
    if name == "truncated-cg-grid":
        return _truncated_cg_nodes(n, opts)
    a, b = _interval(opts, (-1.0, 1.0))
    nodes = _nodes(opts)
    if nodes is None:
        if name == "clenshaw-curtis":
            nodes = mapped_chebyshev_nodes(n, a, b)
        elif name == "chebyshev-gauss":
            nodes = mapped_chebyshev_gauss_nodes(n, a, b)
        else:
            raise ValueError(f"{name} requires explicit nodes")
    if nodes.ndim != 1 or len(nodes) != n:
        raise ValueError(f"array length {n} does not match supplied nodes")
    if name == "clenshaw-curtis" and _mapped_lobatto_reference_degree(nodes, a, b) is None:
        raise ValueError(f"clenshaw-curtis needs the full mapped Lobatto grid on [{a}, {b}]")
    if name == "chebyshev-gauss" and _mapped_gauss_reference_degree(nodes, a, b) is None:
        raise ValueError(f"chebyshev-gauss needs the full mapped Gauss grid on [{a}, {b}]")
    return nodes, (a, b)


def _scheme_and_opts(entry, axis):
    if isinstance(entry, str):
        return entry, {}
    if not isinstance(entry, dict) or len(entry) != 1:
        raise ValueError(
            f"schemes_and_options[{axis}] must be a scheme name or a dict "
            f"with exactly one scheme name, got {entry!r}"
        )
    (name, opts), = entry.items()
    if opts is None:
        opts = {}
    elif not isinstance(opts, dict):
        raise ValueError(
            f"options for {name!r} on axis {axis} must be a dict, "
            f"got {type(opts).__name__}"
        )
    return name, opts


def _is_legacy_dxs(arg) -> bool:
    """True for the pre-scheme second argument: a sequence of numeric spacings."""
    if arg is None or isinstance(arg, (dict, str, bytes)):
        return False
    try:
        seq = list(arg)
    except TypeError:
        return False
    if not seq:
        return False
    return all(isinstance(x, (int, float, np.integer, np.floating)) for x in seq)


def _int_arr_legacy(arr, dxs):
    """Original trapezoidal path used by older scripts: ``int_arr(arr, dxs)``."""
    if dxs is None:
        dxs = [1] * arr.ndim
    integral = np.trapezoid(arr, axis=0)
    if len(dxs) == 1:
        return integral
    return _int_arr_legacy(integral, dxs[1:])


def int_arr(arr, schemes_and_options=None, weight=None, *, domain_local_weight=False, dxs=None):
    """Integrate sampled data along each axis using the selected quadrature rules.

    Call ``int_arr(values, schemes, weight=weight)`` to select a rule per
    axis. Older calls of the form ``int_arr(weighted_values, dxs=dxs)``
    are also accepted. They assume a uniform spacing of array entries, as used
    in the ``trapezoidal`` scheme.

    Parameters
    ----------
    arr : array_like
        Sampled function values. Each axis must correspond to the nodes of
        its selected rule, in the same order. Selecting a rule does not
        resample the data. When passing ``weight=``, supply values that have
        not already been multiplied by that weight.
    schemes_and_options : dict, optional
        Map original array-axis indices to scheme names or one-entry dicts
        ``{scheme_name: options_dict}``. For example, ``{0: "clenshaw-curtis"}``
        selects Clenshaw-Curtis on axis 0 with its default interval [-1, 1].
        ``{0: {"clenshaw-curtis": {"interval": [2, 6]}}}`` selects a full
        Lobatto grid mapped onto [2, 6]. Omitted axes use trapezoidal;
        omitting this argument uses trapezoidal on every axis. The same
        format is used by ``AbstractDataset.schemes_and_options``.

        A numeric sequence in this position instead selects the older
        calling convention: ``int_arr(arr, [dx, dy])`` is equivalent to
        ``int_arr(arr, dxs=[dx, dy])``.
    weight : Weight, optional
        Separable SPIDER polynomial weight, with one polynomial per array
        axis. The function applies these polynomials and ``weight.scale``
        (once overall). Do not also multiply ``arr`` by the weight.
        Omitting ``weight`` or passing None integrates the supplied values
        without applying an additional weight function.
        ``Weight.dxs`` stores physical support half-widths used to scale
        derivatives of the weight. For the mapping from [-1, 1] onto a
        physical interval [a, b], the half-width H = (b-a)/2 is also the
        integration Jacobian: dx = H*ds. The quadrature routines compute
        this factor from ``interval``, not from ``Weight.dxs``. The values
        should agree when both describe the same physical interval.
        ``Weight.dxs`` does not specify sample spacing and is distinct from
        the ``dxs`` argument below. The trapezoidal scheme does not apply
        this integration Jacobian.
        Do not pass ``weight=`` when calling ``int_arr(arr, dxs)`` or
        ``int_arr(arr, dxs=dxs)``. For those calls, multiply the weight into
        ``arr`` beforehand.
    domain_local_weight : bool, optional
        For truncated grids and moment matching, choose which coordinate
        is the argument of each weight polynomial W. With the default
        False, integrate W(x)*f(x) using the supplied node coordinates x.
        With True, use W(s)*f(x), where
        ``s = (x - (a+b)/2) / H`` and ``H = (b-a)/2``. Thus, on [2, 6],
        W(s) means W((x-4)/2), not W(x). The interval must have a < b.
        The nodes are shifted and rescaled from [a, b] to [-1, 1] internally,
        and the quadrature weights are multiplied by H to account for dx = H*ds.
        ``Weight.dxs`` must already contain the physical half-widths for
        correct derivative scaling; setting ``domain_local_weight=True``
        does not update ``Weight.dxs``.

        Full Clenshaw-Curtis and Gauss rules always use W(s), regardless
        of ``domain_local_weight``. The trapezoidal scheme always samples W
        at equally spaced reference coordinates s in [-1, 1] and also ignores
        ``domain_local_weight``. SRDataset passes ``domain_local_weight=True``
        to map its boundary-vanishing weights onto each integration subdomain.
    dxs : sequence of numbers, optional
        Select the older trapezoidal calling convention. Supply one entry
        per array axis: its length controls how many axes are reduced.
        The numeric spacing values are ignored; the calculation uses a spacing
        of 1 between consecutive array entries. For physical-coordinate
        integration on a uniform grid, multiply the result by the product
        of the sample spacings.
        Any weight must already be multiplied into ``arr``. Cannot be combined
        with ``schemes_and_options`` or ``weight``.

    Returns
    -------
    scalar or zero-dimensional ndarray
        Integral after reducing every axis. For the older calling convention,
        this assumes the dxs sequence has one entry per array axis.

    Notes
    -----
    Available schemes and their options:

    - ``trapezoidal``: no options. Uses a spacing of 1 between consecutive
      array entries, ignoring their actual physical spacing, even in the
      scheme-dict calling convention. Does not use supplied coordinates or
      multiply by physical spacings.
    - ``clenshaw-curtis``: Clenshaw-Curtis weights on a full Lobatto grid,
      including both endpoints. Optional ``interval`` defaults to [-1, 1].
      Optional ``nodes`` must match the full Lobatto grid mapped onto that
      interval, in ascending or descending order, with one node per sample.
      Without ``nodes``, the function generates that grid using the interval
      and the sample count along the axis; the data must already be sampled
      there. Includes the integration Jacobian (b-a)/2. With ``weight=``,
      integrates W(s)*f(x), where s maps the interval onto [-1, 1].
    - ``chebyshev-gauss``: Fejer-I weights on a full Chebyshev-Gauss grid,
      excluding both endpoints. Optional ``interval`` defaults to [-1, 1].
      Optional ``nodes`` must match the full Gauss grid mapped onto that
      interval, in ascending or descending order, with one node per sample.
      Without ``nodes``, the function generates that grid using the interval
      and the sample count along the axis; the data must already be sampled
      there. Includes the integration Jacobian (b-a)/2. With ``weight=``,
      integrates W(s)*f(x), where s maps the interval onto [-1, 1].
    - ``truncated-cc-grid``: optional ``interval`` (default [-1, 1]) and
      either explicit retained ``nodes`` or ``num_nodes``, the total node
      count of the parent Lobatto grid on [-1, 1]. With ``num_nodes``,
      retains the parent nodes within ``interval`` rather than mapping a
      new full grid onto it. On [-1, 1], omitting both ``nodes`` and
      ``num_nodes`` defaults the parent count to the sample count along
      that axis, assuming a full Lobatto grid. For a smaller interval,
      supply ``nodes`` or the parent ``num_nodes``. The retained node count
      must match the samples; coordinates are not inferred from data values.
      Optional ``degree`` selects the polynomial degree to match; by default,
      it is the retained sample count minus one.
    - ``truncated-cg-grid``: optional ``interval`` (default [-1, 1]) and
      either explicit retained ``nodes`` or ``num_nodes``, the total node
      count of the parent Gauss grid on [-1, 1]. With ``num_nodes``,
      retains the parent nodes within ``interval`` rather than mapping a
      new full grid onto it. On [-1, 1], omitting both ``nodes`` and
      ``num_nodes`` defaults the parent count to the sample count along
      that axis, assuming a full Gauss grid. For a smaller interval,
      supply ``nodes`` or the parent ``num_nodes``. The retained node count
      must match the samples; coordinates are not inferred from data values.
      Optional ``degree`` selects the polynomial degree to match; by default,
      it is the retained sample count minus one.
    - ``moment-matching``: requires explicit ``nodes`` and ``interval``.
      Uses the supplied coordinates directly unless domain_local_weight
      is True. Optional ``degree`` selects the polynomial degree to match;
      omitting it or passing None uses the sample count minus one. The option
      is forwarded to ``moment_matched_quad_weights`` in either coordinate
      mode. For example, degree=2 asks the solver to reproduce the integrals
      of 1, x, and x**2 (or 1, s, and s**2 in the local coordinate mode).
      A lower degree can make the calculation less sensitive to rounding
      errors, but no longer asks it to reproduce higher-power polynomial
      integrals. Check accuracy for the chosen nodes and degree.

    Every handler reduces the current leading array axis. Scheme keys and
    weight-axis indices refer to the original array axes: after axis 0 is
    removed, original axis 1 is the new leading axis, and so on.

    Examples
    --------
    Select Clenshaw-Curtis to integrate f(x) = x**2 on [0, 2]. The
    interval and five samples determine the full Lobatto nodes:

    >>> nodes = mapped_chebyshev_nodes(5, 0.0, 2.0)
    >>> values = nodes**2
    >>> schemes = {0: {"clenshaw-curtis": {"interval": [0.0, 2.0]}}}
    >>> result = int_arr(values, schemes)
    >>> print(f"{result:.6f}")
    2.666667

    Older trapezoidal calls ignore the spacing values. For five constant
    samples spaced by 0.5, using a spacing of 1 between array entries gives
    an integral of 4; multiply by 0.5 separately to obtain the physical
    integral over an interval of length 2:

    >>> values = np.ones(5)
    >>> result = int_arr(values, dxs=[0.5])
    >>> print(f"{result:.6f}")
    4.000000
    >>> print(f"{0.5 * result:.6f}")
    2.000000

    Integrate samples retained from a nine-node parent Lobatto grid.
    Only three parent nodes lie in [-0.5, 0.5]; ``num_nodes`` describes
    the parent grid, not the retained sample count:

    >>> parent = mapped_chebyshev_nodes(9, -1.0, 1.0)
    >>> np.round(parent, 4)
    array([-1.    , -0.9239, -0.7071, -0.3827,  0.    ,  0.3827,  0.7071,
            0.9239,  1.    ])
    >>> nodes = truncated_chebyshev_nodes(9, -0.5, 0.5)
    >>> np.round(nodes, 4)
    array([-0.3827,  0.    ,  0.3827])
    >>> schemes = {0: {"truncated-cc-grid": {
    ...     "interval": [-0.5, 0.5], "num_nodes": 9,
    ... }}}
    >>> result = int_arr(nodes**2, schemes)
    >>> print(f"{result:.6f}")
    0.083333

    Alternatively, supply the retained nodes directly. This uses the same
    samples and gives the same result without specifying the parent count:

    >>> schemes = {0: {"truncated-cc-grid": {
    ...     "interval": [-0.5, 0.5], "nodes": nodes,
    ... }}}
    >>> result = int_arr(nodes**2, schemes)
    >>> print(f"{result:.6f}")
    0.083333

    Compare the two weight-coordinate choices on [2, 6], using
    W(z) = (z**2 - 1)**2 and f(x) = 1. The default integrates W(x),
    whereas domain_local_weight=True integrates W((x-4)/2):

    >>> from PySPIDER.commons.process_library_terms import Weight
    >>> nodes = np.linspace(2.0, 6.0, 5)
    >>> values = np.ones_like(nodes)
    >>> weight = Weight(m=[2], q=[0], k=[0], dxs=[2.0])
    >>> schemes = {0: {"moment-matching": {"nodes": nodes, "interval": [2.0, 6.0]}}}
    >>> result = int_arr(values, schemes, weight=weight)
    >>> print(f"{result:.6f}")
    1414.133333
    >>> result = int_arr(values, schemes, weight=weight, domain_local_weight=True)
    >>> print(f"{result:.6f}")
    2.133333

    Select a degree explicitly when using moment matching. Here four
    irregular nodes match polynomials through degree two, rather than
    using the default degree three:

    >>> nodes = np.array([-1.0, -0.4, 0.2, 1.0])
    >>> schemes = {0: {"moment-matching": {
    ...     "nodes": nodes, "interval": [-1.0, 1.0], "degree": 2,
    ... }}}
    >>> result = int_arr(nodes**2, schemes)
    >>> print(f"{result:.6f}")
    0.666667

    Here is a possible, but niche use case:

    Suppose a three-dimensional field is sampled on Lobatto nodes along x,
    Gauss nodes along y, and irregular nodes along z. Use Clenshaw-Curtis,
    Fejer-I, and moment matching on axes 0, 1, and 2, respectively. The
    sample array has shape (len(x), len(y), len(z)); each scheme key refers
    to that original axis. Integrate f(x, y, z) = x**2 * y**2 * (1 + z)
    over [0, 2] x [-1, 1] x [0, 1]:

    >>> x = mapped_chebyshev_nodes(5, 0.0, 2.0)
    >>> y = mapped_chebyshev_gauss_nodes(4, -1.0, 1.0)
    >>> z = np.array([0.0, 0.2, 0.7, 1.0])
    >>> values = x[:, None, None]**2 * y[None, :, None]**2 * (1 + z[None, None, :])
    >>> schemes = {
    ...     0: {"clenshaw-curtis": {"interval": [0.0, 2.0]}},
    ...     1: {"chebyshev-gauss": {"interval": [-1.0, 1.0]}},
    ...     2: {"moment-matching": {"nodes": z, "interval": [0.0, 1.0]}},
    ... }
    >>> result = int_arr(values, schemes)
    >>> print(f"{result:.6f}")
    2.666667
    """
    if dxs is not None:
        if schemes_and_options is not None:
            raise ValueError("pass either dxs or schemes_and_options, not both")
        if not _is_legacy_dxs(dxs):
            raise ValueError("dxs must be a sequence of numeric spacings")
        schemes_and_options = dxs

    if _is_legacy_dxs(schemes_and_options):
        if weight is not None:
            raise ValueError(
                "legacy int_arr(arr, dxs) does not take weight=; multiply the "
                "weight into arr first, or use schemes_and_options with weight="
            )
        return _int_arr_legacy(arr, schemes_and_options)

    arr = np.asarray(arr, dtype=float)
    if schemes_and_options is None:
        spec = {}
    elif not isinstance(schemes_and_options, dict):
        raise ValueError(
            "schemes_and_options must be a dict mapping axes to "
            "a scheme name or {scheme: options}, or a legacy dxs list"
        )
    else:
        spec = schemes_and_options
    ndim = arr.ndim
    weight = _prepare_weight(weight, ndim)
    for axis in spec:
        if not isinstance(axis, (int, np.integer)):
            raise ValueError(f"axis keys must be ints, got {axis!r}")
        if not 0 <= int(axis) < ndim:
            raise ValueError(f"axis {axis} out of range for ndim={ndim}")
    for i in range(ndim):
        if i not in spec:
            name, opts = "trapezoidal", {}
        else:
            name, opts = _scheme_and_opts(spec[i], i)
        handler = _HANDLERS.get(name)
        if handler is None:
            raise ValueError(
                f"Unsupported scheme {name!r}. Supported: {', '.join(SUPPORTED_SCHEMES)}."
            )
        arr = handler(
            arr, opts, weight=weight, axis=i, domain_local_weight=domain_local_weight
        )
    return arr
