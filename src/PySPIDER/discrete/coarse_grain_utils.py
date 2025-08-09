import numpy as np
from numba import jit, float64, uint64, prange, int64, uint8
from numba_kdtree import KDTree



@jit(
    signature_or_function="float64(float64, float64, float64)",
    nopython=True,
    cache=False,
    fastmath=True,
    parallel=False,
    debug=False,
    nogil=True,
    boundscheck=False
)
def periodic_distance_1d(x1: float64, x2: float64, box_length: float64) -> float64:
    """
    Calculate the minimum distance between two points in 1D with periodic boundaries.
    :param x1: first coordinate
    :param x2: second coordinate  
    :param box_length: length of the periodic box
    :return: minimum distance considering periodic boundaries
    """
    dx = x1 - x2
    # Apply periodic boundary conditions
    if dx > 0.5 * box_length:
        dx -= box_length
    elif dx < -0.5 * box_length:
        dx += box_length
    return dx


@jit(
    signature_or_function="float64(float64[:], float64[:], float64[:])",
    nopython=True,
    cache=False,
    fastmath=True,
    parallel=False,
    debug=False,
    nogil=True,
    boundscheck=False
)
def periodic_distance_squared_2d(p1: float64[:], p2: float64[:], box_lengths: float64[:]) -> float64:
    """
    Calculate the squared distance between two 2D points with periodic boundaries.
    :param p1: first point [x, y]
    :param p2: second point [x, y]
    :param box_lengths: box lengths [Lx, Ly]
    :return: squared distance considering periodic boundaries
    """
    dx = periodic_distance_1d(p1[0], p2[0], box_lengths[0])
    dy = periodic_distance_1d(p1[1], p2[1], box_lengths[1])
    return dx * dx + dy * dy


@jit(
    signature_or_function="void(float64[:, :], float64[:, :], float64[:])",
    nopython=True,
    cache=False,
    fastmath=True,
    parallel=False,
    debug=False,
    nogil=True,
    boundscheck=False
)
def create_periodic_images_2d(points: float64[:, :], images: float64[:, :], box_lengths: float64[:]):
    """
    Create periodic images of points for 2D system.
    For each original point, creates 8 periodic images (3x3 grid minus center).
    :param points: original points, shape (n, 2)
    :param images: output array for all images, shape (9*n, 2)
    :param box_lengths: box lengths [Lx, Ly]
    """
    n = points.shape[0]
    idx = 0
    
    for i in range(n):
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                images[idx, 0] = points[i, 0] + dx * box_lengths[0]
                images[idx, 1] = points[i, 1] + dy * box_lengths[1]
                idx += 1


@jit(
    signature_or_function=(
        "float64[:](float64[:, :], float64[:], float64[:, :], float64)"
    ),
    nopython=True,
    cache=False,
    fastmath=False,
    parallel=True,
    nogil=True)
def gaussian_coarse_grain2d(points: float64[:, :],
                            values: float64[:],
                            xi: float64[:, :],
                            sigma: float64) -> float64[:]:
    """
    This function implements a gaussian coarse graining algorithm. Heavily inspired 
    by the scipy implementation at 
    https://github.com/scipy/scipy/blob/main/scipy/stats/_stats.pyx
    :param points: the data points to estimate from in 2 dimensions. Shape (n, 2).
    :param values: the multivariate values associated with the data points. (n,)
    :param xi: the coordinates to evaluate the estimate at in 2 dimensions. Shape (m, 2).
    :param sigma: the gaussian kernel width (standard deviation). Float.
    :return: the coarse grained data at the coordinates xi. Shape (m,).
    """
    n: uint64 = points.shape[0]  # number of data points
    d: uint64 = 2  # dimension of the data points
    m: uint64 = xi.shape[0]  # number of evaluation points

    points_: float64[n, d] = points / sigma  # the scaled data points
    xi_: float64[m, d] = xi / sigma  # the scaled evaluation points

    estimate: float64[m] = np.zeros(m)  # the estimate at the evaluation points
    norm: float64 = 1 / (2 * np.pi) / (sigma * sigma)  # the normalization factor of the gaussian kernel

    # Fixed: parallelize over evaluation points to avoid race condition
    for j in prange(m):
        local_sum: float64 = 0.0
        for i in range(n):
            local_sum += (
                np.exp(-np.sum((points_[i, :] - xi_[j, :]) * 
                               (points_[i, :] - xi_[j, :])) / 2) * values[i]
            )
        estimate[j] = local_sum

    return estimate * norm


@jit(
    signature_or_function=(
        "float64[:](float64[:, :], float64[:], float64[:, :], float64, float64)"
    ),
    nopython=True,
    cache=False,
    fastmath=False,
    parallel=True,
    debug=False,
    nogil=True,
    boundscheck=True
)
def kd_gaussian_coarse_grain2d(points: float64[:, :],
                               values: float64[:],
                               xi: float64[:, :],
                               sigma: float64,
                               cutoff: float64) -> float64[:]:
    """
    This function implements a gaussian coarse graining algorithm. Uses a KDTree to 
    only consider nearby points. Heavily inspired by the scipy implementation at 
    https://github.com/scipy/scipy/blob/main/scipy/stats/_stats.pyx
    :param points: the data points to estimate from in 2 dimensions. Shape (n, 2).
    :param values: the multivariate values associated with the data points. (n,)
    :param xi: the coordinates to evaluate the estimate at in 2 dimensions. Shape (m, 2).
    :param sigma: the gaussian kernel width (standard deviation). Float.
    :param cutoff: the cutoff radius for the KDTree in units of sigma. Float.
    :return: the coarse grained data at the coordinates xi. Shape (m,).
    """
    n: uint64 = points.shape[0]  # number of data points
    d: uint64 = 2  # dimension of the data points
    m: uint64 = xi.shape[0]  # number of evaluation points

    points_: float64[n, d] = points / sigma  # the scaled data points
    # the leaf_size of the KDTree (log2(n) is a good heuristic)
    leaf_size: uint64 = np.floor(np.log2(n))
    tree = KDTree(points_, leafsize=leaf_size)  # the KDTree of the scaled data points

    # [TODO] pass in L as a parameter, = c*[(cutoff*sigma)/L]^2*n
    max_neighbors: uint64 = n

    xi_: float64[m, d] = xi / sigma  # the scaled evaluation points

    estimate: float64[m] = np.zeros(m)  # the estimate at the evaluation points
    norm: float64 = 1 / (2 * np.pi) / (sigma * sigma)  # the normalization factor of the gaussian kernel

    for j in prange(m):
        neighbors: uint64[:] = tree.query_radius(xi_[j, :], cutoff)[0]
        n_neighbors: uint64 = len(neighbors)
        # assert n_neighbors <= max_neighbors, 
        # "too many neighbors, increase max_neighbors tolerance"
        est_i : float64[max_neighbors] = np.zeros(max_neighbors)
        for i in prange(n_neighbors):
            est_i[i] = (
                np.exp(-np.sum((points_[i, :] - xi_[j, :]) * 
                               (points_[i, :] - xi_[j, :])) / 2) * values[i]
            )
        estimate[j] = np.sum(est_i)

    return estimate * norm


@jit(
    signature_or_function=(
        "float64[:,:](float64[:, :, :], float64[:, :], float64[:, :], float64, float64)"
    ),
    nopython=True,
    cache=False,
    fastmath=False,
    parallel=False,
    debug=False,
    nogil=True,
    boundscheck=True
)
def coarse_grain_time_slices(points: float64[:, :, :],
                             values: float64[:, :],
                             xi: float64[:, :],
                             sigma: float64,
                             cutoff: float64) -> float64[:, :]:
    """
    This function implements a gaussian coarse graining algorithm. Uses a KDTree to 
    only consider nearby points. Heavily inspired by the scipy implementation at 
    https://github.com/scipy/scipy/blob/main/scipy/stats/_stats.pyx
    :param points: the data points to estimate from in 2 dimensions + time. Shape (n, 2, t).
    :param values: the multivariate values associated with the data points. (n, t)
    :param xi: the coordinates to evaluate the estimate at in 2 dimensions. Shape (m, 2).
    :param sigma: the gaussian kernel width (standard deviation). Float.
    :param cutoff: the cutoff radius for the KDTree in units of sigma. Float.
    :return: the coarse grained data at the coordinates xi. Shape (m, t).
    """
    m: uint64 = xi.shape[0]  # number of evaluation points
    t: uint64 = points.shape[2]  # number of time slices

    estimate: float64[m, t] = np.zeros((m, t))  # the estimate at the evaluation points
    for h in range(t):
        estimate[:, h] = kd_gaussian_coarse_grain2d(
            points[:, :, h], values[:, h], xi[:, :], sigma, cutoff
        )

    return estimate


# WARNING: CURRENTLY DOESN'T WORK WITH PERIODIC BOUNDARIES
@jit(
    signature_or_function=(
        "float64[:,:](float64[:, :, :], float64[:, :], float64[:, :], float64, float64, int64)"
    ),
    nopython=True,
    cache=False,
    fastmath=False,
    parallel=True,
    debug=False,
    nogil=True,
    boundscheck=True
)
def verlet_coarse_grain_time_slices(points: float64[:, :, :],
                                    values: float64[:, :],
                                    xi: float64[:, :],
                                    sigma: float64,
                                    cutoff: float64,
                                    h: int64) -> float64[:, :]:
    """
    This function implements a gaussian coarse graining algorithm. Uses a KDTree to only consider nearby points.
    Heavily inspired by the scipy implementation at https://github.com/scipy/scipy/blob/main/scipy/stats/_stats.pyx
    :param points: the data points to estimate from in 2 dimensions + time. Shape (n, 2, t).
    :param values: the multivariate values associated with the data points. (n, t)
    :param xi: the coordinates to evaluate the estimate at in 2 dimensions. Shape (m, 2).
    :param sigma: the gaussian kernel width (standard deviation). Float.
    :param cutoff: the cutoff radius for the KDTree in units of sigma. Float.
    :param h: How often the verlet list is updated. uint64.
    :return: the coarse grained data at the coordinates xi. Shape (m, t).
    """
    n: uint64 = points.shape[0]  # number of data points
    m: uint64 = xi.shape[0]  # number of evaluation points
    t: uint64 = points.shape[2]  # number of time slices

    points_: float64[:, :, :] = points / sigma  # the scaled data points
    xi_: float64[:, :] = xi / sigma  # the scaled evaluation points

    leaf_size: uint64 = np.floor(np.log2(n))  # the leaf_size of the KDTree (log2(n) is a good heuristic)

    estimate: float64[m, t] = np.zeros((m, t))  # the estimate at the evaluation points
    norm: float64 = 1 / (2 * np.pi) / (sigma * sigma)  # the normalization factor of the gaussian kernel
    for k in np.arange(int64(0), t, h):
        # build the KDTree
        # print("Building KDTree for time slice " + str(k) + " of " + str(t) + 
        #       " time slices.")
        tree = KDTree(points_[:, :, k], leafsize=leaf_size)  # the KDTree of the scaled data points
        for j in prange(m):
            neighbors: uint64[:] = tree.query_radius(xi_[j, :], cutoff)[0]
            for g in range(h):
                if k + g >= t:
                    break
                for i in neighbors:
                    estimate[j, k + g] += np.exp(-np.sum((points_[i, :, k + g] - xi_[j, :]) *
                                                         (points_[i, :, k + g] - xi_[j, :])) / 2) * values[i, k + g]

    return estimate * norm


@jit(
    signature_or_function="float64(float64, uint8)",
    nopython=True,
    cache=False,
    fastmath=True,
    parallel=False,
    debug=False,
    nogil=True,
    boundscheck=False
)
def int_pow(x: float64, n: uint8) -> float64:
    """
    This function implements an integer power function.
    :param x: the base. Float.
    :param n: the exponent. uint64.
    :return: x^n. Float.
    """
    r: float64 = 1
    for i in range(n):
        r *= x

    return r


# [TODO] same changes as Gaussian
@jit(
    signature_or_function=(
        "float64[:](float64[:, :], float64[:], float64[:, :], uint8, float64)"
    ),
    nopython=True,
    cache=False,
    fastmath=True,
    parallel=True,
    debug=False,
    nogil=True,
    boundscheck=False
)
def kd_poly_coarse_grain2d(points: float64[:, :],
                           values: float64[:],
                           xi: float64[:, :],
                           order: uint8,
                           distance: float64) -> float64[:]:
    """
    This function implements a polynomial coarse graining algorithm. Uses a KDTree to only consider nearby points.
    Heavily inspired by the scipy implementation at https://github.com/scipy/scipy/blob/main/scipy/stats/_stats.pyx
    Kernel shape is (a^2- r^2)^n for r < a, 0 otherwise.
    :param points: the data points to estimate from in 2 dimensions. Shape (n, 2).
    :param values: the multivariate values associated with the data points. (n,)
    :param xi: the coordinates to evaluate the estimate at in 2 dimensions. Shape (m, 2).
    :param order: the order of the polynomial to use (n in the formula). uint8.
    :param distance: size of the kernel (a in the formula). Float.
    :return: the coarse grained data at the coordinates xi. Shape (m,).
    """
    n: uint64 = points.shape[0]  # number of data points
    d: uint64 = 2  # dimension of the data points
    m: uint64 = xi.shape[0]  # number of evaluation points

    points_: float64[n, d] = points / distance  # the scaled data points
    leaf_size: uint64 = np.floor(np.log2(n))  # the leaf_size of the KDTree (log2(n) is a good heuristic)
    tree = KDTree(points_, leafsize=leaf_size)  # the KDTree of the scaled data points

    xi_: float64[m, d] = xi / distance  # the scaled evaluation points

    estimate: float64[m] = np.zeros(m)  # the estimate at the evaluation points
    norm: float64 = np.pi * distance * distance / (1 + order)  # the normalization factor of the polynomial kernel
    for j in prange(m):
        neighbors: uint64[:] = tree.query_radius(xi_[j, :], 1)[0]
        for i in neighbors:
            estimate[j] += values[i] * int_pow(
                (1 - np.sum((points_[i, :] - xi_[j, :]) * (points_[i, :] - xi_[j, :]))),
                order
            )

    return estimate / norm


@jit(
    signature_or_function=(
        "float64[:,:](float64[:, :, :], float64[:, :], float64[:, :], uint8, float64)"
    ),
    nopython=True,
    cache=False,
    fastmath=False,
    parallel=True,
    debug=False,
    nogil=True,
    boundscheck=True
)
def poly_coarse_grain_time_slices(points: float64[:, :, :],
                                  values: float64[:, :],
                                  xi: float64[:, :],
                                  order: uint8,
                                  distance: float64) -> float64[:, :]:
    """
    Applies the polynomial coarse graining algorithm to a time series.
    :param points: the data points to estimate from in 2 dimensions. Shape (n, 2).
    :param values: the multivariate values associated with the data points. (n,)
    :param xi: the coordinates to evaluate the estimate at in 2 dimensions. Shape (m, 2).
    :param order: the order of the polynomial to use (n in the formula). uint8.
    :param distance: size of the kernel (a in the formula). Float.
    :return: the coarse grained data at the coordinates xi. Shape (m,).
    """
    m: uint64 = xi.shape[0]  # number of evaluation points
    t: uint64 = points.shape[2]  # number of time slices

    estimate: float64[m, t] = np.zeros((m, t))  # the estimate at the evaluation points
    for h in prange(t):
        estimate[:, h] = kd_poly_coarse_grain2d(points[:, :, h], values[:, h], xi[:, :], order, distance)

    return estimate


@jit(
    signature_or_function=(
        "float64[:](float64[:, :], float64[:], float64[:, :], float64, float64[:])"
    ),
    nopython=True,
    cache=False,
    fastmath=False,
    parallel=True,
    debug=False,
    nogil=True,
    boundscheck=True
)
def periodic_gaussian_coarse_grain2d(points: float64[:, :],
                                     values: float64[:],
                                     xi: float64[:, :],
                                     sigma: float64,
                                     box_lengths: float64[:]) -> float64[:]:
    """
    Gaussian coarse graining with periodic boundary conditions.
    :param points: the data points to estimate from in 2 dimensions. Shape (n, 2).
    :param values: the multivariate values associated with the data points. (n,)
    :param xi: the coordinates to evaluate the estimate at in 2 dimensions. Shape (m, 2).
    :param sigma: the gaussian kernel width (standard deviation). Float.
    :param box_lengths: box lengths [Lx, Ly] for periodic boundaries.
    :return: the coarse grained data at the coordinates xi. Shape (m,).
    """
    n: uint64 = points.shape[0]  # number of data points
    m: uint64 = xi.shape[0]  # number of evaluation points

    estimate: float64[m] = np.zeros(m)  # the estimate at the evaluation points
    norm: float64 = 1 / (2 * np.pi) / (sigma * sigma)  # the normalization factor

    # Fixed: parallelize over evaluation points to avoid race condition
    for j in prange(m):
        local_sum: float64 = 0.0
        for i in range(n):
            # Calculate periodic distance squared
            r_sq = periodic_distance_squared_2d(points[i, :], xi[j, :], box_lengths)
            local_sum += np.exp(-r_sq / (2 * sigma * sigma)) * values[i]
        estimate[j] = local_sum

    return estimate * norm


@jit(
    signature_or_function=(
        "float64[:](float64[:, :], float64[:], float64[:, :], float64, float64, float64[:])"
    ),
    nopython=True,
    cache=False,
    fastmath=False,
    parallel=True,
    debug=False,
    nogil=True,
    boundscheck=True
)
def periodic_kd_gaussian_coarse_grain2d(points: float64[:, :],
                                        values: float64[:],
                                        xi: float64[:, :],
                                        sigma: float64,
                                        cutoff: float64,
                                        box_lengths: float64[:]) -> float64[:]:
    """
    KDTree-accelerated Gaussian coarse graining with periodic boundary conditions.
    Uses periodic images to handle boundary conditions.
    :param points: the data points to estimate from in 2 dimensions. Shape (n, 2).
    :param values: the multivariate values associated with the data points. (n,)
    :param xi: the coordinates to evaluate the estimate at in 2 dimensions. Shape (m, 2).
    :param sigma: the gaussian kernel width (standard deviation). Float.
    :param cutoff: the cutoff radius for the KDTree in units of sigma. Float.
    :param box_lengths: box lengths [Lx, Ly] for periodic boundaries.
    :return: the coarse grained data at the coordinates xi. Shape (m,).
    """
    n: uint64 = points.shape[0]  # number of data points
    m: uint64 = xi.shape[0]  # number of evaluation points

    # Create periodic images (3x3 grid)
    n_images = 9 * n
    images = np.zeros((n_images, 2), dtype=np.float64)
    image_values = np.zeros(n_images, dtype=np.float64)
    
    create_periodic_images_2d(points, images, box_lengths)
    
    # Replicate values for each image
    for i in range(n):
        for j in range(9):
            image_values[i * 9 + j] = values[i]

    # Build KDTree with all images
    leaf_size: uint64 = max(10, np.floor(np.log2(n_images)))
    tree = KDTree(images, leafsize=leaf_size)

    estimate: float64[m] = np.zeros(m)  # the estimate at the evaluation points
    norm: float64 = 1 / (2 * np.pi) / (sigma * sigma)  # the normalization factor

    cutoff_dist = cutoff * sigma
    
    for j in prange(m):
        neighbors: uint64[:] = tree.query_radius(xi[j, :], cutoff_dist)[0]
        for i in neighbors:
            r_sq = np.sum((images[i, :] - xi[j, :]) * (images[i, :] - xi[j, :]))
            estimate[j] += np.exp(-r_sq / (2 * sigma * sigma)) * image_values[i]

    return estimate * norm


@jit(
    signature_or_function=(
        "float64[:](float64[:, :], float64[:], float64[:, :], uint8, float64, float64[:])"
    ),
    nopython=True,
    cache=False,
    fastmath=True,
    parallel=True,
    debug=False,
    nogil=True,
    boundscheck=False
)
def periodic_kd_poly_coarse_grain2d(points: float64[:, :],
                                    values: float64[:],
                                    xi: float64[:, :],
                                    order: uint8,
                                    distance: float64,
                                    box_lengths: float64[:]) -> float64[:]:
    """
    KDTree-accelerated polynomial coarse graining with periodic boundary conditions.
    :param points: the data points to estimate from in 2 dimensions. Shape (n, 2).
    :param values: the multivariate values associated with the data points. (n,)
    :param xi: the coordinates to evaluate the estimate at in 2 dimensions. Shape (m, 2).
    :param order: the order of the polynomial to use. uint8.
    :param distance: size of the kernel. Float.
    :param box_lengths: box lengths [Lx, Ly] for periodic boundaries.
    :return: the coarse grained data at the coordinates xi. Shape (m,).
    """
    n: uint64 = points.shape[0]  # number of data points
    m: uint64 = xi.shape[0]  # number of evaluation points

    # Create periodic images (3x3 grid)
    n_images = 9 * n
    images = np.zeros((n_images, 2), dtype=np.float64)
    image_values = np.zeros(n_images, dtype=np.float64)
    
    create_periodic_images_2d(points, images, box_lengths)
    
    # Replicate values for each image
    for i in range(n):
        for j in range(9):
            image_values[i * 9 + j] = values[i]

    # Build KDTree with all images
    leaf_size: uint64 = max(10, np.floor(np.log2(n_images)))
    tree = KDTree(images, leafsize=leaf_size)

    estimate: float64[m] = np.zeros(m)  # the estimate at the evaluation points
    norm: float64 = np.pi * distance * distance / (1 + order)  # normalization factor
    
    for j in prange(m):
        neighbors: uint64[:] = tree.query_radius(xi[j, :], distance)[0]
        for i in neighbors:
            r_sq = np.sum((images[i, :] - xi[j, :]) * (images[i, :] - xi[j, :]))
            r_sq_normalized = r_sq / (distance * distance)
            if r_sq_normalized < 1.0:
                estimate[j] += image_values[i] * int_pow((1 - r_sq_normalized), order)

    return estimate / norm


@jit(
    signature_or_function=(
        "float64[:,:](float64[:, :, :], float64[:, :], float64[:, :], uint8, float64, float64[:])"
    ),
    nopython=True,
    cache=False,
    fastmath=False,
    parallel=True,
    debug=False,
    nogil=True,
    boundscheck=True
)
def periodic_poly_coarse_grain_time_slices(points: float64[:, :, :],
                                           values: float64[:, :],
                                           xi: float64[:, :],
                                           order: uint8,
                                           distance: float64,
                                           box_lengths: float64[:]) -> float64[:, :]:
    """
    Applies the periodic polynomial coarse graining algorithm to a time series.
    :param points: the data points to estimate from in 2 dimensions + time. Shape (n, 2, t).
    :param values: the multivariate values associated with the data points. (n, t)
    :param xi: the coordinates to evaluate the estimate at in 2 dimensions. Shape (m, 2).
    :param order: the order of the polynomial to use. uint8.
    :param distance: size of the kernel. Float.
    :param box_lengths: box lengths [Lx, Ly] for periodic boundaries.
    :return: the coarse grained data at the coordinates xi. Shape (m, t).
    """
    m: uint64 = xi.shape[0]  # number of evaluation points
    t: uint64 = points.shape[2]  # number of time slices

    estimate: float64[m, t] = np.zeros((m, t))  # the estimate at the evaluation points
    for h in prange(t):
        estimate[:, h] = periodic_kd_poly_coarse_grain2d(
            points[:, :, h], values[:, h], xi[:, :], order, distance, box_lengths
        )

    return estimate
