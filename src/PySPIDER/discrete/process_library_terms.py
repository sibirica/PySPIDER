import warnings
from typing import Any
from dataclasses import dataclass

import numpy as np
import scipy
from scipy.stats._stats import gaussian_kernel_estimate
from scipy.ndimage import gaussian_filter1d
# uncomment the next line if it isn't broken for you
from .coarse_grain_utils import poly_coarse_grain_time_slices #coarse_grain_time_slices, 

from ..commons.process_library_terms import AbstractDataset, IntegrationDomain, LibraryData, int_by_parts, diff
from ..commons.library import LibraryPrime #, Observable
from ..commons.z3base import LiteralIndex, FullRank, Antisymmetric, SymmetricTraceFree
from ..commons.utils import regex_find
from .convolution import gauss1d
from .library import generate_terms_to #, CoarseGrainedProduct

import concurrent.futures
from collections import defaultdict

#function for initializing global variables for each parallel worker process (discrete version)
def discrete_init_domain_worker(dataset_init, current_irrep_init, by_parts_init, debug_init):
    global worker_dataset, worker_current_irrep, worker_by_parts, worker_debug
    worker_dataset = dataset_init
    worker_current_irrep = current_irrep_init
    worker_by_parts = by_parts_init
    worker_debug = debug_init

#function to be executed in parallel to evaluate all terms for a given domain (discrete version with rho handling)
def discrete_parallel_domain_task(domain):
    dataset = worker_dataset
    irrep = worker_current_irrep
    by_parts = worker_by_parts
    debug = worker_debug

    domain_results_dict = defaultdict(float)
    rho_std_for_domain = None

    for t, w, term, tensor_weight in dataset.integrated_terms_tuples:
        if w.scale == 0:
            continue
        value = dataset.eval_on_domain(t,w,domain)
        key = term, tensor_weight
        domain_results_dict[key] += value
    
    # Compute rho standard deviation before cleanup (if cleanup is enabled)
    if dataset.cleanup_cache and dataset.field_dict is not None:
        # Find ρ prime and compute its std for this domain
        for key in dataset.field_dict.keys():
            if len(key) == 2 and key[1] == domain:
                prime = key[0]
                # Check if this prime corresponds to rho (string representation is exactly 'ρ')
                if hasattr(prime, '__str__') and 'ρ'==str(prime):
                    rho_data = dataset.field_dict[key]
                    rho_std_for_domain = np.std(rho_data)
                    break  # Found ρ prime, no need to continue
        
        # Free up memory by removing cached field_dict entries for this domain
        keys_to_remove = [key for key in dataset.field_dict.keys() if len(key) == 2 and key[1] == domain]
        for key in keys_to_remove:
            del dataset.field_dict[key]
    
    return domain, domain_results_dict, rho_std_for_domain

@dataclass(kw_only=True)
class SRDataset(AbstractDataset):  # structures all data associated with a given sparse regression dataset
    particle_pos: np.ndarray[float]  # array of particle positions (particle, spatial index, time)
    kernel_sigma: float # standard deviation of kernel in physical units (scalar for now)
    # subsampling factor when computing coarse-graining, i.e. cg_res points per unit length; should generally just
    # be an integer
    cg_res: float
    deltat: float
    # not sure what the type-hinting was supposed to be here
    domain_neighbors: dict[tuple[IntegrationDomain, float], int] = None # indices of neighbors of each ID at given time
    cutoff: float=6 # how many std deviations to cut off Gaussian weight functions at
    rho_scale: float=1 # density rescaling factor
    time_sigma: float=0 # standard deviation for temporal smoothing kernel (0 = no smoothing)
    #field_dict: dict[tuple[Any], np.ndarray[float]] = None # storage of computed coarse-grained quantities: (cgp, dims, domains) -> array
    
    # Storage for rho statistics computed during parallel processing
    rho_domain_stds: list[float] = None
    
    #cgps: set[CoarseGrainedPrimitive] = None # list of coarse-grained primitives involved

    def __post_init__(self):
        super().__post_init__()
        self.scaled_sigma = self.kernel_sigma * self.cg_res
        self.scaled_pts = self.particle_pos * self.cg_res
        self.dxs = [1 / self.cg_res] * (self.n_dimensions - 1) + [float(self.deltat)]  # spacings of sampling grid
        self.rho_domain_stds = None  # Initialize rho statistics storage
        #self.rho_scale = self.particle_pos.shape[0]/np.prod(self.world_size[:-1]) # mean number density
        #self.cgps = set()

    def make_libraries(self, **kwargs):
        self.libs = dict()
        terms = generate_terms_to(observables=self.observables, **kwargs)
        for irrep in self.irreps:
            match irrep:
                case int():
                    self.libs[irrep] = LibraryData([term for term in terms if term.rank == irrep], irrep)
                case FullRank():
                    self.libs[irrep] = LibraryData([term for term in terms if term.rank == irrep.rank], irrep)
                case Antisymmetric():
                    self.libs[irrep] = LibraryData([term for term in terms if term.rank == irrep.rank 
                                                    and term.symmetry() != 1], irrep)
                case SymmetricTraceFree():
                    self.libs[irrep] = LibraryData([term for term in terms if term.rank == irrep.rank 
                                                    and term.symmetry() != -1], irrep)
                case _:
                    raise NotImplemented

    def make_domains(self, ndomains, domain_size, pad=0, t_pad=0):
        self.domains = []
        scaled_dims = [int(s * self.cg_res) for s in domain_size[:-1]] + [domain_size[-1]]  # self.interp_factor *
        scaled_world_size = [int(s * self.cg_res) for s in self.world_size[:-1]] + [
            self.world_size[-1]]  # self.interp_factor *
        # padding by pad in original units on spatial dims
        self.pad = pad # record the padding used
        pads = [np.ceil(pad * self.cg_res) for s in domain_size[:-1]] + [t_pad] 
        self.domain_size = scaled_dims
        for i in range(ndomains):
            min_corner = []
            max_corner = []
            # define domains on the *scaled* grid
            for (L, max_lim, pad_i) in zip(scaled_dims, scaled_world_size, pads):
                #print(pad_i, max_lim - (L+pad_i) + 1)
                num = np.random.randint(pad_i, max_lim - (L + pad_i) + 1)
                min_corner.append(num)
                max_corner.append(num + L - 1)
            # (potentially) less messy if we fix beginning/end of time extent to the actual measurements
            # time_fraction = min_corner % self.interp_factor
            # min_corner -= time_fraction
            # max_corner -= time_fraction
            self.domains.append(IntegrationDomain(min_corner, max_corner))

    def find_domain_neighbors(self):
        # list of indices corresponding to particles needed to compute quantities on each domain at each t
        self.domain_neighbors = dict()
        for domain in self.domains:
            for t in domain.times:
                self.domain_neighbors[domain, t] = []
                for i, pt in enumerate(self.scaled_pts):
                    dist = domain.distance(pt[:, t])
                    if dist <= self.scaled_sigma * self.cutoff:
                        self.domain_neighbors[domain, t].append(i)

    def eval_prime(self, prime: LibraryPrime, domain: IntegrationDomain, experimental: bool = True, order: int = 4):
        cgp = prime.derivand
        if self.n_dimensions != 3:
            if experimental:
                warnings.warn("Experimental method only implemented for 2D+1 systems")
        
        # Determine time range with buffering for smoothing
        if self.time_sigma > 0:
            # Calculate buffer size: 4*time_sigma in time steps should be sufficient
            time_buffer = int(np.ceil(min(4, self.cutoff) * self.time_sigma))
            # Extend time range for smoothing, but respect data boundaries
            extended_min_time = max(0, domain.min_corner[-1] - time_buffer)
            extended_max_time = min(self.scaled_pts.shape[2] - 1, domain.max_corner[-1] + time_buffer)
            extended_times = list(range(extended_min_time, extended_max_time + 1))
            extended_shape = domain.shape[:-1] + [len(extended_times)]
        else:
            extended_times = domain.times
            extended_shape = domain.shape
        
        data_slice = np.zeros(extended_shape)
        if experimental:
            pt_pos = self.scaled_pts[:, :, extended_times] / self.cg_res  # Unscaled positions
            pt_pos = np.float64(pt_pos)
            weights = np.ones_like(pt_pos[:, 0, :], dtype=np.float64)
            for obs in cgp.observables:
                obs_inds = map(lambda idx: idx.value, obs.indices)
                #if obs.rank == 0:
                #    data = self.data_dict[obs.string][:, 0, extended_times]
                #else:
                data = self.data_dict[obs.string][:, *obs_inds, extended_times]
                weights *= data.astype(np.float64)
                #obs_dim_ind += obs.rank
            sigma = self.scaled_sigma / self.cg_res
            min_corner = domain.min_corner[:-1]
            max_corner = domain.max_corner[:-1]
            xx, yy = np.mgrid[
                         min_corner[0]:(max_corner[0] + 1),
                         min_corner[1]:(max_corner[1] + 1)
                         ]
            xi = np.vstack([
                (xx / self.cg_res).ravel(),
                (yy / self.cg_res).ravel(),
            ]).T
            dist = sigma*np.sqrt(3+2*order)
            # uncomment if this isn't broken for you
            data_slice = poly_coarse_grain_time_slices(pt_pos, weights, xi, order, dist) 
            data_slice = data_slice.reshape(extended_shape)
        else:
            if self.domain_neighbors is None:
                self.find_domain_neighbors()
            for t in range(extended_shape[-1]):
                time_slice = np.zeros(extended_shape[:-1])
                if self.time_sigma > 0:
                    t_shifted = extended_times[t]
                else:
                    t_shifted = t + domain.min_corner[-1]
                if experimental:
                    # experimental method using scipy.stats.gaussian_kde
                    particles = self.domain_neighbors[domain, t_shifted]
                    pt_pos = self.scaled_pts[particles, :, t_shifted] / self.cg_res
                    weights = np.ones_like(particles, dtype=np.float64)
                    #obs_dim_ind = 0
                    for obs in cgp.observables:
                        obs_inds = map(lambda idx: idx.value, obs.indices)
                        #if obs.rank == 0:
                        #    data = self.data_dict[obs.string][:, 0, t_shifted]
                        #else:
                        data = self.data_dict[obs.string][:, *obs_inds, t_shifted]
                        weights *= data.astype(np.float64)
                        #obs_dim_ind += obs.rank
                    sigma = self.scaled_sigma ** 2 / (self.cg_res ** 2)
                    # Check scipy version. If it's lower than 1.10, use inverse_covariance, otherwise use Cholesky
                    if int(scipy.__version__.split(".")[0]) <= 1 and int(scipy.__version__.split(".")[1]) < 10:
                        inv_cov = np.eye(2) / sigma
                    else:
                        inv_cov = np.eye(2) * sigma
                        inv_cov = np.linalg.cholesky(inv_cov[::-1, ::-1]).T[::-1, ::-1]
                    min_corner = domain.min_corner[:-1]
                    max_corner = domain.max_corner[:-1]
                    xx, yy = np.mgrid[min_corner[0]:(max_corner[0] + 1), min_corner[1]:(max_corner[1] + 1)]
                    positions = np.vstack([(xx / self.cg_res).ravel(), (yy / self.cg_res).ravel()]).T
                    density = gaussian_kernel_estimate['double'](pt_pos, weights[:, None], positions, inv_cov,
                                                                 np.float64)
                    time_slice = np.reshape(density[:, 0], xx.shape)

                    data_slice[..., t] = time_slice / (self.cg_res ** 2)
                else:
                    for i in self.domain_neighbors[domain, t_shifted]:
                        pt_pos = self.scaled_pts[i, :, t_shifted]
                        # evaluate observables inside rho[...]
                        coeff = 1
                        for obs in cgp.observables:
                            obs_inds = map(lambda idx: idx.value, obs.indices)
                            # print(obs, i, obs_dims[obs_dim_ind], t_shifted)
                            #if obs.rank == 0:
                            #    data = self.data_dict[obs.string][:, 0, t_shifted]
                            #else:
                            data = self.data_dict[obs.string][:, *obs_inds, t_shifted]
                            coeff *= data.astype(np.float64)
                            # print(coeff)
                        # coarse-graining this particle (one dimension at a time)
                        rngs = []
                        g_nd = 1
                        for coord, d_min, d_max, j in zip(pt_pos, domain.min_corner, domain.max_corner,
                                                          range(self.n_dimensions - 1)):
                            # recenter so that 0 is start of domain
                            g, mn, mx = gauss1d(coord - d_min, self.scaled_sigma, truncate=self.cutoff,
                                                xmin=0, xmax=d_max - d_min)
                            g_nd = np.multiply.outer(g_nd, g)
                            rng_array = np.array(range(mn, mx))  # coordinate range of kernel
                            # now need to add free axes so that the index ends up as an (n-1)-d array
                            n_free_dims = self.n_dimensions - j - 2  # how many np.newaxis to add to index
                            expanded_rng_array = np.expand_dims(rng_array, axis=tuple(range(1, 1 + n_free_dims)))
                            rngs.append(expanded_rng_array)
                        # if len((g_nd*coeff).shape) > len(time_slice.shape):
                        #    print(rngs, g_nd.shape, coeff)
                        time_slice[tuple(rngs)] += g_nd * coeff
                    data_slice[..., t] = time_slice
        if not experimental:
            data_slice *= self.cg_res ** (self.n_dimensions - 1)  # need to scale rho by res^(# spatial dims)!

        # rescale prime to rho=1 units
        data_slice /= self.rho_scale
        
        # Apply temporal smoothing as applicable
        if self.time_sigma > 0:
            # Apply Gaussian smoothing along time axis (last axis)
            data_slice = gaussian_filter1d(data_slice, sigma=self.time_sigma, 
                                           axis=-1, mode='nearest')
            
            # Trim back to original domain size
            time_buffer = int(np.ceil(3 * self.time_sigma))
            start_idx = max(0, domain.min_corner[-1] - extended_times[0])
            end_idx = start_idx + len(domain.times)
            data_slice = data_slice[..., start_idx:end_idx]
        
        # evaluate derivatives
        orders = prime.derivative.get_spatial_orders()
        dimorders = [orders[LiteralIndex(i)] for i in range(self.n_dimensions-1)]
        dimorders += [prime.derivative.torder]
        #print(prime, dimorders, data_slice.shape, self.dxs)
        return diff(data_slice, dimorders, self.dxs) if sum(dimorders)>0 else data_slice

    def find_scales(self, names=None):
        # find mean/std deviation of fields in data_dict that are in names
        self.scale_dict = dict()
        for name in self.data_dict:
            if names is None or name in names:
                self.scale_dict[name] = dict()
                # if these are vector quantities the results could be wonky in the unlikely
                # case a vector field is consistently aligned with one of the axes
                self.scale_dict[name]['mean'] = np.mean(
                    np.linalg.norm(self.data_dict[name]) / np.sqrt(self.data_dict[name].size))
                self.scale_dict[name]['std'] = np.std(self.data_dict[name])
        # also need to handle density separately
        self.scale_dict['rho'] = dict()
        #self.rho_scale = self.particle_pos.shape[0] / np.prod(self.world_size[:-1])
        self.scale_dict['rho']['mean'] = self.particle_pos.shape[0] / np.prod(self.world_size[:-1]) / self.rho_scale

        # Use precomputed rho standard deviations if available (from parallel processing)
        if hasattr(self, 'rho_domain_stds') and self.rho_domain_stds is not None:
            # Compute RMS of per-domain standard deviations
            rho_std = np.sqrt(np.mean(np.array(self.rho_domain_stds) ** 2))
            self.scale_dict['rho']['std'] = rho_std
        else:
            # Fallback to original method if parallel processing wasn't used or cleanup was disabled
            #print(self.field_dict.keys())
            all_cgps = [key[0] for key in self.field_dict.keys()]
            rho_matches = regex_find(all_cgps, r'ρ')
            rho_ind = next(rho_matches)[0]
            rho = all_cgps[rho_ind]
            all_rho_data = np.dstack([self.field_dict[rho, domain] for domain in self.domains])
            rho_std = np.std(all_rho_data)
            self.scale_dict['rho']['std'] = rho_std

    ### TO DO: compute correlation length/time automatically?
    def set_LT_scale(self, L, T):
        self.xscale = L
        self.tscale = T

    def get_char_size(self, term):
        # return characteristic size of a library term
        product = 1
        for prime in term.primes:
            xorder = prime.derivative.xorder
            torder = prime.derivative.torder
            if torder + xorder > 0:
                statistic = 'std'
            else:
                statistic = 'mean'
            for obs in prime.derivand.observables:
                name = obs.string
                product *= self.scale_dict[name][statistic]
            # add in rho contribution (every primitive contains a rho)
            product *= self.scale_dict['rho'][statistic]
            # scale by correlation length (time) / dx (dt)
            product /= self.xscale ** xorder
            product /= self.tscale ** torder
        return product

    def make_Q_parallel(self, irrep, by_parts=True, debug=False, num_processors=None):
        """Override parent method to handle rho statistics for discrete datasets"""
        # Main method logic (adapted from parent)
        init_args = (self, irrep, by_parts, debug)
        domains = self.domains
        all_results = []
        rho_stds = []

        #precompute symbolic manipulations for parallel tasks
        self.integrated_terms_tuples = []
        for term in list(self.libs[irrep].terms):
            if debug:
                print("UNINDEXED TERM:")
                print(term)
                term_symmetry = term.symmetry()
                print("Symmetry:", term_symmetry)
            for weight in list(self.weights):
                for tensor_weight in self.tensor_weight_basis[(irrep, weight)].tw_list:
                    if debug:
                        print("Tensor weight:", tensor_weight)
                    for indexed_term, scalar_weight in self.get_index_assignments(term,tensor_weight): #, debug
                        if debug:
                            print("ASSIGNMENTS:", term, "->")
                            print("Indexed term:", indexed_term)
                            print("Scalar weight:", scalar_weight)
                        for t, w in int_by_parts(indexed_term, scalar_weight, by_parts):
                            if debug:
                                print("INT BY PARTS:", indexed_term, "->")
                                print("Integrated term:", t)
                                print("Integrated weight:", w)
                            self.integrated_terms_tuples.append((t,w,term,tensor_weight))

        #begin parallel task execution
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processors, initializer=discrete_init_domain_worker, initargs=init_args) as executor:
            results = executor.map(discrete_parallel_domain_task, domains)
            for result in results:
                domain, domain_results, rho_std = result
                all_results.append((domain, domain_results))
                if rho_std is not None:
                    rho_stds.append(rho_std)
        
        # Store rho standard deviations for later use in find_scales
        if rho_stds:
            self.rho_domain_stds = rho_stds
                  
        terms = list(self.libs[irrep].terms)
        weights = list(self.weights)
        num_cols = len(terms)
        term_to_col_idx = {term: i for i, term in enumerate(terms)}
        row_map = {}
        current_row_idx = 0
        for weight in weights:
            for tensor_weight in self.tensor_weight_basis[(irrep, weight)].tw_list:
                for domain in domains:
                    row_key = (tensor_weight, domain)
                    if row_key not in row_map:
                        row_map[row_key] = current_row_idx
                        current_row_idx += 1
        num_rows = current_row_idx

        Q_matrix = np.zeros((num_rows, num_cols), dtype=np.float64)

        for domain, domain_results in all_results:
            for (term, tensor_weight), result in domain_results.items():
                col_idx = term_to_col_idx[term]
                row_idx = row_map[(tensor_weight, domain)]

                Q_matrix[row_idx, col_idx] = result

        return Q_matrix