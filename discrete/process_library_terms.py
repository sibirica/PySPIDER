import warnings

import numpy as np
import scipy
from commons.process_library_terms import *
from convolution import *
from commons.library import *
from library import *
from scipy.stats._stats import gaussian_kernel_estimate
# uncomment if this isn't broken for you
from coarse_grain_utils import coarse_grain_time_slices, poly_coarse_grain_time_slices

from commons.utils import regex_find

@dataclass(kw_only=True)
class SRDataset(AbstractDataset):  # structures all data associated with a given sparse regression dataset
    particle_pos: np.ndarray[float]  # array of particle positions (particle, spatial index, time)
    kernel_sigma: float # standard deviation of kernel in physical units (scalar for now)
    # subsampling factor when computing coarse-graining, i.e. cg_res points per unit length; should generally just
    # be an integer
    cg_res: float
    cutoff: float # how many std deviations to cut off weight functions at
    deltat: float
    domain_neighbors: dict[[IntegrationDomain, float], int] = None # indices of neighbors of each ID at given time
    #field_dict: dict[tuple[Any], np.ndarray[float]] = None # storage of computed coarse-grained quantities: (cgp, dims, domains) -> array
    
    #cgps: set[CoarseGrainedPrimitive] = None # list of coarse-grained primitives involved

    def __post_init__(self):
        super().__post_init__()
        self.scaled_sigma = self.kernel_sigma * self.cg_res
        self.scaled_pts = self.particle_pos * self.cg_res
        self.dxs = [1 / self.cg_res] * (self.n_dimensions - 1) + [float(self.deltat)]  # spacings of sampling grid
        #self.cgps = set()

    def make_libraries(self, max_complexity=4, max_observables=3, max_rho=999):
        self.libs = dict()
        terms = generate_terms_to(max_complexity, observables=self.observables,
                                  max_observables=max_observables, max_rho=max_rho)
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
                #case _:
                #    raise NotImplemented

    def make_domains(self, ndomains, domain_size, pad=0):
        self.domains = []
        scaled_dims = [int(s * self.cg_res) for s in domain_size[:-1]] + [domain_size[-1]]  # self.interp_factor *
        scaled_world_size = [int(s * self.cg_res) for s in self.world_size[:-1]] + [
            self.world_size[-1]]  # self.interp_factor *
        # padding by pad in original units on spatial dims
        self.pad = pad # record the padding used
        pads = [np.ceil(pad * self.cg_res) for s in domain_size[:-1]] + [0] 
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
        # experimental: bool = True,
        cgp = prime.derivand
        if self.n_dimensions != 3:
            if experimental:
                warnings.warn("Experimental method only implemented for 2D+1 systems")
        data_slice = np.zeros(domain.shape)
        if experimental:
            pt_pos = self.scaled_pts[:, :, domain.times] / self.cg_res  # Unscaled positions
            pt_pos = np.float64(pt_pos)
            weights = np.ones_like(pt_pos[:, 0, :], dtype=np.float64)
            for obs in cgp.observables:
                obs_inds = map(lambda idx: idx.value, obs.indices)
                #if obs.rank == 0:
                #    data = self.data_dict[obs.string][:, 0, domain.times]
                #else:
                data = self.data_dict[obs.string][:, *obs_inds, domain.times]
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
            data_slice = data_slice.reshape(domain.shape)
        else:
            if self.domain_neighbors is None:
                self.find_domain_neighbors()
            for t in range(domain.shape[-1]):
                time_slice = np.zeros(domain.shape[:-1])
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
                        #    data = self.data_dict[obs.string][:, 0, domain.times]
                        #else:
                        data = self.data_dict[obs.string][:, *obs_inds, domain.times]
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
                            #    data = self.data_dict[obs.string][:, 0, domain.times]
                            #else:
                            data = self.data_dict[obs.string][:, *obs_inds, domain.times]
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
        self.scale_dict['rho']['mean'] = self.particle_pos.shape[0] / np.prod(self.world_size[:-1])
        #rho_ind = find_term(self.cgps, 'rho')
        #rho_cgp = self.cgps[rho_ind]
        
        #rho_cgp = CoarseGrainedPrimitive([])
        #rho_lp = LibraryPrimitive(DerivativeOrder(0,0), rho_cgp)
        #rho_ip = IndexedPrimitive(rho_lp, space_orders=[0]*(self.n_dimensions-1), obs_dims=())

        #print(self.field_dict.keys())
        all_cgps = [key[0] for key in self.field_dict.keys()]
        rho_matches = regex_find(all_cgps, r'ρ')
        rho_ind = next(rho_matches)[0]
        rho = all_cgps[rho_ind]
        all_rho_data = np.dstack([self.field_dict[rho, domain] for domain in self.domains])
        rho_std = np.std(all_rho_data)
        #print(all_rho_data)
        #rho_std = np.std(np.dstack([self.cg_dict[rho_cgp, (), domain] for domain in self.domains]))
        self.scale_dict['rho']['std'] = rho_std

    ### TO DO: compute correlation length/time automatically
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

    # this function is a bit problematic as there isn't a clear correct way to do this, so we will not implement it
    # def find_row_weights(self):
    #     rho_col = find_term(self.libs[0].terms, 'rho')
    #     # integral of rho with the 0 harmonics weight
    #     dom_densities = self.libs[0].Q[0:len(self.domains), rho_col]
    #     row_weights0 = np.tile(dom_densities, len(self.weights))
    #     # scale weights according to square root of density (to cancel CLT noise scaling)
    #     row_weights0 = np.sqrt(row_weights0)
    #     row_weights0 += 1e-6  # don't want it to be exactly zero
    #     # normalize
    #     row_weights0 /= np.max(row_weights0)
    #     row_weights1 = np.tile(row_weights0, self.n_dimensions - 1)  # because each dimension gets its own row
    #     self.libs[0].row_weights = row_weights0
    #     self.libs[1].row_weights = row_weights1