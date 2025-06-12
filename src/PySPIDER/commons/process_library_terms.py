from typing import Any, Union, Iterable
from warnings import warn

import numpy as np
from findiff import FinDiff
from functools import reduce
from operator import mul
from dataclasses import dataclass, replace

import concurrent.futures

from PySPIDER.commons.library import *
from PySPIDER.commons.weight import *

# if we want to use integration domains with different sizes & spacings, it might be
# better to store that information within this object as well
class IntegrationDomain(object):
    def __init__(self, min_corner, max_corner):
        # min_corner - min coordinates in each dimension; sim. max_corner
        self.min_corner = min_corner
        self.max_corner = max_corner
        self.shape = [max_c - min_c + 1 for (min_c, max_c) in zip(self.min_corner, self.max_corner)]
        self.times = list(range(min_corner[-1], max_corner[-1] + 1))

    def __repr__(self):
        return f"IntegrationDomain({self.min_corner}, {self.max_corner})"

    def __hash__(self):  # for use in cg_dict etc.
        # here we use that min_corner and max_corner are both lists of ints
        return hash(tuple(self.min_corner + self.max_corner))

    def distance(self, pt):
        return max(self.line_dist(coord, i) for i, coord in enumerate(pt)) # L_0 norm distance
        # return np.linalg.norm([self.line_dist(coord, dim) for i, coord in enumerate(pt)]) # L_2 norm distance

    def line_dist(self, coord, dim):
        return max(0, self.min_corner[dim] - coord, coord - self.max_corner[dim])
    
    def __eq__(self, other):
        if not isinstance(other, IntegrationDomain):
            return NotImplemented
        return self.min_corner == other.min_corner and self.max_corner == self.max_corner

@dataclass
class Weight(object): # scalar-valued Legendre polynomial weight function (may rename class to LegendreWeight)
    m: List[int]
    q: List[int]
    k: List[int]
    dxs: Iterable[float] = None
    n_dimensions: int = None
    n_spatial_dim: int = None
    scale: float = 1
    ready: bool = False
    weight_objs: List[np.polynomial.Polynomial] = None
        
    def __post_init__(self):
        self.n_dimensions = len(self.m)
        self.n_spatial_dim = self.n_dimensions-1
        if self.dxs is None:
            self.dxs = [1] * self.n_dimensions

    def make_weight_objs(self):
        self.ready = True
        self.weight_objs = [weight_1d(m, q, k, dx) for (m, q, k, dx) in zip(self.m, self.q, self.k, self.dxs)]

    def get_weight_array(self, dims):
        if not self.ready:
            self.make_weight_objs()
        weights_eval = [weight.linspace(dim)[1] for (weight, dim) in zip(self.weight_objs, dims)]
        if self.scale == 0: # short-circuit the zero weight case
            return np.zeros(dims)
        return self.scale * reduce(lambda x, y: np.tensordot(x, y, axes=0), weights_eval)

    def increment(self, dim):  # return new weight with an extra derivative on the dim-th dimension
        knew = self.k.copy()
        knew[dim] += 1
        new_weight = replace(self, k=knew, ready=False)
        return new_weight

    def __neg__(self):
        new_weight = replace(self, scale=-self.scale, ready=False)
        return new_weight

    def __mul__(self, number):
        new_weight = replace(self, scale=self.scale*number, ready=False)
        return new_weight

    __rmul__ = __mul__

    def __repr__(self):
        return f"Weight(m={self.m}, q={self.q}, k={self.k}, coeff={self.scale}, dxs={self.dxs})"

    def __hash__(self):
        return hash(self.__repr__)
    
    def __eq__(self, other):
        if not isinstance(other, Weight):
            return NotImplemented
        return (self.m == other.m and self.q == other.q and self.k == other.k and tuple(self.dxs) == tuple(other.dxs) and self.scale == other.scale)

@dataclass
class GeneralizedWeight(object): # scalar-valued weight function for g(x)*Weight - not currently in use
    base_weight: Weight
    g_fun: Callable[[Iterable[float]], float]
    
    def __repr__(self):
        return f"GeneralizedWeight({self.base_weight}, {self.g_fun.__name__})"

@dataclass(kw_only=True)
class Metric(object):
    n_dimensions: int = None # number of dimensions of space
    tensor: np.ndarray = None # 2d metric tensor

    def __post_init(self):
        if self.tensor is None:
            self.tensor = np.eye(self.n_dimensions)
        elif self.n_dimensions is None:
            self.n_dimensions = self.tensor.shape[0]

    def __repr__(self):
        return f"Metric({self.tensor})"

    def __getitem__(self, input): # return entry of weight at specific multiindex
        return self.tensor[input]

def multi_index_iterator(n_dimensions, rank):
    a = np.arange(n_dimensions**rank).reshape((n_dimensions,)*rank)
    return np.nditer(a, flags=['multi_index'])

@dataclass#(frozen=True)
class TensorWeight: # tensor-valued weight function 
    weight_dict: dict[tuple[int, ...], Weight] # dict mapping tuples to weight functions
    rank: int # rank of tensor/library
    n_spatial_dim: int # number of spatial dimensions of data

    @cached_property
    def n_dimensions(self):
        return self.n_spatial_dim+1

    def __getitem__(self, input): # return entry of weight at specific multiindex
        return self.weight_dict[input]

    def weight_map(self, fun):
        weight_dict = {idx: fun(w) for idx, w in self.weight_dict.items()}
        return replace(self, weight_dict=weight_dict)
        
    def increment(self, dim):  # return new weight with an extra derivative on the dim-th dimension
        return self.weight_map(lambda w: w.increment(dim))

    def __neg__(self):
        return self.weight_map(lambda w: -w)

    def __mul__(self, number):
        return self.weight_map(lambda w: number*w)

    def __repr__(self):
        return repr(self.weight_dict)

    def __hash__(self):
        return hash(repr(self))
    
    def __eq__(self, other):
        if not isinstance(other, TensorWeight):
            return NotImplemented
        return (self.weight_dict == other.weight_dict and self.rank == other.rank and self.n_spatial_dim == other.n_spatial_dim)

    __rmul__ = __mul__

@dataclass#(frozen=True)
class FactoredTensorWeight(TensorWeight): 
# more efficient implementation might be to perform operations on base_weight instead of individual keys,
# but I don't think it matters

    # factorization
    base_weight: Weight
    tensor: np.ndarray

    @staticmethod
    def constant_tw(base_weight, tensor): # create TensorWeight equal to constant Tensor * scalar Weight
        weight_dict = dict()
        n_dimensions, rank = tensor.shape[0], len(tensor.shape)
        it = multi_index_iterator(n_dimensions, rank)
        for x in it:
            weight_dict[it.multi_index] = base_weight * tensor[it.multi_index]
        return FactoredTensorWeight(weight_dict=weight_dict, rank=rank, n_spatial_dim=n_dimensions, 
                                    base_weight=base_weight, tensor=tensor)

    def __repr__(self):
        return f"{self.tensor} * {self.base_weight}"

    def __hash__(self):
        return hash(repr(self))
    
    def __eq__(self, other):
        if not isinstance(other, FactoredTensorWeight):
            return NotImplemented
        if not super().__eq__(other):
            return False
        if not (self.base_weight == other.base_weight):
            return False
        if not np.array_equal(self.tensor, other.tensor):
            return False
        return True

@dataclass
class TensorWeightBasis: # basis of TensorWeights to span desired space
    tw_list: list[TensorWeight]
    nonzero_indices: list[list[int]] # indices that need to be evaluated in product with this TensorWeight      
    
    @staticmethod
    def scalar_basis(base_weight, n_dimensions):
        tw = TensorWeight(weight_dict={(): base_weight}, rank=0, n_spatial_dim=n_dimensions)
        nz_inds = [()]
        return TensorWeightBasis([tw], nz_inds)
    
    @staticmethod
    def vector_basis(base_weight, n_dimensions):
        tw_list = []
        for i in range(n_dimensions): 
            tensor = np.zeros(shape=(n_dimensions,))
            tensor[i] = 1
            tw_list.append(FactoredTensorWeight.constant_tw(base_weight, tensor))
        nz_inds = [(i,) for i in range(n_dimensions)]
        return TensorWeightBasis(tw_list, nz_inds)

    # see https://physics.stackexchange.com/questions/770127/some-intuitions-for-irreducible-representations-so3-in-classical-physics
    @staticmethod
    def stf_rank2_basis(base_weight, n_dimensions): # form all basis elements for rank 2 symmetric trace-free irrep
        tw_list = []
        nz_inds = []
        it = multi_index_iterator(n_dimensions, rank=2)
        for x in it:
            i, j = it.multi_index
            tensor = np.zeros(shape=(n_dimensions, n_dimensions))
            if i==j:
                nz_inds.append((i, i))
                if i==0: # no corresponding basis element
                    continue
                tensor[0, 0] = 1
                tensor[i, i] = -1
                tw_list.append(FactoredTensorWeight.constant_tw(base_weight, tensor))
            elif i<j:
                tensor[i, j] = 1
                tensor[j, i] = 1
                tw_list.append(FactoredTensorWeight.constant_tw(base_weight, tensor))
                nz_inds += [(i, j), (j, i)]
        return TensorWeightBasis(tw_list, nz_inds)

    @staticmethod
    def anti_rank2_basis(base_weight, n_dimensions): # form all basis elements for rank 2 antisymmetric irrep
        tw_list = []
        nz_inds = []
        it = multi_index_iterator(n_dimensions, rank=2)
        for x in it:
            i, j = it.multi_index
            tensor = np.zeros(shape=(n_dimensions, n_dimensions))
            if i<j:
                tensor[i, j] = 1
                tensor[j, i] = -1
                tw_list.append(FactoredTensorWeight.constant_tw(base_weight, tensor))
                nz_inds += [(i, j), (j, i)]
        return TensorWeightBasis(tw_list, nz_inds)

    @staticmethod
    def full_basis(base_weight, n_dimensions, rank): # form all basis elements for rank n
        tw_list = []
        nz_inds = []
        it = multi_index_iterator(n_dimensions, rank)
        for x in it:
            tensor = np.zeros(shape=(n_dimensions,)*rank)
            tensor[it.multi_index] = 1
            tw_list.append(FactoredTensorWeight.constant_tw(base_weight, tensor))
            nz_inds.append(it.multi_index)
        return TensorWeightBasis(tw_list, nz_inds)

    @staticmethod
    def make_basis(base_weight, n_dimensions, irrep): # choose TWS constructor based on irrep
        match irrep:
            case 0:
                return TensorWeightBasis.scalar_basis(base_weight, n_dimensions)
            case 1:
                return TensorWeightBasis.vector_basis(base_weight, n_dimensions) # the general one should probably work fine
            case int():
                return TensorWeightBasis.full_basis(base_weight, n_dimensions, irrep)
            case FullRank(rank=rank):
                return TensorWeightBasis.full_basis(base_weight, n_dimensions, rank) if rank!=0 else \
                       TensorWeightBasis.scalar_basis(base_weight, n_dimensions)
            case Antisymmetric(rank=2):
                return TensorWeightBasis.anti_rank2_basis(base_weight, n_dimensions)
            case SymmetricTraceFree(rank=2):
                return TensorWeightBasis.stf_rank2_basis(base_weight, n_dimensions)
            case _: # shouldn't be reachable under normal conditions
                print("Warning: weird irrep")
                if irrep.rank == 0:
                    return TensorWeightBasis.scalar_basis(base_weight, n_dimensions)
                if irrep.rank == 1:
                    return TensorWeightBasis.vector_basis(base_weight, n_dimensions) # the general one should probably work fine too
                else:
                    warn(RuntimeWarning(f"Rank {irrep.rank} irreps may not be supported!")) 
                    return None  

def lists_for_N(nloops, loop_max):
    if nloops == 0:
        yield []
        return
    for i in range(loop_max + 1):
        for li in lists_for_N(nloops - 1, loop_max):
            yield [i] + li
            

# this class might be absorbed into SRDataset
@dataclass
class LibraryData(object):  # structures information associated with a given rank (or irrep in general) library
    terms: Iterable[LibraryTerm]
    irrep: Irrep
    Q: np.ndarray = None
    col_weights: Iterable[float] = None
    row_weights: Iterable[float] = None

    def clear_results(self): # create a copy of self without results computed
        return replace(self, Q=None, col_weights=None, row_weights=None)

#function for initializing global variables for each parallel worker process
def init_domain_worker(dataset_init, current_irrep_init, by_parts_init, debug_init):
    global worker_dataset, worker_current_irrep, worker_by_parts, worker_debug
    worker_dataset = dataset_init
    worker_current_irrep = current_irrep_init
    worker_by_parts = by_parts_init
    worker_debug = debug_init

#function to be executed in parallel to evaluate all terms for a given domain
def parallel_domain_task(domain):
    dataset = worker_dataset
    irrep = worker_current_irrep
    by_parts = worker_by_parts
    debug = worker_debug

    domain_results_dict = defaultdict(float)

    for t, w, term, tensor_weight in dataset.integrated_terms_tuples:
        if w.scale == 0:
            continue
        value = dataset.eval_on_domain(t,w,domain,debug=debug)
        key = term, tensor_weight
        domain_results_dict[key] += value
    return domain, domain_results_dict

@dataclass(kw_only=True)
class AbstractDataset(object): # template for structure of all data associated with a given sparse regression dataset
    world_size: List[float] # linear dimensions of dataset in physical units (spatial + time)
    data_dict: Dict[Observable, np.ndarray[float]] # observable -> array of values (e.g. discrete - particle, spatial index, time)
    observables: List[Observable]  # list of observables
    # storage of computed quantities: (prim, domains) [not dims] -> array
    cache_primes: bool = True # whether the field_dict is used
    field_dict: dict[tuple[Any, ...], np.ndarray[float]] = None 
    
    dxs: List[float] = None # grid spacings
    weight_dxs: List[float] = None
    scalar_weights: List[Weight] = None
    tensor_weight_basis: Dict[Tuple[Union[int, Irrep, Weight], ...], TensorWeightBasis] = field(default_factory=dict)  # (irrep, weight) -> stack
    # size of domain in grid units (NOT SUBGRID UNITS, AS ACTUALLY USED IN DISCRETE COMPUTATION)
    domain_size: List[float] = None 
    domains: List[IntegrationDomain] = None
    pad: float = 0
    libs: Dict[Union[int, Irrep], LibraryData] = None # irrep label (e.g. 0, 1, "2s" irrep) -> LibraryData object
    irreps: List[Union[int, str]] = (0, 1) # set of irreducible representations to generate libraries for = libs.keys()

    scale_dict: Dict[str, float] = None # dict of characteristic scales of observables -> (mean, std)
    xscale: float = 1  # length scale for correct computation of char scales (default 1)
    tscale: float = 1  # time scale

    metric: Metric = None # we support only constant coeff metrics for now
    metric_is_identity: bool = True

    integrated_terms_tuples: List[Tuple[LibraryTerm,Weight,LibraryTerm,TensorWeight]] = None

    def __post_init__(self):
        self.n_dimensions = len(self.world_size) # number of dimensions (spatial + temporal)
        # consider n_spatial_dim field
        self.field_dict = dict()
        if self.metric is None: 
            self.metric = Metric(n_dimensions=self.n_dimensions)
        else:
            self.metric_is_identity = False

    def resample(self): # should return SRD that is instance of implementing classes, so this is not type-hinted
        new_srd = replace(self, domains=None, libs={irrep: lib.clear_results() for irrep, lib in self.libs.items()})
        # remake domains
        new_srd.make_domains(ndomains=len(self.domains), domain_size=self.domain_size, pad=self.pad)
        # recompute Q etc.
        new_srd.make_library_matrices(debug=False)
        return new_srd

    @classmethod
    def all_rank2_irreps(cls):
        return (FullRank(rank=0), FullRank(rank=1), Antisymmetric(rank=2), SymmetricTraceFree(rank=2)) 

    @classmethod
    def only_rank2_irreps(cls):
        return (Antisymmetric(rank=2), SymmetricTraceFree(rank=2)) 
    
    def make_libraries(self, **kwargs): # populate libs
        pass

    def make_domains(self, ndomains, domain_size, pad=0): # set domain_size/populate domains
        pass

    def make_weights(self, m, qmax): # populate weights/set weight_dxs
        self.weights = []
        self.weight_dxs = [(width - 1) / 2 * dx for width, dx in zip(self.domain_size, self.dxs)]
        for q in lists_for_N(self.n_dimensions, qmax):
            weight = Weight([m] * self.n_dimensions, q, [0] * self.n_dimensions, dxs=self.weight_dxs)
            self.weights.append(weight)
            for irrep in self.irreps:
                # note that we need to count spatial dimensions for this
                self.tensor_weight_basis[irrep, weight] = TensorWeightBasis.make_basis(weight, self.n_dimensions-1, irrep)

    def get_index_assignments(self, term, tensor_weight, debug=False): # ONLY IMPLEMENTING FOR SIMPLER IDENTITY METRIC CASE
        n_spatial_dims = self.n_dimensions-1
        if self.metric_is_identity: # simplest evaluation - just sum over all assignments
            n_indices_to_assign = highest_index(term.all_indices())+1 if term.all_indices() else 0
            if debug:
                print("All indices:", term.all_indices())
                print("# of indices to assign:", n_indices_to_assign)
            for assignment in lists_for_N(n_indices_to_assign, n_spatial_dims-1): # assignments only along spatial indices
                if debug:
                    print("Index assignment:", assignment)
                assigned_term = term.map_all_indices(index_map=lambda idx:LiteralIndex(assignment[idx.value])) if assignment else term
                scalar_weight = tensor_weight[*tuple(assignment)[:term.rank]] #if assignment else scalar_weight
                yield assigned_term, scalar_weight
        else:
            raise NotImplemented
    
    def eval_on_domain(self, term, weight, domain, debug=False):
        #print('weight_array hash', hash(weight.get_weight_array(domain.shape).tostring()))
        term_weight_product = self.eval_term(term, domain, debug) * weight.get_weight_array(domain.shape)
        if debug:
            filtered_flat = list(filter(lambda x: x!=0, term_weight_product.flat))
            lenf = len(filtered_flat)
            if lenf==0:
                print("ARRAY IS 0")
            else:
                print("MIDDLE NZ VALUE OF ARRAY:", '{:.2E}'.format(filtered_flat[lenf//2])) # middle of the array
        result = int_arr(term_weight_product, dxs=self.dxs)
        if debug:
            print('Integrated result', result)
        return result

    # to be used in non-identity metric case
    def metric_effect(self, inds1, inds2):
        product = 1
        for ind1, ind2 in zip(inds1, inds2):
            product *= self.metric[ind1, ind2] 
        return product
    
    def eval_term(self, term, domain, debug=False): # evaluate a term on domain
        # term: LibraryTerm
        # domain: IntegrationDomain corresponding to where the term is evaluated
        # return the evaluated term on the domain grid
        product = np.ones(shape=domain.shape)
        if isinstance(term, ConstantTerm): # short-circuit
            return product
        #if debug:
        #    print(f"LibraryTerm {term}")
        for prime in term.primes:
        #    if debug:
        #        print(f"LibraryPrime {prime}")
            if self.cache_primes and (prime, domain) in self.field_dict.keys():  # field is "cached"
                data_slice = self.field_dict[prime, domain]
            else:
                data_slice = self.eval_prime(prime, domain)
                if self.cache_primes:
                    self.field_dict[prime, domain] = data_slice
            #print(product.shape, data_slice.shape)
            product *= data_slice
            # print(product[0, 0, 0])
        return product

    # evaluate prime on a domain - DIFFERENT IMPLEMENTATIONS for continuous and discrete!
    def eval_prime(self, prime, domain, *args): 
        pass

    # def trace(self, term_values, weight_values): # evaluate inner product of term evaluation and tensor weight using metric
    #     if self.metric_is_identity:
    #         return np.einsum('ij..., ij...->...', term_values, weight_values, optimize=True)
    #     else:
    #         return np.einsum('ij..., jk, ik...->...', term_values, self.metric, weight_values, optimize=True)

    # not sure if it'll explicitly get used
    # def shortcut_trace(self, product_values, tensor_weight): # evaluate inner product with factorizable TensorWeight
    #     if self.metric_is_identity:
    #         return np.einsum('ij..., ij->...', product_values, tensor_weight, optimize=True)
    #     else:
    #         return np.einsum('ij..., jk, ik->...', product_values, self.metric, tensor_weight, optimize=True)

    def make_Q(self, irrep, by_parts=True, debug=False): # compute Q matrix for given irrep
        #debug = True
        #by_parts = False
        cols_list = []
        for term in self.libs[irrep].terms:
            if debug:
                print("UNINDEXED TERM:")
                print(term)
            column = []
            if debug:
                term_symmetry = term.symmetry()
                print("Symmetry:", term_symmetry)
            wd_dict = defaultdict(int) # group by weight/domain, aggregate by assignment & integration by parts term
            for weight in self.weights:
                for tensor_weight in self.tensor_weight_basis[irrep, weight].tw_list:
                    if debug:
                        print("Tensor weight:", tensor_weight)
                    # compute weight(term) on each domain: w(t)|d = sum_(wi'(ti'))|d
                    for indexed_term, scalar_weight in self.get_index_assignments(term, tensor_weight, debug):
                        if debug:
                            print("ASSIGNMENTS:", term, "->")
                            print("Indexed term:", indexed_term)
                            print("Scalar weight:", scalar_weight)
                        for t, w in int_by_parts(indexed_term, scalar_weight, by_parts):
                            if w.scale == 0:
                                continue
                            if debug:
                                print("INT BY PARTS:", indexed_term, "->")
                                print("Integrated term:", t)
                                print("Integrated weight:", w)
                            for domain in self.domains:
                                #if debug:
                                #    print("Domain:", domain)
                                debug_this_value = (domain==self.domains[0] and weight==self.weights[0] and debug)
                                wd_dict[tensor_weight, domain] += self.eval_on_domain(t, w, domain, debug=debug_this_value)
                                if debug_this_value:
                                    print('I_TERM', t, 'I_WEIGHT', w, 'CURR RESULT', wd_dict[tensor_weight, domain])

                    # # fair question: should we integrate and then multiply by the tensor or multiply and then integrate?
                    # for domain in domains:
                    #     arr = self.shortcut_trace(pv_dict[tensor_weight, domain], tensor_weight.tensor)
                    #     column.append(int_arr(arr, self.dxs)) # integrate the product array
            for weight in self.weights:
                for tensor_weight in self.tensor_weight_basis[irrep, weight].tw_list:
                    for domain in self.domains:
                        column.append(wd_dict[tensor_weight, domain])
            cols_list.append(column)
        return np.array(cols_list).transpose() # convert to numpy array
    
    def make_Q_parallel(self, irrep, by_parts=True, debug=False, num_processors=None):
        
        init_args = (self, irrep, by_parts, debug)
        domains = self.domains
        all_results = []

        #precompute symbolic manipulations for parallel tasks
        self.integrated_terms_tuples = []
        for term in list(self.libs[irrep].terms):
            for weight in list(self.weights):
                for tensor_weight in self.tensor_weight_basis[(irrep, weight)].tw_list:
                    for indexed_term, scalar_weight in self.get_index_assignments(term,tensor_weight):
                        for t, w in int_by_parts(indexed_term, scalar_weight, by_parts):
                            self.integrated_terms_tuples.append((t,w,term,tensor_weight))

        #begin parallel task execution
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processors, initializer=init_domain_worker, initargs=init_args) as executor:
            results = executor.map(parallel_domain_task, domains)
            for result in results:
                all_results.append(result)
                  
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

        
    def make_library_matrices(self, by_parts=True, debug=False, parallel=False, num_processors=None): # compute LibraryData Q matrices
        for irrep in self.irreps:
            if debug:
                print(f"***RANK {irrep} LIBRARY***")
            if parallel:
                self.libs[irrep].Q = self.make_Q_parallel(irrep, by_parts, debug, num_processors)
            else:
                self.libs[irrep].Q = self.make_Q(irrep, by_parts, debug)
        self.find_scales()
        for irrep in self.irreps:
            self.libs[irrep].col_weights = [self.get_char_size(term) for term in self.libs[irrep].terms]
            #print('Irrep', irrep, '; weights', self.libs[irrep].col_weights)
        #self.find_row_weights()

    def find_scales(self, names=None): # find mean/std deviation of fields in data_dict that are in names
        pass

    def get_char_size(self, term): # get characteristic size of LibraryTerm 
        pass

    def set_LT_scale(self, L, T): # set correlation length/time -> implement another automatic function?
        self.xscale = L
        self.tscale = T

def get_slice(arr, domain):
    arr_slice = arr
    for (slice_dim, min_c, max_c) in zip(range(arr.ndim), domain.min_corner, domain.max_corner):
        idx = [slice(None)] * arr.ndim
        idx[slice_dim] = slice(min_c, max_c + 1)
        arr_slice = arr_slice[tuple(idx)]
    return arr_slice

def int_arr(arr, dxs=None):  # integrate an array of values on an integration domain
    if dxs is None:
        dxs = [1] * len(arr.shape)
    dx = dxs[0]
    integral = np.trapz(arr, axis=0)
    if len(dxs) == 1:
        return integral
    else:
        return int_arr(integral, dxs[1:])

def int_by_parts(term, weight, by_parts=True, dim=0):
    if weight.scale == 0 or not by_parts: # no point - the weight is zero anyway or we were asked not to
        yield term, weight
        return
    failed = False
    #print("START:", term, weight, dim)
    for te, we, fail in int_by_parts_dim(term, weight, dim):  # try to integrate by parts
        #print(te, we, fail)
        failed = (failed or fail)
        if failed:  # can't continue, so go to next dimension
            if dim=='t':
                yield term, weight
                return
            dim += 1
            if dim==weight.n_dimensions-1: # on the t index
                dim = 't'
        yield from int_by_parts(te, we, by_parts, dim)  # repeat process (possibly on next dimension)

# for integration by parts, check terms that look like x', x*x', and x*x*x' (vs all other terms have derivative orders
# smaller by at least 2) 
# maybe his misses out on opportunities to integrate by parts using a different basis, but this seems
# too difficult to automate; at that point it's probably easier to just write the term out manually.

# match statement for scalar or tensor weight?
def int_by_parts_dim(term, weight, dim, debug=False):
    #debug = True
    # find best prime to base integration off of
    best_prime, next_prime = None, None
    num_next = 0
    for prime in term.primes:
        if prime.nderivs == term.max_prime_derivatives():
            if debug:
                print("Found best prime:", prime)
            if best_prime is None:
                best_prime = prime
                if prime.derivs_along(dim) == 0:  # can't integrate by parts along this dimension
                    if debug:
                        print("Terminate: no derivatives along", dim)
                    yield term, weight, True
                    return
            else:  # multiple candidate terms -> integrating by parts will not help
                if debug:
                    print("Terminate: multiple best primes")
                yield term, weight, True
                return
        elif prime.nderivs == term.max_prime_derivatives() - 1:
            if next_prime is None:
                num_next, next_prime = 1, prime
            elif prime == next_prime:
                num_next += 1
            else:  # not all one-lower terms are successors of best_prime -> can't integrate further
                yield term, weight, True
                return
    if debug:
        print(dim, "&", weight, 'with dimensions', weight.n_dimensions)
    new_weight = weight.increment(dim if dim!='t' else weight.n_dimensions-1)
    new_prime = best_prime.antidiff(dim)
    if debug:
        print("TERM", term, "best_prime", best_prime)
        print("antidiffs to", new_prime)
        print("weight:", weight, '->', new_weight)
    rest = term.drop(best_prime)
    # check viability by cases
    if next_prime is None:  # then all other terms have derivatives up to order n-2, so we are in x' case
        if debug:
            print("Success: other derivative orders are n-2")
        for summand in rest.diff(dim):
            yield new_prime * summand, -weight, False
        #print(new_prime.derivative.x_derivatives, rest)
        yield new_prime * rest, -new_weight, False
        return
    else:
        if debug:
            print('n-1 case?')
            print('rest', rest, 'next_prime', next_prime)
        if next_prime==new_prime: # check if next goes with best
            if debug:
                print("Success: x'*x type case")
            #rest = rest.drop_all(next_prime)
            for i in range(num_next):
                rest = rest.drop(next_prime) # these are not differentiated
            num_dupes = 1 + num_next
            for summand in rest.diff(dim):
                yield ES_prod(*([next_prime]*num_dupes), summand), -1 / num_dupes * weight, False
            yield ES_prod(*([next_prime]*num_dupes), rest), -1 / num_dupes * new_weight, False
        else: # can't integrate by parts
            if debug:
                print("Terminate: next derivative too close and doesn't match")
            yield term, weight, True

def diff(data, dorders, dxs=None, acc=6):
    # for spatial directions can use finite differences or spectral differentiation. For time, only the former.
    # in any case, it's probably best to pre-compute the derivatives on the whole domains (at least up to order 2).
    # with integration by parts, there shouldn't be higher derivatives.
    if dxs is None:
        dxs = [1] * len(dorders)
    diff_list = []
    for i, dx, order in zip(range(len(dxs)), dxs, dorders):
        if order > 0:
            diff_list.append((i, dx, order))
    diff_operator = FinDiff(*diff_list, acc=acc)
    return diff_operator(data)