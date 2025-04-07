from __future__ import annotations

from dataclasses import dataclass, field, replace, KW_ONLY
from typing import List, Dict, Union, Tuple, Iterable, Generator
from functools import cached_property#, lru_cache#, reduce 
from operator import add
from collections import defaultdict, Counter

import re
import unicodedata

import numpy as np
from numpy import prod

from commons.z3base import *

# list of substitutions to go between plaintext and LaTeX output
latex_replacements = {'·': '\\cdot', '²': '^2', '³': '^3', '⁴': '^4', '⁵': '^5', '⁶': '^6',
                      '⁷': '^7', '⁸': '^8', '⁹': '^9', '∂': '\\partial'}
for letter in lowercase_greek_letters:
    latex_replacements[letter] = '\\'+ unicodedata.name(letter).split()[-1].lower()

def multiple_replace(string, rep_dict):
    pattern = re.compile("|".join([re.escape(k) for k in rep_dict]), flags=re.DOTALL)
    return pattern.sub(lambda x: rep_dict[x.group(0)], string)

def wrap_subscripts(string):
    return re.sub(r'_([A-z0-9\\]*)', r'_{\1}', string)

def latexify(string):
    step1 = re.sub(r'\\partial', r'\\partial_', wrap_subscripts(multiple_replace(string, latex_replacements)))
    return re.sub(r']}', r'}]', step1)

# increment all VarIndices in an expression
def inc_inds(expr: EinSumExpr[VarIndex | LiteralIndex], shift=1):
    return expr.map_all_indices(lambda ind: replace(ind, value=ind.value + shift) if isinstance(ind, VarIndex) else ind)

def canonicalize(expr: EinSumExpr[Index] | Equation):
    if isinstance(expr, Equation):
        return expr.canonicalize()
    if not expr.all_indices() or isinstance(expr.all_indices()[0], LiteralIndex):
        return expr.eq_canon()[0]
    #expr = secv_canon(expr)
    indexings = generate_indexings(expr, autocorrect=True) # correctly commute things when reindexing
    try:
        canon = next(indexings)
    except StopIteration:
        assert False, f"Didn't find any indexings for {expr} with rank {expr.rank} :("
    try:
        canon2 = next(indexings)
        assert False, f"Found multiple canonical indexings for {expr} with rank {expr.rank}: {canon} and {canon2}"
    except StopIteration:
        pass
    return canon
    #return canon.eq_canon()[0] # canonicalize structure too

def dt_fun(expr: EinSumExpr[VarIndex]):
    if isinstance(expr, ConstantTerm):
        return LibraryTerm(primes=(), rank=0)
    dted = dt_helper(expr)
    return canonicalize(dted) # make sure it's canonicalized

def dt_helper(expr: EinSumExpr[VarIndex]):
    match expr:
        case Observable():
            return LibraryPrime(derivative=DerivativeOrder.indexed_derivative(1, 0), derivand=expr)
        case DerivativeOrder(torder=to, x_derivatives=xd):
            return DerivativeOrder(torder=to+1, x_derivatives=xd)
        case LibraryPrime(derivative=derivative):
            return replace(expr, derivative=dt_helper(derivative))
        case LibraryTerm():
            subexs = list(expr.sub_exprs())
            # don't canonicalize since we're not in an equation context
            dts = [ES_safe_prod(*subexs[:i], dt_helper(term), *subexs[i+1:]) 
                       for i, term in enumerate(subexs)]
            return dts[0] if len(dts)==1 else ES_sum(*dts)
        case Equation(terms=terms, coeffs=coeffs):
            # note that derivative is an Equation object in general
            components = tuple([coeff * dt_helper(term) for term, coeff in zip(terms, coeffs)
                          if not isinstance(term, ConstantTerm)])
            return ES_sum(*components)

# construct term or equation by taking x derivative with respect to new i index, shifting others up by 1
# NOT GUARANTEED TO BE CANONICAL
def dx_fun(expr: EinSumExpr[VarIndex]):
    # the alternative implementation was to run dx and then use z3 solver to identify index labeling
    if isinstance(expr, ConstantTerm):
        return LibraryTerm(primes=(), rank=0)
    dxed = dx_helper(expr)
    return canonicalize(dxed) # make sure it's canonicalized

# take x derivative without worrying about indices
def dx_helper(expr: EinSumExpr[VarIndex]):
    match expr:
        case Observable():
            return LibraryPrime(derivative=DerivativeOrder.blank_derivative(0, 1), derivand=expr)
        case DerivativeOrder(torder=to, x_derivatives=xd):
            return DerivativeOrder(torder=to, x_derivatives=(IndexHole(), *xd))
        case LibraryPrime(derivative=d):
            return replace(expr, derivative=dx_helper(d))
        case LibraryTerm():
            subexs = list(expr.sub_exprs())
            # don't forget to increment rank by 1 for derivative & don't canonicalize yet - not eqn context
            dxs = [replace(ES_safe_prod(*subexs[:i], dx_helper(term), *subexs[i+1:]), rank=expr.rank+1) 
                       for i, term in enumerate(subexs)]
            dxs = list(dxs)
            return dxs[0] if len(dxs)==1 else ES_sum(*dxs) # return LibraryTerm if it's just one term in the sum
        case Equation():
            components = (coeff * dx_helper(term) for term, coeff in zip(expr.terms, expr.coeffs)
                          if not isinstance(term, ConstantTerm))
            components = list(components)
            return ES_sum(*components)#.canonicalize() # note that rank is already incremented from the term being handled

# contract term or equation along i and j indices, setting j to i (if i<j) and moving others down by 1
# NOT GUARANTEED TO BE CANONICAL
def contract(expr: EinSumExpr[VarIndex] | Equation[VarIndex], i: int, j: int):
    match expr:
        case Equation(terms=ts, coeffs=c):
            return Equation(terms=[contract(t, i, j) for t in ts], coeffs=c)
        case EinSumExpr():
            n_singles = index_rank(expr.all_indices()) 
            assert i<n_singles and j<n_singles, f"Can only contract single indices, not ({i}, {j}) in {expr}"
            new_n_singles = n_singles - 2
            new_double = new_n_singles
            if j<i:
                i, j = j, i
            def contraction_map(ind: VarIndex):
                if ind.value == i or ind.value == j:
                    # index_rank decreases by 2 as a result of contraction, so map new double to ir-1
                    return VarIndex(new_double) 
                if ind.value >= n_singles:
                    # beats 2-1 additional indices
                    return VarIndex(ind.value-1)
                if ind.value > j:
                    # beats 2 additional indices
                    return VarIndex(ind.value-2)
                if ind.value > i:
                    # beats 1 additional index
                    return VarIndex(ind.value-1)
                return ind
            reindexed_expr, sign = expr.map_all_indices(index_map=contraction_map).eq_canon()
            return replace(reindexed_expr, rank=expr.rank-2)

# cast a ConstantTerm or LibraryPrime to LibraryTerm
def cast_to_term(x: ConstantTerm | LibraryPrime | LibraryTerm):
    match x:
        case ConstantTerm():
            return LibraryTerm(primes=(), rank=0)
        case LibraryPrime():
            return LibraryTerm(primes=(x,), rank=x.rank)
        case LibraryTerm():
            return x
        case _:
            assert False, f"Can't cast to term for {type(x)} {x}"
    
# helper function for prime/library term multiplication w/ index collision resolution - OUTPUT IS CANONICAL
def ES_prod(*terms: ConstantTerm | LibraryPrime | LibraryTerm):
    #print("Product of terms:", terms)
    product_rank = 0
    combined_primes = []
    shift = 0
    #literal = False
    for t in terms:
        t = cast_to_term(t)
        product_rank += t.rank
        combined_primes += inc_inds(t, shift).primes
        shift += highest_index(t.all_indices())+1 # since we changed inc_inds to only increment VarIndices
    
    combined_primes = tuple(sorted(combined_primes))
    product = LibraryTerm(primes=combined_primes, rank=product_rank)
    return canonicalize(product) 

# multiplication without index collision resolution or canonicalization - used in derivatives only
def ES_safe_prod(*terms: ConstantTerm | LibraryPrime | LibraryTerm):
    combined_primes = []
    for t in terms:
        t = cast_to_term(t)
        combined_primes += t.primes
    combined_primes = tuple(sorted(combined_primes))
    product_wo_rank = LibraryTerm(primes=combined_primes, rank=0)
    product = replace(product_wo_rank, rank=index_rank(product_wo_rank.all_indices()))
    return product

# helper function for prime/library term addition - OUTPUT IS CANONICAL
def ES_sum(*equations: LibraryTerm | Equation):
    equations = list(equations)
    terms = tuple((term for eq in equations for term in (eq.terms if isinstance(eq, Equation) else (eq,))))
    coeffs = tuple((coeff for eq in equations for coeff in (eq.coeffs if isinstance(eq, Equation) else (1,))))
    return Equation(terms=terms, coeffs=coeffs)

# create a copy of the prime with a new set of derivative orders
def set_order_counts(p: Prime, counts: Counter[int]) -> Prime:
    torder = counts['t']
    ct_copy = counts.copy()
    ct_copy['t'] = 0
    x_derivatives = tuple(sorted(ct_copy.elements()))
    new_derivative = DerivativeOrder(torder=torder, x_derivatives=x_derivatives)
    return replace(p, derivative=new_derivative)

def parse_ind(dim: Union[int, str], literal: bool=True):
    match dim:
        case int():
            return LiteralIndex(value=dim) if literal else VarIndex(value=dim)
        case 't':
            return 't'
        case _:
            return NotImplemented

@dataclass(frozen=True)
class INF:
    def __lt__(self, other):
        return False
    def __gt__(self, other):
        return not isinstance(other, INF)
    def __le__(self, other):
        return isinstance(other, INF)
    def __ge__(self, other):
        return True

def exp_string(num):
    if num == 1:
        return ""
    elif num < 9:
        return "**²³⁴⁵⁶⁷⁸⁹"[num]
    else:
        return f"^{num}"

@dataclass(frozen=True)
class DerivativeOrder[T](EinSumExpr):
    """
    Object to store and manipulate derivative orders.
    """
    _: KW_ONLY
    torder: int = 0
    x_derivatives: tuple[T, ...]
    can_commute_indices: bool = True

    @cached_property
    def complexity(self):
        return self.torder+self.xorder

    @cached_property
    def xorder(self):
        return len(self.x_derivatives)

    @classmethod
    def blank_derivative(cls, torder, xorder):
    # make an abstract x derivative with given orders
        x_derivatives = tuple([IndexHole()]*xorder)
        return DerivativeOrder[IndexHole](torder=torder, x_derivatives=x_derivatives)

    @classmethod
    def indexed_derivative(cls, torder, xorder):
        x_derivatives = tuple([VarIndex(i) for i in range(xorder)])
        return DerivativeOrder[VarIndex](torder=torder, x_derivatives=x_derivatives)

    def __repr__(self):
        if self.torder == 0:
            tstring = ""
        else:
            tstring = f"∂t{exp_string(self.torder)} "
        xstring = ""
        if self.xorder != 0:
            ind_counter = self.get_spatial_orders()
            for ind in sorted(ind_counter.keys()):
                xstring += f"∂{ind}{exp_string(ind_counter[ind])} "
        return (tstring + xstring)[:-1] # get rid of the trailing space

    # @lru_cache
    def get_spatial_orders(self) -> Counter:
        return Counter(self.x_derivatives)

    def get_all_orders(self) -> Counter:
        counter = self.get_spatial_orders()
        counter['t'] = self.torder
        return counter

    def __lt__(self, other):
        if not isinstance(other, DerivativeOrder):
            return NotImplemented
        return self.torder < other.torder if self.torder != other.torder \
            else self.x_derivatives < other.x_derivatives

    def __le__(self, other):
        return self < other or self == other

    def sub_exprs(self) -> Iterable[T]:
        return ()

    def own_indices(self) -> Iterable[T]:
        return self.x_derivatives

    def map[T2](self, *,
                expr_map: Callable[[EinSumExpr[T]], EinSumExpr[T2]] = lambda x: x,
                index_map: Callable[[T], T2] = lambda x: x) -> EinSumExpr[T2]:
        """ Constructs a copy of self replacing (direct) child expressions according to expr_map
            and (direct) child indices according to index_map"""
        return replace(self, x_derivatives=tuple([index_map(index) for index in self.own_indices()]))

    def eq_canon(self):
        xd = tuple(sorted(self.x_derivatives))
        return DerivativeOrder(torder=self.torder, x_derivatives=xd), 1

@dataclass(frozen=True)
class Observable[T](EinSumExpr):
    """
    Data class object that stores a string representation of an observable as well as its rank. For
    documentation purposes, this class will always be refered to as 'Observable' (capitalized), unless
    stated otherwise.
    """
    _: KW_ONLY
    string: str  # String representing the Observable.
    rank: Union[int, SymmetryRep]
    indices: Tuple[T, ...] = None
    can_commute_indices: bool = False # set to true for symmetric or antisymmetric
    antisymmetric: bool = False

    @cached_property
    def complexity(self):
        return max(self.rank, 1)

    # For sorting: convention is in ascending order of name

    def __lt__(self, other):
        if not isinstance(other, Observable):
            raise TypeError(f"Operation not supported between instances of '{type(self)}' and '{type(other)}'")
        return self.string < other.string if self.string != other.string \
            else tuple(self.all_indices()) < tuple(other.all_indices())

    def __repr__(self):
        index_string = ''.join([repr(idx) for idx in self.all_indices()])
        return f"{self.string}" if index_string == "" else f"{self.string}_{index_string}"

    def sub_exprs(self) -> Iterable[T]:
        return []

    def own_indices(self) -> Iterable[T]:
        return tuple([IndexHole()]*self.rank) if self.indices is None  \
               else self.indices

    def map[T2](self, *,
                expr_map: Callable[[EinSumExpr[T]], EinSumExpr[T2]] = lambda x: x,
                index_map: Callable[[T], T2] = lambda x: x) -> EinSumExpr[T2]:
        """ Constructs a copy of self replacing (direct) child expressions according to expr_map
            and (direct) child indices according to index_map"""
        return replace(self, indices=tuple([index_map(index) for index in self.own_indices()]))

    def eq_canon(self):
        inds = self.indices
        if self.can_commute_indices:
            inds = tuple(sorted(inds))
        sign = parity(self.indices, inds) if self.antisymmetric else 1
        return Observable(string=self.string, indices=inds, rank=self.rank, 
                          can_commute_indices=self.can_commute_indices, antisymmetric=self.antisymmetric), sign

def parity(old_list, new_list): # return -1 for odd permutation, +1 for even
    zipped = list(zip(old_list, new_list))
    return prod([-1 for (ox,nx) in zipped for (oy,ny) in zipped if ox<oy and nx>ny], initial=1)

@dataclass(frozen=True)
class ConstantTerm(Observable):
    """ Short-hand for constant term = 1 """

    string: str = "1"
    rank: int = 0

    def __repr__(self):
        return "1"

    def dx(self):
        return 0

    def dt(self):
        return 0

    def __mul__(self, other):
        return other

    def __rmul__(self, other):
        return other

    def eq_canon(self):
        return self, 1

    def derivs_along(self, dim): # number of derivatives along 'dim' index
        return 0

@dataclass(frozen=True)
class LibraryPrime[T, Derivand](EinSumExpr):
    """
    Dataclass representing DerivativeOrder applied to a Derivand (e.g. Observable, CGP)
    """
    _: KW_ONLY
    derivative: DerivativeOrder
    derivand: Derivand
    can_commute_exprs: bool = False

    @cached_property
    def complexity(self):
        return self.derivative.complexity+self.derivand.complexity

    @cached_property
    def rank(self): # now defined for either VarIndex or IndexHole
        return index_rank(self.all_indices())

    # For sorting: convention is in ascending order of name of derivand, then derivative

    def __lt__(self, other):
        if not isinstance(other, LibraryPrime):
            raise TypeError(f"Operation not supported between instances of '{type(self)}' and '{type(other)}'")
        
        # continuous case
        if hasattr(self.derivand, 'string') and self.derivand.string != other.derivand.string:
            return self.derivand.string < other.derivand.string
        # discrete case
        elif hasattr(self.derivand, 'observables') and self.derivand.observables != other.derivand.observables: 
            return self.derivand.observables < other.derivand.observables
            
        self_orders = (self.derivative.torder, self.derivative.xorder)
        other_orders = (other.derivative.torder, other.derivative.xorder)
        if self_orders != other_orders:
            return self_orders < other_orders
        self_derivs = self.derivative.x_derivatives  
        other_derivs = other.derivative.x_derivatives
        if self_derivs != other_derivs:
            return self_derivs < other_derivs
        return self.derivand.all_indices() < other.derivand.all_indices()

    def __repr__(self):
        string1 = repr(self.derivative)
        string2 = repr(self.derivand)
        return f"{string1 + " " if string1 else ""}{string2}"

    def sub_exprs(self) -> Iterable[T]:
        return (self.derivative, self.derivand)

    def own_indices(self) -> Iterable[T]:
        return ()

    def map[T2](self, *,
                expr_map: Callable[[EinSumExpr[T]], EinSumExpr[T2]] = lambda x: x,
                index_map: Callable[[T], T2] = lambda x: x) -> EinSumExpr[T2]:
        """ Constructs a copy of self replacing (direct) child expressions according to expr_map
            and (direct) child indices according to index_map"""
        return replace(self, derivative=expr_map(self.derivative), derivand=expr_map(self.derivand))

    def __mul__(self, other: Union[LibraryPrime, LibraryTerm, Equation]) -> Union[LibraryTerm, Equation]:
        if isinstance(other, Equation): # prime * Equation case is handled within Equation class
            return other * self
        else:
            return ES_prod(self, other)

    def eq_canon(self):
        derivand_ec, sign = self.derivand.eq_canon()
        return LibraryPrime(derivative=self.derivative.eq_canon()[0], derivand=derivand_ec), sign

    @cached_property
    def nderivs(self):
        return self.derivative.xorder+self.derivative.torder

    def derivs_along(self, dim: Union[int, str]) -> int: # number of derivatives along 'dim' index
        return self.derivative.get_all_orders()[parse_ind(dim, literal=True)]

    # (REDUNDANT)
    # def succeeds(self, other: LibraryPrime, dim: Union[int, str]) -> bool: # check if d/d(dim) other==self
    #     if self.derivand != other.derivand:
    #         return False
    #     self_ords = self.derivative.get_all_orders()
    #     other_ords = other.derivative.get_all_orders()
    #     diff = self_ords-other_ords
    #     if len(diff.keys())!=1 or diff.total()!=1: # there should only be one extra index
    #         return False
    #     # check that only derivative in difference is w.r.t dim
    #     return diff.keys()[0] == 't' if dim == 't' else diff.keys()[0].value == dim

    def diff(self, dim: Union[int, str], literal: bool=True) -> LibraryPrime: # add a derivative w.r.t. VarIndex or LiteralIndex dim (or t)
        deriv_counter = self.derivative.get_all_orders()
        index = parse_ind(dim, literal)
        deriv_counter[index] += 1
        return set_order_counts(self, deriv_counter)
        #return replace(self, derivative=replace(derivative, x_derivatives=sorted(x_derivatives+(index,))))

    # remove a derivative w.r.t. VarIndex or LiteralIndex dim (or t)
    def antidiff(self, dim: Union[int, str], literal: bool=True) -> LibraryPrime: 
        deriv_counter = self.derivative.get_all_orders()
        index = parse_ind(dim, literal)
        deriv_counter[index] -= 1
        #print(self, "->", set_order_counts(self, deriv_counter))
        return set_order_counts(self, deriv_counter)
        #old_derivs = self.derivative.x_derivatives
        #position = old_derivs.index(index)
        #return replace(self, derivative=replace(derivative, x_derivatives=old_derivs[:position]+old_derivs[position+1:]))

@dataclass(frozen=True, order=True)
class LibraryTerm[T, Derivand](EinSumExpr):
    """
    Dataclass representing DerivativeOrder applied to a Derivand (e.g. Observable, CGP)
    """
    _: KW_ONLY
    primes: Tuple[LibraryPrime[T, Derivand], ...]
    rank: Union[int, SymmetryRep]

    @cached_property
    def complexity(self):
        return sum((prime.complexity) for prime in self.primes)

    #@lru_cache
    def symmetry(self, free_ind1=0, free_ind2=1) -> Optional[Int]:
        def transpose(ind):
            if ind.value == free_ind1:
                return VarIndex(free_ind2)
            elif ind.value == free_ind2:
                return VarIndex(free_ind1)
            else:
                return ind
            
        transposed_canon, sign = self.map_all_indices(transpose).eq_canon()
        #print(self, transposed_canon)
        #print(self.primes[0].derivative.x_derivatives, transposed_canon.primes[0].derivative.x_derivatives)
        return sign if self==transposed_canon else None

    #@cached_property
    #def rank(self): # only defined for VarIndex at the moment
    #    return index_rank(self)

    # For sorting: convention is in ascending order of name

    # def __lt__(self, other):
    #     if not isinstance(other, Observable):
    #         raise TypeError(f"Operation not supported between instances of '{type(self)}' and '{type(other)}'")
    #     return self. < other.string if self.string != other.string \
    #         else tuple(self.all_indices()) < tuple(other.all_indices())


    def __repr__(self):
        return ' · '.join([repr(prime) for prime in self.primes])

    def sub_exprs(self) -> Iterable[T]:
        return self.primes

    def own_indices(self) -> Iterable[T]:
        return ()

    def map[T2](self, *,
                expr_map: Callable[[EinSumExpr[T]], EinSumExpr[T2]] = lambda x: x,
                index_map: Callable[[T], T2] = lambda x: x) -> EinSumExpr[T2]:
        """ Constructs a copy of self replacing (direct) child expressions according to expr_map
            and (direct) child indices according to index_map"""
        return replace(self, primes=tuple(expr_map(prime) for prime in self.primes))

    def __add__(self, other): # add to LibraryTerm or Equation
        return ES_sum(self, other)
        #if isinstance(other, LibraryTerm):
        #    return Equation(coeffs=(1, 1), terms=(self, other))
        #else:
        #    return Equation(coeffs=(1, *other.coeffs), terms=(self, *other.terms))

    def __mul__(self, other: Union[LibraryPrime, LibraryTerm, float, int]) -> Union[LibraryTerm, Equation]:
        match other:
            case LibraryPrime() | LibraryTerm():
                return ES_prod(self, other)
            case float() | int():
                return Equation(terms=(self,), coeffs=(other,))

    __rmul__ = __mul__

    def eq_canon(self):
        ecs = [prime.eq_canon() for prime in self.primes]
        sign = prod([pair[1] for pair in ecs], initial=1)
        return LibraryTerm(primes=tuple(sorted([pair[0] for pair in ecs])), rank=self.rank), sign

    def drop(self, prime: LibraryPrime) -> LibraryTerm: # pop one copy of given prime from term
        index = self.primes.index(prime)
        return replace(self, primes=self.primes[:index]+self.primes[index+1:])

    def max_prime_derivatives(self): # max number of derivatives among any prime
        return max(prime.nderivs for prime in self.primes)

    def diff(self, dim: Union[int, str], literal: bool=True) -> Generator[LibraryTerm]: # derivative of term wrt specific index
        yield from (ES_prod(*self.primes[:i], prime.diff(dim, literal), *self.primes[i+1:])
                   for i, prime in enumerate(self.primes))

# NOTE: HIGHER-RANK TERMS IN SUM NOT GUARANTEED TO BE CANONICAL - CANONICALIZE WHEN SAMPLING
class Equation[T, Derivand]:  # can represent equation (expression = 0) OR expression
    def __init__(self, terms, coeffs):  # terms are LibraryTerms, coeffs are real numbers
        content = zip(terms, coeffs)
        coeffs_dict = defaultdict(int)
        for term, coeff in content:
            coeffs_dict[term] += coeff
        # remove terms with 0 coefficient
        coeffs_dict = {term: coeff for term, coeff in coeffs_dict.items() if coeff != 0}
        # note that sorting guarantees canonicalization in equation term order
        self.terms = tuple(sorted(coeffs_dict.keys()))
        self.coeffs = tuple(coeffs_dict[term] for term in self.terms)
        self.rank = terms[0].rank if terms else 0
        self.complexity = sum([term.complexity for term in terms])  # another choice is simply the number of terms
        #self.canonicalize() # too risky to canonicalize in constructor, do it manually

    def __add__(self, other):
        return ES_sum(self, other).canonicalize()

    def __mul__(self, other):
        if isinstance(other, EinSumExpr): # multiplication by term or castable to term
            # note: need to canonicalize term * other and as well as the end
            return Equation([ES_prod(term, other) for term in self.terms], self.coeffs).canonicalize()
        else:  # multiplication by number
            return Equation(self.terms, [c * other for c in self.coeffs])

    def __rmul__(self, other): # equation multiplication is commutative
        return self.__mul__(other)

    def __repr__(self):
        repstr = ' + '.join([str(coeff) + ' · ' + str(term) if coeff != 1 else str(term)
                             for coeff, term in zip(self.coeffs, self.terms)])
        return repstr

    def __str__(self):
        return self.__repr__() + " = 0"

    def pstr(self, num_format: str='{0:.3g}', latex_output: bool=False) -> str: # pretty string representation
        pretty_str = ' + '.join([num_format.format(coeff) + ' · ' + str(term) if coeff != 1 else str(term)
                             for coeff, term in zip(self.coeffs, self.terms)]) + " = 0"

        return latexify(pretty_str) if latex_output else pretty_str

    def __eq__(self, other):
        return self.terms == other.terms and self.coeffs == other.coeffs

    def canonicalize(self):
        #print("start of equation canonicalization:", self)
        dx_hole = VarIndex(-1, src=VarIndex(-1)) # for tracking sources in the one hole per term case (dx)
        coeffs = defaultdict(int)
        def update_source(idx):
            new_idx = replace(idx, src=idx) if not isinstance(idx, IndexHole) else dx_hole
            #print('idx', idx, 'src', idx.src if not isinstance(idx, IndexHole) else None, 
            #      'new_idx', new_idx, 'new_src', new_idx.src)
            return new_idx
        for term, coeff in zip(self.terms, self.coeffs):
            #term = term.map_all_indices( # set the source of index as itself
            #    lambda idx: (replace(idx, src=idx) if not isinstance(idx, IndexHole) else dx_hole))
            term = term.map_all_indices(update_source)
            coeffs[term] += coeff
            #print('term at start:', term)
            #print('indices', term.all_indices(), 'srcs', [idx.src for idx in term.all_indices()])
        # canonicalize terms, removing those with 0 coefficient
        # MIGHT BE EQ_CANON?
        
        # def crashless_canonicalize(term):
        #     try:
        #         canon = canonicalize(term)
        #     except AssertionError as ae:
        #         print(f"Canonicalization of {term} failed. Reason:")
        #         print(ae)
        #         return term
        #coeffs = {crashless_canonicalize(term): coeff
        #          for term, coeff in coeffs.items() if coeff != 0}
        
        def eq_canonicalize(coeffs):
            for term, coeff in coeffs.items():
                if coeff != 0:
                    term, sign = term.eq_canon()
                    yield term, sign*coeff
        coeffs = {canonicalize(term): coeff for term, coeff in eq_canonicalize(coeffs)}    

        if not coeffs:
            self.terms, self.coeffs = (), ()
            return

        #print('new terms after first canonicalization:', coeffs.keys())

        # get minimum term for standardizing free indices,
        # e.g. dx(a_\alpha * b) = d\alpha a_\beta * b + a_\beta * d\alpha b
        first : LibraryTerm = min(coeffs.keys())
        indices = list(first.all_indices())
        has_omega = any((idx.src is not None and idx.src.value < 0) for idx in indices)
        #print(indices, "has omega?", has_omega)
        
        key_map = {idx.src : idx for idx in indices
                                 if idx.src and idx.src.value < self.rank - has_omega}

        def mapper(idx):
            return key_map.get(idx.src, idx)
        #print("key_map", key_map)
        # reindex terms using free indices of first
        coeffs = {
            term.map_all_indices(mapper) : coeff
            for term, coeff in coeffs.items()
        }
        #print("->", coeffs.keys())
        # re-sort terms based on their updated indices
        self.terms, self.coeffs = zip(*sorted(coeffs.items()))

        # return canonicalized self as necessary
        return self

    def map[T2](self, *,
                expr_map: Callable[[EinSumExpr[T]], EinSumExpr[T2]] = lambda x: x,
                index_map: Callable[[T], T2] = lambda x: x) -> EinSumExpr[T2]:
        """ Constructs a copy of self replacing (direct) child expressions according to expr_map
            and (direct) child indices according to index_map"""
        return Equation(terms=tuple(expr_map(term) for term in self.terms), coeffs=self.coeffs)

    def map_all_indices[T2](self, index_map: Callable[[T], T2]) -> Equation[T2]:
        def mapper(expr):
            nonlocal index_map
            return expr.map(expr_map=mapper, index_map=index_map)
        return mapper(self)

    # note that LHS should be canonicalized if needed for lookup
    def eliminate_complex_term(self, return_normalization=False):
        if len(self.terms) == 1:
            return self.terms[0], None
        lhs = max(self.terms, key=lambda t: t.complexity)
        lhs_ind = self.terms.index(lhs)
        new_terms = self.terms[:lhs_ind] + self.terms[lhs_ind + 1:]
        new_coeffs = self.coeffs[:lhs_ind] + self.coeffs[lhs_ind + 1:]
        new_coeffs = [-c / self.coeffs[lhs_ind] for c in new_coeffs]
        rhs = Equation(new_terms, new_coeffs)
        #print("LHS", lhs, "->", canonicalize(lhs))
        if return_normalization:
            return canonicalize(lhs), rhs, self.coeffs[lhs_ind]
        return canonicalize(lhs), rhs

    def to_term(self):
        if len(self.terms) != 1:
            raise ValueError("Equation contains more than one distinct term")
        else:
            return canonicalize(self.terms[0]) # may need structural canonicalization too - check

def partition(n: int, k: int, weights: Optional[Tuple[int, ...]]) -> Generator[Tuple[int, ...], None, None]:
    """
    Given k bins (represented by a k-tuple), it yields every possible way to distribute x elements among those bins,
    with x ranging from 0 to n. For example partition(n=3, k=2) -> [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1),
    (1, 2), (2, 0), (2, 1), (3, 0)].
    Optional argument weights indicates that one element in bin i counts for weight[i] elements. 
    NOTE: partition(n, 0) returns None, and partition(n, 1) is similar to range(n + 1), but the yields are wrapped in a
    1-tuple.

    :param n: Max number of elements to distribute.
    :param k: Number of bins to distribute.
    :return: Generator that yields all possible partitions.
    """
    if weights is None:
        weights = [1]*k
    if k < 1:
        return
    max = n // weights[0] + 1
    if k == 1:
        for i in range(max):
            yield i,
        return
    for i in range(max):
        for result in partition(n - weights[0] * i, k - 1, weights[1:]):
            yield (i,) + result