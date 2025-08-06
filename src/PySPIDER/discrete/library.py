import copy
from functools import cached_property # reduce
#from operator import add
#from itertools import permutations
from numpy import prod
import numpy as np
from collections import Counter
from typing import Union, Callable, Iterable, Generator # Tuple, List,
from dataclasses import dataclass, replace, KW_ONLY

from ..commons.z3base import (
    EinSumExpr, index_rank, generate_indexings # VarIndex, IndexHole
)
from ..commons.library import (
    Observable, DerivativeOrder, LibraryPrime, LibraryTerm, ConstantTerm, partition
)

@dataclass(frozen=True, order=True)
class CoarseGrainedProduct[T](EinSumExpr):
    """
    Dataclass representing rho[product]
    """
    _: KW_ONLY
    observables: tuple[Observable]

    @cached_property
    def complexity(self):
        return 1+sum((observable.complexity) for observable in self.observables)
    pass

    @cached_property
    def rank(self): # only defined for VarIndex at the moment
        return index_rank(self.all_indices())

    def __repr__(self):
        return f"ρ[{' · '.join([repr(obs) for obs in self.observables])}]" \
               if len(self.observables)>0 else "ρ"

    def sub_exprs(self) -> Iterable[T]:
        return self.observables

    def own_indices(self) -> Iterable[T]:
        return ()

    def map[T2](self, *,
                expr_map: Callable[[EinSumExpr[T]], EinSumExpr[T2]] = lambda x: x,
                index_map: Callable[[T], T2] = lambda x: x) -> EinSumExpr[T2]:
        """ Constructs a copy of self replacing (direct) child expressions according to expr_map
            and (direct) child indices according to index_map"""
        return replace(self, observables=tuple(expr_map(obs) for obs in self.observables))

    def eq_canon(self):
        ecs = [obs.eq_canon() for obs in self.observables]
        sign = prod([pair[1] for pair in ecs], initial=1)
        return CoarseGrainedProduct(observables=tuple(sorted([pair[0] for pair in ecs]))), sign

def generate_terms_to(max_complexity: int, observables: list[Observable],
                      max_rank: int = 2, max_observables: int = 999, max_rho: int = 999,
                      max_dt: int = 999, max_dx: int = 999,
                      max_observable_counts: dict[Observable, int] = None, **kwargs) -> \
                      list[Union[ConstantTerm, LibraryTerm]]:
    """
    Given a list of Observable objects and a complexity order, returns the list of all LibraryTerms 
    with complexity up to order and rank up to max_rank using at most max_observables copies of the observables.

    :param max_complexity: Max complexity order that terms will be generated to.
    :param observables: list of Observable objects used to construct the terms.
    :param max_rank: maximum rank of a term to construct.
    :param max_observables: Maximum number of Observables in a single term.
    :param max_rho: Maximum number of primes (rhos) in a single term.
    :param max_observable_counts: Maximum count of each Observable in a single term.
    :param max_dt: Maximum t derivative order in a term.
    :param max_dx: Maximum x derivative order in a term.
    :return: List of all possible LibraryTerms whose complexity is less than or equal to max_complexity 
    that can be generated using the given observables.
    """
    max_observable_counts = Counter({obs: 999 for obs in observables}) if max_observable_counts is None \
                            else Counter(max_observable_counts)
    
    libterms = list()
    n = max_complexity  # max number of "blocks" to include
    k = len(observables)
    partitions = [] # to make sure we don't duplicate partitions
    weights = [obs.complexity for obs in observables] + [1, 1] # complexities of each symbol
    # generate partitions in bijection to all possible primes
    for part in partition(n - 1, k + 2, weights=weights):  # k observables + 2 derivative dimensions, plus always 1 rho
        # account for complexities > 1
        if sum(part[:k]) <= max_observables and part[-2]<=max_dt and part[-1]<=max_dx:
            partitions.append(part)

    def partition_to_prime(partition):
        prime_observables = []
        for i in range(k):
            prime_observables += [observables[i]] * partition[i]
        cgp = CoarseGrainedProduct(observables=tuple(prime_observables))
        derivative = DerivativeOrder.blank_derivative(torder=partition[-2], xorder=partition[-1])
        prime = LibraryPrime(derivative=derivative, derivand=cgp)
        return prime
    
    partitions = sorted(partitions)
    primes = [partition_to_prime(partition) for partition in partitions]
    #for pa, pr in zip(partitions, primes):
    #    print(pa, pr)

    # make all possible lists of primes and convert to terms of each rank, then generate labelings
    for prime_list in valid_prime_lists(primes, max_complexity, max_observables, max_rho, max_observable_counts):
        parity = sum(len(prime.all_indices()) for prime in prime_list) % 2
        for rank in range(parity, max_rank + 1, 2):
            term = LibraryTerm(primes=prime_list, rank=rank)
            for labeled in generate_indexings(term):
                # terms should already be in canonical form except eq_canon
                libterms.append(labeled.eq_canon()[0]) 
    return libterms

def valid_prime_lists(primes: list[LibraryPrime],
                      max_complexity: int,
                      max_observables: int,
                      max_rho: int,
                      max_observable_counts: Counter, 
                      non_empty: bool = False) -> Generator[tuple[LibraryPrime, ...], None, None]:
    # starting_ind: int
    """
    Generate components of valid terms from list of primes, with maximum complexity = max_complexity, maximum number of observables = max_observables, max number of primes = max_rho.
    """
    # , and using only primes starting from index starting_ind.
    # base case: yield no primes
    if non_empty:
        yield ()
    for i, prime in enumerate(primes): # relative_i
        complexity = prime.complexity
        n_observables = len(prime.derivand.observables)
        observable_counts = Counter(prime.derivand.observables)
        if complexity <= max_complexity and n_observables <= max_observables and \
           1 <= max_rho and observable_counts <= max_observable_counts:
            max_observable_counts -= observable_counts # temporarily modify the dictionary
            for tail in valid_prime_lists(primes=primes[i:], max_complexity=max_complexity-complexity,
                                          max_observables=max_observables-n_observables, max_rho=max_rho-1,
                                          max_observable_counts=max_observable_counts, non_empty=True):
                yield (prime,) + tail
            max_observable_counts += observable_counts