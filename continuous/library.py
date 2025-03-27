import copy
from functools import reduce
from numbers import Real
from operator import add
from typing import Any, Optional
from warnings import warn

#from numpy import inf

from commons.z3base import *
from commons.library import *

def generate_terms_to(order: int,
                      observables: Iterable[Observable],
                      max_rank: int = 2,
                      max_observables: int = 999) -> List[Union[ConstantTerm, LibraryTerm]]:
    """
    Given a list of Observable objects and a complexity order, returns the list of all LibraryTerms with complexity up to order and rank up to max_rank using at most max_observables copies of the observables.

    :param order: Max complexity order that terms will be generated to.
    :param observables: list of Observable objects used to construct the terms.
    :param max_observables: Maximum number of Observables (and derivatives) in a single term.
    :return: List of all possible LibraryTerms whose complexity is less than or equal to order, that can be generated
    using the given observables.
    """
    libterms = list()
    n = order  # max number of "blocks" to include
    k = len(observables)
    pairs = [] # to make sure we don't duplicate partitions
    weights = [obs.complexity for obs in observables] # complexities of each observable
    # generate partitions in bijection to all possible primes
    for i in range(k):
        for part in partition(n-weights[i], 2, weights=(1, 1)):  # ith observable + 2 derivative dimensions
            pairs.append((observables[i], part))

    def pair_to_prime(observable, part):
        derivative = DerivativeOrder.blank_derivative(torder=part[0], xorder=part[1])
        prime = LibraryPrime(derivative=derivative, derivand=observable)
        return prime
    
    pairs = sorted(pairs)
    primes = [pair_to_prime(observable, part) for (observable, part) in pairs]

    # make all possible lists of primes and convert to terms of each rank, then generate labelings
    for prime_list in valid_prime_lists(primes, order, max_observables):
        parity = sum(len(prime.all_indices()) for prime in prime_list) % 2
        for rank in range(parity, max_rank + 1, 2):
            term = LibraryTerm(primes=prime_list, rank=rank)
            for labeled in generate_indexings(term):
                # terms should already be in canonical form except eq_canon
                libterms.append(labeled.eq_canon()[0]) 
    return libterms

def valid_prime_lists(primes: List[LibraryPrime],
                      order: int,
                      max_observables: int,
                      non_empty: bool = False) -> List[Union[ConstantTerm, LibraryTerm]]:
    # starting_ind: int
    """
    Generate components of valid terms from list of primes, with maximum complexity = order, maximum number of observables = max_observables, max number of primes = max_rho.
    """
    # , and using only primes starting from index starting_ind.
    # base case: yield no primes
    if non_empty:
        yield ()
    for i, prime in enumerate(primes): # relative_i
        complexity = prime.complexity
        if complexity <= order and 1 <= max_observables:
            for tail in valid_prime_lists(primes=primes[i:], order=order-complexity,
                                          max_observables=max_observables-1,
                                          non_empty=True):
                yield (prime,) + tail
