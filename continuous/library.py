import copy
from functools import reduce
from numbers import Real
from operator import add
from typing import Any, Optional
from warnings import warn
from collections import defaultdict

#from numpy import inf

from commons.z3base import *
from commons.library import *

def generate_terms_to(order: int, observables: Iterable[Observable],
                      max_rank: int = 2, max_observables: int = 999,
                      max_observable_counts: dict[Observable, int] = None, max_dt: int = 999, max_dx: int = 999,
                      ) -> List[Union[ConstantTerm, LibraryTerm]]:
    """
    Given a list of Observable objects and a complexity order, returns the list of all LibraryTerms with complexity up to order and rank up to max_rank using at most max_observables copies of the observables.

    :param order: Max complexity order that terms will be generated to.
    :param observables: list of Observable objects used to construct the terms.
    :param max_rank: maximum rank of a term to construct.
    :param max_observables: Maximum number of Observables in a single term.
    :param max_observable_counts: Maximum count of each Observable in a single term.
    :param max_dt: Maximum t derivative order in a term.
    :param max_dx: Maximum x derivative order in a term.
    :return: List of all possible LibraryTerms whose complexity is less than or equal to order, that can be generated
    using the given observables.
    """
    max_observable_counts = defaultdict(lambda: 999) if max_observable_counts is None else max_observable_counts
    
    libterms = list()
    n = order  # max number of "blocks" to include
    k = len(observables)
    pairs = [] # to make sure we don't duplicate partitions
    weights = [obs.complexity for obs in observables] # complexities of each observable
    # generate partitions in bijection to all possible primes
    for i in range(k):
        for part in partition(n-weights[i], 2, weights=(1, 1)):  # ith observable + 2 derivative dimensions
            if part[0]<=max_dt and part[1]<=max_dx: # ensure that highest derivative along spatial/time axes does not exceed max
                pairs.append((observables[i], part))

    def pair_to_prime(observable, part):
        derivative = DerivativeOrder.blank_derivative(torder=part[0], xorder=part[1])
        prime = LibraryPrime(derivative=derivative, derivand=observable)
        return prime
    
    pairs = sorted(pairs)
    primes = [pair_to_prime(observable, part) for (observable, part) in pairs]

    # make all possible lists of primes and convert to terms of each rank, then generate labelings
    for prime_list in valid_prime_lists(primes, order, max_observables, max_observable_counts):
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
                      max_observable_counts: dict[Observable, int],
                      non_empty: bool = False) -> List[Union[ConstantTerm, LibraryTerm]]:
    # starting_ind: int
    """
    Generate components of valid terms from list of primes, with maximum complexity = order, maximum number of observables = max_observables, max number of primes = max_rho, and max of each observable count = max_observable_counts[observable].
    """
    # , and using only primes starting from index starting_ind.
    # base case: yield no primes
    if non_empty:
        yield ()
    for i, prime in enumerate(primes): # relative_i
        complexity = prime.complexity
        if complexity <= order and 1 <= max_observables and 1 <= max_observable_counts[prime.derivand]:
            max_observable_counts[prime.derivand] -= 1 # temporarily modify the dictionary
            for tail in valid_prime_lists(primes=primes[i:], order=order-complexity,
                                          max_observables=max_observables-1,
                                          max_observable_counts=max_observable_counts, non_empty=True):
                yield (prime,) + tail
            max_observable_counts[prime.derivand] += 1 # unmodify