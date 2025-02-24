from __future__ import annotations

#from functools import lru_cache
from typing import Any, Protocol, Union, assert_type
from abc import abstractmethod, ABC
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace, KW_ONLY
from itertools import count
import z3

lowercase_greek_letters = "αβγδεζηθικλμνξοπρστυφχψω"

@dataclass(frozen=True)
class SymmetryRep:
    _: KW_ONLY
    rank: int

    def __add__(self, other):
        print("Warning, SymmetryReps have been added")
        return FullRank(rank=self.rank + other.rank)

@dataclass(frozen=True)
class Antisymmetric(SymmetryRep):
    def __repr__(self):
        return f"Antisymmetric rank {self.rank}"

@dataclass(frozen=True)
class SymmetricTraceFree(SymmetryRep):
    def __repr__(self):
        return f"Symmetric trace-free rank {self.rank}"

@dataclass(frozen=True)
class FullRank(SymmetryRep):
    def __repr__(self):
        return f"Rank {self.rank}"

Irrep = Antisymmetric | SymmetricTraceFree | FullRank

@dataclass(frozen=True)
class IndexHole:
    def __lt__(self, other):
        if isinstance(other, IndexHole):
            return False
        elif isinstance(other, VarIndex):
            return True
        else:
            raise TypeError(f"Operation not supported between instances of '{type(self)}' and '{type(other)}'")

    def __gt__(self, other):
        if isinstance(other, (IndexHole, VarIndex)):
            return False
        else:
            raise TypeError(f"Operation not supported between instances of '{type(self)}' and '{type(other)}'")

    def __eq__(self, other):
        if not isinstance(other, IndexHole):
            return False
        else:
            return True

    def __repr__(self):
        return "{ }"

@dataclass(frozen=True)
class SMTIndex:
    var: z3.ArithRef
    src: Any = None

    def __lt__(self, other):
        return str(self.var) < str(other.var)

    def __eq__(self, other):
        if not isinstance(other, SMTIndex):
            raise TypeError(f"Operation not supported between instances of '{type(self)}' and '{type(other)}'")
        else:
            return self.var == other.var

    def __repr__(self):
        return f"{repr(self.var)}"

    def __hash__(self):
        return hash(self.var)

@dataclass(frozen=True)
class VarIndex:
    value: int
    src: Any = None

    def __lt__(self, other):
        if isinstance(other, str):
            return True # 't' always goes after spatial indices
        elif not isinstance(other, VarIndex):
            return NotImplemented
        else:
            return self.value < other.value

    def __eq__(self, other):
        if isinstance(other, str):
            return False # 't' is different from spatial indices
        elif not isinstance(other, VarIndex):
            return NotImplemented
        else:
            return self.value == other.value

    def __repr__(self):
        return lowercase_greek_letters[self.value]

    def __hash__(self):
        return hash(self.value)

@dataclass(frozen=True)
class LiteralIndex:
    value: int

    def __lt__(self, other):
        if isinstance(other, str):
            return True # 't' always goes after spatial indices
        elif not isinstance(other, LiteralIndex):
            raise TypeError(f"Operation not supported between instances of '{type(self)}' and '{type(other)}'")
        else:
            return self.value < other.value

    def __eq__(self, other):
        if isinstance(other, str):
            return True # 't' always goes after spatial indices
        elif not isinstance(other, LiteralIndex):
            raise TypeError(f"Operation not supported between instances of '{type(self)}' and '{type(other)}'")
        else:
            return self.value == other.value

    def __repr__(self):
        return "xyzt"[self.value]  # if your problem has 5 dimensions you messed up

Index = IndexHole | SMTIndex | VarIndex | LiteralIndex

@dataclass(frozen=True)
class EinSumExpr[T](ABC):
    _: KW_ONLY
    can_commute_indices: bool = False
    can_commute_exprs: bool = True

    @abstractmethod
    def __lt__(self, other):
        ...

    @abstractmethod
    def __eq__(self, other):
        ...

    # may need separate struct_eq if we need to manually check for terms commuting across *

    @abstractmethod
    def __repr__(self):
        ...

    def get_rank(self):
        match self.rank:
            case SymmetryRep(rank=rank):
                return rank
            case _ as rank:
                return rank

    @abstractmethod
    def sub_exprs(self) -> Iterable[EinSumExpr[T]]:
        """ Implementation returns list of sub_exprs (whatever this attribute may be called) """
        ...

    @abstractmethod
    def own_indices(self) -> Iterable[T]:
        """ Implementation returns list of own indices """
        ...

    #@lru_cache(maxsize=10000)
    def all_indices(self) -> list[T]: # make sure these are in depth-first/left-to-right order
        """ List all indices """
        return list(self.own_indices()) + [idx for expr in self.sub_exprs() for idx in expr.all_indices()]

    @abstractmethod
    def eq_canon(self) -> Tuple[EinSumExpr[T], int]:
        """ Returns the canonical form of the term modulo equality rewrites & the sign after rewrites. """
        ...

    @abstractmethod
    def map[T2](self, *,
                expr_map: Callable[[EinSumExpr[T]], EinSumExpr[T2]] = lambda x: x,
                index_map: Callable[[T], T2] = lambda x: x) -> EinSumExpr[T2]:
        """ Constructs a copy of self replacing (direct) child expressions according to expr_map
            and (direct) child indices according to index_map"""
        ...

    def map_all_indices[T2](self, index_map: Callable[[T], T2]) -> EinSumExpr[T2]:
        def mapper(expr):
            nonlocal index_map
            return expr.map(expr_map=mapper, index_map=index_map)
        ms = mapper(self)
        #src_check = lambda x: x.src if hasattr(x, 'src') else None
        return ms

    def purge_indices(self): # return a copy with only IndexHoles
        return self.map_all_indices(index_map=lambda idx: IndexHole())

    def canonical_indexing_problem(self, idx_cache: defaultdict | None = None) -> tuple[EinSumExpr[SMTIndex], list[z3.ExprRef]]:
        base_id = f"i{id(self)}"
        def next_z3_var():
            return free_z3_var(base_id)
        
        idx_cache = defaultdict(next_z3_var) if idx_cache is None else idx_cache
        
        constraints = []
        def emap(expr):
            nonlocal constraints
            updated, cxs = expr.canonical_indexing_problem(idx_cache)
            constraints += cxs
            return updated

        def imap(idx):
            if isinstance(idx, IndexHole):
                return SMTIndex(next_z3_var())
            #print(id(idx_cache), idx, len(idx_cache), idx_cache[idx])
            return SMTIndex(idx_cache[idx], src=idx)

        updated = self.map(expr_map=emap, index_map=imap)
        #print(id(idx_cache), list(idx_cache.items()))

        new_constraints = []
        if self.can_commute_indices:
            # constraint on own_indices
            #own_indices = sorted(list(updated.own_indices()))
            own_indices = list(updated.own_indices())
            for i, i_next in zip(own_indices, own_indices[1:]):
                if i.src is None: # only add constraint if the first index in the pair isn't already constrained
                    new_constraints.append(i.var <= i_next.var)
        if self.can_commute_exprs:
            duplicates = defaultdict(list)
            for e, e_new in zip(self.sub_exprs(), updated.sub_exprs()):
                #print(id(e), hash(e), [(se, hash(se)) for se in e.sub_exprs()], e)
                duplicates[e].append(e_new)
            for dup_list in duplicates.values():
                for e, e_next in zip(dup_list, dup_list[1:]):
                    new_constraints.append(lexico_le(e.all_indices(), e_next.all_indices()))
            #print(duplicates)

        #print(">>", updated, list(updated.own_indices()), new_constraints)
        constraints += new_constraints
        return updated, constraints

def generate_indexings(expr: EinSumExpr[IndexHole | VarIndex]) -> Iterable[EinSumExpr[VarIndex]]:
    indexed_expr, constraints = expr.canonical_indexing_problem() # includes lexicographic constraints
    assert_type(indexed_expr, EinSumExpr[SMTIndex])
    #print(indexed_expr)
    #print(constraints)
    # add global constraints
    indices = indexed_expr.all_indices()
    n_single_inds = expr.rank
    n_total_inds = (len(indices)+n_single_inds)//2
    # use-next-variable constraints
    single_idx_max = 0
    paired_idx_max = n_single_inds
    for j, idx in enumerate(indices):
        s_idx_max_next = z3.Int(f's_idxmax_{j}')
        p_idx_max_next = z3.Int(f'p_idxmax_{j}')
        constraints += [z3.Or(
            z3.And(idx.var == single_idx_max,
                   s_idx_max_next == single_idx_max + 1, 
                   p_idx_max_next == paired_idx_max),
            z3.And(idx.var >= n_single_inds, idx.var <= paired_idx_max,
                   s_idx_max_next == single_idx_max,
                   p_idx_max_next == paired_idx_max + z3.If(idx.var==paired_idx_max, 1, 0)
                  )
        )]
        single_idx_max = s_idx_max_next
        paired_idx_max = p_idx_max_next
    constraints += [single_idx_max == n_single_inds, paired_idx_max == n_total_inds]
    # constrain number of appearances in pair
    for paired_idx in range(n_single_inds, n_total_inds):
       constraints.append(z3.AtMost(*[idx.var == paired_idx for idx in indices], 2))
    # give problem to smt solver
    solver = z3.Solver()
    solver.add(*constraints)
    while (result := solver.check()) == z3.sat: # smt solver finds a new solution
        m = solver.model()
        indexing = {index: m[index.var] for index in indices}
        yield indexed_expr.map_all_indices(
            index_map = lambda index: VarIndex(indexing[index].as_long(), src=index.src))
        # prevent smt solver from repeating solution
        solver.add(z3.Or(*[idx.var != val for idx, val in indexing.items()]))
    if result == z3.unknown:
        raise RuntimeError("Could not solve SMT problem :(")
    #print(solver.to_smt2())

def lexico_le(idsA: list[SMTIndex], idsB: list[SMTIndex]) -> z3.ExprRef:
    lt = True
    for a, b in zip(reversed(idsA), reversed(idsB)):
        lt = z3.Or(a.var < b.var, z3.And(a.var == b.var, lt))
    return lt

def free_z3_var(prefix: str, *, ctr=count()):
    return z3.Int(f"{prefix}_{next(ctr)}")
