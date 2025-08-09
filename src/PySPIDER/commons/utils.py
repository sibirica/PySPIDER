from collections.abc import Iterable
from typing import Any
import re
import numpy as np


def regex_find(lst: Iterable[Any], pattern: re.Pattern) -> Iterable[tuple[int, Any]]:
    yield from ((i, x) for i, x in enumerate(lst) if re.fullmatch(pattern, str(x)))


def save(filename, *args):
    with open(filename, "wb") as f:
        for arr in args:
            np.save(f, arr)


def load(filename, nload):
    to_load = []
    with open(filename, "rb") as f:
        for i in range(nload):
            to_load.append(np.load(f))
    return tuple(to_load)
