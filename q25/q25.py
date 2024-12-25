import dataclasses
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import heapq
import math
from math import gcd
import operator
import re
import textwrap
import time
import numpy as np

from bisect import bisect_left, bisect_right

from dataclasses import dataclass, field, replace
from pathlib import Path

import collections
from collections import (
    Counter,
    defaultdict,
    deque
)

import functools
import itertools

from functools import (
    partial,
)
from itertools import (
    accumulate,
    batched,
    chain,
    combinations,
    count,
    cycle,
    filterfalse,
    groupby,
    islice,
    pairwise,
    permutations,
    product,
    repeat,
    takewhile,
    tee,
)

from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Iterator,
    Literal,
    NamedTuple,
    Never,
    Optional,
    assert_never,
    assert_type,
    overload
)

import pytest
import hypothesis
from hypothesis import (
    given,
    strategies as st
)

from aoc.search import astar, allpaths, bfs #, path_to
from aoc.perf import timer
from aoc.itertools import (
    star,
    nth,
    mapnth,
    observe,
    consume,
    window,
    interleave,
    first,
    only,
    split,
)

def reachable[T](
        start: T,
        neighbours: Callable[[T], Iterable[T]]
) -> set[T]:
    dist, _ = bfs(start, None, neighbours)
    return {
        n for n, d in dist.items() if d != float('inf')
    }

import numpy as np
import numpy.typing as npt
from aoc.np import (
    locate,
    display,
    follow_ray
)

def iterate(f, n) -> Iterator[int]:
    while True:
        yield n
        n = f(n)

def sliding_window[T](
        items: Iterable[T],
        n: int
) -> Iterator[tuple[T, ...]]:
    items = iter(items)
    window = collections.deque(islice(items, n-1), maxlen=n)
    for x in items:
        window.append(x)
        yield tuple(window)

def innermap[T, U](
        f: Callable[[T], U],
        s: Iterable[Iterable[T]]
) -> Iterable[Iterable[U]]:
    return map(partial(map, f), s)

def read_input(
        filename: str
) -> tuple[
    list[np.ndarray[Any, np.dtypes.StringDType]],
    list[np.ndarray[Any, np.dtypes.StringDType]]
]:
    def make_grid(lines: list[str]) -> np.ndarray[Any, np.dtypes.StringDType]:
        return np.array(
            list(map(list, lines)),
            dtype=np.dtypes.StringDType
        )

    with open(filename) as f:
        s = iter(f)
        s = map(str.rstrip, s)
        s = split(lambda line: line == "", s)
        s = map(make_grid, s)
        grids = list(s)

    locks, keys = [], []
    for g in grids:
        assert g.shape == (7, 5)
        if ((g[0, :] == "#") & (g[-1, :] == ".")).all():
            locks.append(g)
        elif ((g[0, :] == ".") & (g[-1, :] == "#")).all():
            keys.append(g)
        else:
            assert False, g
    return locks, keys

def part1(filename):
    locks, keys = read_input(filename)
    rows, cols = map(range, locks[0].shape)
    nlocks, nkeys = len(locks), len(keys)

    print(f"{nlocks, nkeys = }")

    lheights = set(tuple(map(int, np.sum(lock == "#", axis=0) - 1)) for lock in locks)
    kheights = set(tuple(map(int, np.sum(key == "#", axis=0) - 1)) for key in keys)
    assert len(lheights) == nlocks
    assert len(kheights) == nkeys

    can_fit = 0
    for i, lock in enumerate(lheights):
        for j, key in enumerate(kheights):
            if any(lh + kh >= 6 for lh, kh in zip(lock, key)):
                # print(f"*** doesn't fit: {lock = } {key = }")
                continue
            can_fit += 1
    print(f"{can_fit = }")
def part2(filename):
    known, wires, gates = read_input(filename)

def usage(message):
    print(f'usage: {sys.argv[0]} [-1|-2] [--] input_file...')
    print(f'    {message}')
    sys.exit(1)

parts = {
    1: partial(part1),
    2: partial(part2),
}

def main(args):
    from aoc.cmd import argscan
    from aoc.perf import timer
    torun = set()
    options = {}

    for flag in argscan(args):
        if flag in ('-1'): torun.add(1)
        elif flag in ('-2'): torun.add(2)
        else:
            usage(f"{flag}: unexpected option")
    if not torun: torun = {1, 2}

    if len(args) == 0:
        usage("missing input file")
    for infile, part in product(args, sorted(torun)):
        print(f"\n***** START PART {part} -- {infile}")
        with timer() as t:
            parts[part](infile, **options)
        print(f"***** FINISHED PART {part} -- {infile} elapsed {t.elapsed()}s")

if __name__ == '__main__':
    main(sys.argv[1:])