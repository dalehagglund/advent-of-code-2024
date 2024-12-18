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

from dataclasses import dataclass
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

from aoc.search import astar, allpaths, bfs
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

def innermap[T, U](
        f: Callable[[T], U],
        s: Iterable[Iterable[T]]
) -> Iterable[Iterable[U]]:
    return map(partial(map, f), s)

def read_input(
        filename: str
) -> list[tuple[int, int]]:
    with open(filename) as f:
        s = iter(f)
        s = map(str.rstrip, s)
        s = map(partial(str.split, sep=","), s)
        s = innermap(int, s)
        s = map(tuple, s)
        return list(s)

def part1(filename):
    positions = read_input(filename)
    print(positions)

    nrow = 1 + max(r for r, _ in positions)
    ncol = 1 + max(c for _, c in positions)

    assert min(r for r, _ in positions) == 0
    assert min(c for _, c in positions) == 0

    grid = np.full(
        (nrow, ncol),
        fill_value=".",
        dtype=np.dtypes.StringDType
    )

    if (nrow, ncol) == (7, 7):
        blocks = positions[:12]
    elif (nrow, ncol) == (71, 71):
        blocks = positions[:1024]
    else:
        print("unknown size!")
        return

    grid[*np.transpose(blocks)] = "#"
    print(f"after {len(blocks)} ...")
    display(grid)

    all_coords = set(
        (r, c)
        for r, c in product(range(nrow), range(ncol))
        if grid[r, c] == "."
    )

    def neighbours(pos) -> Iterator[tuple[int, int]]:
        r, c = pos
        for dr, dc in [
            (+1,  0),
            (-1,  0),
            ( 0, +1),
            ( 0, -1),
        ]:
            if (r + dr, c + dc) not in all_coords:
                continue
            yield (r + dr, c + dc)

    start = (0, 0)
    end = (nrow-1, ncol-1)

    dist, _ = astar(start, end, neighbours)
    print(dist[end])

def part2(filename):
    ...

def usage(message):
    print(f'usage: {sys.argv[0]} [-1|-2] [--] input_file...')
    print(f'    {message}')
    sys.exit(1)

parts = {
    1: part1,
    2: part2,
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