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
) -> np.ndarray:
    with open(filename) as f:
        s = iter(f)
        s = map(str.rstrip, s)
        s = map(list, s)
        return np.array(
            list(s),
            dtype=np.dtypes.StringDType
        )

all_dirs = [
    (-1,  0),
    (+1,  0),
    ( 0, -1),
    ( 0, +1),
]

def part1(filename):
    grid = read_input(filename)
    rows, cols = map(range, grid.shape)

    positions = set(product(rows, cols))

    def neighbours(pos) -> Iterator[tuple[int, int]]:
        r, c = pos
        for dr, dc in all_dirs:
            if (r + dr, c + dc) not in positions:
                continue
            if grid[r + dr, c + dc] == "#":
                continue
            yield r + dr, c + dc

    def extract_path(prev, n) -> list:
        path = [n]
        while (n := prev[n]) is not None:
            path.append(n)
        path.reverse()
        return path

    start = only(locate(grid == "S"))
    end = only(locate(grid == "E"))
    dist, prev = astar(
        start,
        None,
        neighbours,
    )
    print(f"{end = } {dist[end] = }")
    shortest_path = extract_path(prev, end)
    nodes_on_path = set(shortest_path)
    # print(f"{shortest_path = }")

    def try_cheat(pos, dir) -> int:
        (r, c), (dr, dc) = pos, dir
        assert pos in nodes_on_path

        if (r + dr, c + dc) not in positions:
            return 0
        if grid[r + dr, c + dc] != "#":
            return 0
        if (r + 2 * dr, c + 2 * dc) not in positions:
            return 0
        if (ch := grid[r + 2 * dr, c + 2 * dc]) == "#":
            return 0
        assert ch in ("S", "E", ".")

        posdist = dist[r, c]
        cheatdist = 2 + dist[r + 2 * dr, c + 2 * dc]

        if cheatdist > posdist:
            return 0
        return posdist - cheatdist

    savings: defaultdict[int, set] = defaultdict(set)
    for pos, dir in product(shortest_path, all_dirs):
        saved = try_cheat(pos, dir)
        if saved > 0:
            savings[saved].add((pos, dir))

    for k, v in sorted(savings.items(), key=star(lambda k, _: k)):
        print(f"{k} picoseconds: {len(v)} cheats")

    above_100 = 0
    for saved, cheats in savings.items():
        if saved >= 100: above_100 += len(cheats)
    print(f"{above_100 = }")

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