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
):
    with open(filename) as f:
        s = iter(f)
        s = map(str.rstrip, s)
        s = map(list, s)
        return np.array(list(s), dtype=np.dtypes.StringDType)

directions = {
    "^": (-1,  0),
    ">": ( 0, +1),
    "v": (+1,  0),
    "<": ( 0, -1),
}

class Pos(NamedTuple):
    r: int
    c: int

    @overload
    def __add__(self, other: tuple[int, int]) -> "Pos": ...

    @overload
    def __add__(self, other: "Pos") -> "Pos": ...

    def __add__(self, other: "Pos") -> "Pos":
        if isinstance(other, tuple):
            return Pos(self.r + other[0], self.c + other[1])
        return Pos(self.r + other.r, self.c + other.c)

@dataclass(frozen=True)
class State:
    pos: Pos
    dir: tuple[int, int]

def part1(filename):
    grid = read_input(filename)
    display(grid)
    nrow, ncol = grid.shape

    board: defaultdict[Pos, set[tuple[int, int]]] = defaultdict(set)
    for pos in map(star(Pos), product(range(1, nrow-1), range(1, ncol - 1))):
        for _, dir in directions.items():
            if grid[pos + dir] == "#":
                continue
            board[pos].add(dir)

    # print(sorted(board.items(), key=star(lambda k, v: k)))

    start = Pos(*only(locate(grid == "S")))
    end = Pos(*only(locate(grid == "E")))

    print(f"{start, end = }")

    def edge_cost(u: State, v: State) -> int:
        dotprod = sum(map(operator.mul, u.dir, v.dir))
        match dotprod:
            case  1: return 1
            case  0: return 1000 + 1
            case -1: return 2000 + 2
            case _:
                assert_never(dotprod)

    def moves(u: State) -> Iterator[State]:
        for dir in board[u.pos]:
            yield State(
                pos = u.pos + dir,
                dir = dir
            )

    def distance(u: State, _) -> int:
        return abs(u.pos.r - end.r) + abs(u.pos.c - end.c)

    def is_goal(u: State) -> bool:
        return u.pos == end

    dist, prev = astar(
        State(pos=start, dir=directions["<"]),
        is_goal,
        moves,
        edge_cost=edge_cost,
        est_remaining=distance
    )

    for k in filter(lambda s: s.pos == end, dist.keys()):
        print(f"{k, dist[k] = }")


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