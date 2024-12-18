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

from aoc.search import astar, dijkstra, allpaths, bfs
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

type Dir = complex
type Pos = complex

to_dir = {
    "N": -1 + 0j,
    "S": +1 + 0j,
    "E":  0 - 1j,
    "W":  0 + 1j,
}

from_dir = dict((v, k) for k, v in to_dir.items())

rot_left = 1j
rot_right = 1j * 1j * 1j

def read_input(
        filename: str
):
    with open(filename) as f:
        s = iter(f)
        s = map(str.rstrip, s)
        s = map(list, s)
        return np.array(list(s), dtype=np.dtypes.StringDType)

@dataclass(frozen=True)
class Move:
    from_: tuple[Pos, Dir] | None
    to_: tuple[Pos, Dir]

    def __post_init__(self):
        if self.from_ is not None:
            assert self.from_[1] in to_dir.values(), self
        assert self.to_[1] in to_dir.values(), self
        if self.from_ is None:
            return

        fpos, fdir = self.from_
        tpos, tdir = self.to_

        if fpos == tpos:
            assert tdir in (fdir * rot_left, fdir * rot_right), self
        elif fpos + fdir == tpos:
            assert fdir == tdir, self
        else:
            assert False, self

    def __repr__(self):
        def fmt(end: tuple[Pos, Dir] | None) -> str:
            if end is None: return "None"
            pos, dir = end
            r, c = map(int, (pos.real, pos.imag))
            return f"{from_dir[dir]}@{r, c}"
        return f"{fmt(self.from_)} -> {fmt(self.to_)}"

    def cost(self) -> int:
        if self.from_ is None:
            # a magic *initial* move to bring the reindeer onto the
            # board
            return 0

        fpos, fdir = self.from_
        tpos, tdir = self.to_

        if fpos == tpos:
            return 1000

        if fpos + fdir == tpos:
            return 1

        assert False, ("impossible move state!", self)

def part1(filename):
    grid = read_input(filename)

    rows, cols = map(range, grid.shape)
    start: tuple[int, int] = only(locate(grid == "S"))
    end: tuple[int, int] = only(locate(grid == "E"))

    def inbounds(pos: Pos) -> bool:
        return pos.real in rows and pos.imag in cols
    def at(pos: Pos) -> str:
        r, c = map(int, (pos.real, pos.imag))
        return grid[r, c]
    def to_pos(p: tuple[int, int]) -> complex:
        r, c = p
        return r + c*1j
    def to_pair(p: Pos) -> tuple[int, int]:
        return int(p.real), int(p.imag)

    move_graph: defaultdict[tuple[Pos, Dir], set[Move]] = defaultdict(set)
    for r, c, dir in product(rows, cols, to_dir.values()):
        pos = to_pos((r, c))
        if not inbounds(pos) or at(pos) == "#":
            continue

        # add a forward move
        if inbounds(pos + dir) and at(pos + dir) != "#":
            move_graph[(pos, dir)].add(
                Move((pos, dir), (pos + dir, dir))
            )
        # add rotational moves
        for newdir in [dir * rot_left, dir * rot_right]:
            move_graph[(pos, dir)].add(
                Move((pos, dir), (pos, newdir))
            )

    starting_move = Move(None, (to_pos(start), to_dir["E"]))

    ending_moves = set(
        Move((to_pos(end) + dir, -dir), (to_pos(end), -dir))
        for dir in map(lambda d: to_dir[d], "NSEW")
        if inbounds(to_pos(end) + dir) and at(to_pos(end) + dir) != "#"
    )

    dist, _ = astar(
        starting_move,
        None and (lambda e: e in ending_moves),
        functools.cache(lambda e: move_graph[e.to_]),
        #next_moves,
        edge_cost = lambda m1, m2: m2.cost()
    )
    for e in ending_moves:
        print(f"... dist[{e}] = {dist[e]}")

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