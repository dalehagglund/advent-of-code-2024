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

def p1_try_cheat(
        pos, dir, grid, dist, positions
) -> int:
    (r, c), (dr, dc) = pos, dir

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

def mh_dist(p, q):
    return sum(abs(d) for d in map(operator.sub, p, q))

def p2_cheat_endpoints(
        grid,
        pos: tuple[int, int],
        rad: int
) -> Iterator[tuple[int, int]]:
    md = mh_dist_from(grid, pos)
    return locate((grid != "#") & (md <= rad))

def p2_cheat_mask(grid, pos, rad) -> np.ndarray[Any, np.dtypes.BoolDType]:
    md = mh_dist_from(grid, pos)
    return (grid != "#") & (md <= rad)

def mh_dist_from(grid, pos):
    r, c = pos
    return np.fromfunction(
        lambda i, j: abs(i - r) + abs(j - c),
        grid.shape,
    )

@pytest.mark.parametrize(
    "dist, origin, other",
    [
        [ 0, (0, 0), (0, 0) ],
        [ 2, (0, 0), (1, 1) ],
        [ 6, (0, 0), (3, 3) ],
        [ 2, (0, 0), (0, 2) ],
        [ 3, (0, 0), (1, 2) ],
    ]
)
def test_mh_dist_from(dist, origin, other):
    grid = np.zeros((4, 4))
    dist_to = mh_dist_from(grid, origin)
    assert dist_to[other] == dist

def shared_setup(grid) -> tuple:
    rows, cols = map(range, grid.shape)
    positions = set(product(rows, cols))
    start = only(locate(grid == "S"))
    end = only(locate(grid == "E"))

    def neighbours(pos) -> Iterator[tuple[int, int]]:
        r, c = pos
        for dr, dc in all_dirs:
            if (r + dr, c + dc) not in positions:
                continue
            if grid[r + dr, c + dc] == "#":
                continue
            yield r + dr, c + dc
    def extract_path[T](prev, n: T) -> list[T]:
        path = [n]
        while (n := prev[n]) is not None:
            path.append(n)
        path.reverse()
        return path
    def make_distvec():
        return np.full_like(grid, fill_value=float('inf'), dtype=np.float64)
    dist, prev = astar(
        start,
        None,
        neighbours,
        dist_factory=make_distvec,
    )

    return dist, extract_path(prev, end)

def part1(filename, cheatradius, min_saving=50):
    grid = read_input(filename)
    dist, shortest_path = shared_setup(grid)

    positions = set(product(*map(range, grid.shape)))
    winning_cheats: defaultdict[int, set] = defaultdict(set)
    for pos, dir in product(shortest_path, all_dirs):
        saved = p1_try_cheat(pos, dir, grid, dist, positions)
        if saved > 0:
            winning_cheats[saved].add((pos, dir))

    part1_above_mincheat = 0
    for saved, cheats in winning_cheats.items():
        if saved >= min_saving: part1_above_mincheat += len(cheats)
    print(f"{part1_above_mincheat = }")

def part2(filename, cheatradius, min_saving=50):
    grid = read_input(filename)
    dist, shortest_path = shared_setup(grid)

    winning_cheats = np.zeros((len(shortest_path) + 1,), dtype=int)
    print(f"{min_saving = } {cheatradius = }")
    for i, pos in enumerate(shortest_path):
        if i % 500 == 0: print(f"path step {i} ...")

        cheats = p2_cheat_mask(grid, pos, cheatradius)
        dist_from_pos = mh_dist_from(grid, pos)
        shortcut_savings = dist[pos] - (dist_from_pos + dist)
        for cheat in locate((shortcut_savings > 0) & cheats):
            winning_cheats[int(shortcut_savings[cheat])] += 1

    print(f"{min_saving = } {cheatradius = }")
    part2_above_mincheat = int(winning_cheats[min_saving:].sum())
    print(f"part 2: {part2_above_mincheat = }")

def usage(message):
    print(f'usage: {sys.argv[0]} [-1|-2] [--] input_file...')
    print(f'    {message}')
    sys.exit(1)

parts = {
    1: partial(part1, cheatradius=2, min_saving=100),
    2: partial(part2, cheatradius=20, min_saving=100),
}

def main(args):
    from aoc.cmd import argscan
    from aoc.perf import timer
    torun = set()
    options = {}

    for flag in argscan(args):
        if flag in ('-1'): torun.add(1)
        elif flag in ('-2'): torun.add(2)
        elif flag in ('--mincheat'): options["min_saving"] = int(args.pop(0))
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

@pytest.mark.parametrize(
    "expected, p, q",
    [
        (0, (0, 0), (0, 0)),
        (0, (2, 2), (2, 2)),
        (1, (2, 1), (2, 2)),
        (2, (0, 0), (1, 1)),
        (2, (0, 0), (0, 2)),
        (2, (2, 2), (0, 2)),
    ]
)
def test_mh_dist(expected, p, q):
    assert expected == mh_dist(p, q)

def make_grid(*lines):
    return np.array(
        list(map(list, lines)),
        dtype=np.dtypes.StringDType
    )

def test_p2_cheat_endpoints_1():
    grid = make_grid(
        ".....",
        ".S#S.", # same as dots, but easier to se
        ".SSS.",
        ".S#S.",
        ".....",
    )
    locs = set(p2_cheat_endpoints(grid, (2, 2), 1))
    assert locs == {
        (2, 2),
        (2, 1), (2, 3),
        # (1, 2), (3, 2),
    }
def test_p2_cheat_endpoints_2():
    grid = make_grid(
        ".....",
        ".SSS.", # same as dots, but easier to see
        ".SSS.",
        ".SSS.",
        ".....",
    )
    locs = set(p2_cheat_endpoints(grid, (2, 2), 1))
    assert locs == {
        (2, 2),
        (2, 1), (2, 3),
        (1, 2), (3, 2),
    }

def test_p2_cheat_endpoints_3():
    grid = make_grid(
        "SSS..",
        "SSS..", # same as dots, but easier to see
        "SSS..",
        ".....",
        ".....",
    )
    locs = set(p2_cheat_endpoints(grid, (1, 1), 1))
    assert locs == {
        (1, 1),
        (1, 0), (1, 2),
        (0, 1), (2, 1),
    }

def test_p2_cheat_endpoints_4():
    grid = make_grid(
        "SS...",
        "SS...", # same as dots, but easier to see
        ".....",
        ".....",
        ".....",
    )
    locs = set(p2_cheat_endpoints(grid, (0, 0), 1))
    assert locs == {
        (0, 0), (0, 1), (1, 0)
    }