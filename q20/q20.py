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

def try_cheat(
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

@functools.cache
def md_mask(d: int) -> np.ndarray:
    mask = np.zeros((2 * d + 1, 2 * d + 1), dtype=np.bool)
    center = (d, d)      # center
    for r, c in product(*map(range, mask.shape)):
        if mh_dist((r, c), center) <= d:
            mask[r, c] = 1
    return mask

def part2_possible_cheat_locs(
        pos, d, grid
) -> Iterator[tuple[int, int]]:
    nrow, ncol = grid.shape
    r, c = pos

    for dr, dc in product(range(-d, d+1), repeat=2):
        nr, nc = r + dr, c + dc
        if dr == 0 and dc == 0:
            continue
        if not (0 <= nr < nrow and 0 <= nc < ncol):
            continue
        if grid[nr, nc] == "#":
            continue
        if mh_dist((nr, nc), pos) <= d:
            yield (nr, nc)

def part1(filename, cheatradius, mincheat=50, part2=False):
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

    savings: defaultdict[int, set] = defaultdict(set)
    for pos, dir in product(shortest_path, all_dirs):
        saved = try_cheat(pos, dir, grid, dist, positions)
        if saved > 0:
            savings[saved].add((pos, dir))

    # for k, v in sorted(savings.items(), key=star(lambda k, _: k)):
    #     print(f"{k} picoseconds: {len(v)} cheats")

    part1_above_mincheat = 0
    for saved, cheats in savings.items():
        if saved >= mincheat: part1_above_mincheat += len(cheats)
    print(f"{part1_above_mincheat = }")

    if not part2: return

    savings: defaultdict[int, set] = defaultdict(set)

    print(f"{mincheat = } {cheatradius = }")
    for i, pos in enumerate(shortest_path):
        if i % 250 == 0: print(f"path step {i} ...")
        # print(f"path step {i} pos {pos} ...")
        reachable_cheats = set(part2_possible_cheat_locs(pos, cheatradius, grid))
        # print(f"   reachable cheats", len(reachable_cheats))
        for cheat in reachable_cheats:
            posdist = dist[pos]
            cheatdist = mh_dist(pos, cheat) + dist[cheat]

            if cheatdist >= posdist:
                continue
            savings[posdist - cheatdist].add((pos, cheat))

    # for k, v in sorted(savings.items(), key=star(lambda k, _: k)):
    #     print(f"{k} picoseconds: {len(v)} cheats")

    print(f"{mincheat = } {cheatradius = }")
    part2_above_mincheat = 0
    for saved, cheats in savings.items():
        if saved >= mincheat: part2_above_mincheat += len(cheats)
    print(f"part 2: {part2_above_mincheat = }")

def part2(filename):
    ...

def usage(message):
    print(f'usage: {sys.argv[0]} [-1|-2] [--] input_file...')
    print(f'    {message}')
    sys.exit(1)

parts = {
    1: partial(part1, cheatradius=2, mincheat=100, part2=False),
    2: partial(part1, cheatradius=20, mincheat=100, part2=True),
}

def main(args):
    from aoc.cmd import argscan
    from aoc.perf import timer
    torun = set()
    options = {}

    for flag in argscan(args):
        if flag in ('-1'): torun.add(1)
        elif flag in ('-2'): torun.add(2)
        elif flag in ('--mincheat'): options["mincheat"] = int(args.pop(0))
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

def test_md_mask_d_0():
    mask = md_mask(0)
    assert (
        mask ==
        np.array([
            [1]
        ],
        dtype=np.bool)
    ).all()
def test_md_mask_d_1():
    mask = md_mask(1)
    assert (
        mask ==
        np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=np.bool)
    ).all()
def test_md_mask_d_2():
    mask = md_mask(2)
    assert (
        mask ==
        np.array([
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=np.bool)
    ).all()

def make_grid(*lines):
    return np.array(
        list(map(list, lines)),
        dtype=np.dtypes.StringDType
    )

def test_part2_cheat_locs_with_blocks():
    grid = make_grid(
        ".....",
        ".S#S.", # same as dots, but easier to se
        ".SSS.",
        ".S#S.",
        ".....",
    )
    locs = set(part2_possible_cheat_locs((2, 2), 1, grid))
    assert locs == {
        (2, 2),
        (2, 1), (2, 3),
        # (1, 2), (3, 2),
    }
def test_part2_cheat_locs_away_from_edges():
    grid = make_grid(
        ".....",
        ".SSS.", # same as dots, but easier to see
        ".SSS.",
        ".SSS.",
        ".....",
    )
    locs = set(part2_possible_cheat_locs((2, 2), 1, grid))
    assert locs == {
        (2, 2),
        (2, 1), (2, 3),
        (1, 2), (3, 2),
    }
def test_part2_cheat_locs_at_ul_boundary():
    grid = make_grid(
        "SSS..",
        "SSS..", # same as dots, but easier to see
        "SSS..",
        ".....",
        ".....",
    )
    locs = set(part2_possible_cheat_locs((1, 1), 1, grid))
    assert locs == {
        (1, 1),
        (1, 0), (1, 2),
        (0, 1), (2, 1),
    }

def test_part2_cheat_locs_with_mask_beyond_ul_corner():
    grid = make_grid(
        "SS...",
        "SS...", # same as dots, but easier to see
        ".....",
        ".....",
        ".....",
    )
    locs = set(part2_possible_cheat_locs((0, 0), 1, grid))
    assert locs == {
        (0, 0), (0, 1), (1, 0)
    }