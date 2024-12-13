import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import heapq
import math
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

import numpy as np
import numpy.typing as npt
from aoc.np import (
    locate,
    display,
    follow_ray
)

def read_input(
        filename: str
) -> np.ndarray:
    with open(filename) as f:
        s = f.readlines()
        s = map(str.rstrip, s)
        s = map(list, s)
        return np.array(list(s), dtype=np.dtypes.StringDType)

def perimeter(places, neighbours):
    perim = 0
    for p in places:
        perim += 4 - sum(1 for _ in neighbours(p))
    return perim

def count_corners(places):
    s = product(places, [
            (-1, -1),
            (-1, +1),
            (+1, -1),
            (+1, +1),
    ])
    s = map(star(partial(looking_at_corner, places)), s)
    return sum(s)

def looking_at_corner(cc, pos, dir) -> bool:
    orthogonal_directions = {
        # the directions to the orthogonal neighbours when facing in a
        # given diagonal direction. these are in the order left,
        # right, and the code here assumes that
        (-1, -1): [ (-1, 0), (0, -1) ],
        (-1, +1): [ (-1, 0), (0, +1) ],
        (+1, +1): [ (+1, 0), (0, +1) ],
        (+1, -1): [ (+1, 0), (0, -1) ],
    }

    truthtable = {
        # if we're at pos facing in the diagonal direction, we
        # determine whether or not we're facing a corner by check if
        # each of the left, diagonal, and right positions are in the
        # in the same connected component as the current position.
        #
        # drawing little 2x2 grids with pos in the lower left corner,
        # and filling in the other three positions according to each
        # row below should show the intuition.
        #
        # Note (1): One tricky case is if we're adjacent to a corner
        # but not "on" the corner, so to speak. Consider
        #
        #     . x
        #     X x
        #
        # where pos is the lower left. Certainly we're adjacent to a
        # corner, but since that corner will be observed by the lower
        # right position, we shouldn't count it in this case.

        (False, False, False): True,    # yes: inside corner
        (False, False,  True): False,   # no: wall continues ahead on left
        (False,  True, False): True,    # yes: inside corner, same cc to diag
        (False,  True,  True): False,   # no: see (1)
        ( True, False, False): False,   # no: wall continues ahead on right
        ( True, False,  True): True,    # yes: outside corner
        ( True,  True, False): False,   # no: see (1)
        ( True,  True,  True): False,   # no: no edges in this direction
    }

    def adjust(dir):
        r, c = pos
        dr, dc = dir
        return (r + dr, c + dc)

    assert pos in cc, (pos, cc)
    dirleft, dirright = orthogonal_directions[dir]
    left, diag, right = map(adjust, [dirleft, dir, dirright])

    return truthtable[left in cc, diag in cc, right in cc]

def reachable[T](
        start: T,
        neighbours: Callable[[T], Iterable[T]]
) -> set[T]:
    dist, _ = bfs(start, None, neighbours)
    return {
        n for n, d in dist.items() if d != float('inf')
    }

def part1(filename, part2=False):
    grid = read_input(filename)
    rows, cols = map(range, grid.shape)
    nodes = set(product(rows, cols))

    def neighbours(pos: tuple[int, int]) -> Iterable[tuple[int, int]]:
        r, c = pos
        plant = grid[r, c]
        for dr, dc in [
            (-1, 0), (+1, 0), (0, -1), (0, +1)
        ]:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in nodes:
                continue
            if grid[nr, nc] != plant:
                continue
            yield (nr, nc)

    all_plants = set(
        grid[r, c] for r, c in product(rows, cols)
    ) - {"."}

    components = []
    for plant in all_plants:
        locs = set(locate(grid == plant))
        while locs:
            cc = reachable(locs.pop(), neighbours)
            components.append((plant, cc))
            locs -= cc

    if part2 == False:
        area_multiplier = partial(perimeter, neighbours=neighbours)
    else:
        area_multiplier = count_corners

    fencing_cost = sum(
        len(places) * area_multiplier(places)
        for _, places in
        components
    )

    print(f"{fencing_cost = }")

def part2(filename):
    pass

def usage(message):
    print(f'usage: {sys.argv[0]} [-1|-2] [--] input_file...')
    print(f'    {message}')
    sys.exit(1)

parts = {
    1: partial(part1),
    2: partial(part1, part2=True),
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