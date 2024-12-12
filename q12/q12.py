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

type Edge = tuple[tuple[int, int], tuple[int, int]]

def sides(places: list[tuple[int, int]], neighbours):
    all_edges: set[Edge] = set()

    def add_edge(c1, c2):
        # print(f"... ... edge {c1 = } {c2 = }")
        all_edges.add((c1, c2))

    for r, c in places:
        adj = set(neighbours((r, c)))
        # print(f"... adding edges for {r, c = } {adj = }")
        # check for non-neighbours in the top, bottom, left, right
        # directions. any non-neighbour corresponds to an edge along
        # that side of this cell.
        if (r-1, c) not in adj: add_edge( (r,   c),   (r,   c+1) )
        if (r+1, c) not in adj: add_edge( (r+1, c),   (r+1, c+1) )
        if (r, c-1) not in adj: add_edge( (r,   c),   (r+1, c)   )
        if (r, c+1) not in adj: add_edge( (r,   c+1), (r+1, c+1) )

    # now, start pulling out arbitrary *edges* and extending af as we
    # can go in each direction (either up/down or left/right). note
    # that every edge is part of *exactly* one side, so there's no
    # double counting to be worried about.

    side_count = 0
    remaining_edges = set(all_edges)
    while remaining_edges:
        start = remaining_edges.pop()

        def follow_edge(e: Edge, dir: int) -> Iterator[Edge]:
            c1, c2 = map(np.array, e)
            delta = c2 - c1
            while True:
                if dir > 0:
                    c1, c2 = c2, c2 + delta
                else:
                    c1, c2 = c1 - delta, c1
                e = ( tuple(map(int, c1)), tuple(map(int, c2)) )
                if e not in remaining_edges:
                    break
                yield e

        def has_cross_edges(e1, e2) -> bool:
            e1 = tuple(map(np.array, e1))
            e2 = tuple(map(np.array, e2))
            assert (e1[1] == e2[0]).all()

            dir = e1[1] - e1[0]
            if (dir == [0, 1]).all():
                crossdir = np.array([1, 0])
            elif (dir == [1, 0]).all():
                crossdir = np.array([0, 1])
            else:
                assert False, dir
            center = e1[1]

            def to_edge(a, b):
                return (
                    tuple(map(int, a)),
                    tuple(map(int, b))
                )

            return all(
                crossedge in all_edges
                for crossedge in [
                    to_edge(center - crossdir, center),
                    to_edge(center, center + crossdir)
                ]
            )

        edges = {start}
        for e1 in follow_edge(start, +1):
            edges.add(e1)
        for e1 in follow_edge(start, -1):
            edges.add(e1)

        # print(f"... found a chain: {sorted(edges)}")

        final_edges = set()
        for e1, e2 in pairwise(chain(sorted(edges), [None])):
            if e2 is None:
                final_edges.add(e1)
                break
            final_edges.add(e1)
            if has_cross_edges(e1, e2):
                # print(f"... cross edges at {e1, e2}")
                break

        # print(f"... final side: {sorted(final_edges)}")
        remaining_edges -= final_edges
        side_count += 1

    # print(f"... side count: {side_count}")
    return side_count


def part1(filename, cost: Callable):
    grid = read_input(filename)
    rows, cols = map(range, grid.shape)

    # display(grid)

    nodes = set(product(rows, cols))
    def neighbours(pos):
        r, c = pos
        plant = grid[r, c]
        for dr, dc in [
            (-1, 0),
            (+1, 0),
            (0, -1),
            (0, +1)
        ]:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in nodes:
                continue
            if grid[nr, nc] != plant:
                continue
            yield (nr, nc)

    all_plants = set(
        grid[r, c] for r, c in product(rows, cols)
    )
    # print(all_plants)

    components = []
    for plant in all_plants:
        locs = set(locate(grid == plant))
        while locs:
            start = locs.pop()
            distvec, _ = bfs(start, None, neighbours)
            reached = set(
                pos
                for pos, dist
                in distvec.items()
                if dist != float('inf')
            )

            components.append((plant, reached))
            locs -= reached

    fencing_cost = 0
    for plant, places in components:
        cc_cost = cost(places, neighbours)
        # print(f"{plant} cost(places, neighbours) = {cc_cost}: {places}")
        fencing_cost += len(places) * cc_cost
    print(fencing_cost)


def part2(filename):
    pass

def usage(message):
    print(f'usage: {sys.argv[0]} [-1|-2] [--] input_file...')
    print(f'    {message}')
    sys.exit(1)

parts = {
    1: partial(part1, cost=perimeter),
    2: partial(part1, cost=sides),
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