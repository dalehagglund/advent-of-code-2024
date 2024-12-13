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

def sides_with_edgewalk(
        places: list[tuple[int, int]],
        neighbours
):
    # notes:
    #
    # - if you draw lines on the map grid so that there is
    #
    #   - a horizontal line between every pair of adjacent rows and at
    #     the top and bottom, and a
    #   - a vertical line between every pair of of adjacent columns
    #     and at the far left and right
    #
    #   these form a lattice of lines and points *between* the cells
    #   in the grid.
    #
    # - Note that the lattice points are distinct from *positions*,
    #   which are indices into the grid. Among others:
    #   - lattices points are in {(r, c) | 0 <= r <= nrow, 0 <= c <=
    #     ncol}, while
    #   - positions are in {(r, c) | 0 <= r < nrow, 0 <= c < ncol}
    #
    # - an *edge* is a pair of lattice points.
    #
    #   - IDEA: represent a latticepoint as a frozenset so that it's
    #     hashable and comparable without ordering issues.
    #   - Do we need edges except as adjacency information in a graph?
    #
    # - we know that "places" form a connected component.
    #
    # - we want to identify all the lattice points that are part of a
    #   boundary of `places`, and the adjacencies between them.
    #
    # - CLAIM?: every lattice point discovered above has exactly two
    #   neighbours, because that has to be true of a closed perimeter
    #
    #   - I think this is false, eg in the case of `ab0.txt`: The
    #     center lattice point (diagonally between the two Bs) is
    #     adjacent to all 4 of its neighbouring points.
    #   - Let's ignore the above for now.
    #
    # - how about this...
    #
    #   1. find all edges that separate a position from a different
    #      connected component. (note that the exterior of the grid
    #      counts as a connected component for this purpose.)
    #   2. choose one of those edges and note the other cc.
    #   3. now, follow edges separating the same two connected
    #      component until you return to the starting point, recording
    #      the turns along the way.
    #   4. the number of turns is the number of sides.
    #

    all_edges = set()
    edge_graph = defaultdict(set)
    for r, c in places:
        cc = ccnums[r, c]
        for dr, dc in [...]:
            ncc = ccnums[r + dr, c + dc]
            if cc == ncc: continue
            match (dr, dc):
                case (-1,  0): p1, p2 = (r,   c), (r,   c+1)
                case ( 1,  0): p1, p2 = (r+1, c), (r+1, c+1)
                case ( 0, -1): p1, p2 = (r,   c), (r+1, c)
                case ( 0,  1): p1, p2 = (r+1, c), (r+1, c+1)
                case _:
                    assert_never((dr, dc))
            edge_graph[p1].add((p1, cc, ncc, p2))
            all_edges.add((p1, cc, ncc, p2))

    def next_edges(p: tuple[int, int], cc: int) -> Iterable[tuple[int, int]]:
        for e in edge_graph[p]:
            p1, _, othercc, _ = e
            assert p1 == p, (p, p1)
            if othercc == cc:
                yield e
        return

    def walk_perimeter(start):
        visited = {start}
        _, _, othercc, p2 = start
        e = start
        while True:
            choices = list(
                filter(visited.__contains__, next_edges(p2, othercc))
            )
            assert len(choices) <= 1, len(choices)
            if len(choices) == 0:
                break
            e = only(choices)
            visited.add(e)

    def count_corners(perimeter: list):
        assert len(perimeter) >= 1
        count = 0
        def is_corner(e1, e2):
            pass

        for e1, e2 in pairwise(chain(perimeter, perimeter[-1])):

            if is_corner(e1, e2):
                count += 1
        return count

def looking_at_corner(cc, pos, dir) -> bool:
    neighbours = {
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
        # where pos is the lower left the diagonal direction.
        # Certainly we're adjacent to a corner, but since that corner
        # will be observed by the lower position, we shouldn't count
        # it in this case.
        (False, False, False): True,    # yes: inside corner
        (False, False,  True): False,   # no: wall continues ahead on left
        (False,  True, False): True,    # yes: inside corner, same cc to diag
        (False,  True,  True): False,   # no: see (1)
        ( True, False, False): False,   # no: wall continues ahead on right
        ( True, False,  True): True,    # yes: outside corner
        ( True,  True, False): False,   # no: see (1)
        ( True,  True,  True): False,   # no: no edges in this direction
    }

    def adjustpos(dir):
        r, c = pos
        dr, dc = dir
        return (r + dr, c + dc)

    assert pos in cc, (pos, cc)
    dirleft, dirright = neighbours[dir]
    s = [dirleft, dir, dirright]
    s = map(adjustpos, s)
    left, diag, right = s

    return truthtable[left in cc, diag in cc, right in cc]

def add_boundary(
        m: np.ndarray,
        fill,
        dtype=None,
) -> np.ndarray:
    nr, nc = m.shape
    newm = np.full((nr + 2, nc + 2), fill_value=fill, dtype=dtype)
    newm[1:-1, 1:-1] = m
    return newm

def part1(filename, cost: Callable):
    grid = add_boundary(
        read_input(filename),
        ".", dtype=np.dtypes.StringDType
    )
    ccnum = np.zeros(grid.shape)
    rows, cols = map(range, grid.shape)

    # display(grid)

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

    def set_ccnum(nodes: set[tuple[int, int]], n: int):
        ccnum[*np.transpose(list(nodes))] = n

    cccount = count(0)
    boundary = reachable((0, 0), neighbours)
    set_ccnum(boundary, next(cccount))

    all_plants = set(
        grid[r, c] for r, c in product(rows, cols)
    ) - {"."}
    print(all_plants)

    components = []
    for plant in all_plants:
        locs = set(locate(grid == plant))
        while locs:
            cc = reachable(locs.pop(), neighbours)
            components.append((plant, cc))
            set_ccnum(cc, next(cccount))
            locs -= cc

    print(f"{len(components) = }")
    perimeter_cost = 0
    corner_cost = 0
    for plant, places in components:
        pcost = perimeter(places, neighbours)
        ccost = 0
        for pos, dir in product(places, [
            (-1, -1),
            (-1, +1),
            (+1, -1),
            (+1, +1),
        ]):
            if looking_at_corner(places, pos, dir):
                ccost += 1
        print(f"{plant} perim = {pcost} corners = {ccost} area = {len(places)}")
        perimeter_cost += len(places) * pcost
        corner_cost += len(places) * ccost
    print(f"{perimeter_cost = } {corner_cost = }")

def reachable[T](
        start: T,
        neighbours: Callable[[T], Iterable[T]]
) -> set[T]:
    dist, _ = bfs(start, None, neighbours)
    return {
        n for n, d in dist.items() if d != float('inf')
    }

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