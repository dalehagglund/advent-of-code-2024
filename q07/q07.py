import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import heapq
import math
import operator
import re
import textwrap
import time
import numpy as np

from dataclasses import dataclass
from pathlib import Path

import collections
from collections import (
    Counter,
    defaultdict,
    deque
)

import functools
from functools import partial

import itertools
from itertools import (
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
    tee
)

from typing import (
    Any,
    Callable, 
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

def star(f):
    return lambda t: f(*t)

def nth(n, t):
    return t[n]

def mapnth(f, n):
    def update(t: tuple) -> tuple:
        return t[:n] + (f(t[n]),) + t[n+1:]
    return update

def observe[T](
        f: Callable[[T], Any], 
        items: Iterable[T]
) -> Iterator[T]:
    for item in items:
        f(item)
        yield item

def consume(iterator, n=None):
    # Use functions that consume iterators at C speed.
    if n is None:
        collections.deque(iterator, maxlen=0)
    else:
        next(islice(iterator, n, n), None)

def window[T](
        items: Iterable[T], 
        n: int
) -> Iterator[tuple[T, ...]]:
    iters = tee(items, n)
    for it, skip in zip(iters, count()):
        consume(it, skip)
    return zip(*iters)

def first[T](
        items: Iterable[T], 
        strict: bool = False
) -> T | None:
    items = iter(items)
    if not strict:
        return next(items, None)
    if (item := next(items, None)) is None:
        raise ValueError("no first item")
    return item

def only[T](
        items: Iterable[T]
) -> T:
    items = iter(items)
    item = next(items, None)
    print(f"{item = }")
    if item is None:
        raise ValueError("no first item")
    if next(items, None) is not None:
        raise ValueError("expecting exactly one item")
    return item

def split[T](
        separator: Callable[[T], bool],
        items: Iterable[T]
) -> Iterator[list[T]]:
    batch = []
    for item in items:
        if separator(item):
            yield batch
            batch = []
            continue
        batch.append(item)
    if len(batch) > 0:
        yield batch

def locate(g):
    # return non-zero locations in g in a more useful form for 
    # my python code. shoudl probably be called with an array resulting
    # from a boolean operation of some sort.
 
    s = zip(*g.nonzero())
    s = map(partial(map, int), s)
    s = map(tuple, s)
    return s

def display(g, prefix="    "):
    nrow, _ = g.shape
    for r in range(nrow):
        print(prefix, "".join(g[r, :].flat), sep="")

def astar[T](
        start: T, 
        end: T | None, 
        neighbours: Callable[[T], Iterator[T]], 
        edge_cost: Callable[[T, T], int] = lambda n1, n2: 1, 
        est_remaining: Callable[[T, T], int] = lambda n1, n2: 0,
        verbose: int = 0,
        statsevery: int = 500,
    ):
    from collections import defaultdict
    import functools

    seen: set[T] = set()
    dist: defaultdict[T, float | int] = defaultdict(lambda: float('inf'))
    prev: defaultdict[T, T | None] = defaultdict(lambda: None)

    est_remaining = functools.cache(est_remaining)
    edge_cost = functools.cache(edge_cost)

    q = []
    heapq.heapify(q)

    sequence = count(1)
    def push(item, prio):
        heapq.heappush(q, (prio, next(sequence), item))
    def pop():
        _, _, item = heapq.heappop(q)
        return item

    loops, updates, rescans = 0, 0, 0
    def display_stats():
        print(
            f"..."
            f" loops {loops}"
            f" updates {updates}"
            f" rescans {rescans}"
            f" assigned {len(dist)}"
            f" seen {len(seen)}"
        )

    dist[start] = 0
    push(start, dist[start])
    while len(q) > 0:
        node = pop()
        verbose > 1 and print(f"pop: {node = }")
        if node == end:
            verbose > 1 and print(f"found goal: {end}")
            break

        if node in seen:
            rescans += 1
        
        loops += 1
        if verbose > 0 and loops % statsevery == 0:
            display_stats()

        for n in neighbours(node):
            verbose > 1 and print(f"... neighbour {n}")
            cost = edge_cost(n, node)
            ndist = dist[node] + cost
            verbose > 1 and print(f"... {node} -> {n}: delay {cost} newdist {ndist}")
            if n not in dist or ndist < dist[n]:
                if n in dist: updates += 1
                prev[n] = node
                dist[n] = ndist
                push(n, dist[n] + est_remaining(n, end))

        seen.add(node)

    verbose > 1 and print("after search")
    display_stats()
    return dist, prev

#
# spent a while working this out over in the experiments directory.
#
def follow_ray(g, start, dir):
    dr, dc = dir
    nr, nc = g.shape
    r, c = start

    if (dr, dc) == (0, 0):
        raise ValueError("direction vector is (0, 0)")

    # note that `flipud(fliplr(g)) == fliplr(flipud(g))`. see below
    # for a test confirming that.

    if dc == -1:
        # if the column direction is to the left, flip each row and
        # flip the column index correspondingly
        g = np.fliplr(g)
        c = nc - 1 - c
    if dr == -1:
        # similarly, if the row direction is upwards, flip each column
        # and flip the row index correspondingly
        g = np.flipud(g)
        r = nr - 1 - r

    if 0 not in (dr, dc):
        return np.diag(g, k=c-r)[min(r,c):]
    elif dc == 0:
        return g[r:, c]
    elif dr == 0:
        return g[r, c:]
    else:
        assert_never(dir)

def read_input(
        filename: str
) -> np.ndarray:
    with open(filename) as f:
        s = f.readlines()
        s = map(str.rstrip, s)
        s = map(list, s)
        return np.array(list(s), np.dtypes.StringDType)

def part1(filename):
    print("*** part1 ***")
    grid = read_input(filename)
    print(grid)

    directions = {
        "^": (-1,  0),
        ">": ( 0, +1),
        "v": (+1,  0),
        "<": ( 0, -1),
    }

    right_turn = {
        (-1,  0): ( 0, +1),
        ( 0, +1): (+1,  0),
        (+1,  0): ( 0, -1),
        ( 0, -1): (-1,  0),
    }

    guard = only(locate(
        (grid == "^") |
        (grid == ">") |
        (grid == "v") |
        (grid == "<")
    ))

    def track(
            pos: tuple[int, int],
            dir: tuple[int, int]
    ) -> Iterator[tuple[int, int]]:
        rows, cols = map(range, grid.shape)
        r, c = pos
        dr, dc = dir

        while True:
            yield r, c
            if not(r + dr in rows and c + dc in rows):
                break
            if grid[r + dr, c + dc] == "#":
                dr, dc = right_turn[dr, dc]
            r, c = r + dr, c + dc

    dir = directions[grid[guard]]
    print(f"{(guard, dir) = }")

    locations = list(track(guard, dir))
    print(locations)
    print(len(set(locations)))
    

class Pos(NamedTuple):
    r: int
    c: int
    def __add__(self, other: "Pos"):
        return Pos(self.r + other.r, self.c + other.c)
    def __mul__(self, scale: int):
        return Pos(self.r * scale, self.c * scale)

def part2(filename):
    print("******** PART 2 ********")
    grid = read_input(filename)

    dir_to_pos = {
        "^": Pos(-1,  0),
        ">": Pos( 0, +1),
        "v": Pos(+1,  0),
        "<": Pos( 0, -1),
    }

    pos_to_dir = dict((v, k) for k, v in dir_to_pos.items())

    right_turn = {
        Pos(-1,  0): Pos( 0, +1),
        Pos( 0, +1): Pos(+1,  0),
        Pos(+1,  0): Pos( 0, -1),
        Pos( 0, -1): Pos(-1,  0),
    }

    guard = Pos(*only(locate(
        (grid == "^") |
        (grid == ">") |
        (grid == "v") |
        (grid == "<")
    )))

    def track(
            grid,
            pos: Pos,
            dir: Pos
    ) -> Iterator[tuple[Pos, Pos]]:
        rows, cols = map(range, grid.shape)
        while True:
            yield pos, dir
            while True:
                nextpos = pos + dir
                if not (nextpos.r in rows and nextpos.c in cols):
                    return
                if grid[nextpos] not in "#O":
                    break
                dir = right_turn[dir]
            pos = pos + dir

    def cycle(path: Iterable[tuple[Pos, Pos]]) -> bool:
        visited = set()
        for state in path:
            if state in visited:
                return True
            visited.add(state)
        return False

    dir = dir_to_pos[grid[guard]]
    grid[guard] = "."

    locations: set[Pos] = set()
    possible_obstacles = [
        pos 
        for pos, _
        in track(grid.copy(), guard, dir)
    ]
    for i, pos in enumerate(possible_obstacles[1:]):
        if i % 1000 == 0:
            print(f"{i}: {len(locations)}")
        assert (grid != "O").all() 
        grid[pos] = "O"
        if cycle(track(grid, guard, dir)):
            locations.add(pos)
        grid[pos] = "."

    if len(locations) < 20: 
        print(f"{locations = }")
    print(f"{len(locations) = }")

def part2broken(filename):
    print("******** PART 2 ********")
    grid = read_input(filename)
    # display(grid)

    directions = {
        "^": Pos(-1,  0),
        ">": Pos( 0, +1),
        "v": Pos(+1,  0),
        "<": Pos( 0, -1),
    }

    right_turn = {
        Pos(-1,  0): Pos( 0, +1),
        Pos( 0, +1): Pos(+1,  0),
        Pos(+1,  0): Pos( 0, -1),
        Pos( 0, -1): Pos(-1,  0),
    }

    guard = only(locate(
        (grid == "^") |
        (grid == ">") |
        (grid == "v") |
        (grid == "<")
    ))

    def turns(
            pos: tuple[int, int],
            dir: tuple[int, int]
    ) -> Iterator[tuple[
        Pos,
        Pos
    ]]:
        rows, cols = map(range, grid.shape)
        r, c = pos
        dr, dc = dir

        while True:
            if not(r + dr in rows and c + dc in rows):
                break
            if grid[r + dr, c + dc] == "#":
                dr, dc = right_turn[dr, dc]
                yield Pos(r, c), Pos(dr, dc)
            r, c = r + dr, c + dc

    def dist(p1, p2) -> int: # mahattan distance
        r1, c1 = p1
        r2, c2 = p2
        return abs(r1 - r2) + abs(c1 - c2)

    dir = Pos(*directions[grid[guard]])
    # print(f"{(guard, dir) = }")
    grid[guard] = "." # guard symbol gets in the way later

    def minmax[T](items: Iterable[T]) -> tuple[T, T]:
        items = list(items)
        return min(items), max(items)

    def missing_corner(p1, p2, p3) -> tuple[int, int]:
        minc, maxc = minmax(c for _, c in (p1, p2, p3))
        minr, maxr = minmax(r for r, _ in (p1, p2, p3))

        for c in [
            (minr, minc),
            (minr, maxc),
            (maxr, minc),
            (maxr, maxc),
        ]:
            if c not in (p1, p2, p3):
                return Pos(c[0], c[1])
            
        raise ValueError("no missing corner?")

    def advance(p1, target, dir):
        r, c = p1
        dr, dc = dir
        while grid[r, c] == "." and (r, c) != target: 
            r, c = r + dr,c + dc
        if (r, c) != target:
            r, c = r - dr, c - dc
        return Pos(r, c)

    locations = list(turns(guard, dir))
    windows = list(window(locations, 3))
    obstacles: list[Pos] = []

    print(f"{len(locations), len(windows) = }")
    # print(f"{locations = }")
    # print(f"{windows = }")
    for i, ((p1, d1), (p2, d2), (p3, d3)) in enumerate(windows):
        print(f"*** {i}:points {p1} {p2} {p3}...")

        for i in range(5):
            print("... pass ",i)
            g2 = grid.copy()
            g2[p1] = '1'
            g2[p2] = '2'
            g2[p3] = '3'
            # display(g2)

            corner = missing_corner(p1, p2, p3)
            reachable = advance(p3, corner, d3)

            # print(f"... ... missing corner {corner}")
            # print(f"... ... stopped at {reachable}")

            g2[corner] = '*'
            g2[reachable] = 'R'
            display(g2)

            if reachable == corner and grid[reachable + d3] == ".":
                o = reachable + d3
                if o not in obstacles:
                    obstacles.append(reachable + d3)
                print(f"... POSSIBLE! {o}")
                break

            p1, p2, p3 = p2, p3, reachable
            d3 = right_turn[d3]

    print(obstacles)
    
def main():
    dispatch = {
        "1": part1, 
        "2": part2,
    }
    parts = sys.argv[1]
    filename = sys.argv[2]
    for part in sorted(parts):
        dispatch[part](filename)

if __name__ == '__main__':
    main()