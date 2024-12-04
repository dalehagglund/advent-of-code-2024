import sys

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
    if item is None:
        raise ValueError("no first item")
    if next(items, None) is None:
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

def read_input(filename: str) -> np.typing.NDArray[str]:
     with open(filename) as f:
        s = f.readlines()
        s = map(str.rstrip, s)
        s = map(list, s)
        return np.array(list(s), dtype=np.dtypes.StringDType)

def part1(filename, handledo=False):
    print("*** part1 ***")
    grid = read_input(filename)
    print(grid)
    nrow, ncol = grid.shape

    target = "XMAS"
    dirs = list(
        product([-1, 0, +1], [-1, 0, +1])
    )

    xlocs = locate(grid == "X")

    def looking_at(
            target: str,
            xpos: tuple[int, int], 
            dir: tuple[int, int]
    ) -> bool:
        r, c = xpos
        dr, dc = dir

        print(f"> looking_at {(target, xpos, dir) = }")

        def ray(start, dir):
            r, c = start
            dr, dc = dir
            while r in range(nrow) and c in range(ncol):
                yield r, c
                r, c = r + dr, c + dc

        for i, (char, coord) in enumerate(zip(target, ray(xpos, dir))):
            # print(f"... {(i, char, coord, grid[coord]) = }")
            if char != grid[coord]:
                return False
        return i == len(target) - 1

    matches = 0
    for xpos, dir in product(xlocs, dirs):
        if looking_at(target, xpos, dir):
            matches += 1
    print(matches)

def part2(filename):
    print("*** part2 ***")
    grid = read_input(filename)
    print(grid)
    nrow, ncol = grid.shape

    target = "MAS"
    corners = [     # cyclic order
        (-1, -1),
        (-1, +1),
        (+1, +1),
        (+1, -1),
    ]

    s = zip(cycle(corners), islice(cycle(corners), 1, None))
    cornerpairs = list(islice(s, 4))

    dirfrom = {
        (-1, -1): (+1, +1),
        (-1, +1): (+1, -1),
        (+1, +1): (-1, -1),
        (+1, -1): (-1, +1),
    }

    print(f"{corners = }")
    print(f"{cornerpairs = }")
    print(f"{dirfrom = }")

    alocs = locate(grid == "A")

    def looking_at(
            target: str,
            xpos: tuple[int, int], 
            dir: tuple[int, int]
    ) -> bool:
        r, c = xpos
        dr, dc = dir

        # print(f"> looking_at {(target, xpos, dir) = }")

        def ray(start, dir):
            r, c = start
            dr, dc = dir
            while r in range(nrow) and c in range(ncol):
                yield r, c
                r, c = r + dr, c + dc

        i = -1
        for i, (char, coord) in enumerate(zip(target, ray(xpos, dir))):
            # print(f"... {(i, char, coord, grid[coord]) = }")
            if char != grid[coord]:
                return False
        return i == len(target) - 1

    matches = 0
    
    for apos, (c1, c2) in product(alocs, cornerpairs):
        ar, ac = apos
        dr1, dc1 = c1
        dr2, dc2 = c2

        c1good = looking_at(target, (ar + dr1, ac + dc1), dirfrom[c1])
        c2good = looking_at(target, (ar + dr2, ac + dc2), dirfrom[c2])

        if c1good and c2good: matches += 1
    
    print(matches)

def main():
    dispatch = {
        "1": part1, 
        "2": part2,
    }
    parts = sys.argv[1]
    filename = sys.argv[2]
    for part in parts:
        dispatch[part](filename)

if __name__ == '__main__':
    main()