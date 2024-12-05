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

def read_input(filename: str) -> np.typing.NDArray[str]:
     with open(filename) as f:
        s = f.readlines()
        s = map(str.rstrip, s)
        rules, editions = split(lambda line: line == "", s)

        s = map(partial(str.split, sep="|"), rules)
        s = map(partial(map, int), s)
        s = map(tuple, s)
        rules = list(s)

        s = map(partial(str.split, sep=","), editions)
        s = map(partial(map, int), s)
        s = map(list, s)
        editions = list(s)

        return rules, editions

def part2(filename):
    print("*** part1 ***")
    rules, editions = read_input(filename)
    print(rules, editions)
    rules = set(rules)

    def topsort(ed: list[int]):
        ed = ed[:]
        n = len(ed)
        changed = True
        while changed:
            changed = False
            for i in range(n-1):
                for j in range(i+1, n):
                    pi, pj = ed[i], ed[j]
                    if not (pi, pj) in rules:
                        ed[i], ed[j] = pj, pi
                        changed = True

        return ed

    total = 0
    print(f"{rules = }")
    for ed in editions:
        print(f"{ed = }")
        correct = True
        for i in range(len(ed) - 1):
            if not correct: break
            for j in range(i+1, len(ed)):
                pi, pj = ed[i], ed[j]
                if (pi, pj) not in rules:
                    print(f"... {(pi, pj)} in the wrong order")
                    correct = False
                    break
        print(f"... {correct = }")
        if not correct:
            ed = topsort(ed)
            print(f" ... sorted {ed = }")
            total += ed[len(ed) //2]

    print(total)

def part1(filename):
    print("*** part1 ***")
    rules, editions = read_input(filename)
    print(rules, editions)

    rules = set(rules)

    total = 0
    print(f"{rules = }")
    for ed in editions:
        print(f"{ed = }")
        correct = True
        for i in range(len(ed) - 1):
            for j in range(i+1, len(ed)):
                pi, pj = ed[i], ed[j]
                if (pi, pj) not in rules:
                    print(f"... {(pi, pj)} in the wrong order")
                    correct = False
                    break
        print(f"... {correct = }")
        if correct:
            total += ed[len(ed) //2]

    print(total)

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