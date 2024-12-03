import collections
from dataclasses import dataclass
import functools
import heapq
import math
import operator
import re
import sys
import pytest
import hypothesis
from hypothesis import given, strategies as st

from pathlib import Path
from functools import partial
from collections import Counter, defaultdict, deque
from itertools import batched, chain, combinations, count, cycle, filterfalse, groupby, islice, pairwise, permutations, product, repeat, takewhile, tee
import textwrap
import time
from typing import Callable, Iterable, Iterator, NamedTuple, Never, Optional, assert_never
import itertools
import numpy as np

def star(f):
    return lambda t: f(*t)

def nth(n, t):
    return t[n]

def mapnth(f, n):
    def update(t: tuple) -> tuple:
        return t[:n] + (f(t[n]),) + t[n+1:]
    return update

def observe[T](f: Callable, items: Iterable[T]) -> Iterator[T]:
    for item in items:
        f(item)
        yield item

def first[T](items: Iterable[T]) -> T | None:
    items = iter(items)
    return next(items, None)

def only[T](items: Iterable[T]) -> T:
    items = iter(items)
    try:
        item = next(items)
    except StopIteration:
        raise ValueError("empty iterable")
    try:
        next(items)
        raise ValueError("more than one item")
    except StopIteration:
        pass
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

def read_input(filename: str) -> list[str]:
     with open(filename) as f:
        s = f.readlines()
        s = map(str.rstrip, s)
        return list(s)

def part1(filename, trydeletes=False):
    print("*** part1 ***")
    lines = read_input(filename)

    ops = r"""(?x)
        (mul|do|don't)\(((\d+),(\d+))?\)
    """

    total = 0
    enabled = True

    for prog in lines:
        for m in re.finditer(ops, prog):
            # print(f"{enabled = }")
            # print(f"{m = }")
            op = m.group(1)
            # print(op)
            if op == "do":
                enabled = True
            elif op == "don't":
                enabled = False
            elif op == "mul":
                n1, n2 = map(int, (m.group(3), m.group(4)))
                if enabled:
                    total += n1 * n2
            else:
                assert False

    print(total)

def part2(filename):
    print("*** part2 ***")
    grid = read_input(filename)

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