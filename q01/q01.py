import collections
from dataclasses import dataclass
import functools
import heapq
import math
import operator
import sys

from pathlib import Path
from functools import partial
from collections import Counter, defaultdict, deque
from itertools import batched, chain, combinations, count, cycle, groupby, islice, pairwise, permutations, product, repeat, takewhile, tee
import textwrap
import time
from typing import Callable, Iterable, Iterator, NamedTuple, Never, Optional, assert_never
import itertools
import numpy as np

def star(f):
    return lambda t: f(*t)

def observe[T](f: Callable, items: Iterable[T]) -> Iterator[T]:
    for item in items:
        f(item)
        yield item

def divide[T](
        cutpoint: Callable[[T], bool],
        items: Iterable[T]
) -> Iterator[list[T]]:
    batch = []
    for item in items:
        if cutpoint(item):
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

def mapnth(f, n):
    def update(t: tuple) -> tuple:
        return t[:n] + (f(t[n]),) + t[n+1:]
    return update

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
        verbose: int = 0
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

    updates, loops, rescans = 0, 0, 0

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
        if verbose > 0 and loops % 500 == 0:
            print(
                f"..."
                f" loops {loops}"
                f" updates {updates}"
                f" rescans {rescans}"
                f" assigned {len(dist)}"
                f" seen {len(seen)}"
            )

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
    return dist, prev

def nth(n, t):
    return t[n]

def read_input(filename: str) -> tuple[list[int], list[int]]:
     with open(filename) as f:
        s = f.readlines()
        s = map(str.rstrip, s)
        s = map(str.split, s)
        s = map(partial(map, int), s)
        s = map(tuple, s)
        u, v = tee(s)
        return list(map(partial(nth, 0), u)), list(map(partial(nth, 1), v))

def part1(filename):
    locs1, locs2 = read_input(filename)
    locs1, locs2 = sorted(locs1), sorted(locs2)

    counts = Counter(locs2)

    distance = sum(abs(n1 - n2) for n1, n2 in zip(locs1, locs2))
    similarity = sum(n * counts[n] for n in locs1) 
    print(f"{(distance, similarity) = }")

def part2(filename):
    grid = read_input(filename)

def main():
    dispatch = dict(
        p1=part1, 
        p2=part2,
    )
    filename = sys.argv[1]
    dispatch[Path(filename).name[:2]](filename)

if __name__ == '__main__':
    main()