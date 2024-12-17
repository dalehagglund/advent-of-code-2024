import collections
import contextlib
from itertools import chain, count, cycle, product
from typing import (
    Callable,
    Iterable,
    Iterator,
    assert_never
)
from collections import Counter, defaultdict
from functools import partial

import heapq
import functools

type Neighbours[T] = Callable[[T], Iterable[T]]

def topsort[T](
        nodes: set[T],
        neighbours: Neighbours[T]
) -> Iterator[T]:
    s = iter(nodes)
    s = map(neighbours, s)
    s = chain.from_iterable(s)
    indeg = Counter(s)

    zeros = collections.deque(n for n, c in indeg.items() if c == 0)
    while zeros:
        u = zeros.popleft()
        assert indeg[u] == 0
        del indeg[u]
        yield u

        for v in neighbours(u):
            if indeg[v] == 0:
                continue
            indeg[v] -= 1
            if indeg[v] == 0:
                zeros.append(v)

    if len(indeg) > 0:
        raise ValueError("cycle detected")

def bfs[T](
        start: T,
        end: T | None,
        neighbours: Neighbours[T],
        verbose: int = 0,
        statsevery: int = 500
):
    return astar(
        start, end, neighbours,
        verbose=verbose, statsevery=statsevery
    )

def dijkstra[T](
        start: T,
        end: T | None,
        neighbours: Neighbours[T],
        edge_cost: Callable[[T, T], int] = lambda n1, n2: 1,
        verbose: int = 0,
        statsevery: int = 500
):
    return astar(
        start, end, neighbours, edge_cost,
        verbose=verbose, statsevery=statsevery
    )

def astar[T](
        start: set[T] | T,
        end: T | Callable[[T], bool] | None,
        neighbours: Neighbours[T],
        edge_cost: Callable[[T, T], int] | None = None,
        est_remaining: Callable[[T, T], int] | None = None,
        verbose: int = 0,
        statsevery: int = 500,
):
    if end is None and est_remaining is not None:
        raise ValueError("est_remaining not allowed if end is None")

    if edge_cost is None: edge_cost = lambda n1, n2: 1
    if est_remaining is None: est_remaining = lambda n1, n2: 0
    if end is None:
        reached_goal = lambda n: False
    elif callable(end):
        reached_goal = end
    else:
        reached_goal = lambda n: n == end

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
        if verbose: print(
            f"..."
            f" loops {loops}"
            f" updates {updates}"
            f" rescans {rescans}"
            f" assigned {len(dist)}"
            f" seen {len(seen)}"
        )

    if isinstance(start, set):
        for s in start:
            dist[s] = 0
            push(s, dist[s])
    else:
        dist[start] = 0
        push(start, dist[start])

    while len(q) > 0:
        node = pop()
        verbose > 1 and print(f"pop: {node = }")
        if reached_goal(node):
            verbose > 1 and print(f"found goal: {end}")
            break

        if node in seen:
            rescans += 1

        loops += 1
        if verbose > 0 and loops % statsevery == 0:
            display_stats()

        for n in neighbours(node):
            verbose > 1 and print(f"... neighbour {n}")
            cost = edge_cost(node, n)
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

@contextlib.contextmanager
def _around(before, after):
    before()
    try:
        yield
    finally:
        after()

def allpaths[T](
        path: list[T],
        nodes: set[T],
        neighbours: Callable[[T], Iterable[T]],
        keep: Callable[[list[T]], bool] = lambda path: True,
) -> Iterator[list[T]]:
    if keep(path):
        yield path.copy()
    for n in neighbours(path[-1]):
        if n in nodes:
            continue
        with (
            _around(partial(path.append, n), path.pop),
            _around(partial(nodes.add, n), partial(nodes.remove, n))
        ):
            yield from allpaths(path, nodes, neighbours, keep)