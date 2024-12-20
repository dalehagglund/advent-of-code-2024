import collections
import contextlib
from itertools import chain, count, cycle, product
from typing import (
    Callable,
    Iterable,
    Iterator,
    Protocol,
    assert_never,
    assert_type
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

class Indexable[K, V](Protocol):
    def __getitem__(self, k: K, /) -> V: ...
    def __setitem__(self, k: K, v: V, /): ...

type DistVector[K] = Indexable[K, float]

def astar[T](
        start: set[T] | T,
        end: T | Callable[[T], bool] | None,
        neighbours: Neighbours[T],
        edge_cost: Callable[[T, T], int] | None = None,
        est_remaining: Callable[[T, T], int] | None = None,
        verbose: int = 0,
        statsevery: int = 500,
        dist_factory: Callable[[], DistVector[T]] | None = None,
):
    if end is None and est_remaining is not None:
        raise ValueError("est_remaining not allowed if end is None")

    if edge_cost is None:
        edge_cost = lambda n1, n2: 1
    if est_remaining is None:
        est_remaining = lambda n1, n2: 0
    if dist_factory is None:
        dist_factory = lambda: defaultdict[T, float](lambda: float('inf'))

    if end is None:
        reached_goal = lambda n: False
    elif callable(end):
        reached_goal = end
    else:
        reached_goal = lambda n: n == end

    seen: set[T] = set()
    # dist: defaultdict[T, float | int] = defaultdict(lambda: float('inf'))
    dist: DistVector[T] = dist_factory()
    prev: defaultdict[T, T | None] = defaultdict(lambda: None)

    est_remaining = functools.cache(est_remaining)
    edge_cost = functools.cache(edge_cost)

    q = []
    heapq.heapify(q)
    sequence = count(1)
    def push(item, prio):
        heapq.heappush(q, (prio, next(sequence), item))
    def pop() -> T:
        _, _, item = heapq.heappop(q)
        return item

    loops, assigned, updates, rescans = 0, 0, 0, 0
    def display_stats():
        if verbose: print(
            f"..."
            f" loops {loops}"
            f" updates {updates}"
            f" rescans {rescans}"
            f" assigned {assigned}"
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
            # before introducing the `DistVec` idea and the
            # `make_distvec` parameter, `dist` was always a
            # `defaultdict`, and the next line was
            #
            #     if n in distvec and ndist < dist[n]: ...
            #
            # now, we can't assume that the `in` operator works in the
            # way we wanted (consider a numpy array indexed by
            # co-ordinate pairs).
            if ndist < dist[n]:
                if dist[n] == float('inf'):
                    assigned += 1   # the first assignment of dist[n]
                else:
                    updates += 1    # a re-assignment of dist[n]
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