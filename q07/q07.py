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

class timer[T]:
    def __init__(self, clock: Callable[[], T] = time.perf_counter):
        self._laps = []
        self._clock = clock
        self._state = "stopped"
    def _click(self):
        self._laps.append(self._clock())
    def start(self) -> "timer":
        if self._state == "running":
            raise ValueError("timer already running")
        self._state = "running"
        self._laps = []
        self._click()
        return self
    def lap(self):
        if self._state != "running":
            raise ValueError("timer not running")
        self._click()
    def stop(self):
        if self._state == "stopped":
            raise ValueError("timer already stopped")
        self._click()
        self._state = "stopped"
    def elapsed(self):
        if self._state == "stopped":
            return self._laps[-1] - self._laps[0]
        else:
            return self._clock() - self._laps[0]

    def __enter__(self):
        self.start()
        return self
    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.stop()
        # let the exception, if any, happen normally
        return False

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

def interleave(*iterables: Iterable) -> Iterator:
    iterators = cycle(iter(it) for it in iterables)
    sentinel = object()

    remaining = len(iterables)
    while remaining > 0:
        while True:
            it = next(iterators)
            value = next(it, sentinel) 
            if value == sentinel:
                break
            yield value
        remaining -= 1
        nexts = cycle(islice(iterators, remaining))

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
) -> list[tuple[int, list[int]]]:
    with open(filename) as f:
        s = f.readlines()
        s = map(str.rstrip, s)
        s = map(partial(str.split, sep=": "), s)
        s = map(
            star(lambda left, right: (
                int(left), 
                [int(s) for s in right.split(" ")]
            )),
            s
        )
        return list(s)

def iter_solvable(target, operands, dummy, part2) -> bool:
    def gen():
        target, position = (yield)
        if position == 0:
            yield ("return", target == operands[position])
            return
        
        quot, rem = divmod(target, operands[position])
        if rem == 0 and (yield ("recur", quot, position-1)):
            yield ("return", True)
            return
        
        reduced = target - operands[position]
        if reduced >= 0 and (yield ("recur", reduced, position-1)):
            yield ("return", True)
            return
        
        suffix, t = str(operands[position]), str(target)
        if (
            part2
            and t.endswith(suffix)
            and len(prefix := t[:-len(suffix)]) > 0
            and (yield ("recur", int(prefix), position-1))
        ):
            yield ("return", True)
            return
        
        yield ("return", False)

    @dataclass
    class Task:
        gen: Generator
        sendval: Any

    def new():
        g = gen()
        next(g)
        return g


    stack: list[Task] = []
    stack.append(Task(new(), (target, len(operands) - 1)))
    while True:
        task = stack[-1]
        action, *response = task.gen.send(task.sendval)
        if action == "recur":
            stack.append(Task(new(), response))
            continue
        stack.pop()
        task.gen.close()
        if len(stack) == 0:
            return response[0]
        stack[-1].sendval = response[0]

def recur_solvable(target, operands, position, part2) -> bool:
    if position == 0:
        return target == operands[position]
    
    args = dict(
        operands=operands,
        position=position-1,
        part2=part2,
    )
    
    quot, rem = divmod(target, operands[position])
    if rem == 0 and recur_solvable(quot, **args):
        return True
    
    reduced = target - operands[position]
    if reduced >= 0 and recur_solvable(reduced, **args):
        return True
    
    # I found understanding this just a bit tricky: we want to find
    # some prefix such that _prefix || operand_ is equal to _target_.
    # This is the same as saying that
    # 
    # - the operand must be a *suffix* of the current target, and 
    # - the prefix can't be empty
    # 
    # if those are both true, then the new target is the integer value
    # of the prefix
    suffix, t = str(operands[position]), str(target)
    if (
        part2
        and t.endswith(suffix)
        and len(prefix := t[:-len(suffix)]) > 0
        and recur_solvable(int(prefix), **args)
    ):
        return True
    
    return False

def part1(filename, part2=False):
    print("*** part1 ***")
    equations = read_input(filename)
    # print(equations)

    calibration = 0
    with timer() as t:
        for i, (result, operands) in enumerate(equations):
            assert len(operands) >= 2
            if iter_solvable(result, operands, len(operands) - 1, part2=part2):
                calibration += result
    print(calibration)
    print(f"{t.elapsed() = }")

def part2(filename):
    print("******** PART 2 ********")

def main():
    dispatch = {
        "1": part1, 
        "2": partial(part1, part2=True),
    }
    parts = sys.argv[1]
    filename = sys.argv[2]
    for part in sorted(parts):
        dispatch[part](filename)

if __name__ == '__main__':
    main()