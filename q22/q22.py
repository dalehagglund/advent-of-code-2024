import dataclasses
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import heapq
import math
from math import gcd
import operator
import re
import textwrap
import time
import numpy as np

from bisect import bisect_left, bisect_right

from dataclasses import dataclass, field, replace
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

from aoc.search import astar, allpaths, bfs #, path_to
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

def reachable[T](
        start: T,
        neighbours: Callable[[T], Iterable[T]]
) -> set[T]:
    dist, _ = bfs(start, None, neighbours)
    return {
        n for n, d in dist.items() if d != float('inf')
    }

import numpy as np
import numpy.typing as npt
from aoc.np import (
    locate,
    display,
    follow_ray
)

def innermap[T, U](
        f: Callable[[T], U],
        s: Iterable[Iterable[T]]
) -> Iterable[Iterable[U]]:
    return map(partial(map, f), s)

def read_input(
        filename: str
) -> list[int]:
    with open(filename) as f:
        s = iter(f)
        s = map(str.rstrip, s)
        s = map(int, s)
        return list(s)

def mix(n: int, m: int) -> int:
    return n ^ m
def prune(n: int) -> int:
    return n & 0xffffff
def next_secret(s: int) -> int:
    s = prune(mix(s, s * 64))
    s = prune(mix(s, s // 32))
    s = prune(mix(s, s * 2048))
    return s

def part1(filename):
    buyers = read_input(filename)
    for _ in range(2000):
        for i in range(len(buyers)):
            buyers[i] = next_secret(buyers[i])
    secret_total = 0
    for i, s in enumerate(buyers):
        secret_total += s
        print(f"{i}: {s}")
    print(f"{secret_total = }")

def iterate(f, n) -> Iterator[int]:
    while True:
        yield n
        n = f(n)

def sliding_window[T](
        items: Iterable[T],
        n: int
) -> Iterator[tuple[T, ...]]:
    items = iter(items)
    window = collections.deque(islice(items, n-1), maxlen=n)
    for x in items:
        window.append(x)
        yield tuple(window)

def part2(filename):
    buyers = read_input(filename)

    s = iterate(next_secret, 123)
    s = islice(s, 100)
    s = map(lambda s: s%10, s)
    s = pairwise(s)
    s = map(star(lambda p1, p2: (p2, p2 - p1)), s)

    c = dict()
    for i, quad in sliding_window(s, 4):
        (_, d1), (_, d2), (_, d3), (p4, d4) = quad


def usage(message):
    print(f'usage: {sys.argv[0]} [-1|-2] [--] input_file...')
    print(f'    {message}')
    sys.exit(1)

parts = {
    1: partial(part1),
    2: partial(part2),
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