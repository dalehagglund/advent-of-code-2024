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
) -> defaultdict[str, set[str]]:
    with open(filename) as f:
        s = iter(f)
        s = map(str.rstrip, s)
        s = map(partial(str.split, sep="-"), s)

        graph = defaultdict(set)
        for c1, c2 in s:
            graph[c1].add(c2)
            graph[c2].add(c1)
        return graph

def part1(filename):
    network = read_input(filename)

    def connected(n1: str, n2: str, n3: str) -> bool:
        return (
            {n1, n2} <= network[n3]
            and {n1, n3} <= network[n2]
            and {n2, n3} <= network[n1]
        )

    s = combinations(network.keys(), r=3)
    s = filter(star(connected), s)
    s = filter(lambda t: any(n[0] == "t" for n in t), s)
    s = map(sorted, s)
    s = map(",".join, s)
    for i, item in enumerate(s):
        print(i, item)

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
    network = read_input(filename)

    # See the Bron-Kerbosch Algorithm page in wikipedia.
    # This code implements the simplest version, named
    # `BronKerbosch1`.
    #
    # There are a couple confusing things about the pseudo-code when
    # you translate it into python.
    #
    # (1) The pseudocode contains this iteration:
    #
    #   ```
    #   for each vertex v in P:
    #       bron_kerbosch1(...)
    #       P := P \ {v}
    #       X := X \cup {v}
    #   ```
    #
    # If you translate this to a python for loop, P is being modified
    # while it's be iterated over, which can lead to surprising
    # behaviour. I used a for loop initially (not even noticing the
    # problem), but then changed the the while-loop form below which I
    # believe is the *intent* of the pseudocode. Oddly, when I
    # implemented this as a for loop, it still worked: I suspect
    # because we're, in effect, deleting just *behind* the iteration,
    # and that probably works out ok just by co-incidence.
    #
    # (2) As written below, all of R, P, and X are mutable sets, and
    # so P is being modified as the function proceeds. This is
    # apparently ok, I think because each recursive call has freshly
    # computed sets to process.
    def bron_kerbosch(
            R: set[str],
            P: set[str],
            X: set[str]
    ) -> Iterator[set[str]]:
        if len(P) == 0 and len(X) == 0:
            yield R
            return
        while P:
            v = next(iter(P))
            yield from bron_kerbosch(
                R | {v},
                P & network[v],
                X & network[v]
            )
            P = P - {v}
            X = X | {v}

    max_clique = max(
        bron_kerbosch(set(), set(network.keys()), set()),
        key=len
    )
    # print(max_clique)
    print("password:", ",".join(sorted(max_clique)))

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