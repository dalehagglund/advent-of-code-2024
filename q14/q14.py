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

from dataclasses import dataclass
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

from aoc.search import astar, allpaths, bfs
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

type Pos = complex
type Vel = complex

def innermap[T, U](
        f: Callable[[T], U],
        s: Iterable[Iterable[T]]
) -> Iterable[Iterable[U]]:
    return map(partial(map, f), s)

def read_input(
        filename: str
) -> list[tuple[
    tuple[int, int],
    tuple[int, int]
]]:
    findall: Callable[[str, str], list[str]] = re.findall
    with open(filename) as f:
        s = iter(f)
        s = map(partial(findall, r"(-?\d+)"), s)
        s = innermap(int, s)
        s = map(partial(batched, n=2), s)
        s = map(tuple, s)
        return list(s)

def part1(filename, seconds=100):
    guards = read_input(filename)

    xmod = 1 + max(x for (x, _), _ in guards)
    ymod = 1 + max(y for (_, y), _ in guards)
    print(f"{xmod, ymod = }")
    assert xmod % 2 == ymod % 2 == 1

    pos = [ complex(*pos) for pos, _ in guards ]
    vel = [ complex(*vel) for _, vel in guards ]
    final_positions = [
        p + seconds * v
        for p, v in zip(pos, vel)
    ]

    ox, oy = xmod // 2, ymod // 2
    print(f"{ox, oy = }")

    ul, ur, ll, lr = 0, 0, 0, 0
    for c in final_positions:
        x, y = c.real % xmod, c.imag % ymod
        # print(f"{x, y = }", end="")
        if   x < ox and y < oy: ul += 1; #print(" ... ul")
        elif x < ox and y > oy: ll += 1; #print(" ... ll")
        elif x > ox and y < oy: ur += 1; #print(" ... ur")
        elif x > ox and y > oy: lr += 1; #print(" ... lr")
        else:
            print(f"... skipped {x, y}")

    print(f"{ul, ur, ll, lr = }")
    print(ul * ur * ll * lr)

def part2(filename):
    guards = read_input(filename)

    xmod = 1 + max(x for (x, _), _ in guards)
    ymod = 1 + max(y for (_, y), _ in guards)
    print(f"{xmod, ymod = }")

    pos = [ complex(*pos) for pos, _ in guards ]
    vel = [ complex(*vel) for _, vel in guards ]

    def consecutive_runs[T](
            items: Iterable[T],
            key: Callable[[T], int]
    ) -> Iterable[list[T]]:
        item = first(items)
        if item is None:
            return
        run, kprev = [item], key(item)
        for item in items:
            kcur = key(item)
            if kcur != kprev + 1:
                yield run
                run, kprev = [item], kcur
                continue
            run.append(item)
            kprev = kcur
        if len(run) > 0:
            yield run

    def reduce(c: complex) -> complex:
        return complex(c.real % xmod, c.imag % ymod)

    def vlines(pos: list[complex], minlen=10):
        s = sorted(pos, key=lambda c: (c.real, c.imag))
        for x, col in groupby(s, key=lambda c: c.real):
            yield from filter(
                lambda r: len(r) >= minlen,
                consecutive_runs(col, key=lambda c: int(c.imag))
            )

    def hlines(pos: list[complex], minlen=10):
        s = sorted(pos, key=lambda c: (c.imag, c.real))
        for y, row in groupby(s, key=lambda c: c.imag):
            yield from filter(
                lambda r: len(r) >= minlen,
                consecutive_runs(row, key=lambda c: int(c.real))
            )

    def update_positions(step:int = 1):
        for i in range(len(pos)):
            pos[i] = reduce(pos[i] + step * vel[i])

    def display_pos():
        m = np.full((ymod, xmod), fill_value=".", dtype=np.dtypes.StringDType)
        for p in pos:
            m[int(p.imag), int(p.real)] = "X"
        display(m)

    display_pos()
    for i in range(1, 10000):
        update_positions()
        hl = list(hlines(pos, minlen=5))
        vl = list(vlines(pos, minlen=5))
        nh = len(hl)
        nl = len(vl)

        if nh and nl:
            print(f"... candiate {i}")
            display_pos()


def usage(message):
    print(f'usage: {sys.argv[0]} [-1|-2] [--] input_file...')
    print(f'    {message}')
    sys.exit(1)

parts = {
    1: part1,
    2: part2,
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