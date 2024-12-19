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

def innermap[T, U](
        f: Callable[[T], U],
        s: Iterable[Iterable[T]]
) -> Iterable[Iterable[U]]:
    return map(partial(map, f), s)

def read_input(
        filename: str
):
    with open(filename) as f:
        s = iter(f)
        s = map(str.rstrip, s)
        left, designs = split(lambda line: line == "", s)
        towels = only(left).split(", ")
        return towels, designs

def part1(filename):
    towels, designs = read_input(filename)
    min_towel = min(len(t) for t in towels)
    #print(towels, designs)

    # note: part 1 is solveable by bfs search, moving backwards
    # through the state space. I wrote this cached cached dynamic
    # programming solution first anyway, though, in part because I
    # expected part 2 to be about the total *number* of ways, which it
    # was.

    @functools.cache
    def count_solutions(design: str, depth:int = 0) -> int:
        # def trace(*args): print(" " * depth, *args, sep="")
        def trace(*args): pass

        trace(f"> {design = }")
        if design == "":
            return 1

        count = 0
        for t in towels:
            if not design.endswith(t):
                continue
            trace(f"= trying {t}")
            count += count_solutions(design[:-len(t)], depth+1)
        trace(f"< {design = } -> {count}")
        return count

    print(towels)
    possible = 0
    total_ways = 0

    for i, d in enumerate(designs):
        ways = count_solutions(d)
        # print(f"{d}: {ways}")
        total_ways += ways
        if ways > 0:
            possible += 1

    print(f"{total_ways = }")
    print(f"{possible = }")

def part2(filename):
    ...

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