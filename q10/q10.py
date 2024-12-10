import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aoc.search import astar

import heapq
import math
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
from functools import partial

import itertools
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

import numpy as np
import numpy.typing as npt
from aoc.np import (
    locate,
    display,
    follow_ray
)

def read_input(
        filename: str
) -> Any:
    with open(filename) as f:
        s = f.readlines()
        s = map(str.rstrip, s)
        s = map(list, s)
        return np.array(list(s), dtype=np.dtypes.StringDType)


def part1(filename, all=False):
    grid = read_input(filename)
    display(grid)
    nrow, ncol = grid.shape

    trailheads = list(locate(grid == "0"))
    endpoints = list(locate(grid == "9"))

    nodes = {
        loc
        for loc in product(range(nrow), range(ncol))
        if grid[loc] != "."
    }

    def neighbours(pos):
        r, c = pos
        for dr, dc in [
            (+1,  0),
            (-1,  0),
            ( 0, +1),
            ( 0, -1),
        ]:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in nodes:
                continue
            if int(grid[nr, nc]) - int(grid[pos]) != 1:
                continue
            yield (nr, nc)

    def allpaths(path: list, nodes: set):
        if grid[path[-1]] == "9": yield path.copy()
        for n in neighbours(path[-1]):
            if n in nodes: continue
            path.append(n)
            nodes.add(n)
            yield from allpaths(path, nodes)
            nodes.remove(n)
            path.pop()

    if all:
        rating = 0
        for start in trailheads:
            paths = list(allpaths([start], {start}))
            print(f"... trail {start} rating {len(paths)}")
            rating += len(paths)
        print("rating", rating)
        return

    total_score = 0
    for start in trailheads:
        dist, _ = astar(start, None, neighbours)
        score = sum(
            (end in dist)
            for end in endpoints
        )
        print(f"... scores for trail {start}: {score}")
        total_score += score
    print("total score", total_score)


def part2(filename):
    pass

def usage(message):
    print(f'usage: {sys.argv[0]} [-1|-2] [--] input_file...')
    print(f'    {message}')
    sys.exit(1)

parts = {
    1: part1,
    2: partial(part1, all=True),
}

options = {
}

def main(args):
    from aoc.cmd import argscan
    from aoc.perf import timer
    infile = None
    run1 = run2 = False

    for flag in argscan(args):
        if flag in ('-1'): run1 = True
        elif flag in ('-2'): run2 = True
        else:
            usage(f"{flag}: unexpected option")
    if not (run1 or run2): run1 = run2 = True

    if len(args) == 0:
        usage("missing input file")

    to_run = [
        i
        for i, flag
        in enumerate([run1, run2], start=1)
        if flag
    ]
    for infile, part in product(args, to_run):
        print(f"\n***** START PART {part} -- {infile}")
        with timer() as t:
            parts[part](infile)
        print(f"***** FINISHED PART {part} -- {infile} elapsed {t.elapsed()}s")

if __name__ == '__main__':
    main(sys.argv[1:])