import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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

import numpy as np
import numpy.typing as npt
from aoc.np import (
    locate,
    display,
    follow_ray
)

def read_input(
        filename: str
) -> np.ndarray:
    with open(filename) as f:
        s = f.readlines()
        s = map(str.rstrip, s)
        s = map(list, s)
        return np.array(list(s), dtype=np.dtypes.StringDType)

def part1(filename):
    grid = read_input(filename)
    rows, cols = map(range, grid.shape)

    display(grid)

    nodes = set(product(rows, cols))
    def neighbours(pos):
        r, c = pos
        plant = grid[r, c]
        for dr, dc in [
            (-1, 0),
            (+1, 0),
            (0, -1),
            (0, +1)
        ]:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in nodes:
                continue
            if grid[nr, nc] != plant:
                continue
            yield (nr, nc)

    all_plants = set(
        grid[r, c] for r, c in product(rows, cols)
    )
    print(all_plants)

    components = []
    for plant in all_plants:
        locs = set(locate(grid == plant))
        while locs:
            start = locs.pop()
            distvec, _ = bfs(start, None, neighbours)
            reached = set(
                pos
                for pos, dist
                in distvec.items()
                if dist != float('inf')
            )

            components.append((plant, reached))
            locs -= reached

    def perimeter(places):
        perim = 0
        for p in places:
            perim += 4 - sum(1 for _ in neighbours(p))
        return perim

    fencing_cost = 0
    for plant, places in components:
        print(f"{plant} {perimeter(places) = }: {places}")
        fencing_cost += len(places) * perimeter(places)
    print(fencing_cost)


def part2(filename):
    pass

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