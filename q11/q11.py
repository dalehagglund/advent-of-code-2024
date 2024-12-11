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

from aoc.search import astar, allpaths
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
) -> list[int]:
    with open(filename) as f:
        s = f.readlines()[0].rstrip().split()
        return list(map(int, s))

def update_stone(n: int) -> list[int]:
    assert isinstance(n, int), n
    nstr = str(n)
    if n == 0:
        return [ 1 ]
    elif len(nstr) % 2 == 0:
        mid = len(nstr) // 2
        return [ int(nstr[:mid]), int(nstr[mid:]) ]
    else:
        return [ n * 2024 ]

def expand_fast(counts: Counter[int]) -> Counter[int]:
    new_counts = Counter()
    for s, scount in counts.items():
        for new in update_stone(s):
            new_counts[new] += scount
    return new_counts

def part1(filename, count=25):
    stones = read_input(filename)
    print(stones)

    stone_counts = Counter(stones)
    for i in range(count):
        stone_counts = expand_fast(stone_counts)
        # print(f"{i}: {stone_counts.total() = }")
    print(stone_counts.total())

def part2(filename):
    pass

def usage(message):
    print(f'usage: {sys.argv[0]} [-1|-2] [--] input_file...')
    print(f'    {message}')
    sys.exit(1)

parts = {
    1: part1,
    2: partial(part1, count=75),
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