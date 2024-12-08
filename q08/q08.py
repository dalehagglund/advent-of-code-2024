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
):
    with open(filename) as f:
        s = f.readlines()
        s = map(str.rstrip, s)
        s = map(list, s)
        return np.array(
            list(s),
            dtype=np.dtypes.StringDType
        )
        
def part1(filename, resonance=False):
    grid = read_input(filename)
    rows, cols = map(range, grid.shape)

    # display(grid)
    # print(rows, cols)

    all_locations: set[tuple[int, int]] = set()
    antenna_types: defaultdict[str, set[tuple[int, int]]] = defaultdict(set)

    for r, c in product(rows, cols):
        all_locations.add((r, c))
        ch = grid[r, c]
        if ch != ".":
            antenna_types[ch].add((r, c))

    # print(f"{antenna_types = }")

    nodes: set[tuple[int, int]] = set()
    for ch, locs in antenna_types.items():
        # print(f"... {ch}: {locs}")
        for p1, p2 in combinations(locs, r=2):
            p1r, p1c = p1
            p2r, p2c = p2
            dr, dc = p2r - p1r, p2c - p1c

            def inbounds(r, c): return (r, c) in all_locations
            def move(p, dir, i):
                r, c = p
                dr, dc = dir
                return (r + i * dr, c + i * dc)
            
            for start, step in [(p2, +1), (p1, -1)]:
                s = [step] if not resonance else count(0, step)
                s = map(partial(move, start, (dr,dc)), s)
                s = takewhile(star(inbounds), s)
                nodes.update(s)

    print(len(nodes))

def part2(filename):
    pass

def usage(message):
    print(f'usage: {sys.argv[0]} [-1|-2] [--] input_file...')
    print(f'    {message}')
    sys.exit(1)

parts = {
    1: part1, 
    2: partial(part1, resonance=True),
}

options = {
}

def main(args):
    infile = None
    run1 = run2 = False
    
    while args and args[0].startswith('-'):
        arg = args.pop(0)
        if arg in ('--'): break
        elif re.match(r'^-[A-Za-z0-9]{2,}$', arg):
            args[:0] = list(map(partial(operator.add, '-'), arg[1:]))
        elif arg in ('-1'): run1 = True
        elif arg in ('-2'): run2 = True
        else:
            usage(f'{arg}: unexpected option')

    if not (run1 or run2): run1 = run2 = True

    if len(args) == 0:
        usage("missing input file")
        
    if run1:
        for infile in args:
            print(f"\n***** START PART 1 ({infile})")
            with timer() as t:
                parts[1](infile)
            print(f"***** FINISHED PART 1 ({infile}) {t.elapsed()}s")

    if run2:
        for infile in args:
            print(f"\n*** PART 2 ({infile})")
            with timer() as t:
                parts[2](infile)
            print(f"*** FINISHED PART 2 ({infile}) {t.elapsed()}s")

if __name__ == '__main__':
    main(sys.argv[1:])