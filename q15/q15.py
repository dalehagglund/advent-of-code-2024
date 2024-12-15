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
    last,
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

# should be generalized slightly...
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

def read_input(
        filename: str
) -> tuple[np.ndarray, str]:
    with open(filename) as f:
        s = iter(f)
        s = map(str.rstrip ,s)
        rows, moves = split(lambda line: line == "", s)

        return (
            np.array(list(map(list, rows)), dtype=np.dtypes.StringDType),
            "".join(moves)
        )

def part1(filename):
    grid, moves = read_input(filename)


    movedir = {
        "^": (-1,  0),
        ">": ( 0, +1),
        "v": (+1,  0),
        "<": ( 0, -1),
    }

    r, c = only(locate(grid == "@"))
    print(f">>> initial grid, robot {r, c}")
    display(grid)

    for i, move in enumerate(moves):
        dr, dc = movedir[move]
        assert (dr == 0) != (dc == 0), (dr, dc)
        match grid[r + dr, c + dc]:
            case "#":
                pass
            case ".":
                grid[r, c] = "."
                grid[r + dr, c + dc] = "@"
                r, c = r + dr, c + dc
            case "O":
                if dr == 0 and dc > 0:
                    cols = slice(c, None)
                    dot = first(locate(grid[r, cols] == "."))
                    wall = first(locate(grid[r, cols] == "#"), strict=True)
                    if dot is None or wall < dot: continue
                    dotpos = (r, cols.start + dot[0])
                elif dr == 0 and dc < 0:
                    cols = slice(0, c)
                    dot = last(locate(grid[r, cols] == "."))
                    wall = last(locate(grid[r, cols] == "#"), strict=True)
                    if dot is None or wall > dot: continue
                    dotpos = (r, cols.start + dot[0])
                elif dc == 0 and dr > 0:
                    rows = slice(r, None)
                    dot = first(locate(grid[rows, c] == "."))
                    wall = first(locate(grid[rows, c] == "#"), strict=True)
                    if dot is None or wall < dot: continue
                    dotpos = (rows.start + dot[0], c)
                elif dc == 0 and dr < 0:
                    rows = slice(0, r)
                    dot = last(locate(grid[rows, c] == "."))
                    wall = last(locate(grid[rows, c] == "#"), strict=True)
                    if dot is None or wall > dot: continue
                    dotpos = (rows.start + dot[0], c)
                else:
                    assert False, "either dr or dc should be zero!"
                # print("curr, dotpos, new",  [ (r, c), dotpos, (r + dr, c + dc) ])
                # print("transpose: ", np.transpose([ (r, c), dotpos, (r + dr, c + dc) ]))
                grid[*np.transpose(
                    [ (r, c), dotpos, (r + dr, c + dc) ]
                )] = [ ".", "O", "@" ]
                r, c = r + dr, c + dc
        # print(f">>> {i}: move {move} robot {r, c}")
        # display(grid)

    print(f">>> final grid:")
    display(grid)

    gps_total= sum(
        r * 100 + c
        for r, c
        in locate(grid == "O")
    )

    print(gps_total)

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