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

class Pair(NamedTuple):
    x: int
    y: int
@dataclass(frozen=True)
class Machine:
    asteps: Pair
    bsteps: Pair
    prize: Pair

def read_input(
        filename: str
) -> list[Machine]:
    with open(filename) as f:
        s = iter(f)
        s = map(str.rstrip, s)
        s = split(lambda line: line == "", s)
        blocks: list[list[str]] = list(s)

    machines = []
    for blk in blocks:
        assert len(blk) == 3, blk
        a, b, p = blk
        ax, ay = map(lambda t: int(t[1]), map(partial(str.split, sep="+"), a.split(": ")[1].split(", ")))
        bx, by = map(lambda t: int(t[1]), map(partial(str.split, sep="+"), b.split(": ")[1].split(", ")))
        px, py = map(lambda t: int(t[1]), map(partial(str.split, sep="="), p.split(": ")[1].split(", ")))
        machines.append(Machine(
            Pair(ax, ay),
            Pair(bx, by),
            Pair(px, py)
        ))
    return machines

def extended_gcd(a, b) -> tuple[int, int, int]:
    """Compute gcd(a, b), x, and y such that

        gcd(a, b) = x * a + y * b
    """
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y

def extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y

def solve_diophantine(a, b, c):
    gcd, x, y = extended_gcd(a, b)
    if c % gcd != 0:
        raise ValueError("No solution exists")
    x *= c // gcd
    y *= c // gcd
    return gcd, x, y

def part1(filename, part2=False):
    machines = read_input(filename)

    def solveable(m: Machine):
        ax, ay = m.asteps
        bx, by = m.bsteps
        px, py = m.prize

        return (
            px % gcd(ax, bx) == 0 and
            py % gcd(ay, by) == 0
        )

    def solution_cost(amoves, bmoves):
        return 3 * amoves + 1 * bmoves

    total_tokens = 0
    eps = 1e-6

    for m in machines:
        if not solveable(m):
            continue

        M = np.array([
            [m.asteps.x, m.bsteps.x],
            [m.asteps.y, m.bsteps.y]
        ])
        y = np.array([m.prize.x, m.prize.y])
        # print("inputs: ", M, y)
        na, nb = v = np.linalg.solve(M,y)

        if abs(na - round(na)) > eps or abs(nb - round(nb)) > eps:
            print("ignoring: ", v, na - round(na), nb - round(nb))
            continue
        na = round(na)
        nb = round(nb)

        # print(v, solution_cost(*v))
        total_tokens += solution_cost(*v)

    print(total_tokens)

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