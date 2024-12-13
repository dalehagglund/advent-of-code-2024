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
        filename: str,
        part2: bool
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
        ax, ay = map(
            lambda t: int(t[1]),
            map(partial(str.split, sep="+"), a.split(": ")[1].split(", ")))
        bx, by = map(
            lambda t: int(t[1]),
            map(partial(str.split, sep="+"), b.split(": ")[1].split(", ")))
        px, py = map(
            lambda t: int(t[1]),
            map(partial(str.split, sep="="), p.split(": ")[1].split(", ")))
        if part2:
            px += 10000000000000
            py += 10000000000000
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

def solve_int_2x2(
        coeff: list[list[int]],
        goal: list[int]
) -> tuple[int, int] | None:
    # We could still  easily do this with np.linalg.solve, except
    # that you have to deal with floating point rounding error.
    #
    # But we know that our inputs are only integers, so we can take
    # advantage of that to get precise results with no roundoff error.
    # This equations in this code are those you get from using
    # high-school algebra to solve a system of two variables, with the
    # divisions deferred until the very end. Each division is checked
    # for a non-zero remainder, and if one is found, we return None to
    # indicate no integer result.
    #
    # other we return the precise integer quotients.

    (a, b), (d, e) = coeff
    c, f = goal

    det = a*e - b*d
    if det == 0:
        return None

    x0, r0 = divmod(c*e - b*f, det)
    x1, r1 = divmod(a*f - c*d, det)
    if r0 or r1:
        return None

    return x0, x1

def part1(filename, part2=False):
    machines = read_input(filename, part2)

    def solution_cost(amoves, bmoves):
        return 3 * amoves + 1 * bmoves

    total_tokens = 0
    ignored = 0
    eps = 1e-4

    for (i, m) in enumerate(machines):
        result = solve_int_2x2(
            [
                [m.asteps.x, m.bsteps.x],
                [m.asteps.y, m.bsteps.y],
            ],
            [m.prize.x, m.prize.y]
        )
        if result is None:
            # no integer result, or a singular matrix.
            continue
        na, nb = result

        # print(v, solution_cost(*v))
        total_tokens += solution_cost(na, nb)

    print(f"{ignored = }")
    print(total_tokens)

def part2(filename):
    pass

def usage(message):
    print(f'usage: {sys.argv[0]} [-1|-2] [--] input_file...')
    print(f'    {message}')
    sys.exit(1)

parts = {
    1: part1,
    2: partial(part1, part2=True),
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