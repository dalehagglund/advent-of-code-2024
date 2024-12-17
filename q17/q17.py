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
) -> tuple[list[int], dict[str, int]]:
    with open(filename) as f:
        s = iter(f)
        s = map(str.rstrip, s)
        s = split(lambda line: line == "", s)
        data, prog = s

        regs = {}
        for reg in data:
            ns, vs = re.findall(r"([ABC]|\d+)", reg)
            regs[ns] = int(vs)
        prog = list(map(int, only(prog).split(" ")[1].split(",")))
        return prog, regs

names = "adv,bxl,bst,jnz,bxc,out,bdv,cdv".split(",")

def dis(prog: list[int]):
    def combo(op):
        if 0 <= op <= 3: return f"#{op}"
        if op == 4: return "A"
        if op == 5: return "B"
        if op == 6: return "C"
        assert False, f"invalid combo op: {op}"
    def lit(op): return f"{op}"
    def ignore(op): return f""
    fmtop = {
        "adv": combo,
        "bxl": lit,
        "bst": combo,
        "jnz": lit,
        "bxc": ignore,
        "out": combo,
        "bdv": combo,
        "cdv": combo,
    }

    for i, (instr, op) in enumerate(batched(prog, 2)):
        print(
            f"{i*2:<5}"
            f"{names[instr]:<8}"
            f"{fmtop[names[instr]](op)}"
        )

def combo(op, regs):
    if 0 <= op <= 3:
        return op
    if op == 4: return regs["A"]
    if op == 5: return regs["B"]
    if op == 6: return regs["C"]
    assert False, f"invalid combo op: {op}"

def simstep(step, ip, instr, op, regs, trace=False) -> tuple[int, int | None]:
    if trace: print(
        f"\nstep {step}: "
        f"{ip = } "
        f"{names[instr]} "
        f"lit {op} "
        f"combo {str(combo(op, regs)) if op <= 6 else "?"}"
    )
    output: int | None = None
    if trace: print(f"... regs before ", *regs.items())
    match instr:
        case 0: # adv
            num = regs["A"]
            den = 2 ** combo(op, regs)
            regs["A"] = num // den
            ip += 2
        case 1: # bxl
            regs["B"] = regs["B"] ^ op
            ip += 2
        case 2: #  bst
            regs["B"] = combo(op, regs) % 8
            ip += 2
        case 3: # jnz
            ip = op if regs["A"] != 0 else ip + 2
        case 4: # bxc
            regs["B"] = regs["B"] ^ regs["C"]
            ip += 2
        case 5: # out
            if trace: print(f"... out {combo(op, regs) % 8}")
            output = combo(op, regs) % 8
            ip += 2
        case 6: # bdv
            num = regs["A"]
            den = 2 ** combo(op, regs)
            regs["B"] = num // den
            ip += 2
        case 7: # cdv
            num = regs["A"]
            den = 2 ** combo(op, regs)
            regs["C"] = num // den
            ip += 2
    if trace: print(f"... regs after", *regs.items())

    return ip, output

def runprog(
        prog: list[int],
        regs: dict[str, int],
        trace: bool = True,
) -> Iterator[int]:
    output = []
    stopped = False
    ip = 0

    visited = set()

    for step in count(0):
        if ip >= len(prog):
            break
        state = (ip, *regs.values())
        if state in visited:
            raise ValueError(f"cycle detected: {step = } {state = }")
        visited.add(state)

        instr, op = prog[ip], prog[ip+1]
        ip, output = simstep(step, ip, instr, op, regs, trace)
        if output is not None: yield output

    yield -1

@pytest.mark.parametrize(
    "init, regsout, out, prog",
    [
        [ (0, 0, 9),  (-1, 1, -1), None, [2, 6] ],
        [ (10, 0, 0), None, [0, 1, 2, -1], [5,0,5,1,5,4] ],
        [ (2024, 0, 0), (0, -1, -1), [4,2,5,6,7,7,7,7,3,1,0,-1], [0,1,5,4,3,0] ],
        [ (0, 29, 0), (-1, 26, -1), None, [1,7] ],
        [ (0, 2024, 43690), (-1, 44354, -1), None, [4, 0] ],
    ]
)
def test_runprog(init, regsout, out, prog):
    regs = dict(zip("ABC", init))
    output = list(runprog(prog, regs, trace=False))
    if out is not None:
        assert out == output
    if regsout is not None:
        for r, expected in zip("ABC", regsout):
            if expected == -1:
                continue
            assert regs[r] == expected

def part1(filename, part1=True):
    prog, regs = read_input(filename)
    print(prog, regs)

    dis(prog)

    if part1:
        output = list(runprog(prog, regs.copy(), trace=False))[:-1]
        print(",".join(map(str, output)))
        return

    for n in count(2 ** (3*len(prog))):
        if n % 250000 == 0: print(f"{n = }")
        input = regs.copy()
        input["A"] = n
        output = runprog(prog, input, trace=False)
        for i, (expected, out) in enumerate(zip(prog + [-1], output)):
            # print(f"*** step {i}: {expected, out = }")
            if expected != out:
                # print(f"*** mismatch at step {i}:", expected, out)
                break
        else:
            print("... matched!")
            break

    print(n)


def part2(filename):
    prog, origregs = read_input(filename)

    assert prog[-2:] == [3, 0]      # jnz 0
    assert 1 == sum(
        (ip, op) == (0, 3)          # adv 3
        for ip, op in batched(prog, 2)
    )

    dis(prog)
    print()

    # states are tuples with
    #
    # - accumulated "a" value,
    # - remaining outputs to produce

    first_found = None
    start = (0, tuple(prog))
    def end(state) -> bool:
        nonlocal first_found
        a, rem = state
        if len(rem) == 0:
            first_found = a
            return True
        return False

    def moves(state) -> Iterator[tuple[int, tuple[int, ...]]]:
        aprev, remaining = state
        if len(remaining) == 0:
            return
        target = remaining[-1]
        # the order in which we look for potential solutions has an
        # impact on the order in which overall search finds the
        # terminal solutions (one where we get through all of the
        # output).
        #
        # in the standard order, use below, even bfs will find the
        # smallest solution first. but if the order is shuffled, then
        # it's necessary to use a* with edge-weights to find the best
        # solution first.
        for lowbits in range(8):
            regs = origregs.copy()
            anew = aprev * 8 + lowbits
            regs["A"] = anew
            output, _ = runprog(prog[:-2], regs, False)
            if output == target:
                yield anew, remaining[:-1]

    def movecost(s1, s2):
        # see the comment in `moves` for more details on why this
        # function is necessary (sometimes). but as noted there, we
        # can throw this away and go back to bfs and for this puzzle
        # it really doesn't matter much.
        (a1, _), (a2, _) = s1, s2
        assert a2 - a1 >= 0, (s1, s2)
        return 1 + a2 - a1

    dist, _ = astar(start, end, moves, movecost)

    print(f"{first_found = }")
    solutions = { a for (a, rem), _ in dist.items() if len(rem) == 0 }
    distances = { d for (a, rem), d in dist.items() if len(rem) == 0 }
    print(f"{len(solutions), len(distances) = }")
    amin = min(solutions)
    print(amin)

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