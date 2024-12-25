import dataclasses
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

from dataclasses import dataclass, field, replace
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

from aoc.search import astar, allpaths, bfs #, path_to
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

def iterate(f, n) -> Iterator[int]:
    while True:
        yield n
        n = f(n)

def sliding_window[T](
        items: Iterable[T],
        n: int
) -> Iterator[tuple[T, ...]]:
    items = iter(items)
    window = collections.deque(islice(items, n-1), maxlen=n)
    for x in items:
        window.append(x)
        yield tuple(window)

def innermap[T, U](
        f: Callable[[T], U],
        s: Iterable[Iterable[T]]
) -> Iterable[Iterable[U]]:
    return map(partial(map, f), s)

@dataclass(order=True, frozen=True)
class Gate:
    in1: str
    op: str
    in2: str
    out: str

    def eval(self, known: dict[str, int]):
        ops = {
            "AND": operator.__and__,
            "OR": operator.__or__,
            "XOR": operator.__xor__
        }
        return ops[self.op](known[self.in1], known[self.in2])

    def __repr__(self):
        return f"{self.out} <- {self.in1} {self.op} {self.in2}"

def read_input(
        filename: str
) -> tuple[
    dict[str, int],
    set[str],
    set[Gate]
]:
    with open(filename) as f:
        s = iter(f)
        s = map(str.rstrip, s)
        decls, second = split(lambda line: line == "", s)

    s = decls
    s = map(partial(str.split, sep=": "), s)
    s = map(star(lambda w, v: (w, int(v))), s)
    givens = dict(s)
    wires = set(givens.keys())

    gates: set[Gate] = set()
    s = second
    s = map(partial(re.split, r" -> | "), s)
    for (w1, op, w2, out) in s:
        wires |= {w1, w2, out}
        gates.add( Gate(min(w1, w2), op, max(w1, w2), out) )

    return givens, wires, gates

def eval_circuit(known, wires, gates) -> list[int]:
    outputs = set(filter(lambda w: w.startswith("z"), wires))

    # print(sorted(outputs, reverse=True))
    def inputs_ready(g: Gate):
        return g.in1 in known and g.in2 in known

    ready = set(filter(inputs_ready, gates))
    waiting = defaultdict(set)
    for g in filterfalse(inputs_ready, gates):
        if g.in1 not in known: waiting[g.in1].add(g)
        if g.in2 not in known: waiting[g.in2].add(g)

    assert ready | set(chain.from_iterable(waiting.values())) == gates
    assert ready & set(chain.from_iterable(waiting.values())) == set()

    while ready:
        g = ready.pop()
        known[g.out] = g.eval(known)

        for g2 in waiting[g.out]:
            if not inputs_ready(g2):
                continue
            ready.add(g2)
        if g.out in waiting: del waiting[g.out]

    assert len(waiting) == 0
    assert outputs <= known.keys()
    assert known.keys() == wires

    return [known[o] for o in sorted(outputs, reverse=True)]

def part1(filename):
    known, wires, gates = read_input(filename)
    zreg = eval_circuit(known, wires, gates)
    print(int("".join(map(str, zreg)), base=2))

def topsort(gates: set[Gate]) -> Iterator[Gate]:
    wires = set(chain.from_iterable([g.in1, g.in2, g.out] for g in gates))
    outs = set(g.out for g in gates)

    waiting = defaultdict(set)
    for g in gates:
        waiting[g.in1].add(g)
        waiting[g.in2].add(g)
    known = wires - outs
    ready = { g for g in gates if g.in1 in known and g.in2 in known }

    while ready:
        g = sorted(
            list(ready), key=lambda g: (g.op, g.in1[::-1], g.in2[::-1])
        )[0]
        ready.remove(g)
        yield g
        known.add(g.out)

        for g2 in waiting[g.out]:
            if g2.in1 in known and g2.in2 in known:
                ready.add(g2)
        if g.out in waiting: del waiting[g.out]

def part2(filename):
    known, wires, gates = read_input(filename)

    zwires = set(filter(partial(re.match, r"^z"), wires))
    xwires = set(filter(partial(re.match, r"^x"), wires))
    ywires = set(filter(partial(re.match, r"^y"), wires))
    xywires = xwires | ywires

    # every gate has a unique output
    assert len(gates) == len(set(g.out for g in gates))
    # every zwire comes from exactly one gate (implied by the above)
    assert len(set(filter(lambda g: g.out in zwires, gates))) == len(zwires)

    swaps = [
        ("z39", "ckb"),
        ("z20", "tqq"),
        ("nbd", "kbs"),
        ("z06", "ksv"),
    ]

    src: dict[str, Gate] = { g.out: g for g in gates }

    if do_swaps:
        print("swaps:")
        for w1, w2 in swaps:
            print(f"    swapping {w1}, {w2}")
            g1, g2 = src[w1], src[w2]
            g1clone, g2clone = replace(g1, out=w2), replace(g2, out=w1)
            print(f"    previous: {g1}, {g2}")
            print(f"    fixed:    {g1clone}, {g2clone}")
            gates -= {g1, g2}
            gates |= {g1clone, g2clone}
            src[w1], src[w2] = g2clone, g1clone

    preds: dict[Gate, set[Gate]] = {
        g: reachable(
            g,
            lambda g: set(
                src[wire]
                for wire in [g.in1, g.in2]
                if wire not in xywires
            )
        )
        for g in gates
    }

    # print("predecessors...")
    # for g in sorted(predecessors.keys()):
    #     print("   ", g)
    #     for pred in predecessors[g]:
    #         print("      ", pred)

    conf, nonconf = 0, 0
    for w1, w2 in combinations(wires - xywires, r=2):
        g1, g2 = src[w1], src[w2]
        g1preds, g2preds = preds[g1], preds[g2]
        if g1 in g2preds or g2 in g1preds:
            # print(f"... {w1, w2}: conflicting")
            # print(f"... ...", g1)
            # for g in g1preds: print("... ... ... ", g)
            # print(f"... ...", g2)
            # for g in g2preds: print("... ... ... ", g)
            conf += 1
        else:
            # print(f"... {w1, w2}: not conflicting")
            nonconf += 1
    print(f"{conf, nonconf = }")

    interesting: set[Gate] = set()
    for w1, w2 in pairwise(sorted(zwires, reverse=True)):
        delta_gates = preds[src[w1]] - preds[src[w2]]
        print(f"{w1} - {w2}: {len(delta_gates)} gates")
        if len(delta_gates) != 5:
            interesting |= delta_gates
        for g in topsort(delta_gates):
            print("   ", g)
    print("z00:")
    for g in preds[src["z00"]]:
        print("   ", g)
    print(f"{len(interesting) = }")

    def check_zi_block(i: int):
        zi = f"z{i:02}"
        print(f"*** checking {zi}")
        xi, yi = zi.replace("z", "x"), zi.replace("z", "y")

        if zi in ("z00", "z01", "z45"):
            print(f"    {zi} is special: skipping...")
            return
        A = src[zi]

        if A.op != "XOR":
            print(f"    A: expected XOR, not {A = }")
            return
        B, C = src[A.in1], src[A.in2]

        if {B.op, C.op} != {"XOR", "OR"}:
            print(f"    A: expecting XOR and OR, not {B = }, {C = }")
            return

        X, O = (B, C) if B.op == "XOR" else (C, B)
        assert X.op == "XOR", X
        assert O.op == "OR", O

        if {X.in1, X.in2} != { xi, yi }:
            print(f"    X: expecting {xi}, {yi} inputs, not {X.in1, X.in2}")

        D, E = src[O.in1], src[O.in2]
        if {D.op, E.op} != { "AND" }:
            print(f"    O: expecting AND inputs, not {D = }, {E = }")
            return

        def prev_xy_inputs(g: Gate) -> bool:
            assert g.op == "AND"
            xprev, yprev = f"x{i-1:02}", f"y{i-1:02}"
            return {g.in1, g.in2} == {xprev, yprev}

        candidates = list(filter(prev_xy_inputs, [D, E]))
        if len(candidates) != 1:
            print(f"    D, E: expecting only one with prev inputs: {D, E = }")
            return


    for zi in range(len(zwires)):
        check_zi_block(zi)

    def to_int(reg: list[int]) -> int:
        return int("".join(map(str, reg)), base=2)

    xreg = [known[o] for o in sorted(xwires, reverse=True)]
    yreg = [known[o] for o in sorted(ywires, reverse=True)]
    zreg = eval_circuit(known, wires, gates)

    x, y, z = map(to_int, (xreg, yreg, zreg))
    if x + y == z:
        print(f"*** success: {swaps}")
        print(f"        code {','.join(sorted(chain(*swaps)))}")
    else:
        diffs = bin((x + y) ^ z)[::-1]
        print(
            f"!!! failure: "
            f" differences at {[i for i in range(len(diffs)) if diffs[i] == '1']}"
        )

def usage(message):
    print(f'usage: {sys.argv[0]} [-1|-2] [--] input_file...')
    print(f'    {message}')
    sys.exit(1)

parts = {
    1: partial(part1),
    2: partial(part2),
}

do_swaps = False

def main(args):
    from aoc.cmd import argscan
    from aoc.perf import timer
    torun = set()
    options = {}
    global do_swaps

    for flag in argscan(args):
        if flag in ('-1'): torun.add(1)
        elif flag in ('-2'): torun.add(2)
        elif flag == "--swaps": do_swaps = True
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