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
) -> str:
    with open(filename) as f:
        s = f.readlines()
        return s[0].rstrip()
        
def to_image(lengths: list[int]) -> list[int]:
    s = zip(
        lengths,
        interleave(count(), repeat(-1))
    )
    # s = observe(partial(print, "#1:"), s)
    s = map(star(lambda n, value: [value] * n), s)
    # s = observe(partial(print, "#2:"), s)
    s = chain.from_iterable(s)
    return list(s)

def adjacent_gaps(
        gaps: list[range], 
        frange: range
) -> tuple[ int | None, int | None ]:
    key = lambda r: r.stop
    before = bisect_right(gaps, frange.start, key=key) - 1
    after = bisect_left(gaps, frange.stop, key=key)

    if before == -1 or gaps[before].stop < frange.start:
        before = None
    if after == len(gaps) or gaps[after].start > frange.stop:
        after = None

    return before, after

@pytest.mark.parametrize(
    "frange, expected, gaps",
    [
        [range(10, 20), (0   ,    1), [range(0, 10), range(20, 30)]],
        [range(10, 20), (None,    1), [range(0, 9),  range(20, 30)]],
        [range(10, 20), (   0, None), [range(0, 10), range(21, 30)]],
        [range(10, 20), (None, None), [range(0, 9),  range(21, 30)]],
    ]
)
def test_free_around(frange, expected, gaps):
    assert expected == adjacent_gaps(gaps, frange)

def find_first_fit(
        frange: range,
        gaps: list[range]
) -> int | None:
    for gno, grange in enumerate(gaps):
        if len(grange) >= len(frange):
            return gno
    return None

def coalesce_around(
        r: range,
        gaps: list[range]
):
    left, right = adjacent_gaps(gaps, r)
    if   left is None     and right is None:
        pass
    elif left is None     and right is not None:
        gaps[right] = range(r.start, gaps[right].stop)
    elif left is not None and right is None:
        gaps[left] = range(gaps[left].start, r.stop)
    elif left is not None and right is not None:
        gaps[left: right+1] = [
            range(gaps[left].start, gaps[right].stop)
        ]

def copy_blocks(
        image: list[int],
        fill: int,
        fold: range,
        fnew: range
):
    image[fnew.start: fnew.stop] = [fill] * len(fnew)        
    image[fold.start: fold.stop] = [-1]   * len(fold)

def trim_left(gap, removed):
    return range(gap.start + removed, gap.stop)

def defrag_firstfit(
        image: list[int], 
        short: bool,
        files: list[range], 
        gaps: list[range]
):
    def image_str():
        return "".join(
            "." if val == -1 else str(val)
            for val in image
        )
    
    if short: print(image_str())
    for fno in reversed(range(len(files))):
        fold = files[fno]
        if short: print(f"... considering file {fno}: {fold = } {len(gaps) = }")

        gno = find_first_fit(fold, gaps)
        if gno is None:
            continue 

        gap = gaps[gno] 
        if short: print(f"... found gap {gno}: {gap}")
        if gap.start >= fold.stop:
            if short: print(f"... gap to the right of file: skipping")
            continue
        
        fnew = range(gap.start, gap.start + len(fold))
        copy_blocks(image, fno, fold, fnew)
        
        files[fno] = fnew
        gaps[gno] = trim_left(gap, len(fnew))
        if len(gaps[gno]) == 0:
            del gaps[gno]
            
        coalesce_around(fold, gaps)
        if short: print(image_str())

def defrag_dense(image, short, files, gaps):
    moveto = chain.from_iterable(gaps)
    movefrom = chain.from_iterable(map(reversed, reversed(files)))

    if short: print(image)
    for f, t in zip(movefrom, moveto):
        if f < t:
            break
        if short: print(f"    {f, t = }")
        if short: print(f"    {image[f], image[t] = }")
        assert image[t] == -1
        image[t] = image[f]
        image[f] = -1
        if short: print(image)

def part1(filename, defrag):
    encoded = list(map(int, read_input(filename)))

    if len(encoded) < 50: print(f"{encoded =}")
    print(f"{len(encoded) = }")
    image = to_image(encoded)
    short = len(image) < 50
    if short: 
        print(f"{image = }")
    else:
        print(f"{len(image) = }")

    files: list[range] = []
    gaps: list[range] = []
    o = 0
    for n, fill in zip(
        encoded,
        interleave(count(), repeat(-1))
    ):
        if n == 0:
            continue
        if fill != -1:
            files.append(range(o, o+n))
        else:
            gaps.append(range(o, o+n))
        o += n
    if short: print(f"{files = }")
    if short: print(f"{gaps  = }")

    print(f"{len(files) = } zeros = {sum(len(f) == 0 for f in files)}")
    print(f"{len(gaps) = } zeros = {sum(len(g) == 0 for g in gaps)}")

    with timer() as dtime:
        defrag(image, short, files, gaps)
    with timer() as cktime:
        checksum = sum(
            i * fileno
            for i, fileno in enumerate(image)
            if fileno != -1
        )
    print(f"{(dtime.elapsed(), cktime.elapsed()) = }")
    print(checksum)

def part2(filename):
    pass

def usage(message):
    print(f'usage: {sys.argv[0]} [-1|-2] [--] input_file...')
    print(f'    {message}')
    sys.exit(1)

parts = {
    1: partial(part1, defrag=defrag_dense), 
    2: partial(part1, defrag=defrag_firstfit),
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
        print(f"\n***** START PART {part} ({infile})")
        with timer() as t:
            parts[part](infile)
        print(f"***** FINISHED PART {part} ({infile}) {t.elapsed()}s")

if __name__ == '__main__':
    main(sys.argv[1:])