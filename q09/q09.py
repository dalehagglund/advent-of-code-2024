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

def defrag(image, short, files, gaps):
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

def part1(filename):
    encoded = list(map(int, read_input(filename)))
    # print(encoded)

    if len(encoded) < 50: print(f"{encoded =}")
    print(f"{len(encoded) = }")
    image = to_image(encoded)
    short = len(image) < 30
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
        if fill != -1:
            files.append(range(o, o+n))
        else:
            gaps.append(range(o, o+n))
        o += n
    if short: print(f"{files = }")
    if short: print(f"{gaps  = }")

    print(f"{len(files) = }")
    print(f"{len(gaps) = }")

    defrag(image, short, files, gaps)

    checksum = sum(
        i * fileno
        for i, fileno in enumerate(image)
        if fileno != -1
    )
    print(checksum)

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