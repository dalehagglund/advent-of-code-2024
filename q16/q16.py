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
):
    with open(filename) as f:
        s = iter(f)
        s = map(str.rstrip, s)
        s = map(list, s)
        return np.array(list(s), dtype=np.dtypes.StringDType)

dirs = {
    "^": (-1,  0),
    ">": ( 0, +1),
    "v": (+1,  0),
    "<": ( 0, -1),
}

dirsym = dict((v, k) for k, v in dirs.items())

class Pos(NamedTuple):
    r: int
    c: int

    @overload
    def __add__(self, other: tuple[int, int]) -> "Pos": ...
    @overload
    def __add__(self, other: "Pos") -> "Pos": ...

    def __add__(self, other: "Pos") -> "Pos":
        if isinstance(other, tuple):
            return Pos(self.r + other[0], self.c + other[1])
        return Pos(self.r + other.r, self.c + other.c)

    @overload
    def __sub__(self, other: tuple[int, int]) -> "Pos": ...
    @overload
    def __sub__(self, other: "Pos") -> "Pos": ...
    def __sub__(self, other: "Pos") -> "Pos":
        if isinstance(other, tuple):
            return Pos(self.r - other[0], self.c - other[1])
        return Pos(self.r - other.r, self.c - other.c)


@dataclass(frozen=True)
class State:
    pos: Pos
    dir: tuple[int, int]
    def __repr__(self):
        r, c = self.pos
        s = dirsym[self.dir]
        return f"({r}, {c}, {s})"




def part1(filename, part2=False):
    grid = read_input(filename)
    # display(grid)
    nrow, ncol = grid.shape

    board: defaultdict[Pos, set[tuple[int, int]]] = defaultdict(set)
    for pos in map(star(Pos), product(range(1, nrow-1), range(1, ncol - 1))):
        for _, dir in dirs.items():
            if grid[pos + dir] == "#":
                continue
            board[pos].add(dir)

    # print(sorted(board.items(), key=star(lambda k, v: k)))

    start = Pos(*only(locate(grid == "S")))
    end = Pos(*only(locate(grid == "E")))

    print(f"{start, end = }")

    def edge_cost(u: State, v: State) -> int:
        if u.pos != v.pos:
            assert u.dir == v.dir
            assert u.pos + u.dir == v.pos
            return 1
        dotprod = sum(map(operator.mul, u.dir, v.dir))
        match dotprod:
            # case  1: return 1
            case  0: return 1000
            case -1: return 2000
            case _:
                assert_never(dotprod)

    def moves(u: State) -> Iterator[State]:
        if u.dir in board[u.pos]:
            # move forward in the current direction
            yield State(u.pos + u.dir, u.dir)
        for dir in board[u.pos] - {u.dir}:
            # turn in place
            yield State(u.pos, dir)

    def extract_path(prev, s) -> Iterable[State]:
        path = [s]
        while prev[s] is not None:
            path.append(prev[s])
            s = prev[s]
        path.reverse()
        return path

    def distance(u: State, _) -> int:
        return abs(u.pos.r - end.r) + abs(u.pos.c - end.c)

    def is_goal(target: Pos, u: State) -> bool:
        return u.pos == target

    from_start, prev_from_end = astar(
        State(pos=start, dir=dirs["<"]),
        None,
        moves,
        edge_cost=edge_cost,
        # est_remaining=distance
    )

    min_state = min(
        filter(lambda s: s.pos == end, from_start.keys()),
        key=lambda s: from_start[s]
    )
    min_cost = from_start[min_state]
    print(f"{part2, min_state, from_start[min_state] = }")

    # pathgrid = grid.copy()
    # for s in best_seats:
    #     pathgrid[s.pos] = "@"
    # display(pathgrid)

    # print(len(best_seats))

    if not part2:
        return

    # def all_shortest_paths(
    #         path: list[State],
    #         visited: set[State],
    #         cost: int,
    #         depth: int = 0,
    #         trace: bool = False
    # ) -> Iterable[list[State]]:
    #     pref = " " * depth
    #     v = path[-1]
    #     vprev = prev_from_end[v]
    #     assert isinstance(vprev, State)
    #     if trace: print(
    #         f"{pref}> {cost = } {v = !s} {vprev = !s}"
    #     )
    #     if vprev is None:
    #         assert path[-1].pos == start
    #         assert path[0].pos == end
    #         for s, t in pairwise(reversed(path)):
    #             assert t in moves(v)
    #         if trace: print(
    #             f"{pref}> yield {path}"
    #         )
    #         yield path.copy()
    #         return

    #     assert v in set(moves(vprev))
    #     for u in backward_moves(v):
    #         if trace: print(
    #             f"{pref}>"
    #             f" backward move {u!s}"
    #         )
    #         if v not in set(moves(u)):
    #             continue
    #         uvcost = edge_cost(u, v)
    #         if from_start[u] + uvcost != cost:
    #             continue
    #         path.append(u)
    #         visited.add(u)
    #         yield from all_shortest_paths(
    #             path, visited, cost - uvcost, depth+1, trace=trace
    #         )
    #         visited.remove(u)
    #         path.pop()

    print("!!! HELLO!!!")

    def all_best_paths(
            path: list[State],
            trace: bool=False,
            depth: int=0,
            trlim: int=1000000000
    ) -> Iterator[list[State]]:
        def out(*args):
            if trace and depth <= trlim:
                print(f'{" "*depth}', *args)
        def tr(*args): out("...", *args)
        def enter(*args): out(">", *args)
        def exit(*args): out("<", *args)

        v = path[-1]
        vprev = prev_from_end[v]
        vcost = from_start[v]
        enter(f"{v = } {vcost = } {vprev = }")
        if vprev is None:
            tr(f"found {path = }")
            yield path[::-1]

        recur = partial(
            all_best_paths,
            trace=trace,
            depth=depth+1,
            trlim=trlim
        )

        # we can end up at v from a previous state u via one of
        # several possible turns in place, or a move a neighbouring
        # node *in the v.dir* direction. so, compute the candidate
        # predecessors. some of these might not be valid, but we'll
        # discard those later.

        ucand = set()
        # back up a potential forward move
        if v.pos - v.dir in board:
            ucand.add(State(v.pos - v.dir, v.dir))
        # for each *other* direction at this position, we could have
        # rotated to our current direction
        for dir in set(dirs.values()) - {v.dir}:
            ucand.add(State(v.pos, dir))

        tr(f"vprev {vprev} ucand {list(ucand)}")
        for u in ucand:
            uvcost = edge_cost(u, v)
            if from_start[u] + uvcost != vcost:
                tr(f"{u, v = }: costs don't work: {from_start[u], uvcost, vcost = }")
                # u can't be on a best path to v
                continue
            path.append(u)
            yield from recur(path)
            path.pop()

        exit()


    tiles = set()
    for path in all_best_paths([min_state]):
        assert path[0].pos == start
        assert path[-1].pos == end
        for u, v in pairwise(path):
            assert v in moves(u)
        assert min_cost == sum(
            edge_cost(u, v) for u, v in pairwise(path)
        )
        for u, v in pairwise(path):
            tiles.add(u.pos)
            tiles.add(v.pos)

    print(f"{tiles = }")
    pathgrid = grid.copy()
    if len(tiles) > 0: pathgrid[*np.transpose(list(tiles))] = "O"
    display(pathgrid)
    print(len(tiles))
    return

    from_end, _ = astar(
        { State(pos=end, dir=dir) for dir in dirs.values() },
        partial(is_goal, start),
        moves,
        edge_cost=edge_cost,
    )

    flip = dict(batched("<>><^vv^", n=2))
    tiles = set()
    for pos in map(star(Pos), product(range(1,nrow-1), range(1,ncol-1))):
        for dir in "^>v<":
            sdir = State(pos, dirs[dir])
            sflip = State(pos, dirs[flip[dir]])

            if sdir not in from_start or sflip not in from_end:
                if pos == (13, 13):
                    print(f"... {pos, dir, flip[dir] = }")
                    print("... ... from_start", list(filter(lambda s: s.pos == pos, from_start.keys())))
                    print("... ... from_end", list(filter(lambda s: s.pos == pos, from_end.keys())))
                continue
            if from_start[sdir] + from_end[sflip] == min_cost:
                print(f"... {pos = }")
                tiles.add(pos)
    print(f"{len(tiles) = }")

    print(nrow, ncol)
    pathgrid = grid.copy()
    pathgrid[*np.transpose(list(tiles))] = "O"
    display(pathgrid)

    tiles = set()
    # for i, path in enumerate(allpaths(
    #     [ State(pos=start, dir=directions["<"]) ],
    #     { State(pos=start, dir=directions["<"]) },
    #     moves,
    #     # keep = lambda p: p[-1].pos == end,
    # )):
    #     if i % 5000== 0: print(f"... {i}")
    #     if path[-1].pos == end:
    #         continue
    #     if dist[path[-1]] != best_cost:
    #         continue

    #     state = path[-1]
    #     tiles.add(state.pos)
    #     while prev[state] is not None:
    #         tiles.add(prev[state].pos)
    #         state = prev[state]

    print(len(tiles))

def part2(filename):
    ...

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