from typing import (
    Iterator,
    assert_never
)
import numpy as np
import pytest
import hypothesis
from hypothesis import (
    given,
    strategies as st
)

def naive_ray(g, start, dir) -> Iterator[int]:
    rows, cols = map(range, g.shape)
    r, c = start
    dr, dc = dir
    
    while r in rows and c in cols:
        yield g[r, c]
        r, c = r + dr, c + dc

#
# numpy_ray: find the ray extending from r,c in a given direction
#

def numpy_ray(g, start, dir) -> list[int]:
    dr, dc = dir
    nr, nc = g.shape
    r, c = start

    if (dr, dc) == (0, 0):
        raise ValueError("direction vector is (0, 0)")

    # normalize dr, dc to +1, 0, -1 
    dr = dr / abs(dr) if dr != 0 else 0
    dc = dc / abs(dc) if dc != 0 else 0

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
        return list(np.diag(g, k=c-r)[min(r,c):])
    elif dc == 0:
        return list(g[r:, c])
    elif dr == 0:
        return list(g[r, c:])
    else:
        assert_never(dir)

square = np.array(
    [[ 0,  1,  2,  3,  4],
     [ 5,  6,  7,  8,  9],
     [10, 11, 12, 13, 14],
     [15, 16, 17, 18, 19],
     [20, 21, 22, 23, 24]],
    dtype=np.int32
)

rect = np.array(
    [[ 0,  1,  2,  3,  4],
     [ 5,  6,  7,  8,  9],
     [10, 11, 12, 13, 14],
     [15, 16, 17, 18, 19],
     [20, 21, 22, 23, 24],
     [26, 27, 28, 29, 30],
     [31, 32, 33, 34, 35]],
    dtype=np.int32
)

directions = st.one_of(
    # diagonal
    st.just((-1, -1)),
    st.just((-1, +1)),
    st.just((+1, +1)),
    st.just((+1, -1)),
    # horizontal and vertical
    st.just((+1,  0)),
    st.just((-1,  0)),
    st.just(( 0, +1)),
    st.just(( 0, -1)),
)
    
@given(
    st.tuples(
        st.integers(min_value=0, max_value=4), 
        st.integers(min_value=0, max_value=4),
    ),
    directions
)
def test_numpy_ray_square(start, dir):
    expected = list(naive_ray(square, start, dir))
    assert expected == numpy_ray(square, start, dir)

@given(
    st.tuples(
        st.integers(min_value=0, max_value=6), 
        st.integers(min_value=0, max_value=4),
    ),
    directions
)
def test_numpy_ray_rect(start, dir):
    expected = list(naive_ray(rect, start, dir))
    assert expected == numpy_ray(rect, start, dir)

@pytest.mark.parametrize(
    "start, dir, expected", [
        ((0, 0), (+1, +1), [0, 6, 12, 18, 24]),
        ((0, 1), (+1, +1), [1, 7, 13, 19]),
        ((3, 1), (+1, +1), [16, 22]),
        ((3, 1), (-1, -1), [16, 10]),
        ((3, 1), (+1, -1), [16, 20]),
        ((3, 1), (-1, +1), [16, 12, 8, 4]), 
    ]
)
def test_naive_ray(start, dir, expected):
    assert expected == list(naive_ray(square, start, dir))

@given(st.lists(
    st.integers(min_value=-100, max_value=100),
    min_size=25, max_size=25
))
def test_flipud_flipud_order_doesnt_matter(values: list[int]):
    g = np.array(values, dtype=np.int32).reshape((5, 5))
    assert (np.fliplr(np.flipud(g)) == np.flipud(np.fliplr(g))).all()