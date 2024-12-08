import collections
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator
)
from itertools import (
    count,
    cycle,
    islice,
    tee
)

__all__ = [ "star", "nth" ]

def star(f):
    return lambda t: f(*t)

def nth(n, t):
    return t[n]

def mapnth(f, n):
    def update(t: tuple) -> tuple:
        return t[:n] + (f(t[n]),) + t[n+1:]
    return update

def observe[T](
        f: Callable[[T], Any], 
        items: Iterable[T]
) -> Iterator[T]:
    for item in items:
        f(item)
        yield item

def consume(iterator, n=None):
    # Use functions that consume iterators at C speed.
    if n is None:
        collections.deque(iterator, maxlen=0)
    else:
        next(islice(iterator, n, n), None)

def window[T](
        items: Iterable[T], 
        n: int
) -> Iterator[tuple[T, ...]]:
    iters = tee(items, n)
    for it, skip in zip(iters, count()):
        consume(it, skip)
    return zip(*iters)

def interleave(*iterables: Iterable) -> Iterator:
    iterators = cycle(iter(it) for it in iterables)
    sentinel = object()

    remaining = len(iterables)
    while remaining > 0:
        while True:
            it = next(iterators)
            value = next(it, sentinel) 
            if value == sentinel:
                break
            yield value
        remaining -= 1
        nexts = cycle(islice(iterators, remaining))

def first[T](
        items: Iterable[T], 
        strict: bool = False
) -> T | None:
    items = iter(items)
    if not strict:
        return next(items, None)
    if (item := next(items, None)) is None:
        raise ValueError("no first item")
    return item

def only[T](
        items: Iterable[T]
) -> T:
    items = iter(items)
    item = next(items, None)
    print(f"{item = }")
    if item is None:
        raise ValueError("no first item")
    if next(items, None) is not None:
        raise ValueError("expecting exactly one item")
    return item

def split[T](
        separator: Callable[[T], bool],
        items: Iterable[T]
) -> Iterator[list[T]]:
    batch = []
    for item in items:
        if separator(item):
            yield batch
            batch = []
            continue
        batch.append(item)
    if len(batch) > 0:
        yield batch