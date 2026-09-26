"""Async reductions and scans backed by the Mojo kernels.

The coroutine plumbing is deliberately borrowed from the real ``aioitertools``:
``aioitertools.iter`` and ``ait.iter`` do the mixed sync/async iteration, and
this module only replaces the arithmetic that happens once the values are in a
buffer.

There is a deliberate fast path. ``aioitertools`` accepts a plain list but still
runs it through an async generator, paying a coroutine round trip per item. A
synchronous iterable here is handed straight to NumPy's C-level conversion and
then to the kernel, so the per-item cost drops by more than an order of
magnitude. An async iterable is drained exactly as upstream drains it - one
``await`` per item - so for those the two are close.
"""

from __future__ import annotations

import builtins
from typing import Any, AsyncIterator, Iterable

import numpy as np

from aioitertools import itertools as ait

from . import _lib
from ._lib import ADD, ALL, ANY, CMAX, CMIN, MAX, MIN, MUL, SUM

__all__ = [
    "ADD", "ALL", "ANY", "CMAX", "CMIN", "MAX", "MIN", "MUL", "SUM",
    "accumulate", "all", "any", "chunk_offsets", "chunked_sizes", "compress",
    "max", "min", "reduce", "sum",
]


def _materialise(itr: Any) -> np.ndarray | None:
    """Return a contiguous float64 buffer for a synchronous iterable, else None.

    Anything NumPy can convert in C is converted in C; an async iterable, and
    anything that is not iterable at all, returns None so the caller takes the
    awaiting path.
    """
    if isinstance(itr, np.ndarray):
        return np.ascontiguousarray(itr, dtype=np.float64).reshape(-1)
    if hasattr(itr, "__aiter__"):
        return None
    if isinstance(itr, (list, tuple, range)):
        return np.asarray(itr, dtype=np.float64).reshape(-1)
    if hasattr(itr, "__iter__"):
        return np.fromiter(builtins.iter(itr), dtype=np.float64)
    return None


async def values(itr: Any) -> np.ndarray:
    """Drain any mixed iterable into a contiguous float64 buffer."""
    buf = _materialise(itr)
    if buf is not None:
        return buf
    items = [item async for item in ait.iter(itr)]
    return np.asarray(items, dtype=np.float64).reshape(-1)


async def reduce(itr: Any, op: int = SUM) -> float:
    """One reduction over a mixed iterable. ``op`` is one of the module codes."""
    return _lib.reduce(await values(itr), op)


async def sum(itr: Any, start: float | None = None) -> float:  # noqa: A001
    """Total of a mixed iterable of numbers, ``0`` by default.

    Same contract as ``aioitertools.builtins.sum``: ``start`` replaces the
    implicit zero initialiser.
    """
    total = 0.0 if start is None else float(start)
    return total + _lib.reduce(await values(itr), SUM)


async def min(itr: Any, default: Any = None) -> float:  # noqa: A001
    """Smallest item, or ``default`` when the iterable is empty."""
    buf = await values(itr)
    if buf.size == 0:
        if default is None:
            raise ValueError("min() of an empty iterable with no default")
        return default
    return _lib.reduce(buf, MIN)


async def max(itr: Any, default: Any = None) -> float:  # noqa: A001
    """Largest item, or ``default`` when the iterable is empty."""
    buf = await values(itr)
    if buf.size == 0:
        if default is None:
            raise ValueError("max() of an empty iterable with no default")
        return default
    return _lib.reduce(buf, MAX)


async def all(itr: Any) -> bool:  # noqa: A001
    """True when every item is truthy. Always drains the whole iterable."""
    return bool(_lib.reduce(await values(itr), ALL))


async def any(itr: Any) -> bool:  # noqa: A001
    """True when at least one item is truthy. Always drains the whole iterable."""
    return bool(_lib.reduce(await values(itr), ANY))


async def accumulate(itr: Any, op: int = ADD) -> AsyncIterator[float]:
    """Running accumulation, one value per item, seeded with the first item.

    Mirrors ``aioitertools.itertools.accumulate`` with ``operator.add``; ``op``
    selects add, mul, min or max. Upstream also accepts an arbitrary sync or
    async binary callable, which cannot be expressed as a kernel op code, so
    only the four numeric operators are available here.
    """
    buf = await values(itr)
    if buf.size == 0:
        return
    for total in _lib.accumulate(buf, op):
        yield float(total)


def accumulate_array(x: Iterable[float], op: int = ADD) -> np.ndarray:
    """Running accumulation of an already-materialised buffer, as an array."""
    return _lib.accumulate(x, op)


async def compress(itr: Any, selectors: Any) -> AsyncIterator[float]:
    """Yield the elements of ``itr`` whose matching selector is truthy.

    Stops at the shorter of the two, as ``aioitertools.itertools.compress``
    does through ``zip``. Pure data movement, so the values come out bit for
    bit.
    """
    buf = await values(itr)
    sel = await values(selectors)
    n = builtins.min(buf.size, sel.size)
    for value in _lib.compress(buf[:n], sel[:n]):
        yield float(value)


def chunk_offsets(n: int, size: int) -> np.ndarray:
    """Start offsets of consecutive ``size``-item chunks of ``n`` items."""
    return _lib.chunk_offsets(n, size)


async def chunked_sizes(itr: Any, n: int) -> list[int]:
    """Lengths of the chunks ``aioitertools.more_itertools.chunked`` would yield."""
    if n <= 0:
        raise ValueError("n needs to be greater than 0")
    buf = await values(itr)
    starts = _lib.chunk_offsets(buf.size, n)
    if starts.size == 0:
        return []
    ends = np.append(starts[1:], buf.size)
    return [int(e - s) for s, e in builtins.zip(starts, ends, strict=True)]
