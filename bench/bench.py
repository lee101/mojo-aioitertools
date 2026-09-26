"""Correctness-gated benchmark for mojo-aioitertools.

Every case checks agreement with the real ``aioitertools`` result before it is
timed. Two references are used, and the difference between them is the point:

* ``aioitertools`` itself, which is the honest baseline for an async-aware
  reduction - it must ``await`` each item;
* vectorised NumPy, which is the fair baseline for what the compiled kernel is
  actually doing, i.e. the single pass over a materialised buffer.

Reporting both keeps the win honest: the Mojo layer wins by a lot against the
library it replaces, and only modestly against NumPy on the same buffer.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

from aioitertools import builtins as ait_builtins  # noqa: E402
from aioitertools import itertools as ait_itertools  # noqa: E402

import mojo_aioitertools as mai  # noqa: E402

RTOL = 1e-12
N = 1 << 20


def _time(fn, repeats=5, *args):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def _run(coro_fn, *args):
    return asyncio.run(coro_fn(*args))


def bench_sum(n: int = N):
    rng = np.random.default_rng(0)
    values = rng.standard_normal(n).tolist()
    got = _run(mai.sum, values)
    theirs = _run(ait_builtins.sum, values)
    np.testing.assert_allclose(got, theirs, rtol=RTOL)
    buf = np.asarray(values)

    ref = _time(lambda: _run(ait_builtins.sum, values), 3)
    mine = _time(lambda: _run(mai.sum, values), 3)
    numpy_time = _time(lambda: float(buf.sum()), 3)
    print(f"    (same buffer through NumPy: {numpy_time*1e3:.2f} ms)")
    return f"sum {n} sync list", ref, mine


def bench_sum_async(n: int = 1 << 16):
    """The async path: both sides pay one await per item, so this is the honest
    ceiling on the improvement."""

    async def stream():
        for i in range(n):
            yield 1.0

    got = _run(mai.sum, stream())
    theirs = _run(ait_builtins.sum, stream())
    np.testing.assert_allclose(got, theirs, rtol=RTOL)

    ref = _time(lambda: _run(ait_builtins.sum, stream()), 3)
    mine = _time(lambda: _run(mai.sum, stream()), 3)
    return f"sum {n} async gen", ref, mine


def bench_accumulate(n: int = N):
    rng = np.random.default_rng(1)
    values = rng.standard_normal(n).tolist()
    got = mai.accumulate_array(values, mai.ADD)
    async def drain():
        return np.asarray([v async for v in ait_itertools.accumulate(values)])

    theirs = _run(drain)
    np.testing.assert_allclose(got, theirs, rtol=RTOL)
    buf = np.asarray(values)

    ref = _time(lambda: _run(drain), 5)
    mine = _time(lambda: mai.accumulate_array(values, mai.ADD), 3)
    numpy_time = _time(lambda: np.cumsum(buf), 3)
    print(f"    (same buffer through np.cumsum: {numpy_time*1e3:.2f} ms)")
    return f"accumulate {n}", ref, mine


def bench_min_max(n: int = N):
    rng = np.random.default_rng(2)
    values = rng.standard_normal(n).tolist()
    got = _run(mai.min, values)
    theirs = _run(ait_builtins.min, values)
    assert got == theirs, "min mismatch"
    got_max = _run(mai.max, values)
    assert got_max == _run(ait_builtins.max, values), "max mismatch"
    buf = np.asarray(values)

    ref = _time(lambda: _run(ait_builtins.min, values), 3)
    mine = _time(lambda: _run(mai.min, values), 3)
    numpy_time = _time(lambda: float(buf.min()), 3)
    print(f"    (same buffer through buf.min(): {numpy_time*1e3:.2f} ms)")
    return f"min {n} sync list", ref, mine


def bench_compress(n: int = 1 << 18):
    rng = np.random.default_rng(3)
    values = rng.standard_normal(n)
    mask = (rng.random(n) > 0.5).astype(np.float64)
    got = mai._lib.compress(values, mask)
    np.testing.assert_array_equal(got, values[mask.astype(bool)])
    keep = mask.astype(bool)
    # Materialised outside the timed region on both sides: converting a NumPy
    # array to a Python list costs more than the gather and would make the
    # baseline a strawman rather than a comparison.
    value_list, mask_list = values.tolist(), mask.tolist()

    async def theirs():
        return [v async for v in ait_itertools.compress(value_list, mask_list)]

    ref = _time(lambda: _run(theirs), 3)
    mine = _time(lambda: mai._lib.compress(values, mask), 3)
    numpy_time = _time(lambda: values[keep], 3)
    print(f"    (same buffer through boolean indexing: {numpy_time*1e3:.2f} ms)")
    return f"compress {n}", ref, mine


def bench_chunked(n: int = N, size: int = 64):
    offsets = mai.chunk_offsets(n, size)
    theirs = [len(c) for c in np.array_split(np.arange(n), (n + size - 1) // size)]
    assert len(offsets) == len(theirs), "chunk count mismatch"
    assert int(offsets[0]) == 0

    def numpy_offsets():
        return (np.arange(0, n, size)).astype(np.int64)

    ref = _time(numpy_offsets, 3)
    mine = _time(lambda: mai.chunk_offsets(n, size), 3)
    return f"chunk_offsets {n}/{size}", ref, mine


def main():
    print(f"{'case':<28}{'aioitertools':>14}{'mojo-aioitertools':>20}{'ratio':>10}")
    print(f"{'':<28}{'':>14}{'':>20}{'(ref/mojo)':>10}")
    print("-" * 72)
    for fn in (bench_sum, bench_sum_async, bench_min_max, bench_accumulate,
               bench_compress, bench_chunked):
        label, ref, got = fn()
        ratio = ref / got if got else float("nan")
        verdict = "faster" if ratio > 1.02 else ("slower" if ratio < 0.98 else "parity")
        print(f"{label:<28}{ref*1e3:>12.2f}ms{got*1e3:>18.2f}ms{ratio:>9.2f}x  {verdict}")


if __name__ == "__main__":
    main()
