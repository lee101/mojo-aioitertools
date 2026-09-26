# mojo-aioitertools

`mojo-aioitertools` is the compute-oriented subset of
[aioitertools](https://github.com/ammythespy/aioitertools): the reductions,
scans and gathers, compiled into one Mojo shared library.

The Python package is named `mojo_aioitertools`, so it installs alongside the
real `aioitertools` and the tests compare the two directly.

```python
import asyncio
import mojo_aioitertools as mai

async def main():
    await mai.sum([1.5, 2.5, -1.0])          # 3.0
    await mai.min(stream_of_floats())        # smallest item
    [v async for v in mai.accumulate([1, 2, 3, 4], mai.MUL)]   # 1, 2, 6, 24
    [v async for v in mai.compress(range(8), [1, 0, 0, 1])]   # 0, 3
    await mai.chunked_sizes(range(7), 3)     # [3, 3, 1]

asyncio.run(main())
```

## Why this is the compute core

`aioitertools` is 3060 lines, 790 of which are its own test suite. The package
reimplements `itertools` and the numeric builtins over mixed sync/async
iterables, and the coroutine machinery is the package. Four operations in it are
loops over values:

| upstream | formula |
| --- | --- |
| `builtins.sum` | `value += item` |
| `builtins.min` / `max` | `value = item if item < value else value` |
| `builtins.all` / `any` | truthiness reduction over the gathered values |
| `itertools.accumulate` | `total = func(total, item)`, seeded with the first item |
| `itertools.compress` | yield `item` when `selector` is truthy |
| `more_itertools.chunked` | split at every n-th item |

Those are the loops that are compiled here, plus a deliberate fast path: a
synchronous iterable is converted by NumPy's C-level conversion and handed
straight to the kernel, where `aioitertools` runs it through an async generator
and pays a coroutine round trip per item. That difference, not the reduction
itself, is where most of the speedup comes from - and it is why the async path
is benchmarked separately and reported as the loss it is.

## Covered subset

| area | implemented API |
| --- | --- |
| Reductions | `sum`, `min`, `max`, `all`, `any`, `reduce` |
| Scans | `accumulate` (add / mul / min / max), `accumulate_array` |
| Gathers and index scans | `compress`, `chunk_offsets`, `chunked_sizes` |
| Kernels | `ait_reduce`, `ait_accumulate`, `ait_compress`, `ait_chunk_offsets` |

`values` exposes the drain contract, and the mixed sync/async iteration itself
is borrowed from the real `aioitertools.iter` rather than reimplemented.

### Not implemented, and why

* **The iterator algebra** - `chain`, `tee`, `islice`, `zip_longest`,
  `batched`, `combinations`, `permutations`, `product`, `cycle`, `count`,
  `repeat`, `groupby`, `before_and_after`, `starmap`, and the rest. These are
  coroutine generators with no inner loop over values worth compiling; use the
  real `aioitertools`.
* **`accumulate` with an arbitrary binary callable.** Upstream accepts any sync
  or async `func`; a kernel can only take an op code, so the four numeric
  operators are available and the general form is not. It is a real gap, not an
  oversight - a callable would have to come back to Python per element.
* **`min`/`max` with a `key=` function.** Same reason.
* **`take`, `dropwhile`, `takewhile`, `filterfalse`, `enumerate`, `map`,
  `set`, `tuple`, `list`** - no arithmetic.
* **NaN in `min`/`max`.** The kernel propagates whichever operand the
  comparison selects, which is *not* the left-to-right rule CPython uses. This
  is not covered and not tested; do not reduce a buffer that may contain NaN
  with these functions.

## Install

```bash
bash build/build.sh          # -> dist/libmojo-aioitertools.so
PYTHONPATH=python python -m pytest tests -q
PYTHONPATH=python python bench/bench.py
```

The repository pins its own Mojo toolchain in `pixi.toml`
(`mojo == 1.2.0.dev2026092605`); use the shared environment rather than
`pixi install`. Set `PYTHONPATH=python` when using the package outside a task.

`pytest-asyncio` is not installed in the shared test environment, so
`tests/conftest.py` runs coroutine tests with `asyncio.run` through a
`pytest_pyfunc_call` hook. Every `async def test_*` really runs.

## Performance

Best-of-N wall clock, same process, on the shared 36-core build box, which is
also running other builds - absolute numbers move between runs, the ratios much
less so. Every case checks its result against `aioitertools` *before* timing.
The second line of each case is the same buffer through vectorised NumPy, which
is the fair comparison for the kernel itself; the first column is the library
being replaced.

| case | aioitertools | mojo-aioitertools | result | same buffer via NumPy |
| --- | ---: | ---: | ---: | ---: |
| sum, 1M sync list | 572.41 ms | 138.93 ms | 4.12x faster | 0.60 ms |
| sum, 65536 async generator | 36.60 ms | 41.02 ms | **0.89x - slower** | - |
| min, 1M sync list | 687.64 ms | 78.36 ms | 8.78x faster | 1.45 ms |
| accumulate (add), 1M | 3972.40 ms | 85.38 ms | 46.52x faster | 9.35 ms (cumsum) |
| compress, 262144 | 24908.37 ms | 2.07 ms | 12053x faster | 2.63 ms |
| chunk_offsets, 1M / 64 | 0.03 ms | 0.07 ms | **0.44x - slower** | - |

Three of those need explaining, and two are losses.

**The wins against `aioitertools` are mostly wins against coroutine overhead.**
`aioitertools.compress` walks its two inputs through an async `zip` that awaits
each pair: 95 microseconds per element at n = 1M. Nothing to do with arithmetic.
The same buffer through a compiled gather is 2.6 ms, and the Mojo kernel is
2.1 ms - a real 0.5 ms of kernel, against a reference that is three orders of
magnitude away.

**The async-generator case is a genuine loss.** Draining an async iterable costs
one `await` per item in both libraries. `aioitertools` adds each item to a
running total in that same loop; this port collects into a list and converts,
which is one extra pass. There is no arithmetic to win, so it loses. The honest
use of this package is on materialised data.

**`chunk_offsets` loses too.** It computes at most n/64 = 16384 indices. The
Mojo loop is a few microseconds and the ctypes call is about 5, so NumPy's
`arange` wins. The kernel is there because `chunked_sizes` needs the same scan,
not because it is faster than `arange`.

**Against NumPy, the kernels are not a win and are not meant to be.** A single
pass over contiguous float64 is already at memory speed in NumPy, and
`np.cumsum` beats a serial dependent-add scan. The port's value is that it keeps
the aioitertools *interface* while removing the per-item coroutine, not that it
beats NumPy at NumPy.

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-aioitertools.so`.

`python/mojo_aioitertools` owns every array. `_materialise` sends a synchronous
iterable through `np.asarray`/`np.fromiter` (C-level), and an async iterable
through `aioitertools.iter` with one `await` per item; either way the kernel
sees a contiguous `float64` buffer. Buffers cross the C ABI as 64-bit addresses
(`ctypes.c_int64`; `c_int` truncates and segfaults) and are rebuilt inside the
kernel as `Pointer[Float64, AnyOrigin[mut=True]]`.

**These kernels are memory-bound and are left serial.** A reduction and a scan
over contiguous float64 are one pass at memory speed; a thread pool would only
add call overhead. The `accumulate` scan is additionally a dependent chain that
cannot be split without a parallel-prefix rewrite that would change the
summation order, and therefore the result. No `parallelize` is used: 1.2.0 cannot
pass pointers into a parallel body.

## Numerical parity

| operation | agreement | asserted |
| --- | --- | --- |
| `sum`, `accumulate` | same left-to-right order as the Python reference; the only expected difference is multiply-add contraction | `rtol=1e-12` |
| `min`, `max`, `all`, `any` | selections, no arithmetic | exact equality |
| `compress`, `chunk_offsets` | data movement and index arithmetic | exact equality |

`all`/`any` use Python's float truthiness `x != 0.0`, under which NaN is
truthy, matching the reference on the values tested.

## Tests

`tests/test_builtins.py`, 43 tests. Each reduction is checked against the
corresponding `aioitertools` builtin over a plain list *and* over a real async
generator, so a bug that only shows on the awaiting path cannot hide. Covered:
every reduce op code, the four accumulate operators, the empty-iterable
identities (`sum([]) == 0`, `all([]) is True`, `any([]) is False`), the
input-order sensitivity of a cancelling sum, the shorter-input rule of
`compress`, the boundary offsets of `chunked` for nine chunk sizes, and the
`ValueError` raised for an empty `min`/`max`.

## License

MIT
