"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay ``c_int64`` for addresses; ``c_int``
truncates them and segfaults.
"""

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-aioitertools.so"

_A = ctypes.c_int64
_I = ctypes.c_int64

SUM, MIN, MAX, ALL, ANY = 0, 1, 2, 3, 4
ADD, MUL, CMIN, CMAX = 0, 1, 2, 3


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(f"{_LIB_PATH} not found; run `bash build/build.sh` first")
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.ait_reduce.restype = None
    lib.ait_reduce.argtypes = [_A, _A, _I, _I]
    lib.ait_accumulate.restype = None
    lib.ait_accumulate.argtypes = [_A, _A, _I, _I]
    lib.ait_compress.restype = None
    lib.ait_compress.argtypes = [_A, _A, _A, _I, _A]
    lib.ait_chunk_offsets.restype = None
    lib.ait_chunk_offsets.argtypes = [_I, _I, _A, _I, _A]
    return lib


lib = _load()

# Bound once: a CDLL attribute lookup is material next to a reduction that
# only takes a few microseconds.
_reduce = lib.ait_reduce
_accumulate = lib.ait_accumulate
_compress = lib.ait_compress
_chunk_offsets = lib.ait_chunk_offsets


def _v(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def _addr(a: np.ndarray) -> int:
    return a.ctypes.data


def reduce(x, op: int) -> float:
    """Reduce a 1-D float64 buffer to a scalar. See the module for op codes."""
    buf = _v(x).reshape(-1)
    res = np.empty(1, dtype=np.float64)
    _reduce(_addr(buf), _addr(res), buf.size, op)
    return float(res[0])


def accumulate(x, op: int) -> np.ndarray:
    """Inclusive prefix scan of a 1-D float64 buffer, ``operator.add`` by default."""
    buf = _v(x).reshape(-1)
    out = np.empty(buf.size, dtype=np.float64)
    if buf.size:
        _accumulate(_addr(buf), _addr(out), buf.size, op)
    return out


def compress(x, mask) -> np.ndarray:
    """Gather the elements of ``x`` whose ``mask`` entry is non-zero.

    Pure data movement: the result is a subsequence of the input, bit for bit.
    """
    buf = _v(x).reshape(-1)
    msk = _v(mask).reshape(-1)
    if msk.size != buf.size:
        raise ValueError("mask and x must have the same length")
    out = np.empty(buf.size, dtype=np.float64)
    count = np.empty(1, dtype=np.float64)
    _compress(_addr(buf), _addr(msk), _addr(out), buf.size, _addr(count))
    return out[: int(count[0])]


def chunk_offsets(n: int, size: int) -> np.ndarray:
    """Start offsets of consecutive ``size``-item chunks of ``n`` items.

    A final short chunk is included, matching
    ``aioitertools.more_itertools.chunked``.
    """
    if size <= 0:
        raise ValueError("chunk size must be positive")
    n = int(n)
    capacity = (n + size - 1) // size
    out = np.empty(max(capacity, 1), dtype=np.float64)
    count = np.empty(2, dtype=np.float64)
    _chunk_offsets(n, size, _addr(out), capacity, _addr(count))
    return out[: int(count[0])].astype(np.int64)
