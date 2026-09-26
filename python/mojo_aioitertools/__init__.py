"""mojo-aioitertools: the reductions and scans of aioitertools, compiled in Mojo.

Installable alongside the real ``aioitertools``, which it is tested against for
parity and from which it borrows the mixed sync/async iteration helpers.
"""

from .core import (
    ADD,
    ALL,
    ANY,
    CMAX,
    CMIN,
    MAX,
    MIN,
    MUL,
    SUM,
    accumulate,
    accumulate_array,
    all,
    any,
    chunk_offsets,
    chunked_sizes,
    compress,
    max,
    min,
    reduce,
    sum,
    values,
)
from . import _lib

__all__ = [
    "ADD", "ALL", "ANY", "CMAX", "CMIN", "MAX", "MIN", "MUL", "SUM",
    "accumulate", "accumulate_array", "all", "any", "chunk_offsets",
    "chunked_sizes", "compress", "max", "min", "reduce", "sum", "values",
]
__version__ = "0.1.0"
