"""Reductions and scans for the numeric subset of aioitertools.

`aioitertools` reimplements `itertools` and the numeric builtins over mixed
sync/async iterables. The coroutine plumbing is the package; the arithmetic is
four loops:

    builtins.sum          total += item
    builtins.min/max      value = item if item < value else value
    builtins.all/any      truthiness reduction
    itertools.accumulate  total = func(total, item)   (operator.add by default)
    itertools.compress    yield item when selector
    more_itertools.chunked  split at every n-th item

Each is a single pass over an already-materialised buffer, and each is compiled
here. `compress` and `chunked` are pure data movement - a gather and an index
scan - and are exact; the arithmetic cases are not, because Mojo emits FMA and
its own transcendental routines.

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric.
"""

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


@export("ait_reduce")
def ait_reduce(
    x_addr: Int, result_addr: Int, n: Int, op: Int
) abi("C"):
    """Reduce `x[0:n]` to a single float64 written at `result_addr`.

    op 0 sum, 1 min, 2 max, 3 all, 4 any. `all`/`any` use Python's float
    truthiness, `x != 0.0`, which is also true for NaN.
    """
    var x = fp(x_addr)
    var res = fp(result_addr)
    if n <= 0:
        # `all([])` is True and `any([])` is False; every other reduction of an
        # empty buffer is 0.0, which is also what `sum([])` is.
        res[unsafe_offset=0] = 1.0 if op == 3 else 0.0
        return
    if op == 0:
        var total = x[unsafe_offset=0]
        for i in range(1, n):
            total = total + x[unsafe_offset=i]
        res[unsafe_offset=0] = total
    elif op == 1:
        var acc = x[unsafe_offset=0]
        for i in range(1, n):
            var v = x[unsafe_offset=i]
            if v < acc:
                acc = v
        res[unsafe_offset=0] = acc
    elif op == 2:
        var acc = x[unsafe_offset=0]
        for i in range(1, n):
            var v = x[unsafe_offset=i]
            if v > acc:
                acc = v
        res[unsafe_offset=0] = acc
    elif op == 3:
        var acc = 1.0
        for i in range(n):
            if x[unsafe_offset=i] == 0.0:
                acc = 0.0
        res[unsafe_offset=0] = acc
    else:
        var acc = 0.0
        for i in range(n):
            if x[unsafe_offset=i] != 0.0:
                acc = 1.0
        res[unsafe_offset=0] = acc


@export("ait_accumulate")
def ait_accumulate(
    x_addr: Int, out_addr: Int, n: Int, op: Int
) abi("C"):
    """Inclusive prefix scan of `x[0:n]` into `out[0:n]`.

    op 0 add, 1 mul, 2 min, 3 max. The first output is the first input, which is
    what `itertools.accumulate` yields: it seeds the total with the first item
    rather than with an identity.
    """
    var x = fp(x_addr)
    var out = fp(out_addr)
    if n <= 0:
        return
    var acc = x[unsafe_offset=0]
    out[unsafe_offset=0] = acc
    for i in range(1, n):
        var v = x[unsafe_offset=i]
        if op == 0:
            acc = acc + v
        elif op == 1:
            acc = acc * v
        elif op == 2:
            if v < acc:
                acc = v
        else:
            if v > acc:
                acc = v
        out[unsafe_offset=i] = acc


@export("ait_compress")
def ait_compress(
    x_addr: Int, mask_addr: Int, out_addr: Int, n: Int, count_addr: Int
) abi("C"):
    """Gather `x[i]` into `out` for every non-zero `mask[i]`; count goes to
    `count_addr`.

    Pure data movement, so the result is bit-identical to the input. A selector
    that is NaN is truthy in Python and non-zero here, so it is selected.
    """
    var x = fp(x_addr)
    var mask = fp(mask_addr)
    var out = fp(out_addr)
    var count = fp(count_addr)
    var k = 0
    for i in range(n):
        if mask[unsafe_offset=i] != 0.0:
            out[unsafe_offset=k] = x[unsafe_offset=i]
            k += 1
    count[unsafe_offset=0] = Float64(k)


@export("ait_chunk_offsets")
def ait_chunk_offsets(
    n: Int, size: Int, out_addr: Int, capacity: Int, count_addr: Int
) abi("C"):
    """Write the start offsets of consecutive `size`-item chunks of `n` items.

    The count of chunks is written to `count_addr`; offsets beyond `capacity`
    are not written, so the caller can size the buffer with a cheap upper bound
    and slice afterwards. The total item count is written to `count_addr + 1`.
    """
    var out = fp(out_addr)
    var count = fp(count_addr)
    if size <= 0:
        count[unsafe_offset=0] = 0.0
        count[unsafe_offset=1] = Float64(n)
        return
    var nchunks = 0
    var i = 0
    while i < n:
        if nchunks < capacity:
            out[unsafe_offset=nchunks] = Float64(i)
        nchunks += 1
        i += size
    count[unsafe_offset=0] = Float64(nchunks)
    count[unsafe_offset=1] = Float64(n)
