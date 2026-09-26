"""Parity against the real aioitertools reductions, scans and gathers.

Every numeric result is compared with the corresponding ``aioitertools`` builtin
or ``itertools`` function, over both a plain list and a genuinely async
generator, so a bug that only shows on the async path cannot hide. Coroutine
tests are run for real by the ``pytest_pyfunc_call`` hook in ``conftest.py``;
``pytest-asyncio`` is not needed and not installed.

Tolerances. The reductions and scans accumulate in the same left-to-right order
as the pure-Python reference, so the only expected difference is the compiler's
freedom to contract multiply-add, and the parity assertions use ``rtol=1e-12``.
``min``, ``max``, ``all``, ``any``, ``compress`` and ``chunk_offsets`` select or
move values without arithmetic and are asserted exactly.
"""

import asyncio
import operator

import numpy as np
import pytest

import aioitertools as ait
from aioitertools import builtins as ait_builtins
from aioitertools import more_itertools as ait_more

import mojo_aioitertools as mai

RTOL = 1e-12

VALUES = [3.5, -1.25, 0.0, 8.0, 2.5, -7.0, 4.25, 0.5]


async def agen(values, delay=0.0):
    """A real async generator, so the tests exercise the awaiting path."""
    for v in values:
        if delay:
            await asyncio.sleep(delay)
        yield v


# --------------------------------------------------------------------------
# sum


@pytest.mark.parametrize("start", [None, 0.0, 10.5, -3.25])
async def test_sum_matches_upstream(start):
    theirs = (
        await ait_builtins.sum(VALUES, start) if start is not None else await ait_builtins.sum(VALUES)
    )
    np.testing.assert_allclose(await mai.sum(VALUES, start), theirs, rtol=RTOL)


async def test_sum_matches_upstream_on_an_async_iterable():
    theirs = await ait_builtins.sum(agen(VALUES))
    mine = await mai.sum(agen(VALUES))
    np.testing.assert_allclose(mine, theirs, rtol=RTOL)


async def test_sum_of_an_empty_iterable_is_the_start_value():
    assert await mai.sum([]) == 0.0
    assert await mai.sum([], 4.5) == 4.5


async def test_sum_accumulates_in_input_order():
    """Cancellation: a reordered sum would give a visibly different answer."""
    values = [1e16, 1.0, -1e16, 1.0]
    np.testing.assert_allclose(
        await mai.sum(values), await ait_builtins.sum(values), rtol=RTOL
    )
    assert await mai.sum(values) != pytest.approx(sum(reversed(values)), rel=RTOL)


async def test_sum_over_a_generator_matches_upstream():
    np.testing.assert_allclose(
        await mai.sum(v for v in VALUES), await ait_builtins.sum(v for v in VALUES), rtol=RTOL
    )


# --------------------------------------------------------------------------
# min / max / all / any


async def test_min_and_max_match_upstream():
    assert await mai.min(VALUES) == await ait_builtins.min(VALUES)
    assert await mai.max(VALUES) == await ait_builtins.max(VALUES)


async def test_min_and_max_match_upstream_on_an_async_iterable():
    assert await mai.min(agen(VALUES)) == await ait_builtins.min(agen(VALUES))
    assert await mai.max(agen(VALUES)) == await ait_builtins.max(agen(VALUES))


async def test_min_and_max_on_single_and_duplicate_values():
    for case in ([4.0], [2.0, 2.0, 2.0], [-1.0, -1.0, 0.0]):
        assert await mai.min(case) == await ait_builtins.min(case)
        assert await mai.max(case) == await ait_builtins.max(case)


async def test_min_and_max_of_empty_raise_like_upstream():
    with pytest.raises(ValueError):
        await mai.min([])
    with pytest.raises(ValueError):
        await mai.max([])
    assert await mai.min([], default=9.0) == 9.0
    assert await ait_builtins.min([], default=9.0) == 9.0


async def test_all_and_any_match_upstream():
    truthy, falsy = [1.0, -2.0, 0.5], [0.0, 1.0, 0.0]
    assert await mai.all(truthy) is await ait_builtins.all(truthy) is True
    assert await mai.all(falsy) is await ait_builtins.all(falsy) is False
    assert await mai.any(falsy) is await ait_builtins.any(falsy) is True
    assert await mai.any([0.0, 0.0]) is await ait_builtins.any([0.0, 0.0]) is False
    assert await mai.all([]) is await ait_builtins.all([]) is True
    assert await mai.any([]) is await ait_builtins.any([]) is False


async def test_all_and_any_match_upstream_on_an_async_iterable():
    assert await mai.all(agen([1.0, 2.0])) is await ait_builtins.all(agen([1.0, 2.0]))
    assert await mai.any(agen([0.0, 0.0])) is await ait_builtins.any(agen([0.0, 0.0]))


# --------------------------------------------------------------------------
# accumulate


async def test_accumulate_add_matches_upstream():
    theirs = [v async for v in ait.itertools.accumulate(VALUES, operator.add)]
    mine = [v async for v in mai.accumulate(VALUES, mai.ADD)]
    np.testing.assert_allclose(mine, theirs, rtol=RTOL)


async def test_accumulate_on_an_async_iterable():
    theirs = [v async for v in ait.itertools.accumulate(agen(VALUES), operator.add)]
    mine = [v async for v in mai.accumulate(agen(VALUES), mai.ADD)]
    np.testing.assert_allclose(mine, theirs, rtol=RTOL)


@pytest.mark.parametrize(
    "op,theirs_op",
    [
        (mai.ADD, operator.add),
        (mai.MUL, operator.mul),
        (mai.CMIN, min),
        (mai.CMAX, max),
    ],
)
async def test_accumulate_operators_match_upstream(op, theirs_op):
    theirs = [v async for v in ait.itertools.accumulate(VALUES, theirs_op)]
    mine = [v async for v in mai.accumulate(VALUES, op)]
    np.testing.assert_allclose(mine, theirs, rtol=RTOL)


async def test_accumulate_of_a_single_item_is_that_item():
    np.testing.assert_array_equal(mai.accumulate_array([7.5]), [7.5])


async def test_accumulate_of_nothing_is_empty():
    assert mai.accumulate_array([]).size == 0


async def test_accumulate_of_an_empty_iterable_yields_nothing():
    assert [v async for v in mai.accumulate([])] == []
    assert [v async for v in ait.itertools.accumulate([])] == []


# --------------------------------------------------------------------------
# compress


async def test_compress_matches_upstream():
    selectors = [1, 0, 0, 1, 1, 0, 1, 0]
    theirs = [v async for v in ait.itertools.compress(range(8), selectors)]
    mine = [v async for v in mai.compress(range(8), selectors)]
    assert mine == theirs


async def test_compress_stops_at_the_shorter_input():
    theirs = [v async for v in ait.itertools.compress(range(8), [1, 1, 1])]
    mine = [v async for v in mai.compress(range(8), [1, 1, 1])]
    assert mine == theirs == [0, 1, 2]


def test_compress_is_bit_exact_data_movement():
    rng = np.random.default_rng(4)
    values = rng.standard_normal(1000)
    mask = (rng.random(1000) > 0.5).astype(np.float64)
    got = mai._lib.compress(values, mask)
    np.testing.assert_array_equal(got, values[mask.astype(bool)])


def test_compress_rejects_a_length_mismatch():
    with pytest.raises(ValueError):
        mai._lib.compress([1.0, 2.0, 3.0], [1.0, 0.0])


# --------------------------------------------------------------------------
# chunked


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 8, 9, 100])
async def test_chunked_sizes_match_upstream(n):
    theirs = [len(c) async for c in ait_more.chunked(VALUES, n)]
    assert await mai.chunked_sizes(VALUES, n) == theirs, f"n={n}"


def test_chunk_offsets_are_the_real_boundaries():
    np.testing.assert_array_equal(mai.chunk_offsets(7, 3), [0, 3, 6])
    assert mai.chunk_offsets(0, 3).size == 0
    assert mai.chunk_offsets(6, 3).size == 2


async def test_chunk_offsets_reject_a_non_positive_size():
    with pytest.raises(ValueError):
        mai.chunk_offsets(10, 0)
    with pytest.raises(ValueError):
        await mai.chunked_sizes(VALUES, 0)


async def test_chunked_sizes_on_an_async_iterable():
    theirs = [len(c) async for c in ait_more.chunked(agen(VALUES), 3)]
    assert await mai.chunked_sizes(agen(VALUES), 3) == theirs


# --------------------------------------------------------------------------
# the buffer contract


async def test_non_contiguous_and_integer_inputs_are_normalised():
    base = np.arange(20, dtype=np.float64)
    strided = base[::2]
    np.testing.assert_allclose(
        await mai.sum(strided), await ait_builtins.sum(strided.tolist()), rtol=RTOL
    )
    assert await mai.sum(np.arange(5)) == 10.0
    assert await mai.min(np.array([[3.0, 1.0], [2.0, 9.0]])) == 1.0


async def test_values_drains_both_iterable_kinds_to_float64():
    from_list = await mai.values(VALUES)
    assert from_list.dtype == np.float64
    assert from_list.shape == (len(VALUES),)
    from_async = await mai.values(agen(VALUES))
    np.testing.assert_array_equal(from_async, from_list)


def test_reduce_dispatches_every_op_code():
    buf = np.array([0.0, 2.0, 0.0])
    assert mai._lib.reduce(buf, mai.SUM) == 2.0
    assert mai._lib.reduce(buf, mai.MIN) == 0.0
    assert mai._lib.reduce(buf, mai.MAX) == 2.0
    assert mai._lib.reduce(buf, mai.ALL) == 0.0
    assert mai._lib.reduce(buf, mai.ANY) == 1.0
    assert mai._lib.reduce(np.array([]), mai.SUM) == 0.0


def test_the_real_aioitertools_still_imports_alongside():
    """The whole point of the name: both packages are importable at once."""
    assert ait.__name__ == "aioitertools"
    assert mai.__name__ == "mojo_aioitertools"
    assert ait_builtins.sum is not mai.sum
