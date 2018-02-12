"""
Batcher's Odd-even sorting network
Adapted from https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort
"""
import math

from dask import delayed

import pygdf


def get_oversized(length):
    """
    The oddeven network requires a power-of-2 length.
    This method computes the next power-of-2 from the *length* if
    *length* is not a power-of-2 value.
    """
    return 2 ** math.ceil(math.log2(length))


def is_power_of_2(length):
    return math.log2(length).is_integer()


def oddeven_merge(lo, hi, r):
    step = r * 2
    if step < hi - lo:
        for each in oddeven_merge(lo, hi, step):
            yield each
        for each in oddeven_merge(lo + r, hi, step):
            yield each
        for i in range(lo + r, hi - r, step):
            yield (i, i + r)
    else:
        yield (lo, lo + r)


def oddeven_merge_sort_range(lo, hi):
    """ sort the part of x with indices between lo and hi.

    Note: endpoints (lo and hi) are included.
    """
    if (hi - lo) >= 1:
        # if there is more than one element, split the input
        # down the middle and first sort the first and second
        # half, followed by merging them.
        mid = lo + ((hi - lo) // 2)
        for each in oddeven_merge_sort_range(lo, mid):
            yield each
        for each in oddeven_merge_sort_range(mid + 1, hi):
            yield each
        for each in oddeven_merge(lo, hi, 1):
            yield each


def oddeven_merge_sort(length):
    """ "length" is the length of the list to be sorted.
    Returns a list of pairs of indices starting with 0 """
    assert is_power_of_2(length)
    for each in oddeven_merge_sort_range(0, length - 1):
        yield each


def _pad_data_to_length(parts):
    parts = list(parts)
    needed = get_oversized(len(parts))
    padn = needed - len(parts)
    return parts + [None] * padn, len(parts)


def _compare_frame(a, b, by):
    if a is not None and b is not None:
        joint = pygdf.concat([a, b])
        sorten = joint.sort_values(by=by)
        return sorten[:len(a)], sorten[len(a):]
    elif a is None:
        return b, None
    else:
        return a, None


def _compare_and_swap_frame(parts, a, b, by):
    compared = delayed(_compare_frame)(parts[a], parts[b], by=by)
    parts[a] = compared[0]
    parts[b] = compared[1]


def sort_delayed_frame(parts, column):
    if len(parts) == 0:
        return parts
    parts, valid = _pad_data_to_length(parts)
    for a, b in oddeven_merge_sort(len(parts)):
        _compare_and_swap_frame(parts, a, b, by=column)
    return parts[:valid]

