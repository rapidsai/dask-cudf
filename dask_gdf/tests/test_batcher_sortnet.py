import pytest

import numpy as np

import pygdf
from dask_gdf import batcher_sortnet


@pytest.mark.parametrize('n', list(range(1, 40)))
def test_padding(n):
    data = list(range(n))
    padded, valid = batcher_sortnet._pad_data_to_length(data)
    assert len(data) == valid
    assert batcher_sortnet.is_power_of_2(len(padded))
    assert valid > len(padded) / 2
    assert all(x is not None for x in padded[:valid])
    assert all(x is None for x in padded[valid:])


@pytest.mark.parametrize('seed', [43, 120])
@pytest.mark.parametrize('nelem', [2, 10, 100])
def test_compare_frame(seed, nelem):
    np.random.seed(seed)
    # Make LHS
    lhs = pygdf.DataFrame()
    lhs['a'] = lhs_a = np.random.random(nelem)
    lhs['b'] = lhs_b = np.random.random(nelem)

    # Make RHS
    rhs = pygdf.DataFrame()
    rhs['a'] = rhs_a = np.random.random(nelem)
    rhs['b'] = rhs_b = np.random.random(nelem)

    # Sort by column "a"
    got_a = batcher_sortnet._compare_frame(lhs, rhs, by='a')
    # Check
    expect_a = np.hstack([lhs_a, rhs_a])
    expect_a.sort()
    np.testing.assert_array_equal(got_a[0].a.to_array(), expect_a[:nelem])
    np.testing.assert_array_equal(got_a[1].a.to_array(), expect_a[nelem:])

    # Sort by column "b"
    got_b = batcher_sortnet._compare_frame(lhs, rhs, by='b')
    # Check
    expect_b = np.hstack([lhs_b, rhs_b])
    expect_b.sort()
    np.testing.assert_array_equal(got_b[0].b.to_array(), expect_b[:nelem])
    np.testing.assert_array_equal(got_b[1].b.to_array(), expect_b[nelem:])


def test_compare_frame_with_none():
    df = pygdf.DataFrame()
    res = batcher_sortnet._compare_frame(df, None, by='a')
    assert res == (df, None)
    res = batcher_sortnet._compare_frame(None, df, by='a')
    assert res == (df, None)
    res = batcher_sortnet._compare_frame(None, None, by='a')
    assert res == (None, None)
