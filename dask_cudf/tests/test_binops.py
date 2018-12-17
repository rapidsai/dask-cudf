import operator

import numpy as np
import pandas as pd
import pytest

import cudf as gd
import dask_cudf as dgd


def _make_empty_frame(npartitions=2):
    df = pd.DataFrame({"x": [], "y": []})
    gdf = gd.DataFrame.from_pandas(df)
    dgf = dgd.from_cudf(gdf, npartitions=npartitions)
    return dgf


def _make_random_frame(nelem, npartitions=2):
    df = pd.DataFrame(
        {"x": np.random.random(size=nelem), "y": np.random.random(size=nelem)}
    )
    gdf = gd.DataFrame.from_pandas(df)
    dgf = dgd.from_cudf(gdf, npartitions=npartitions)
    return df, dgf


def _make_random_frame_float(nelem, npartitions=2):
    df = pd.DataFrame(
        {
            "x": np.random.randint(0, 5, size=nelem),
            "y": np.random.normal(size=nelem) + 1,
        }
    )
    gdf = gd.DataFrame.from_pandas(df)
    dgf = dgd.from_cudf(gdf, npartitions=npartitions)
    return df, dgf


_binops = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.eq,
    operator.ne,
    operator.gt,
    operator.ge,
    operator.lt,
    operator.le,
]


@pytest.mark.parametrize("binop", _binops)
def test_series_binops_empty(binop):
    with pytest.raises(ValueError, match=r".*size=0.*"):
        gdf = _make_empty_frame()
        binop(gdf.x, gdf.y)


@pytest.mark.parametrize("binop", _binops)
def test_series_binops_integer(binop):
    np.random.seed(0)
    size = 1000000
    lhs_df, lhs_gdf = _make_random_frame(size)
    rhs_df, rhs_gdf = _make_random_frame(size)
    got = binop(lhs_gdf.x, rhs_gdf.y)
    exp = binop(lhs_df.x, rhs_df.y)
    np.testing.assert_array_almost_equal(got.compute().to_array(), exp)


@pytest.mark.parametrize("binop", _binops)
def test_series_binops_float(binop):
    np.random.seed(0)
    size = 1000000
    lhs_df, lhs_gdf = _make_random_frame_float(size)
    rhs_df, rhs_gdf = _make_random_frame_float(size)
    got = binop(lhs_gdf.x, rhs_gdf.y)
    exp = binop(lhs_df.x, rhs_df.y)
    np.testing.assert_array_almost_equal(got.compute().to_array(), exp)
