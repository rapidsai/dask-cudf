import operator

import numpy as np
import pandas as pd
import pytest

import pygdf as gd
import dask_gdf as dgd


def _make_random_frame(nelem, npartitions=2):
    df = pd.DataFrame({'x': np.random.randint(0, 5, size=nelem),
                       'y': np.random.normal(size=nelem)})
    gdf = gd.DataFrame.from_pandas(df)
    dgf = dgd.from_pygdf(gdf, npartitions=npartitions)
    return df, dgf


_binops = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.eq,
    operator.ne,
    operator.gt,
    operator.ge,
    operator.lt,
    operator.le,
]


@pytest.mark.parametrize('binop', _binops)
def test_binops(binop):
    np.random.seed(0)
    size = 10
    lhs_df, lhs_gdf = _make_random_frame(size)
    rhs_df, rhs_gdf = _make_random_frame(size)
    got = binop(lhs_gdf.x, rhs_gdf.x)
    exp = binop(lhs_df.x, rhs_df.x)
    np.testing.assert_array_equal(got.compute().to_array(), exp)
