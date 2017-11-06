"""
Test IO with dask.delayed API
"""
import numpy as np
from pandas.util.testing import assert_frame_equal

import pygdf as gd
import dask_gdf as dgd
from dask.delayed import delayed


@delayed
def load_data(nelem, ident):
    df = gd.DataFrame()
    df['x'] = np.arange(nelem)
    df['ident'] = np.asarray([ident] * nelem)
    return df


@delayed
def get_combined_column(df):
    return df.x * df.ident


def test_dataframe_from_delayed():
    delays = [load_data(10 * i, i) for i in range(1, 3)]
    out = dgd.from_delayed(delays)
    res = out.compute()
    assert isinstance(res, gd.DataFrame)

    expected = gd.concat([d.compute() for d in delays])
    assert_frame_equal(res.to_pandas(), expected.to_pandas())


def test_series_from_delayed():
    delays = [get_combined_column(load_data(10 * i, i))
              for i in range(1, 3)]
    out = dgd.from_delayed(delays)
    res = out.compute()
    assert isinstance(res, gd.Series)

    expected = gd.concat([d.compute() for d in delays])
    np.testing.assert_array_equal(res.to_pandas(), expected.to_pandas())
