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


def test_dataframe_to_delayed():
    nelem = 100

    df = gd.DataFrame()
    df['x'] = np.arange(nelem)
    df['y'] = np.random.randint(nelem, size=nelem)

    ddf = dgd.from_pygdf(df, npartitions=5)

    delays = ddf.to_delayed()

    assert len(delays) == 5

    # Concat the delayed partitions
    got = gd.concat([d.compute() for d in delays])
    assert_frame_equal(got.to_pandas(), df.to_pandas())

    # Check individual partitions
    divs = ddf.divisions
    assert len(divs) == len(delays) + 1

    for i, part in enumerate(delays):
        s = divs[i]
        # The last divisions in the last index
        e = None if i + 1 == len(delays) else divs[i + 1]
        expect = df[s:e].to_pandas()
        got = part.compute().to_pandas()
        assert_frame_equal(got, expect)


def test_series_to_delayed():
    nelem = 100

    sr = gd.Series(np.random.randint(nelem, size=nelem))

    dsr = dgd.from_pygdf(sr, npartitions=5)

    delays = dsr.to_delayed()

    assert len(delays) == 5

    # Concat the delayed partitions
    got = gd.concat([d.compute() for d in delays])
    assert isinstance(got, gd.Series)
    np.testing.assert_array_equal(got.to_pandas(), sr.to_pandas())

    # Check individual partitions
    divs = dsr.divisions
    assert len(divs) == len(delays) + 1

    for i, part in enumerate(delays):
        s = divs[i]
        # The last divisions in the last index
        e = None if i + 1 == len(delays) else divs[i + 1]
        expect = sr[s:e].to_pandas()
        got = part.compute().to_pandas()
        np.testing.assert_array_equal(got, expect)

