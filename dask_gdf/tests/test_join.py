import pytest
import numpy as np

import pygdf as gd
import dask_gdf as dgd
from functools import partial

param_nrows = [5, 10, 100, 400]


@pytest.mark.parametrize('left_nrows', param_nrows)
@pytest.mark.parametrize('right_nrows', param_nrows)
@pytest.mark.parametrize('left_nkeys', [4, 5])
@pytest.mark.parametrize('right_nkeys', [4, 5])
def test_join_inner(left_nrows, right_nrows, left_nkeys, right_nkeys):
    chunksize = 50

    np.random.seed(0)

    # PyGDF
    left = gd.DataFrame({'x': np.random.randint(0, left_nkeys,
                                                size=left_nrows),
                         'a': np.arange(left_nrows)}.items())
    right = gd.DataFrame({'x': np.random.randint(0, right_nkeys,
                                                 size=right_nrows),
                          'a': 1000 * np.arange(right_nrows)}.items())

    expect = left.set_index('x').join(right.set_index('x'), how='inner',
                                      sort=True, lsuffix='l', rsuffix='r')
    expect = expect.to_pandas()

    # Dask GDf
    left = dgd.from_pygdf(left, chunksize=chunksize)
    right = dgd.from_pygdf(right, chunksize=chunksize)

    joined = left.set_index('x').join(right.set_index('x'), how='inner',
                                      lsuffix='l', rsuffix='r')
    got = joined.compute().to_pandas()

    # Check index
    np.testing.assert_array_equal(expect.index.values,
                                  got.index.values)

    # Check rows in each groups
    expect_rows = {}
    got_rows = {}

    def gather(df, grows):
        grows[df['index'].values[0]] = (set(df.al), set(df.ar))

    expect.reset_index().groupby('index')\
        .apply(partial(gather, grows=expect_rows))

    expect.reset_index().groupby('index')\
        .apply(partial(gather, grows=got_rows))

    assert got_rows == expect_rows


@pytest.mark.parametrize('left_nrows', param_nrows)
@pytest.mark.parametrize('right_nrows', param_nrows)
@pytest.mark.parametrize('left_nkeys', [4, 5])
@pytest.mark.parametrize('right_nkeys', [4, 5])
@pytest.mark.parametrize('how', ['left', 'right'])
def test_join_left(left_nrows, right_nrows, left_nkeys, right_nkeys, how):
    chunksize = 50

    np.random.seed(0)

    # PyGDF
    left = gd.DataFrame({'x': np.random.randint(0, left_nkeys,
                                                size=left_nrows),
                         'a': np.arange(left_nrows, dtype=np.float64)}.items())
    right = gd.DataFrame({'x': np.random.randint(0, right_nkeys,
                                                 size=right_nrows),
                          'a': 1000 * np.arange(right_nrows,
                                                dtype=np.float64)}.items())

    expect = left.set_index('x').join(right.set_index('x'), how=how,
                                      sort=True, lsuffix='l', rsuffix='r')
    expect = expect.to_pandas()

    # Dask GDf
    left = dgd.from_pygdf(left, chunksize=chunksize)
    right = dgd.from_pygdf(right, chunksize=chunksize)

    joined = left.set_index('x').join(right.set_index('x'), how=how,
                                      lsuffix='l', rsuffix='r')
    got = joined.compute().to_pandas()

    # Check index
    np.testing.assert_array_equal(expect.index.values,
                                  got.index.values)

    # Check rows in each groups
    expect_rows = {}
    got_rows = {}

    def gather(df, grows):
        cola = np.sort(np.asarray(df.al))
        colb = np.sort(np.asarray(df.ar))

        grows[df['index'].values[0]] = (cola, colb)

    expect.reset_index().groupby('index')\
        .apply(partial(gather, grows=expect_rows))

    expect.reset_index().groupby('index')\
        .apply(partial(gather, grows=got_rows))

    for k in expect_rows:
        np.testing.assert_array_equal(expect_rows[k][0],
                                      got_rows[k][0])
        np.testing.assert_array_equal(expect_rows[k][1],
                                      got_rows[k][1])
