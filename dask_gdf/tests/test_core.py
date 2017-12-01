import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal

import pytest

import pygdf as gd
import dask_gdf as dgd
import dask.dataframe as dd


def test_from_pygdf():
    np.random.seed(0)

    df = pd.DataFrame({'x': np.random.randint(0, 5, size=10000),
                       'y': np.random.normal(size=10000)})

    gdf = gd.DataFrame.from_pandas(df)

    # Test simple around to/from dask
    ingested = dgd.from_pygdf(gdf, npartitions=2)
    assert_frame_equal(ingested.compute().to_pandas(), df)

    # Test conversion to dask.dataframe
    ddf = ingested.to_dask_dataframe()
    assert_frame_equal(ddf.compute(), df)


def _fragmented_gdf(df, nsplit):
    n = len(df)

    # Split dataframe in *nsplit*
    subdivsize = n // nsplit
    starts = [i * subdivsize for i in range(nsplit)]
    ends = starts[1:] + [None]
    frags = [df[s:e] for s, e in zip(starts, ends)]
    return frags


def test_concat():
    np.random.seed(0)

    n = 1000
    df = pd.DataFrame({'x': np.random.randint(0, 5, size=n),
                       'y': np.random.normal(size=n)})

    gdf = gd.DataFrame.from_pandas(df)
    frags = _fragmented_gdf(gdf, nsplit=13)

    # Combine with concat
    concated = dgd.concat(frags)
    assert_frame_equal(df, concated.compute().to_pandas())


def test_append():
    np.random.seed(0)

    n = 1000
    df = pd.DataFrame({'x': np.random.randint(0, 5, size=n),
                       'y': np.random.normal(size=n)})

    gdf = gd.DataFrame.from_pandas(df)
    frags = _fragmented_gdf(gdf, nsplit=13)

    # Combine with .append
    head = frags[0]
    tail = frags[1:]

    appended = dgd.from_pygdf(head, npartitions=1)
    for each in tail:
        appended = appended.append(each)

    assert_frame_equal(df, appended.compute().to_pandas())


def test_series_concat():
    np.random.seed(0)

    n = 1000
    df = pd.DataFrame({'x': np.random.randint(0, 5, size=n),
                       'y': np.random.normal(size=n)})

    gdf = gd.DataFrame.from_pandas(df)
    frags = _fragmented_gdf(gdf, nsplit=13)

    frags = [df.x for df in frags]

    concated = dgd.concat(frags).compute().to_pandas()
    assert isinstance(concated, pd.Series)
    np.testing.assert_array_equal(concated, df.x)


def test_series_append():
    np.random.seed(0)

    n = 1000
    df = pd.DataFrame({'x': np.random.randint(0, 5, size=n),
                       'y': np.random.normal(size=n)})

    gdf = gd.DataFrame.from_pandas(df)
    frags = _fragmented_gdf(gdf, nsplit=13)

    frags = [df.x for df in frags]

    appending = dgd.from_pygdf(frags[0], npartitions=1)
    for frag in frags[1:]:
        appending = appending.append(frag)

    appended = appending.compute().to_pandas()
    assert isinstance(appended, pd.Series)
    np.testing.assert_array_equal(appended, df.x)


def test_query():
    np.random.seed(0)

    df = pd.DataFrame({'x': np.random.randint(0, 5, size=10),
                       'y': np.random.normal(size=10)})
    gdf = gd.DataFrame.from_pandas(df)
    expr = 'x > 2'

    assert_frame_equal(gdf.query(expr).to_pandas(), df.query(expr))

    queried = (dgd.from_pygdf(gdf, npartitions=2).query(expr))

    got = queried.compute().to_pandas()
    expect = gdf.query(expr).to_pandas()

    assert_frame_equal(got, expect)


def test_head():
    np.random.seed(0)
    df = pd.DataFrame({'x': np.random.randint(0, 5, size=100),
                       'y': np.random.normal(size=100)})
    gdf = gd.DataFrame.from_pandas(df)
    dgf = dgd.from_pygdf(gdf, npartitions=2)

    assert_frame_equal(dgf.head().to_pandas(), df.head())


def test_from_dask_dataframe():
    np.random.seed(0)
    df = pd.DataFrame({'x': np.random.randint(0, 5, size=20),
                       'y': np.random.normal(size=20)})
    ddf = dd.from_pandas(df, npartitions=2)
    dgdf = dgd.from_dask_dataframe(ddf)
    got = dgdf.compute().to_pandas()
    expect = df

    np.testing.assert_array_equal(got.index.values, expect.index.values)
    np.testing.assert_array_equal(got.x.values, expect.x.values)
    np.testing.assert_array_equal(got.y.values, expect.y.values)


@pytest.mark.parametrize('seed', list(range(5)))
def test_set_index(seed):
    np.random.seed(seed)
    nelem = 20

    # Use unique index range as the sort may not be stable-ordering
    x = np.arange(nelem)
    np.random.shuffle(x)
    df = pd.DataFrame({'x': x,
                       'y': np.random.randint(0, nelem, size=nelem)})
    ddf = dd.from_pandas(df, npartitions=2)
    dgdf = dgd.from_dask_dataframe(ddf)

    expect = ddf.set_index('x').compute()
    got = dgdf.set_index('x').compute().to_pandas()

    np.testing.assert_array_equal(got.index.values, expect.index.values)
    np.testing.assert_array_equal(got.y.values, expect.y.values)
    assert got.columns == expect.columns



@pytest.mark.parametrize('nelem,nparts', [(10, 1),
                                          (100, 10),
                                          (1000, 10)])
def test_take(nelem, nparts):
    np.random.seed(0)

    # # Use unique index range as the sort may not be stable-ordering
    x = np.random.randint(0, nelem, size=nelem)
    y = np.random.random(nelem)

    selected = np.random.randint(0, nelem - 1, size=nelem // 2)

    df = pd.DataFrame({'x': x, 'y': y})

    ddf = dd.from_pandas(df, npartitions=nparts)
    dgdf = dgd.from_dask_dataframe(ddf)
    out = dgdf.take(gd.Series(selected), npartitions=5)
    got = out.compute().to_pandas()

    expect = df.take(selected)
    assert 1 < out.npartitions <= 5
    np.testing.assert_array_equal(got.index, np.arange(len(got)))
    np.testing.assert_array_equal(got.x, expect.x)
    np.testing.assert_array_equal(got.y, expect.y)


def test_assign():
    np.random.seed(0)
    df = pd.DataFrame({'x': np.random.randint(0, 5, size=20),
                       'y': np.random.normal(size=20)})

    dgf = dgd.from_pygdf(gd.DataFrame.from_pandas(df), npartitions=2)
    pdcol = pd.Series(np.arange(20) + 1000)
    newcol = dgd.from_pygdf(gd.Series(pdcol),
                            npartitions=dgf.npartitions)
    out = dgf.assign(z=newcol)

    got = out.compute().to_pandas()
    assert_frame_equal(got.loc[:, ['x', 'y']], df)
    np.testing.assert_array_equal(got['z'], pdcol)
