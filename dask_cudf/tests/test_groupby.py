import numpy as np
import pandas as pd
import pytest
from numba import cuda

import cudf as gd
import dask_cudf as dgd


def _gen_skewed_keys(nelem):
    """Skewed keys to check a key will not split across multiple
    partitions even if the makes it unbalanced.
    """
    skewed_size = int(nelem * 0.95)
    assert nelem > skewed_size
    reminaing_size = nelem - skewed_size

    xs = np.hstack([np.random.randint(0, 2, size=skewed_size),
                    np.random.randint(2, 10, size=reminaing_size)])

    np.random.shuffle(xs)
    return xs


def _gen_uniform_keys(nelem):
    xs = np.random.randint(0, 20, size=nelem)
    return xs


@pytest.mark.xfail(reason="pandas/cudf groupbys are not consistent")
@pytest.mark.parametrize('keygen', [_gen_skewed_keys, _gen_uniform_keys])
def test_groupby_single_key(keygen):
    np.random.seed(0)

    nelem = 500
    npartitions = 10

    # Generate the keys
    xs = keygen(nelem)

    assert xs.size == nelem
    df = pd.DataFrame({'x': xs,
                       'z': np.random.normal(size=nelem) + 1})
    gdf = gd.DataFrame.from_pandas(df)
    dgf = dgd.from_cudf(gdf, npartitions=npartitions)

    groups = dgf.groupby(by=['x']).count()
    got = groups.compute().to_pandas()

    # Check against expectation
    expect = df.groupby(by=['x'], as_index=False).count()
    # Check keys
    np.testing.assert_array_equal(got.x, expect.x)
    # Check values
    np.testing.assert_array_equal(got.count_z, expect.z)


@pytest.mark.xfail(reason="pandas/cudf groupbys are not consistent")
@pytest.mark.parametrize('keygen', [_gen_skewed_keys, _gen_uniform_keys])
def test_groupby_multi_keys(keygen):
    np.random.seed(0)

    nelem = 500
    npartitions = 10

    # Generate the keys
    xs = keygen(nelem)
    ys = keygen(nelem)

    assert xs.size == nelem
    assert ys.size == nelem
    df = pd.DataFrame({'x': xs,
                       'y': ys,
                       'z': np.random.normal(size=nelem) + 1})

    gdf = gd.DataFrame.from_pandas(df)
    dgf = dgd.from_cudf(gdf, npartitions=npartitions)

    groups = dgf.groupby(by=['x', 'y']).count()
    got = groups.compute().to_pandas()

    # Check against expectation
    expect = df.groupby(by=['x', 'y'], as_index=False).count()
    # Check keys
    np.testing.assert_array_equal(got.x, expect.x)
    np.testing.assert_array_equal(got.y, expect.y)
    # Check values
    np.testing.assert_array_equal(got.count_z, expect.z)


def check_groupby_agg(agg):
    np.random.seed(0)

    nelem = 100
    npartitions = 3
    xs = _gen_uniform_keys(nelem)
    df = pd.DataFrame({'x': xs,
                       'v1': np.random.normal(size=nelem),
                       'v2': np.random.normal(size=nelem)})

    gdf = gd.DataFrame.from_pandas(df)
    dgf = dgd.from_cudf(gdf, npartitions=npartitions)

    gotgroup = dgf.groupby(by='x')
    expgroup = df.groupby(by='x', as_index=False)

    def add_prefix(x):
        return '_'.join([agg, x])

    got = getattr(gotgroup, agg)().compute().to_pandas()
    exp = getattr(expgroup, agg)()
    np.testing.assert_array_almost_equal(getattr(got, add_prefix('v1')),
                                         exp.v1)
    np.testing.assert_array_almost_equal(getattr(got, add_prefix('v2')),
                                         exp.v2)


@pytest.mark.xfail(reason="pandas/cudf groupbys are not consistent")
@pytest.mark.parametrize('agg', ['count', 'sum', 'max', 'min'])
def test_groupby_agg(agg):
    check_groupby_agg(agg)


@pytest.mark.skip(reason="Not implemented by cudf yet")
@pytest.mark.parametrize('agg', ['mean', 'std'])
def test_groupby_harder_agg(agg):
    check_groupby_agg(agg)


@pytest.mark.skip(reason="Fix needed from \
    https://github.com/gpuopenanalytics/dask_cudf/pull/26")
@pytest.mark.skip(reason="Groupby apply not implemented in libgdf")
def test_groupby_apply_grouped():
    np.random.seed(0)

    nelem = 100
    xs = _gen_uniform_keys(nelem)
    ys = _gen_uniform_keys(nelem)
    df = pd.DataFrame({'x': xs,
                       'y': ys,
                       'idx': np.arange(nelem),
                       'v1': np.random.normal(size=nelem),
                       'v2': np.random.normal(size=nelem)})

    gdf = gd.DataFrame.from_pandas(df)
    dgf = dgd.from_cudf(gdf, npartitions=2)

    def transform(y, v1, v2, out1):
        for i in range(cuda.threadIdx.x, y.size, cuda.blockDim.x):
            out1[i] = y[i] * (v1[i] + v2[i])

    grouped = dgf.groupby(by=['x', 'y']).apply_grouped(
        transform,
        incols=['y', 'v1', 'v2'],
        outcols={'out1': np.float64},
        )

    # Compute with dask
    dgd_grouped = grouped.compute().to_pandas()
    binning = {}
    for _, row in dgd_grouped.iterrows():
        binning[row.idx] = row

    # Emulate the operation with pandas
    def emulate(df):
        df['out1'] = df.y * (df.v1 + df.v2)
        return df

    pd_groupby = df.groupby(
        by=['x', 'y'], sort=True, as_index=True
    ).apply(emulate)

    # Check the result
    for _, expect in pd_groupby.iterrows():
        got = binning[expect.idx]

        attrs = ['x', 'y', 'v1', 'v2', 'out1']
        for a in attrs:
            np.testing.assert_equal(getattr(got, a),
                                    getattr(expect, a))


@pytest.mark.skip(reason="Fix needed from \
    https://github.com/gpuopenanalytics/dask_cudf/pull/26")
@pytest.mark.skip(reason="Groupby apply not implemented in libgdf")
def test_groupby_apply():
    np.random.seed(0)

    nelem = 100
    xs = _gen_uniform_keys(nelem)
    ys = _gen_uniform_keys(nelem)
    df = pd.DataFrame({'x': xs,
                       'y': ys,
                       'idx': np.arange(nelem),
                       'v1': np.random.normal(size=nelem),
                       'v2': np.random.normal(size=nelem)})

    gdf = gd.DataFrame.from_pandas(df)
    dgf = dgd.from_cudf(gdf, npartitions=2)

    def transform(df):
        df['out1'] = df.y * (df.v1 + df.v2)
        return df

    grouped = dgf.groupby(by=['x', 'y']).apply(transform)

    # Compute with dask
    dgd_grouped = grouped.compute().to_pandas()
    binning = {}
    for _, row in dgd_grouped.iterrows():
        binning[row.idx] = row

    # Emulate the operation with pandas
    pd_groupby = df.groupby(
        by=['x', 'y'], sort=True, as_index=True
    ).apply(transform)

    # Check the result
    for _, expect in pd_groupby.iterrows():
        got = binning[expect.idx]

        attrs = ['x', 'y', 'v1', 'v2', 'out1']
        for a in attrs:
            np.testing.assert_equal(getattr(got, a),
                                    getattr(expect, a))


@pytest.mark.skip(reason="Fix needed from \
    https://github.com/gpuopenanalytics/dask_cudf/pull/26")
def test_repeated_groupby():
    np.random.seed(0)

    nelem = 100
    df = pd.DataFrame()
    df['a'] = _gen_uniform_keys(nelem)
    df['b'] = _gen_uniform_keys(nelem)

    ref_df = gd.DataFrame.from_pandas(df)
    df = dgd.from_cudf(ref_df, npartitions=3)
    assert df.known_divisions

    df2 = df.groupby('a').apply(lambda x: x)
    assert not df2.known_divisions

    got = df2.groupby('a').apply(lambda x: x).compute().to_pandas()
    expect = ref_df.groupby('a').apply(lambda x: x).to_pandas()

    def sort_content(df):
        return sorted(list(df.b))

    got = got.groupby('a').apply(sort_content)
    expect = expect.groupby('a').apply(sort_content)

    pd.util.testing.assert_series_equal(got, expect)


@pytest.mark.xfail(reason="pandas/cudf groupbys are not consistent")
@pytest.mark.parametrize('nelem', [50, 100, 1000])
@pytest.mark.parametrize('npart', [3, 4, 5, 10])
def test_groupby_tree_reduce_max(nelem, npart):
    np.random.seed(0)
    df = pd.DataFrame()
    df['a'] = _gen_uniform_keys(nelem)
    df['b'] = _gen_uniform_keys(nelem)

    ref_df = gd.DataFrame.from_pandas(df)
    dgf = dgd.from_cudf(ref_df, npartitions=npart)
    got = dgf.groupby('a').max().compute().to_pandas()
    expect = df.groupby('a', as_index=False).max()

    pd.util.testing.assert_series_equal(expect.a, got.a)
    pd.util.testing.assert_series_equal(
        expect.b,
        got.max_b,
        check_names=False,
        )


@pytest.mark.parametrize('nelem', [50, 100, 1000])
@pytest.mark.parametrize('npart', [3, 4, 5, 10])
def test_groupby_tree_reduce_multi_agg(nelem, npart):
    np.random.seed(0)
    df = pd.DataFrame()
    df['a'] = _gen_uniform_keys(nelem)
    df['b'] = _gen_uniform_keys(nelem)
    df['c'] = _gen_uniform_keys(nelem)

    ref_df = gd.DataFrame.from_pandas(df)
    dgf = dgd.from_cudf(ref_df, npartitions=npart)
    got = dgf.groupby('a').agg({'b': 'max', 'c': 'min'}).compute().to_pandas()
    expect = df.groupby('a', as_index=False).agg({'b': 'max', 'c': 'min'})

    pd.util.testing.assert_series_equal(expect.a, got.a)
    pd.util.testing.assert_series_equal(
        expect.b,
        got.max_b,
        check_names=False,
        )
    pd.util.testing.assert_series_equal(
        expect.c,
        got.min_c,
        check_names=False,
        )
