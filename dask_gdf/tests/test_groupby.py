import numpy as np
import pandas as pd
import pytest

import pygdf as gd
import dask_gdf as dgd


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
    dgf = dgd.from_pygdf(gdf, npartitions=npartitions)

    groups = dgf.groupby(by=['x']).count()
    got = groups.compute().to_pandas()

    # Check against expectation
    expect = df.groupby(by=['x'], as_index=False).count()
    # Check keys
    np.testing.assert_array_equal(got.x, expect.x)
    # Check values
    np.testing.assert_array_equal(got.z, expect.z)


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
    dgf = dgd.from_pygdf(gdf, npartitions=npartitions)

    groups = dgf.groupby(by=['x', 'y']).count()
    got = groups.compute().to_pandas()

    # Check against expectation
    expect = df.groupby(by=['x', 'y'], as_index=False).count()
    # Check keys
    np.testing.assert_array_equal(got.x, expect.x)
    np.testing.assert_array_equal(got.y, expect.y)
    # Check values
    np.testing.assert_array_equal(got.z, expect.z)


@pytest.mark.parametrize('agg', ['mean', 'count', 'max', 'min', 'std'])
def test_groupby_agg(agg):
    np.random.seed(0)

    nelem = 100
    npartitions = 3
    xs = _gen_uniform_keys(nelem)
    df = pd.DataFrame({'x': xs,
                       'v1': np.random.normal(size=nelem),
                       'v2': np.random.normal(size=nelem)})

    gdf = gd.DataFrame.from_pandas(df)
    dgf = dgd.from_pygdf(gdf, npartitions=npartitions)

    gotgroup = dgf.groupby(by='x')
    expgroup = df.groupby(by='x', as_index=False)

    got = getattr(gotgroup, agg)().compute().to_pandas()
    exp = getattr(expgroup, agg)()

    np.testing.assert_array_almost_equal(got.v1, exp.v1)
    np.testing.assert_array_almost_equal(got.v2, exp.v2)