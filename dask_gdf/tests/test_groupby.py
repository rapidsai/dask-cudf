import operator

import numpy as np
import pandas as pd
import pytest

import pygdf as gd
import dask_gdf as dgd


def _gen_skewed_keys(nelem):
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


@pytest.mark.parametrize('keygen', [_gen_skewed_keys])
def test_groupby_skewed(keygen):
    np.random.seed(0)

    nelem = 3000
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
    expect = df.groupby(by=['x']).count()
    np.testing.assert_array_equal(got.z, expect.z)

