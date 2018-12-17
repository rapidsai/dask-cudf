import pytest

import numpy as np
import pandas as pd

import cudf as gd
import dask
import dask_cudf as dgd


@pytest.mark.parametrize("by", ["a", "b"])
@pytest.mark.parametrize("nelem", [10, 100, 1000])
@pytest.mark.parametrize("nparts", [1, 2, 5, 10])
def test_sort_values(nelem, nparts, by):
    df = gd.DataFrame()
    df["a"] = np.ascontiguousarray(np.arange(nelem)[::-1])
    df["b"] = np.arange(100, nelem + 100)
    ddf = dgd.from_cudf(df, npartitions=nparts)

    with dask.config.set(scheduler="single-threaded"):
        got = ddf.sort_values(by=by).compute().to_pandas()
    expect = df.sort_values(by=by).to_pandas().reset_index(drop=True)
    pd.util.testing.assert_frame_equal(got, expect)


def test_sort_values_binned():
    np.random.seed(43)
    nelem = 100
    nparts = 5
    by = "a"
    df = gd.DataFrame()
    df["a"] = np.random.randint(1, 5, nelem)
    ddf = dgd.from_cudf(df, npartitions=nparts)

    parts = ddf.sort_values_binned(by=by).to_delayed()
    part_uniques = []
    for i, p in enumerate(parts):
        part = dask.compute(p)[0]
        part_uniques.append(set(part.a.unique()))

    # Partitions do not have intersecting keys
    for i in range(len(part_uniques)):
        for j in range(i + 1, len(part_uniques)):
            assert not (
                part_uniques[i] & part_uniques[j]
            ), "should have empty intersection"
