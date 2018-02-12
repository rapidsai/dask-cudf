import pytest

import numpy as np
import pandas as pd

import pygdf
import dask_gdf as dgd


@pytest.mark.parametrize('by', ['a', 'b'])
@pytest.mark.parametrize('nelem', [10, 100, 1000])
@pytest.mark.parametrize('nparts', [1, 2, 5, 10])
def test_sort_values(nelem, nparts, by):
    df = pygdf.DataFrame()
    df['a'] = np.ascontiguousarray(np.arange(nelem)[::-1])
    df['b'] = np.arange(100, nelem + 100)
    ddf = dgd.from_pygdf(df, npartitions=nparts)

    got = ddf.sort_values(by=by).compute().to_pandas()
    expect = df.sort_values(by=by).to_pandas().reset_index(drop=True)
    pd.util.testing.assert_frame_equal(got, expect)

