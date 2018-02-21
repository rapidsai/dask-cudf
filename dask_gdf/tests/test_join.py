import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal

import pytest

import pygdf as gd
import dask_gdf as dgd
import dask.dataframe as dd


param_nrows = [5, 10, 100, 400]

@pytest.mark.parametrize('left_nrows', param_nrows)
@pytest.mark.parametrize('right_nrows', param_nrows)
def test_join_inner(left_nrows, right_nrows):
    chunksize = 50

    np.random.seed(0)

    left = gd.DataFrame({'x': np.random.randint(0, 5, size=left_nrows),
                         'a': np.arange(left_nrows)}.items())
    right = gd.DataFrame({'x': np.random.randint(0, 5, size=right_nrows),
                          'b': np.arange(right_nrows)}.items())

    expect = left.set_index('x').join(right.set_index('x'), how='inner', sort=True)
    expect = expect.to_pandas()

    left = dgd.from_pygdf(left, chunksize=chunksize)
    right = dgd.from_pygdf(right, chunksize=chunksize)

    joined = left.set_index('x').join(right.set_index('x'), how='inner')
    got = joined.compute().to_pandas()

    np.testing.assert_array_equal(expect.index.values,
                                  got.index.values)
