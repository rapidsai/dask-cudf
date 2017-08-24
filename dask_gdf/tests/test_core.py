import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal

import pygdf as gd
import dask_gdf as dgd


def test_from_pygdf():
    df = pd.DataFrame({'x': np.random.randint(0, 5, size=10000),
                       'y': np.random.normal(size=10000)})

    gdf = gd.DataFrame.from_pandas(df)

    # Test simple around to/from dask
    ingested = dgd.from_pygdf(gdf, npartitions=2)
    assert_frame_equal(ingested.compute().to_pandas(), df)

    # Test conversion to dask.dataframe
    ddf = ingested.to_dask_dataframe()
    assert_frame_equal(ddf.compute(), df)
