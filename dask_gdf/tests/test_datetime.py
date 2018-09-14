import pytest 
import numpy as np 
from pandas.util.testing import assert_series_equal
from pygdf.dataframe import Series
import dask_gdf as dgd 
import pandas as pd 


def data():
    return pd.date_range('20010101', '20020215', freq='400h')


fields = ['year', 'month', 'day', 'hour', 'minute', 'second']


@pytest.mark.parametrize('data', [data()])
def test_series(data):
    pd_data = pd.Series(data.copy())
    gdf_data = Series(pd_data)
    dask_gdf_data = dgd.from_pygdf(gdf_data, npartitions=5)
    np.testing.assert_equal(
        np.array(pd_data), 
        np.array(dask_gdf_data.compute()),
        )

@pytest.mark.parametrize('data', [data()])
@pytest.mark.parametrize('field', fields)
def test_dt_series(data, field):
    pd_data = pd.Series(data.copy())
    gdf_data = Series(pd_data)
    dask_gdf_data = dgd.from_pygdf(gdf_data, npartitions=5)
    base = getattr(pd_data.dt, field)
    test = getattr(dask_gdf_data.dt, field).to_dask_dataframe().compute().astype('int64')
    assert_series_equal(base, test)
