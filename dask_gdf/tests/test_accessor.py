import pytest 
import numpy as np 
from pandas.util.testing import assert_series_equal
from pygdf.dataframe import Series
import dask_gdf as dgd 
import pandas as pd 


def data():
    return pd.date_range('20010101', '20020215', freq='400h')


def data1():
    return np.random.randn(100)


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

@pytest.mark.parametrize('data', [data1()])
@pytest.mark.xfail(raises=AttributeError)
def test_datetime_accessor_initialization(data):
    pd_data = pd.Series(data.copy())
    gdf_data = Series(pd_data)
    dask_gdf_data = dgd.from_dask_dataframe(gdf_data, npartitions=5)
    dask_gdf_data.dt 