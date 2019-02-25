import dask
import dask_cudf
import dask.dataframe as dd
import pandas as pd
import numpy as np

import pytest

def test_read_csv(tmp_path):
    df = dask.datasets.timeseries(dtypes={"x": int, "y": int}, freq="120s").reset_index(
        drop=True
    )
    df.to_csv(tmp_path / "data-*.csv", index=False)

    df2 = dask_cudf.read_csv(tmp_path / "*.csv")
    dd.assert_eq(df, df2)


def test_read_csv_w_bytes(tmp_path):
    df = dask.datasets.timeseries(dtypes={"x": int, "y": int}, freq="120s").reset_index(
        drop=True
    )
    df = pd.DataFrame(dict(x=np.arange(20), y=np.arange(20)))
    df = dd.from_pandas(df, npartitions=1)
    df.to_csv(tmp_path / "data-*.csv", index=False)

    df2 = dask_cudf.read_csv(tmp_path / "*.csv", chunksize="50 B")

    result = df2.compute().to_pandas()
    expected = df.compute()
    assert df2.npartitions is 3
    dd.assert_eq(result, expected, check_index=False)

def test_read_csv_compression(tmp_path):
    df = dask.datasets.timeseries(dtypes={"x": int, "y": int}, freq="120s").reset_index(
        drop=True
    )
    df = pd.DataFrame(dict(x=np.arange(20), y=np.arange(20)))
    ddf = dd.from_pandas(df, npartitions=1)
    df.to_csv(tmp_path / "data.csv.gz", index=False, compression='gzip')

    with pytest.warns(UserWarning) as w:
        df2 = dask_cudf.read_csv(tmp_path / "*.csv.gz", chunksize="50 B", compression='gzip')

    assert len(w) == 1
    msg = str(w[0].message)
    assert 'gzip' in msg

    df2 = dask_cudf.read_csv(tmp_path / "*.csv.gz", chunksize=None, compression='gzip')
    assert df2.npartitions is 1

    result = df2.compute().to_pandas()
    expected = ddf.compute()
    dd.assert_eq(result, expected, check_index=False)