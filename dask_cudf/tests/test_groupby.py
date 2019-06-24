import dask.dataframe as dd
import dask_cudf
import pandas as pd
import cudf
import numpy as np

import pytest


@pytest.mark.parametrize(
    "func,check_dtype",
    [
        (lambda df: df.groupby("x").sum(), True),
        (lambda df: df.groupby("x").mean(), True),
        (lambda df: df.groupby("x").count(), False),
        (lambda df: df.groupby("x").min(), True),
        (lambda df: df.groupby("x").max(), True),
        (lambda df: df.groupby("x").y.sum(), True),
        (lambda df: df.groupby("x").agg({"y": "max"}), True),
        pytest.param(
            lambda df: df.groupby("x").y.agg(["sum", "max"]),
            True,
            marks=pytest.mark.skip
        )
    ],
)
def test_groupby(func, check_dtype):
    pdf = pd.DataFrame(
        {"x": np.random.randint(0, 5, size=10000), "y": np.random.normal(size=10000)}
    )

    gdf = cudf.DataFrame.from_pandas(pdf)

    ddf = dask_cudf.from_cudf(gdf, npartitions=5)

    a = pd.DataFrame(func(gdf).to_pandas())
    b = pd.DataFrame(func(ddf).compute().to_pandas())

    a.index.name = None
    a.name = None
    b.index.name = None
    b.name = None

    dd.assert_eq(
        a,
        b,
        check_dtype=check_dtype
    )


@pytest.mark.xfail(reason="cudf issues")
@pytest.mark.parametrize(
    "func", [lambda df: df.groupby("x").std(), lambda df: df.groupby("x").y.std()]
)
def test_groupby_std(func):
    pdf = pd.DataFrame(
        {"x": np.random.randint(0, 5, size=10000), "y": np.random.normal(size=10000)}
    )

    gdf = cudf.DataFrame.from_pandas(pdf)

    ddf = dask_cudf.from_cudf(gdf, npartitions=5)

    a = func(gdf.to_pandas())
    b = func(ddf).compute().to_pandas()

    a.index.name = None
    a.name = None
    b.index.name = None

    dd.assert_eq(a, b)


# reason gotattr in cudf
@pytest.mark.parametrize(
    "func",
    [
        lambda df: df.groupby(["a", "b"]).x.sum(),
        pytest.param(
            lambda df: df.groupby(["a", "b"]).sum(), marks=pytest.mark.xfail
        ),
        pytest.param(
            lambda df: df.groupby(["a", "b"]).agg({'x', "sum"}), marks=pytest.mark.xfail
        )
    ],
)
def test_groupby_multi_column(func):
    pdf = pd.DataFrame(
        {
            "a": np.random.randint(0, 20, size=1000),
            "b": np.random.randint(0, 5, size=1000),
            "x": np.random.normal(size=1000),
        }
    )

    gdf = cudf.DataFrame.from_pandas(pdf)

    ddf = dask_cudf.from_cudf(gdf, npartitions=5)

    a = pd.DataFrame(func(gdf).to_pandas())
    b = pd.DataFrame(func(ddf).compute().to_pandas())

    dd.assert_eq(a, b)
