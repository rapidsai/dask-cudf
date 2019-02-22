from functools import partial

import numpy as np
import pytest

import cudf
import cudf as gd
import dask_cudf as dgd
import dask.dataframe as dd

param_nrows = [5, 10, 50, 100]


@pytest.mark.parametrize("left_nrows", param_nrows)
@pytest.mark.parametrize("right_nrows", param_nrows)
@pytest.mark.parametrize("left_nkeys", [4, 5])
@pytest.mark.parametrize("right_nkeys", [4, 5])
def test_join_inner(left_nrows, right_nrows, left_nkeys, right_nkeys):
    chunksize = 50

    np.random.seed(0)

    # cuDF
    left = gd.DataFrame(
        {
            "x": np.random.randint(0, left_nkeys, size=left_nrows),
            "a": np.arange(left_nrows),
        }.items()
    )
    right = gd.DataFrame(
        {
            "x": np.random.randint(0, right_nkeys, size=right_nrows),
            "a": 1000 * np.arange(right_nrows),
        }.items()
    )

    expect = left.set_index("x").join(
        right.set_index("x"), how="inner", sort=True, lsuffix="l", rsuffix="r"
    )
    expect = expect.to_pandas()

    # dask_cudf
    left = dgd.from_cudf(left, chunksize=chunksize)
    right = dgd.from_cudf(right, chunksize=chunksize)

    joined = left.set_index("x").join(
        right.set_index("x"), how="inner", lsuffix="l", rsuffix="r"
    )
    got = joined.compute().to_pandas()

    # Check index
    np.testing.assert_array_equal(expect.index.values, got.index.values)

    # Check rows in each groups
    expect_rows = {}
    got_rows = {}

    def gather(df, grows):
        grows[df["index"].values[0]] = (set(df.al), set(df.ar))

    expect.reset_index().groupby("index").apply(partial(gather, grows=expect_rows))

    expect.reset_index().groupby("index").apply(partial(gather, grows=got_rows))

    assert got_rows == expect_rows


@pytest.mark.parametrize("left_nrows", param_nrows)
@pytest.mark.parametrize("right_nrows", param_nrows)
@pytest.mark.parametrize("left_nkeys", [4, 5])
@pytest.mark.parametrize("right_nkeys", [4, 5])
@pytest.mark.parametrize("how", ["left", "right"])
def test_join_left(left_nrows, right_nrows, left_nkeys, right_nkeys, how):
    chunksize = 50

    np.random.seed(0)

    # cuDF
    left = gd.DataFrame(
        {
            "x": np.random.randint(0, left_nkeys, size=left_nrows),
            "a": np.arange(left_nrows, dtype=np.float64),
        }.items()
    )
    right = gd.DataFrame(
        {
            "x": np.random.randint(0, right_nkeys, size=right_nrows),
            "a": 1000 * np.arange(right_nrows, dtype=np.float64),
        }.items()
    )

    expect = left.set_index("x").join(
        right.set_index("x"), how=how, sort=True, lsuffix="l", rsuffix="r"
    )
    expect = expect.to_pandas()

    # dask_cudf
    left = dgd.from_cudf(left, chunksize=chunksize)
    right = dgd.from_cudf(right, chunksize=chunksize)

    joined = left.set_index("x").join(
        right.set_index("x"), how=how, lsuffix="l", rsuffix="r"
    )
    got = joined.compute().to_pandas()

    # Check index
    np.testing.assert_array_equal(expect.index.values, got.index.values)

    # Check rows in each groups
    expect_rows = {}
    got_rows = {}

    def gather(df, grows):
        cola = np.sort(np.asarray(df.al))
        colb = np.sort(np.asarray(df.ar))

        grows[df["index"].values[0]] = (cola, colb)

    expect.reset_index().groupby("index").apply(partial(gather, grows=expect_rows))

    expect.reset_index().groupby("index").apply(partial(gather, grows=got_rows))

    for k in expect_rows:
        np.testing.assert_array_equal(expect_rows[k][0], got_rows[k][0])
        np.testing.assert_array_equal(expect_rows[k][1], got_rows[k][1])


@pytest.mark.parametrize("left_nrows", param_nrows)
@pytest.mark.parametrize("right_nrows", param_nrows)
@pytest.mark.parametrize("left_nkeys", [4, 5])
@pytest.mark.parametrize("right_nkeys", [4, 5])
def test_merge_left(left_nrows, right_nrows, left_nkeys, right_nkeys, how="left"):
    chunksize = 3

    np.random.seed(0)

    # cuDF
    left = gd.DataFrame(
        {
            "x": np.random.randint(0, left_nkeys, size=left_nrows),
            "y": np.random.randint(0, left_nkeys, size=left_nrows),
            "a": np.arange(left_nrows, dtype=np.float64),
        }.items()
    )
    right = gd.DataFrame(
        {
            "x": np.random.randint(0, right_nkeys, size=right_nrows),
            "y": np.random.randint(0, right_nkeys, size=right_nrows),
            "a": 1000 * np.arange(right_nrows, dtype=np.float64),
        }.items()
    )

    expect = left.merge(right, on=("x", "y"), how=how)

    def normalize(df):
        return (
            df.to_pandas().sort_values(["x", "y", "a_x", "a_y"]).reset_index(drop=True)
        )

    # dask_cudf
    left = dgd.from_cudf(left, chunksize=chunksize)
    right = dgd.from_cudf(right, chunksize=chunksize)

    result = left.merge(right, on=("x", "y"), how=how).compute(
        scheduler="single-threaded"
    )

    dd.assert_eq(normalize(expect), normalize(result))


@pytest.mark.parametrize("left_nrows", [2, 5])
@pytest.mark.parametrize("right_nrows", [5, 10])
@pytest.mark.parametrize("left_nkeys", [4])
@pytest.mark.parametrize("right_nkeys", [4])
def test_merge_1col_left(left_nrows, right_nrows, left_nkeys, right_nkeys, how="left"):
    chunksize = 3

    np.random.seed(0)

    # cuDF
    left = gd.DataFrame(
        {
            "x": np.random.randint(0, left_nkeys, size=left_nrows),
            "a": np.arange(left_nrows, dtype=np.float64),
        }.items()
    )
    right = gd.DataFrame(
        {
            "x": np.random.randint(0, right_nkeys, size=right_nrows),
            "a": 1000 * np.arange(right_nrows, dtype=np.float64),
        }.items()
    )

    expect = left.merge(right, on=["x"], how=how)
    expect = expect.to_pandas().sort_values(["x", "a_x", "a_y"]).reset_index(drop=True)

    # dask_cudf
    left = dgd.from_cudf(left, chunksize=chunksize)
    right = dgd.from_cudf(right, chunksize=chunksize)

    joined = left.merge(right, on=["x"], how=how)

    got = joined.compute().to_pandas()

    got = got.sort_values(["x", "a_x", "a_y"]).reset_index(drop=True)

    dd.assert_eq(expect, got)


@pytest.mark.parametrize("how", ["left", "inner"])
def test_how(how):
    left = cudf.DataFrame({"x": [1, 2, 3, 4, None], "y": [1.0, 2.0, 3.0, 4.0, 0.0]})
    right = cudf.DataFrame({"x": [2, 3, None, 2], "y": [20, 30, 0, 20]})

    dleft = dd.from_pandas(left, npartitions=2)
    dright = dd.from_pandas(right, npartitions=3)

    expected = left.merge(right, how=how, on="x")
    result = dleft.merge(dright, how=how, on="x")

    dd.assert_eq(
        result.compute().to_pandas().sort_values("x"),
        expected.to_pandas().sort_values("x"),
        check_index=False,
    )
