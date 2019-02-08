from dask.dataframe.methods import concat_dispatch
from dask.dataframe.core import get_parallel_type, meta_nonempty, make_meta
import cudf

from .core import DataFrame, Series, Index


get_parallel_type.register(cudf.DataFrame, lambda _: DataFrame)
get_parallel_type.register(cudf.Series, lambda _: Series)
get_parallel_type.register(cudf.Index, lambda _: Index)


@meta_nonempty.register((cudf.DataFrame, cudf.Series, cudf.Index))
def _(x):
    y = meta_nonempty(x.to_pandas())  # TODO: add iloc[:5]
    return cudf.from_pandas(y)


@make_meta.register((cudf.Series, cudf.DataFrame))
def _(x):
    return x.head(0)


@make_meta.register(cudf.Index)
def _(x):
    return x[:0]


@concat_dispatch.register((cudf.DataFrame, cudf.Series, cudf.Index))
def _(dfs, axis=0, join="outer", uniform=False, filter_warning=True):
    assert axis == 0
    assert join == "outer"
    assert filter_warning is True
    return cudf.concat(dfs)
