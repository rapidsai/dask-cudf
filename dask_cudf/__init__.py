from .core import (
    DataFrame,
    Series,
    from_cudf,
    from_dask_dataframe,
    concat,
    from_delayed,
)
from .io import read_csv

from cudf._version import get_versions

__version__ = get_versions()["version"]
del get_versions

__all__ = [
    "DataFrame",
    "Series",
    "from_cudf",
    "from_dask_dataframe",
    "concat",
    "from_delayed",
]
