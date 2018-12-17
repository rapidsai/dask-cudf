import pandas as pd
import cudf
import dask.dataframe as dd
from dask.utils import asciitable


def make_meta(x):
    """Create an empty cudf object containing the desired metadata.

    Parameters
    ----------
    x : dict, tuple, list, pd.Series, pd.DataFrame, pd.Index, dtype, scalar
        To create a DataFrame, provide a `dict` mapping of `{name: dtype}`, or
        an iterable of `(name, dtype)` tuples. To create a `Series`, provide a
        tuple of `(name, dtype)`. If a cudf object, names, dtypes, and index
        should match the desired output. If a dtype or scalar, a scalar of the
        same dtype is returned.

    Examples
    --------
    >>> make_meta([('a', 'i8'), ('b', 'O')])
    Empty DataFrame
    Columns: [a, b]
    Index: []
    >>> make_meta(('a', 'f8'))
    Series([], Name: a, dtype: float64)
    >>> make_meta('i8')
    1
    """
    if hasattr(x, "_meta"):
        return x._meta
    if isinstance(x, (cudf.Series, cudf.DataFrame, cudf.Index)):
        out = x[:2]
        return out.copy() if hasattr(out, "copy") else out

    meta = dd.utils.make_meta(x)

    if isinstance(meta, (pd.DataFrame, pd.Series, pd.Index)):
        meta2 = dd.utils.meta_nonempty(meta)
        if isinstance(meta2, pd.DataFrame):
            return cudf.DataFrame.from_pandas(meta2)
        elif isinstance(meta2, pd.Series):
            return cudf.Series(meta2)
        else:
            if isinstance(meta2, pd.RangeIndex):
                return cudf.RangeIndex(meta2.start, meta2.stop)
            return cudf.dataframe.GenericIndex(meta2)

    return meta


def check_meta(x, meta, funcname=None):
    """Check that the dask metadata matches the result.
    If metadata matches, ``x`` is passed through unchanged. A nice error is
    raised if metadata doesn't match.

    Parameters
    ----------
    x : DataFrame, Series, or Index
    meta : DataFrame, Series, or Index
        The expected metadata that ``x`` should match
    funcname : str, optional
        The name of the function in which the metadata was specified. If
        provided, the function name will be included in the error message to be
        more helpful to users.
    """

    if not isinstance(meta, (cudf.Series, cudf.Index, cudf.DataFrame)):
        raise TypeError(
            "Expected partition to be DataFrame, Series, or "
            "Index of cudf, got `%s`" % type(meta).__name__
        )

    if type(x) != type(meta):
        errmsg = "Expected partition of type `%s` but got " "`%s`" % (
            type(meta).__name__,
            type(x).__name__,
        )
    elif isinstance(meta, cudf.DataFrame):

        extra_cols = set(x.columns) ^ set(meta.columns)
        if extra_cols:
            errmsg = "extra columns"
        else:
            bad = [
                (col, x[col].dtype, meta[col].dtype)
                for col in x.columns
                if not series_type_eq(x[col], meta[col])
            ]

            if not bad:
                return x
            errmsg = "Partition type: `%s`\n%s" % (
                type(meta).__name__,
                asciitable(["Column", "Found", "Expected"], bad),
            )
    else:
        if series_type_eq(x, meta):
            return x

        errmsg = "Partition type: `%s`\n%s" % (
            type(meta).__name__,
            asciitable(["", "dtype"], [("Found", x.dtype), ("Expected", meta.dtype)]),
        )

    raise ValueError(
        "Metadata mismatch found%s.\n\n"
        "%s" % ((" in `%s`" % funcname if funcname else ""), errmsg)
    )


def series_type_eq(a, b):
    """Are the two Series type equivalent?

    Parameters
    ----------
    a, b : cudf.Series

    Returns
    -------
    res : boolean
    """
    return a._column.is_type_equivalent(b._column)
