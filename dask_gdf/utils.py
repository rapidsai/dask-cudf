import pandas as pd
import pygdf as gd
import dask.dataframe as dd


def make_meta(x):
    """Create an empty pygdf object containing the desired metadata.

    Parameters
    ----------
    x : dict, tuple, list, pd.Series, pd.DataFrame, pd.Index, dtype, scalar
        To create a DataFrame, provide a `dict` mapping of `{name: dtype}`, or
        an iterable of `(name, dtype)` tuples. To create a `Series`, provide a
        tuple of `(name, dtype)`. If a pygdf object, names, dtypes, and index
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
    if hasattr(x, '_meta'):
        return x._meta
    if isinstance(x, (gd.Series, gd.DataFrame, gd.index.Index)):
        out = x[:1]
        return out.copy() if hasattr(out, 'copy') else out

    meta = dd.utils.make_meta(x)

    if isinstance(meta, (pd.DataFrame, pd.Series, pd.Index)):
        meta2 = dd.utils.meta_nonempty(meta)
        if isinstance(meta2, pd.DataFrame):
            return gd.DataFrame.from_pandas(meta2.iloc[:1])
        elif isinstance(meta2, pd.Series):
            return gd.Series.from_any(meta2.iloc[:1])
        else:
            meta2 = meta2[:1]
            if isinstance(meta2, pd.RangeIndex):
                return gd.index.RangeIndex(meta2.start, meta2.stop)
            return gd.index.GenericIndex(meta2)

    return meta
