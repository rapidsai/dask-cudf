import os
from glob import glob
from warnings import warn

import pandas as pd

from dask.base import tokenize
from dask.compatibility import apply
import dask.dataframe as dd
from dask.utils import parse_bytes

import cudf
from libgdf_cffi import GDFError


def read_csv(path, chunksize="256 MiB", **kwargs):
    if isinstance(chunksize, str):
        chunksize = parse_bytes(chunksize)
    filenames = sorted(glob(str(path)))  # TODO: lots of complexity
    name = "read-csv-" + tokenize(
        path, tokenize, **kwargs
    )  # TODO: get last modified time

    compression = kwargs.get("compression", False)
    if compression:
        # compressed CSVs reading must read the entire file
        kwargs.pop("byte_range", None)
        warn(
            "Warning %s compression does not support breaking apart files\n"
            "Please ensure that each individual file can fit in memory and\n"
            "use the keyword ``chunksize=None to remove this message``\n"
            "Setting ``chunksize=(size of file)``" % compression
        )
        chunksize = None

    if chunksize is None:
        return read_csv_without_chunksize(path, **kwargs)

    meta = cudf.read_csv(filenames[0], **kwargs)
    dsk = {}
    i = 0
    dtypes = meta.dtypes.values

    for fn in filenames:
        size = os.path.getsize(fn)
        for start in range(0, size, chunksize):
            kwargs2 = kwargs.copy()
            kwargs2["byte_range"] = (
                start,
                chunksize,
            )  # specify which chunk of the file we care about
            if start != 0:
                kwargs2["names"] = meta.columns  # no header in the middle of the file
                kwargs2["header"] = None
            dsk[(name, i)] = (apply, _read_csv, [fn, dtypes], kwargs2)

            i += 1

    divisions = [None] * (len(dsk) + 1)
    return dd.core.new_dd_object(dsk, name, meta, divisions)


def _read_csv(fn, dtypes=None, **kwargs):
    try:
        cdf = cudf.read_csv(fn, **kwargs)
    except GDFError:
        # end of file check https://github.com/rapidsai/dask-cudf/issues/103
        # this should be removed when CUDF has better dtype/parse_date support
        dtypes = dict(zip(kwargs["names"], dtypes))
        df = dd.core.make_meta(dtypes)
        cdf = cudf.from_pandas(df)
    return cdf


def read_csv_without_chunksize(path, **kwargs):
    """Read entire CSV with optional compression (gzip/zip)

    Parameters
    ----------
    path : str
        path to files (support for glob)
    """
    filenames = sorted(glob(str(path)))
    name = "read-csv-" + tokenize(path, **kwargs)

    meta = cudf.read_csv(filenames[0], **kwargs)

    graph = {
        (name, i): (apply, cudf.read_csv, [fn], kwargs)
        for i, fn in enumerate(filenames)
    }

    divisions = [None] * (len(filenames) + 1)

    return dd.core.new_dd_object(graph, name, meta, divisions)
