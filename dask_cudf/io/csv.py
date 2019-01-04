from glob import glob

import cudf
from dask.base import tokenize
from dask.compatibility import apply
import dask.dataframe as dd


def read_csv(path, **kwargs):
    filenames = sorted(glob(str(path)))  # TODO: lots of complexity
    name = "read-csv-" + tokenize(path, **kwargs)  # TODO: get last modified time

    meta = cudf.read_csv(filenames[0], **kwargs)

    graph = {
        (name, i): (apply, cudf.read_csv, [fn], kwargs)
        for i, fn in enumerate(filenames)
    }

    divisions = [None] * (len(filenames) + 1)

    return dd.core.new_dd_object(graph, name, meta, divisions)
