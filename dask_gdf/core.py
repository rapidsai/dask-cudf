import operator
from uuid import uuid4
from math import ceil

import numpy as np
import pandas as pd
import pygdf as gd
from toolz import merge

from dask.base import Base, tokenize, normalize_token
from dask.context import _globals
from dask.core import flatten
from dask.optimize import cull, fuse
from dask.threaded import get as threaded_get

from .utils import make_meta


def optimize(dsk, keys, **kwargs):
    flatkeys = list(flatten(keys)) if isinstance(keys, list) else [keys]
    dsk, dependencies = cull(dsk, flatkeys)
    dsk, dependencies = fuse(dsk, keys, dependencies=dependencies,
                             ave_width=_globals.get('fuse_ave_width', 1))
    dsk, _ = cull(dsk, keys)
    return dsk


def finalize(results):
    return gd.concat(results)


class _Frame(Base):
    """ Superclass for DataFrame and Series

    Parameters
    ----------
    dsk : dict
        The dask graph to compute this DataFrame
    name : str
        The key prefix that specifies which keys in the dask comprise this
        particular DataFrame / Series
    meta : pygdf.DataFrame, pygdf.Series, or pygdf.Index
        An empty pygdf object with names, dtypes, and indices matching the
        expected output.
    divisions : tuple of index values
        Values along which we partition our blocks on the index
    """
    _default_get = staticmethod(threaded_get)
    _optimize = staticmethod(optimize)
    _finalize = staticmethod(finalize)

    def __init__(self, dsk, name, meta, divisions):
        self.dask = dsk
        self._name = name
        meta = make_meta(meta)
        if not isinstance(meta, self._partition_type):
            raise TypeError("Expected meta to specify type {0}, got type "
                            "{1}".format(self._partition_type.__name__,
                                         type(meta).__name__))
        self._meta = meta
        self.divisions = tuple(divisions)

    def __repr__(self):
        s = "<dask_gdf.%s | %d tasks | %d npartitions>"
        return s % (type(self).__name__, len(self.dask), self.npartitions)

    @property
    def npartitions(self):
        """Return number of partitions"""
        return len(self.divisions) - 1

    @property
    def index(self):
        """Return dask Index instance"""
        name = self._name + '-index'
        dsk = {(name, i): (getattr, key, 'index')
               for i, key in enumerate(self._keys())}
        return Index(merge(dsk, self.dask), name,
                     self._meta.index, self.divisions)

    def _keys(self):
        return [(self._name, i) for i in range(self.npartitions)]


normalize_token.register(_Frame, lambda a: a._name)


class DataFrame(_Frame):
    _partition_type = gd.DataFrame

    @property
    def columns(self):
        return self._meta.columns

    @property
    def dtypes(self):
        return self._meta.dtypes

    def __dir__(self):
        o = set(dir(type(self)))
        o.update(self.__dict__)
        o.update(c for c in self.columns if
                 (isinstance(c, pd.compat.string_types) and
                  pd.compat.isidentifier(c)))
        return list(o)

    def __getattr__(self, key):
        if key in self.columns:
            return self[key]
        raise AttributeError("'DataFrame' object has no attribute %r" % key)

    def __getitem__(self, key):
        if isinstance(key, str) and key in self.columns:
            meta = self._meta[key]
            name = 'getitem-%s' % tokenize(self, key)
            dsk = {(name, i): (operator.getitem, (self._name, i), key)
                   for i in range(self.npartitions)}
            return Series(merge(self.dask, dsk), name, meta, self.divisions)

        raise NotImplementedError("Indexing with %r" % key)


class Series(_Frame):
    _partition_type = gd.Series

    @property
    def dtype(self):
        return self._meta.dtype


class Index(Series):
    _partition_type = gd.index.Index

    @property
    def index(self):
        raise AttributeError("'Index' object has no attribute 'index'")


def splits_divisions_sorted_pygdf(df, chunksize):
    segments = df.index.find_segments()
    segments.append(len(df) - 1)

    splits = [0]
    last = current_size = 0
    for s in segments:
        size = s - last
        last = s
        current_size += size
        if current_size >= chunksize:
            splits.append(s)
            current_size = 0
    # Ensure end is included
    if splits[-1] != segments[-1]:
        splits.append(segments[-1])
    divisions = tuple(df.index.take(np.array(splits)).values)
    splits[-1] += 1  # Offset to extract to end

    return splits, divisions


def from_pygdf(data, npartitions=None, chunksize=None, sort=True, name=None):
    """Create a dask_gdf from a pygdf object

    Parameters
    ----------
    data : pygdf.DataFrame or pygdf.Series
    npartitions : int, optional
        The number of partitions of the index to create. Note that depending on
        the size and index of the dataframe, the output may have fewer
        partitions than requested.
    chunksize : int, optional
        The number of rows per index partition to use.
    sort : bool
        Sort input first to obtain cleanly divided partitions or don't sort and
        don't get cleanly divided partitions
    name : string, optional
        An optional keyname for the dataframe. Defaults to a uuid.

    Returns
    -------
    dask_gdf.DataFrame or dask_gdf.Series
        A dask_gdf DataFrame/Series partitioned along the index
    """
    if not isinstance(data, (gd.Series, gd.DataFrame)):
        raise TypeError("Input must be a pygdf DataFrame or Series")

    if ((npartitions is None) == (chunksize is None)):
        raise ValueError('Exactly one of npartitions and chunksize must '
                         'be specified.')

    nrows = len(data)

    if chunksize is None:
        chunksize = int(ceil(nrows / npartitions))

    name = name or ('from_pygdf-' + uuid4().hex)

    if sort:
        data = data.sort_index(ascending=True)
        splits, divisions = splits_divisions_sorted_pygdf(data, chunksize)
    else:
        splits = list(range(0, nrows, chunksize)) + [len(data)]
        divisions = (None,) * len(splits)

    dsk = {(name, i): data[start:stop]
           for i, (start, stop) in enumerate(zip(splits[:-1], splits[1:]))}

    if isinstance(data, gd.Series):
        return Series(dsk, name, data, divisions)
    return DataFrame(dsk, name, data, divisions)
