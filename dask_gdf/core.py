import operator
from uuid import uuid4
from math import ceil

import numpy as np
import pandas as pd
import pygdf as gd
from libgdf_cffi import libgdf
from toolz import merge, partition_all

from dask.base import Base, tokenize, normalize_token
from dask.context import _globals
from dask.core import flatten
from dask.compatibility import apply
from dask.optimize import cull, fuse
from dask.threaded import get as threaded_get
from dask.utils import funcname, M
from dask.dataframe.utils import raise_on_meta_error
from dask.dataframe.core import Scalar

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

    def _keys(self):
        return [(self._name, i) for i in range(self.npartitions)]

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

    @classmethod
    def _get_unary_operator(cls, op):
        return lambda self: map_partitions(op, self)

    @classmethod
    def _get_binary_operator(cls, op, inv=False):
        if inv:
            return lambda self, other: map_partitions(op, other, self)
        else:
            return lambda self, other: map_partitions(op, self, other)

    def __len__(self):
        return reduction(self, len, np.sum, meta=int,
                         split_every=False).compute()

    def map_partitions(self, func, *args, **kwargs):
        """ Apply Python function on each DataFrame partition.

        Note that the index and divisions are assumed to remain unchanged.

        Parameters
        ----------
        func : function
            Function applied to each partition.
        args, kwargs :
            Arguments and keywords to pass to the function. The partition will
            be the first argument, and these will be passed *after*.
        """
        return map_partitions(func, self, *args, **kwargs)

    def head(self, n=5, npartitions=1, compute=True):
        """ First n rows of the dataset

        Parameters
        ----------
        n : int, optional
            The number of rows to return. Default is 5.
        npartitions : int, optional
            Elements are only taken from the first ``npartitions``, with a
            default of 1. If there are fewer than ``n`` rows in the first
            ``npartitions`` a warning will be raised and any found rows
            returned. Pass -1 to use all partitions.
        compute : bool, optional
            Whether to compute the result, default is True.
        """
        if npartitions <= -1:
            npartitions = self.npartitions
        if npartitions > self.npartitions:
            raise ValueError("only %d partitions, received "
                             "%d" % (self.npartitions, npartitions))

        name = 'head-%d-%d-%s' % (npartitions, n, self._name)

        if npartitions > 1:
            name_p = 'head-partial-%d-%s' % (n, self._name)
            dsk = {(name_p, i): (M.head, (self._name, i), n)
                   for i in range(npartitions)}
            dsk[(name, 0)] = (M.head, (gd.concat, sorted(dsk)), n)
        else:
            dsk = {(name, 0): (M.head, (self._name, 0), n)}

        res = new_dd_object(merge(self.dask, dsk), name, self._meta,
                            (self.divisions[0], self.divisions[npartitions]))

        return res.compute() if compute else res


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


def sum_of_squares(x):
    x = x.astype('f8')
    return gd._gdf.apply_reduce(libgdf.gdf_sum_squared_generic, x)


def var_aggregate(x2, x, n, ddof=1):
    try:
        result = (x2 / n) - (x / n)**2
        if ddof != 0:
            result = result * n / (n - ddof)
        return result
    except ZeroDivisionError:
        return np.float64(np.nan)


def nlargest_agg(x, **kwargs):
    return gd.concat(x).nlargest(**kwargs)


def nsmallest_agg(x, **kwargs):
    return gd.concat(x).nsmallest(**kwargs)


def unique_k_agg(x, **kwargs):
    return gd.concat(x).unique_k(**kwargs)


class Series(_Frame):
    _partition_type = gd.Series

    @property
    def dtype(self):
        return self._meta.dtype

    def astype(self, dtype):
        if dtype == self.dtype:
            return self
        return self.map_partitions(M.astype, dtype=dtype)

    def sum(self, split_every=False):
        return reduction(self, chunk=M.sum, aggregate=np.sum,
                         split_every=split_every, meta=self.dtype)

    def count(self, split_every=False):
        return reduction(self, chunk=M.count, aggregate=np.sum,
                         split_every=split_every, meta='i8')

    def mean(self, split_every=False):
        sum = self.sum(split_every=split_every)
        n = self.count(split_every=split_every)
        return sum / n

    def var(self, ddof=1, split_every=False):
        sum2 = reduction(self, chunk=sum_of_squares, aggregate=np.sum,
                         split_every=split_every, meta='f8')
        sum = self.sum(split_every=split_every)
        n = self.count(split_every=split_every)
        return map_partitions(var_aggregate, sum2, sum, n, ddof=ddof,
                              meta='f8')

    def std(self, ddof=1, split_every=False):
        var = self.var(ddof=ddof, split_every=split_every)
        return map_partitions(np.sqrt, var, dtype=np.float64)

    def min(self, split_every=False):
        return reduction(self, chunk=M.min, aggregate=np.min,
                         split_every=split_every, meta=self.dtype)

    def max(self, split_every=False):
        return reduction(self, chunk=M.max, aggregate=np.max,
                         split_every=split_every, meta=self.dtype)

    def ceil(self):
        return self.map_partitions(M.ceil)

    def floor(self):
        return self.map_partitions(M.floor)

    def nlargest(self, n=5, split_every=None):
        return reduction(self, chunk=M.nlargest, aggregate=nlargest_agg,
                         meta=self._meta, token='series-nlargest',
                         split_every=split_every, n=n)

    def nsmallest(self, n=5, split_every=None):
        return reduction(self, chunk=M.nsmallest, aggregate=nsmallest_agg,
                         meta=self._meta, token='series-nsmallest',
                         split_every=split_every, n=n)

    def unique_k(self, k, split_every=None):
        return reduction(self, chunk=M.unique_k, aggregate=unique_k_agg,
                         meta=self._meta, token='unique-k',
                         split_every=split_every, k=k)


for op in [operator.abs, operator.add, operator.eq, operator.gt, operator.ge,
           operator.lt, operator.le, operator.mod, operator.mul, operator.ne,
           operator.sub, operator.truediv, operator.floordiv]:
    Series._bind_operator(op)


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

    return new_dd_object(dsk, name, data, divisions)


def _get_return_type(meta):
    if isinstance(meta, gd.Series):
        return Series
    elif isinstance(meta, gd.DataFrame):
        return DataFrame
    elif isinstance(meta, gd.index.Index):
        return Index
    return Scalar


def new_dd_object(dsk, name, meta, divisions):
    return _get_return_type(meta)(dsk, name, meta, divisions)


def _extract_meta(x):
    """
    Extract internal cache data (``_meta``) from dask_gdf objects
    """
    if isinstance(x, (Scalar, _Frame)):
        return x._meta
    elif isinstance(x, list):
        return [_extract_meta(_x) for _x in x]
    elif isinstance(x, tuple):
        return tuple([_extract_meta(_x) for _x in x])
    elif isinstance(x, dict):
        return {k: _extract_meta(v) for k, v in x.items()}
    return x


def _emulate(func, *args, **kwargs):
    """
    Apply a function using args / kwargs. If arguments contain dd.DataFrame /
    dd.Series, using internal cache (``_meta``) for calculation
    """
    with raise_on_meta_error(funcname(func)):
        return func(*_extract_meta(args), **_extract_meta(kwargs))


def align_partitions(args):
    """Align partitions between dask_gdf objects.

    Note that if all divisions are unknown, but have equal npartitions, then
    they will be passed through unchanged."""
    dfs = [df for df in args if isinstance(df, _Frame)]
    if not dfs:
        return args

    divisions = dfs[0].divisions
    if not all(df.divisions == divisions for df in dfs):
        raise NotImplementedError("Aligning mismatched partitions")
    return args


def map_partitions(func, *args, **kwargs):
    """ Apply Python function on each DataFrame partition.

    Parameters
    ----------
    func : function
        Function applied to each partition.
    args, kwargs :
        Arguments and keywords to pass to the function. At least one of the
        args should be a dask_gdf object.
    """
    meta = kwargs.pop('meta', None)
    if meta is not None:
        meta = make_meta(meta)

    if 'token' in kwargs:
        name = kwargs.pop('token')
        token = tokenize(meta, *args, **kwargs)
    else:
        name = funcname(func)
        token = tokenize(func, meta, *args, **kwargs)
    name = '{0}-{1}'.format(name, token)

    args = align_partitions(args)

    if meta is None:
        meta = _emulate(func, *args, **kwargs)
    meta = make_meta(meta)

    if all(isinstance(arg, Scalar) for arg in args):
        dask = {(name, 0):
                (apply, func, (tuple, [(x._name, 0) for x in args]), kwargs)}
        return Scalar(merge(dask, *[x.dask for x in args]), name, meta)

    dfs = [df for df in args if isinstance(df, _Frame)]
    dsk = {}
    for i in range(dfs[0].npartitions):
        values = [(x._name, i if isinstance(x, _Frame) else 0)
                  if isinstance(x, (_Frame, Scalar)) else x for x in args]
        dsk[(name, i)] = (apply, func, values, kwargs)

    dasks = [arg.dask for arg in args if isinstance(arg, (_Frame, Scalar))]
    return new_dd_object(merge(dsk, *dasks), name, meta, args[0].divisions)


def reduction(args, chunk=None, aggregate=None, combine=None,
              meta=None, token=None, chunk_kwargs=None,
              aggregate_kwargs=None, combine_kwargs=None,
              split_every=None, **kwargs):
    """Generic tree reduction operation.

    Parameters
    ----------
    args :
        Positional arguments for the `chunk` function. All `dask.dataframe`
        objects should be partitioned and indexed equivalently.
    chunk : function [block-per-arg] -> block
        Function to operate on each block of data
    aggregate : function list-of-blocks -> block
        Function to operate on the list of results of chunk
    combine : function list-of-blocks -> block, optional
        Function to operate on intermediate lists of results of chunk
        in a tree-reduction. If not provided, defaults to aggregate.
    $META
    token : str, optional
        The name to use for the output keys.
    chunk_kwargs : dict, optional
        Keywords for the chunk function only.
    aggregate_kwargs : dict, optional
        Keywords for the aggregate function only.
    combine_kwargs : dict, optional
        Keywords for the combine function only.
    split_every : int, optional
        Group partitions into groups of this size while performing a
        tree-reduction. If set to False, no tree-reduction will be used,
        and all intermediates will be concatenated and passed to ``aggregate``.
        Default is 8.
    kwargs :
        All remaining keywords will be passed to ``chunk``, ``aggregate``, and
        ``combine``.
    """
    if chunk_kwargs is None:
        chunk_kwargs = dict()
    if aggregate_kwargs is None:
        aggregate_kwargs = dict()
    chunk_kwargs.update(kwargs)
    aggregate_kwargs.update(kwargs)

    if combine is None:
        if combine_kwargs:
            raise ValueError("`combine_kwargs` provided with no `combine`")
        combine = aggregate
        combine_kwargs = aggregate_kwargs
    else:
        if combine_kwargs is None:
            combine_kwargs = dict()
        combine_kwargs.update(kwargs)

    if not isinstance(args, (tuple, list)):
        args = [args]

    npartitions = set(arg.npartitions for arg in args
                      if isinstance(arg, _Frame))
    if len(npartitions) > 1:
        raise ValueError("All arguments must have same number of partitions")
    npartitions = npartitions.pop()

    if split_every is None:
        split_every = 8
    elif split_every is False:
        split_every = npartitions
    elif split_every < 2 or not isinstance(split_every, int):
        raise ValueError("split_every must be an integer >= 2")

    token_key = tokenize(token or (chunk, aggregate), meta, args,
                         chunk_kwargs, aggregate_kwargs, combine_kwargs,
                         split_every)

    # Chunk
    a = '{0}-chunk-{1}'.format(token or funcname(chunk), token_key)
    if len(args) == 1 and isinstance(args[0], _Frame) and not chunk_kwargs:
        dsk = {(a, 0, i): (chunk, key)
               for i, key in enumerate(args[0]._keys())}
    else:
        dsk = {(a, 0, i): (apply, chunk,
                           [(x._name, i) if isinstance(x, _Frame)
                            else x for x in args], chunk_kwargs)
               for i in range(args[0].npartitions)}

    # Combine
    b = '{0}-combine-{1}'.format(token or funcname(combine), token_key)
    k = npartitions
    depth = 0
    while k > split_every:
        for part_i, inds in enumerate(partition_all(split_every, range(k))):
            conc = (list, [(a, depth, i) for i in inds])
            dsk[(b, depth + 1, part_i)] = (
                    (apply, combine, [conc], combine_kwargs)
                    if combine_kwargs else (combine, conc))
        k = part_i + 1
        a = b
        depth += 1

    # Aggregate
    b = '{0}-agg-{1}'.format(token or funcname(aggregate), token_key)
    conc = (list, [(a, depth, i) for i in range(k)])
    if aggregate_kwargs:
        dsk[(b, 0)] = (apply, aggregate, [conc], aggregate_kwargs)
    else:
        dsk[(b, 0)] = (aggregate, conc)

    if meta is None:
        meta_chunk = _emulate(apply, chunk, args, chunk_kwargs)
        meta = _emulate(apply, aggregate, [[meta_chunk]],
                        aggregate_kwargs)
    meta = make_meta(meta)

    for arg in args:
        if isinstance(arg, _Frame):
            dsk.update(arg.dask)

    return new_dd_object(dsk, b, meta, (None, None))
