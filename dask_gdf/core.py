import operator
from uuid import uuid4
from math import ceil
from collections import OrderedDict
from functools import reduce

import numpy as np
import pandas as pd
import pygdf as gd
from libgdf_cffi import libgdf
from toolz import merge, partition_all, merge_with

import dask.dataframe as dd
from dask.base import tokenize, normalize_token, DaskMethodsMixin
from dask.context import _globals
from dask.core import flatten
from dask.compatibility import apply
from dask.optimization import cull, fuse
from dask.threaded import get as threaded_get
from dask.utils import funcname, M, OperatorMethodMixin
from dask.dataframe.utils import raise_on_meta_error
from dask.dataframe.core import Scalar
from dask.delayed import delayed
from dask import compute

from .utils import make_meta, check_meta
from . import batcher_sortnet


def optimize(dsk, keys, **kwargs):
    flatkeys = list(flatten(keys)) if isinstance(keys, list) else [keys]
    dsk, dependencies = cull(dsk, flatkeys)
    dsk, dependencies = fuse(dsk, keys, dependencies=dependencies,
                             ave_width=_globals.get('fuse_ave_width', 1))
    dsk, _ = cull(dsk, keys)
    return dsk


def finalize(results):
    return gd.concat(results)


class _Frame(DaskMethodsMixin, OperatorMethodMixin):
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
    __dask_scheduler__ = staticmethod(threaded_get)
    __dask_optimize__ = staticmethod(optimize)

    def __dask_postcompute__(self):
        return finalize, ()

    def __dask_postpersist__(self):
        return type(self), (self._name, self._meta, self.divisions)

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

    def __dask_keys__(self):
        return [(self._name, i) for i in range(self.npartitions)]

    def __dask_graph__(self):
        return self.dask

    def __getstate__(self):
        return (self.dask, self._name, self._meta, self.divisions)

    def __setstate__(self, state):
        self.dask, self._name, self._meta, self.divisions = state

    def __repr__(self):
        s = "<dask_gdf.%s | %d tasks | %d npartitions>"
        return s % (type(self).__name__, len(self.dask), self.npartitions)

    @property
    def known_divisions(self):
        """Is divisions known?
        """
        return len(self.divisions) > 0 and self.divisions[0] is not None


    @property
    def npartitions(self):
        """Return number of partitions"""
        return len(self.divisions) - 1

    @property
    def index(self):
        """Return dask Index instance"""
        name = self._name + '-index'
        dsk = {(name, i): (getattr, key, 'index')
               for i, key in enumerate(self.__dask_keys__())}
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

    def to_dask_dataframe(self):
        """Create a dask.dataframe object from a dask_gdf object"""
        meta = self._meta.to_pandas()
        dummy = self.map_partitions(M.to_pandas, meta=self._meta)
        return dd.core.new_dd_object(dummy.dask, dummy._name, meta,
                                     dummy.divisions)

    def to_delayed(self):
        """See dask_gdf.to_delayed docstring for more information."""
        return to_delayed(self)

    def append(self, other):
        """Add rows from *other*
        """
        return concat([self, other])


def _daskify(obj, npartitions=None, chunksize=None):
    """Convert input to a dask-gdf object.
    """
    npartitions = npartitions or 1
    if isinstance(obj, _Frame):
        return obj
    elif isinstance(obj, (pd.DataFrame, pd.Series, pd.Index)):
        return _daskify(dd.from_pandas(obj, npartitions=npartitions))
    elif isinstance(obj, (gd.DataFrame, gd.Series, gd.index.Index)):
        return from_pygdf(obj, npartitions=npartitions)
    elif isinstance(obj, (dd.DataFrame, dd.Series, dd.Index)):
        return from_dask_dataframe(obj)
    else:
        raise TypeError("type {} is not supported".format(type(obj)))


def concat(objs):
    """Concantenate dask gdf objects

    Parameters
    ----------

    objs : sequence of DataFrame, Series, Index
        A sequence of objects to be concatenated.
    """
    objs = [_daskify(x) for x in objs]
    meta = gd.concat(_extract_meta(objs))

    name = "concat-" + uuid4().hex
    dsk = {}
    divisions = [0]
    base = 0
    lastdiv = 0
    for obj in objs:
        for k, i in obj.__dask_keys__():
            dsk[name, base + i] = k, i
        base += obj.npartitions
        divisions.extend([d + lastdiv for d in obj.divisions[1:]])
        lastdiv = obj.divisions[-1]

    dasks = [o.dask for o in objs]
    dsk = merge(dsk, *dasks)
    return new_dd_object(dsk, name, meta, divisions)


normalize_token.register(_Frame, lambda a: a._name)


def query(df, expr, callenv):
    boolmask = gd.queryutils.query_execute(df, expr, callenv)

    selected = gd.Series(boolmask)
    newdf = gd.DataFrame()
    for col in df.columns:
        newseries = df[col][selected]
        newdf[col] = newseries
    return newdf


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
        elif isinstance(key, list):
            def slice_columns(df, key):
                return df.loc[:, key]

            meta = slice_columns(self._meta, key)
            return self.map_partitions(slice_columns, key, meta=meta)
        raise NotImplementedError("Indexing with %r" % key)

    def drop_columns(self, *args):
        cols = list(self.columns)
        for k in args:
            del cols[cols.index(k)]
        return self[cols]

    def rename(self, columns):
        op = self
        for k, v in columns.items():
            op = op._rename_column(k, v)
        return op

    def _rename_column(self, k, v):
        def replace(df, k, v):
            sr = df[k]
            del df[k]
            df[v] = sr
            return df

        meta = replace(self._meta, k, v)
        return self.map_partitions(replace, k, v, meta=meta)

    def assign(self, **kwargs):
        """Add columns to the dataframe.

        Parameters
        ----------
        **kwargs : dict
            The keys are used for the column names.
            The values are Series for the new column.
        """
        op = self
        for k, v in kwargs.items():
            op = op._assign_column(k, v)
        return op

    def _assign_column(self, k, v):
        if not isinstance(v, Series):
            msg = 'cannot column {!r} of type: {}'
            raise TypeError(msg.format(k, type(v)))

        def assigner(df, k, v):
            out = df.copy()
            out.add_column(k, v)
            return out

        meta = assigner(self._meta, k, make_meta(v))
        return self.map_partitions(assigner, k, v, meta=meta)

    def apply_rows(self, func, incols, outcols, kwargs={}, cache_key=None):
        import uuid
        if cache_key is None:
            cache_key = uuid.uuid4()
        def do_apply_rows(df, func, incols, outcols, kwargs):
            return df.apply_rows(func, incols, outcols, kwargs,
                                 cache_key=cache_key)

        meta = do_apply_rows(self._meta, func, incols, outcols, kwargs)
        return self.map_partitions(do_apply_rows, func, incols, outcols, kwargs,
                                   meta=meta)

    def query(self, expr):
        """Query with a boolean expression using Numba to compile a GPU kernel.

        See pandas.DataFrame.query.

        Parameters
        ----------
        expr : str
            A boolean expression.  Names in the expression refers to the
            columns.

        Returns
        -------
        filtered :  DataFrame
        """
        if "@" in expr:
            raise NotImplementedError("Using variables from the calling "
                                      "environment")
        # Empty calling environment
        callenv = {
            'locals': {},
            'globals': {},
        }
        return self.map_partitions(query, expr, callenv, meta=self._meta)

    def groupby(self, by):
        from .groupby import Groupby

        return Groupby(df=self, by=by)

    def merge(self, other, on=None, how='left', lsuffix='_x', rsuffix='_y'):
        assert how == 'left', 'left join is impelemented'
        if on is None or len(on) == 1:
            return self.join(other, how=how, lsuffix=lsuffix, rsuffix=rsuffix)
        else:
            return self._merge(other, on=on, how=how, lsuffix=lsuffix,
                               rsuffix=rsuffix)

    def _merge(self, other, on, how, lsuffix, rsuffix):
        left_val_names = [k for k in self.columns if k not in on]
        right_val_names = [k for k in other.columns if k not in on]
        same_names = set(left_val_names) & set(right_val_names)
        if same_names and not (lsuffix or rsuffix):
            raise ValueError('there are overlapping columns but '
                             'lsuffix and rsuffix are not defined')

        assert how == 'left'

        def build_hashtable(frame):
            mod = 1300511
            subset = frame.loc[:, on].to_pandas()
            multihash = subset.values
            multihash = multihash * (1 + np.arange(multihash.ndim))
            hashed = pd.util.hash_array(multihash.sum(axis=1))
            hashtable = pd.util.hash_array(hashed % mod)
            return hashtable

        def build_whohas_map(frame, partid):
            ht = build_hashtable(frame)
            whohas_map = {v: set([partid]) for v in ht}
            return whohas_map

        def combine_whohas_map(a, b):
            return merge_with(lambda vs: reduce(operator.or_, vs), a, b)

        def build_depends(frame, whohas):
            ht = build_hashtable(frame)
            common = frozenset(ht) & frozenset(whohas.keys())
            return ht, {v for k in common for v in whohas[k]}

        def concat(ht, *frames):
            return gd.concat(frames)

        def get_empty_frame(df):
            return df[:0]

        def merge(left, right):
            return left.merge(right, how=how, on=on)

        def tree_reduce(fn, seq):
            def chunking(seq):
                for a, b in zip(seq[::2], seq[1::2]):
                    yield a, b
                if len(seq) % 2 != 0:
                    yield (seq[-1],)

            while len(seq) > 1:
                seq = [reduce(fn, pair)
                       for pair in chunking(seq)]
            return seq[0]

        # Determine which right partitions has what
        whohas = [delayed(build_whohas_map)(p, i)
                  for i, p in enumerate(other.to_delayed())]
        whohas = tree_reduce(delayed(combine_whohas_map), whohas)
        hts_depends = [delayed(build_depends)(p, whohas=whohas)
                       for p in self.to_delayed()]

        hts = list(map(operator.itemgetter(0), hts_depends))
        depends = list(map(operator.itemgetter(1), hts_depends))

        def do_reparts(ht, dep):
            if not dep:
                return delayed(get_empty_frame)(other.to_delayed()[0])
            else:
                return delayed(concat)(ht, *[other.to_delayed()[i] for i in dep])

        reparts = [do_reparts(ht, dep)
                   for ht, dep in zip(hts, compute(*depends))]

        res = [delayed(merge)(x, y)
               for x, y in zip(self.to_delayed(), reparts)]

        return from_delayed(res, prefix='join_result')


    def join(self, other, how='left', lsuffix='', rsuffix=''):
        """Join two datatframes

        *on* is not supported.
        """
        if how == 'right':
            return other.join(other=self, how='left', lsuffix=rsuffix,
                              rsuffix=lsuffix)

        same_names = set(self.columns) & set(other.columns)
        if same_names and not (lsuffix or rsuffix):
            raise ValueError('there are overlapping columns but '
                             'lsuffix and rsuffix are not defined')

        left, leftuniques = self._align_divisions()
        right, rightuniques = other._align_to_indices(leftuniques)

        leftparts = left.to_delayed()
        rightparts = right.to_delayed()

        @delayed
        def part_join(left, right, how):
            return left.join(right, how=how, sort=True,
                             lsuffix=lsuffix, rsuffix=rsuffix)

        def inner_selector():
            pivot = 0
            for i in range(len(leftparts)):
                for j in range(pivot, len(rightparts)):
                    if leftuniques[i] & rightuniques[j]:
                        yield leftparts[i], rightparts[j]
                        pivot = j + 1
                        break

        def left_selector():
            pivot = 0
            for i in range(len(leftparts)):
                for j in range(pivot, len(rightparts)):
                    if leftuniques[i] & rightuniques[j]:
                        yield leftparts[i], rightparts[j]
                        pivot = j + 1
                        break
                else:
                    yield leftparts[i], None

        selector = {
            'left': left_selector,
            'inner': inner_selector,
        }[how]

        rhs_dtypes = [(k, other._meta.dtypes[k])
                      for k in other._meta.columns]

        @delayed
        def fix_column(lhs):
            df = gd.DataFrame()
            for k in lhs.columns:
                df[k + lsuffix] = lhs[k]

            for k, dtype in rhs_dtypes:
                data = np.zeros(len(lhs), dtype=dtype)
                mask_size = gd.utils.calc_chunk_size(data.size,
                                                     gd.utils.mask_bitsize)
                mask = np.zeros(mask_size, dtype=gd.utils.mask_dtype)
                sr = gd.Series.from_masked_array(data=data,
                                                 mask=mask,
                                                 null_count=data.size)

                df[k + rsuffix] = sr.set_index(df.index)

            return df

        joinedparts = [(part_join(lhs, rhs, how=how)
                        if rhs is not None
                        else fix_column(lhs))
                       for lhs, rhs in selector()]

        meta = self._meta.join(other._meta, how=how,
                               lsuffix=lsuffix, rsuffix=rsuffix)
        return from_delayed(joinedparts, meta=meta)

    def _align_divisions(self):
        """Align so that the values do not split across partitions
        """
        parts = self.to_delayed()
        uniques = self._get_unique_indices(parts=parts)
        originals = list(map(frozenset, uniques))

        changed = True
        while changed:
            changed = False
            for i in range(len(uniques))[:-1]:
                intersect = uniques[i] & uniques[i + 1]
                if intersect:
                    smaller = min(uniques[i], uniques[i+1], key=len)
                    bigger = max(uniques[i], uniques[i+1], key=len)
                    smaller |= intersect
                    bigger -= intersect
                    changed = True

        # Fix empty partitions
        uniques = list(filter(bool, uniques))

        return self._align_to_indices(uniques,
                                      originals=originals,
                                      parts=parts)

    def _get_unique_indices(self, parts=None):
        if parts is None:
            parts = self.to_delayed()

        @delayed
        def unique(x):
            return set(x.index.as_column().unique().to_array())

        parts = self.to_delayed()
        return compute(*map(unique, parts))

    def _align_to_indices(self, uniques, originals=None, parts=None):
        uniques = list(map(set, uniques))

        if parts is None:
            parts = self.to_delayed()

        if originals is None:
            originals = self._get_unique_indices(parts=parts)
            allindices = set()
            for x in originals:
                allindices |= x
            for us in uniques:
                us &= allindices
            uniques = list(filter(bool, uniques))

        extras = originals[-1] - uniques[-1]
        extras = {x for x in extras if x > max(uniques[-1])}

        if extras:
            uniques.append(extras)

        remap = OrderedDict()
        for idxset in uniques:
            remap[tuple(sorted(idxset))] = bins = []
            for i, orig in enumerate(originals):
                if idxset & orig:
                    bins.append(parts[i])

        @delayed
        def take(indices, depends):
            first = min(indices)
            last = max(indices)
            others = []
            for d in depends:
                # TODO: this can be replaced with searchsorted
                # Normalize to index data in range before selection.
                firstindex = d.index[0]
                lastindex = d.index[-1]
                s = max(first, firstindex)
                e = min(last, lastindex)
                others.append(d.loc[s:e])
            return gd.concat(others)

        newparts = []
        for idx, depends in remap.items():
            newparts.append(take(idx, depends))

        divisions = list(map(min, uniques))
        divisions.append(max(uniques[-1]))

        newdd = from_delayed(newparts, meta=self._meta)
        return newdd, uniques

    def _compute_divisions(self):
        if self.known_divisions:
            return self

        @delayed
        def first_index(df):
            return df.index[0]

        @delayed
        def last_index(df):
            return df.index[-1]

        parts = self.to_delayed()
        divs = [first_index(p) for p in parts] + [last_index(parts[-1])]
        divisions = compute(*divs)
        return type(self)(self.dask, self._name, self._meta, divisions)

    def set_index(self, index, drop=True, sorted=False):
        """Set new index.

        Parameters
        ----------
        index : str or Series
            If a ``str`` is provided, it is used as the name of the
            column to be made into the index.
            If a ``Series`` is provided, it is used as the new index
        drop : bool
            Whether the first original index column is dropped.
        sorted : bool
            Whether the new index column is already sorted.
        """
        if not drop:
            raise NotImplementedError('drop=False not supported yet')

        if isinstance(index, str):
            tmpdf = self.sort_values(index)
            return tmpdf._set_column_as_sorted_index(index, drop=drop)
        elif isinstance(index, Series):
            indexname = '__dask_gdf.index'
            df = self.assign(**{indexname: index})
            return df.set_index(indexname, drop=drop, sorted=sorted)
        else:
            raise TypeError('cannot set_index from {}'.format(type(index)))

    def _set_column_as_sorted_index(self, colname, drop):
        def select_index(df, col):
            return df.set_index(col)

        return self.map_partitions(select_index, col=colname,
                                   meta=self._meta.set_index(colname))

    def _argsort(self, col, sorted=False):
        """
        Returns
        -------
        shufidx : Series
            Positional indices to be used with .take() to
            put the dataframe in order w.r.t ``col``.
        """
        # Get subset with just the index and positional value
        subset = self[col].to_dask_dataframe()
        subset = subset.reset_index(drop=False)
        ordered = subset.set_index(0, sorted=sorted)
        shufidx = from_dask_dataframe(ordered)['index']
        return shufidx

    def _set_index_raw(self, indexname, drop, sorted):
        shufidx = self._argsort(indexname, sorted=sorted)
        # Shuffle the GPU data
        shuffled = self.take(shufidx, npartitions=self.npartitions)
        out = shuffled.map_partitions(lambda df: df.set_index(indexname))
        return out

    def reset_index(self, force=False):
        """Reset index to range based
        """
        if force:
            dfs = self.to_delayed()
            sizes = np.asarray(compute(*map(delayed(len), dfs)))
            prefixes = np.zeros_like(sizes)
            prefixes[1:] = np.cumsum(sizes[:-1])

            @delayed
            def fix_index(df, startpos):
                stoppos = startpos + len(df)
                return df.set_index(gd.index.RangeIndex(start=startpos,
                                                        stop=stoppos))

            outdfs = [fix_index(df, startpos)
                      for df, startpos in zip(dfs, prefixes)]
            return from_delayed(outdfs, meta=self._meta.reset_index())
        else:
            def reset_index(df):
                return df.reset_index()
            return self.map_partitions(reset_index,
                                       meta=reset_index(self._meta))

    def sort_values(self, by, ignore_index=False):
        """Sort by the given column

        Parameter
        ---------
        by : str
        """
        parts = self.to_delayed()
        sorted_parts = batcher_sortnet.sort_delayed_frame(parts, by)
        return from_delayed(sorted_parts, meta=self._meta).reset_index(force=not ignore_index)

    def sort_values_binned(self, by):
        """Sorty by the given column and ensure that the same key
        doesn't spread across multiple partitions.
        """
        # Get sorted partitions
        parts = self.sort_values(by=by).to_delayed()
        # Get unique keys in each partition
        @delayed
        def get_unique(p):
            return set(p[by].unique())
        uniques = list(compute(*map(get_unique, parts)))

        joiner = {}
        for i in range(len(uniques)):
            joiner[i] = to_join = {}
            for j in range(i + 1, len(uniques)):
                intersect = uniques[i] & uniques[j]
                # If the keys intersect
                if intersect:
                    # Remove keys
                    uniques[j] -= intersect
                    to_join[j] = frozenset(intersect)
                else:
                    break

        @delayed
        def join(df, other, keys):
            others = [other.query('{by}==@k'.format(by=by))
                      for k in sorted(keys)]
            return gd.concat([df] + others)

        @delayed
        def drop(df, keep_keys):
            locvars = locals()
            for i, k in enumerate(keep_keys):
                locvars['k{}'.format(i)] = k

            conds = ['{by}==@k{i}'.format(by=by, i=i)
                     for i in range(len(keep_keys))]
            expr = ' or '.join(conds)
            return df.query(expr)

        for i in range(len(parts)):
            if uniques[i]:
                parts[i] = drop(parts[i], uniques[i])
                for joinee, intersect in joiner[i].items():
                    parts[i] = join(parts[i], parts[joinee], intersect)

        results = [p for i, p in enumerate(parts) if uniques[i]]
        return from_delayed(results, meta=self._meta).reset_index()

    def _shuffle_sort_values(self, by):
        """Slow shuffle based sort by the given column

        Parameter
        ---------
        by : str
        """
        shufidx = self._argsort(by)
        return self.take(shufidx)

    def take(self, indices, npartitions=None, chunksize=None):
        """Take elements from the positional indices.

        Parameters
        ----------
        indices : Series

        Note
        ----
        Difference from pandas:
            * We reset the index to 0..N to maintain the property that
              the indices must be sorted.

        """
        indices = _daskify(indices, npartitions=npartitions,
                           chunksize=chunksize)

        def get_parts(idxs, divs):
            parts = [p for i in idxs
                       for p, (s, e) in enumerate(zip(divs, divs[1:]))
                       if s <= i and (i < e or e == divs[-1])]
            return parts

        @delayed
        def partition(sr, divs):
            return sorted(frozenset(get_parts(sr.to_array(), divs)))

        @delayed
        def first_index(df):
            return df.index[0]

        @delayed
        def last_index(df):
            return df.index[-1]

        parts = self.to_delayed()
        # get parts
        if self.known_divisions:
            divs = self.divisions
        else:
            divs = [first_index(p) for p in parts] + [last_index(parts[-1])]

        sridx = indices.to_delayed()
        # drop empty partitions in sridx
        sridx_sizes = compute(*map(delayed(len), sridx))
        sridx = [sr for sr, n in zip(sridx, sridx_sizes) if n > 0]
        # compute partitioning
        partsel = compute(*(partition(sr, divs) for sr in sridx))

        grouped_parts = [tuple(parts[j] for j in sel)
                         for sel in partsel]

        # compute sizes of each partition
        sizes = compute(*map(delayed(len), parts))
        prefixes = np.zeros_like(sizes)
        prefixes[1:] = np.cumsum(sizes)[:-1]

        # shuffle them
        @delayed
        def shuffle(sr, prefixes, divs, *deps):
            idxs = sr.to_array()
            parts = np.asarray(get_parts(idxs, divs))

            partdfs = []
            for p, df in zip(sorted(frozenset(parts)), deps):
                cond = parts == p
                valididxs = idxs[cond]
                ordering = np.arange(len(idxs))[cond]
                selected = valididxs - prefixes[p]
                sel = df.take(selected).set_index(ordering)
                partdfs.append(sel)

            joined = gd.concat(partdfs).sort_index()
            return joined

        shuffled = [shuffle(sr, prefixes, divs, *deps)
                    for sr, deps in zip(sridx, grouped_parts)]
        out = from_delayed(shuffled)

        out = out.reset_index(force=True)
        return out


def sum_of_squares(x):
    x = x.astype('f8')._column
    outcol = gd._gdf.apply_reduce(libgdf.gdf_sum_squared_generic, x)
    return gd.Series(outcol)


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

    def fillna(self, value):
        if not np.can_cast(value, self.dtype):
            raise TypeError("fill value must match dtype of series")
        return self.map_partitions(M.fillna, value, meta=self)

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
    segments = list(df.index.find_segments().to_array())
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


def _from_pandas(df):
    return gd.DataFrame.from_pandas(df)


def from_delayed(dfs, meta=None, prefix='from_delayed'):
    """ Create Dask GDF DataFrame from many Dask Delayed objects
    Parameters
    ----------
    dfs : list of Delayed
        An iterable of ``dask.delayed.Delayed`` objects, such as come from
        ``dask.delayed`` These comprise the individual partitions of the
        resulting dataframe.
    meta : pygdf.DataFrame, pygdf.Series, or pygdf.Index
        An empty pygdf object with names, dtypes, and indices matching the
        expected output.
    prefix : str, optional
        Prefix to prepend to the keys.
    """
    from dask.delayed import Delayed, delayed

    if isinstance(dfs, Delayed):
        dfs = [dfs]

    dfs = [delayed(df)
           if not isinstance(df, Delayed) and hasattr(df, 'key')
           else df
           for df in dfs]

    for df in dfs:
            if not isinstance(df, Delayed):
                raise TypeError("Expected Delayed object, got {}".format(
                                type(df).__name__))

    if meta is None:
        meta = dfs[0].compute()
    meta = make_meta(meta)

    name = prefix + '-' + tokenize(*dfs)

    dsk = merge(df.dask for df in dfs)
    dsk.update({(name, i): (check_meta, df.key, meta, 'from_delayed')
                for (i, df) in enumerate(dfs)})

    divs = [None] * (len(dfs) + 1)
    df = new_dd_object(dsk, name, meta, divs)

    return df


def to_delayed(df):
    """ Create Dask Delayed objects from a Dask GDF Dataframe
    Returns a list of delayed values, one value per partition.
    """
    from dask.delayed import Delayed

    keys = df.__dask_keys__()
    dsk = df.__dask_optimize__(df.dask, keys)
    return [Delayed(k, dsk) for k in keys]


def from_dask_dataframe(df):
    """Create a `dask_gdf.DataFrame` from a `dask.dataframe.DataFrame`

    Parameters
    ----------
    df : dask.dataframe.DataFrame
    """
    bad_cols = df.select_dtypes(include=['O', 'M', 'm'])
    if len(bad_cols.columns):
        raise ValueError("Object, datetime, or timedelta dtypes aren't "
                         "supported by pygdf")

    meta = _from_pandas(df._meta)
    dummy = DataFrame(df.dask, df._name, meta, df.divisions)
    return dummy.map_partitions(_from_pandas, meta=meta)


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
               for i, key in enumerate(args[0].__dask_keys__())}
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
