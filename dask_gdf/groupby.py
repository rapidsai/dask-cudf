import operator

import numpy as np

from dask.delayed import delayed
from dask import compute, persist
import pygdf

from .core import from_delayed


class Groupby(object):
    """The object returned by ``df.groupby()``.
    """
    def __init__(self, df, by):
        self._df = df
        self._by = tuple([by]) if isinstance(by, str) else tuple(by)
        self._grouped_cache = None

    @property
    def _grouped(self):
        """Get the groups.

        The actual groupby operation is executed once and
        then cached for future use.
        """
        if self._grouped_cache is None:
            self._grouped_cache = self._do_grouping()
        return self._grouped_cache

    def _do_grouping(self):
        """Group the dataframe
        """
        # First, do groupby on the first key by sorting on the first key.
        # This will sort & shuffle the partitions.
        firstkey = self._by[0]
        df = self._df.sort_values(firstkey)
        groups = df.to_delayed()
        # Second, do groupby internally for each partition.
        @delayed
        def _groupby(df, by):
            grouped = df.groupby(by=by)
            return grouped

        # Get the groupby objects
        grouped = [_groupby(g, self._by) for g in groups]
        return grouped

    def _aggregation(self, chunk, combine, split_every=8):
        by = self._by

        def cat_and_group(*dfs):
            return pygdf.concat(dfs).reset_index().groupby(by)

        parts = [delayed(chunk)(g) for g in self._grouped]
        while len(parts) > 1:
            chunked = _chunk_every(parts, split_every)
            parts = [delayed(cat_and_group)(*c) for c in chunked]
            parts = [delayed(combine)(g) for g in parts]
        return from_delayed(parts).reset_index()

    def apply(self, function):
        """Transform each group using a python function.
        """
        @delayed
        def apply_to_group(grp):
            return grp.apply(function)

        grouped = [apply_to_group(g) for g in self._grouped]
        return from_delayed(grouped).reset_index()

    def apply_grouped(self, *args, **kwargs):
        """Transform each group using a GPU function.

        Calls ``pygdf.Groupby.apply_grouped`` concurrently
        """
        @delayed
        def apply_to_group(grp):
            return grp.apply_grouped(*args, **kwargs)

        grouped = [apply_to_group(g) for g in self._grouped]
        return from_delayed(grouped).reset_index()

    # Aggregation APIs

    def count(self):
        return self._aggregation(lambda g: g.count(),
                                 lambda g: g.sum())

    def sum(self):
        return self._aggregation(lambda g: g.count(),
                                 lambda g: g.sum())

    def mean(self):
        valcols = set(self._df.columns) - set(self._by)

        def combine(df):
            outdf = df[:1].loc[:, list(self._by)]
            for k in valcols:
                sumk = '{}_sum'.format(k)
                countk = '{}_count'.format(k)
                outdf[k] = df[sumk].sum() / df[countk].sum()
            return outdf

        return self._aggregation(lambda g: g.agg(['sum', 'count']),
                                 lambda g: g.apply(combine),
                                 split_every=None)

    def max(self):
        return self._aggregation(lambda g: g.max(),
                                 lambda g: g.max())

    def min(self):
        return self._aggregation(lambda g: g.min(),
                                 lambda g: g.min())

    def std(self, ddof=1):
        valcols = set(self._df.columns) - set(self._by)

        def combine(df):
            outdf = df[:1].loc[:, list(self._by)]
            for k in valcols:
                sosk = '{}_sum_of_squares'.format(k)
                sumk = '{}_sum'.format(k)
                countk = '{}_count'.format(k)
                the_sos = df[sosk].sum()
                the_sum = df[sumk].sum()
                the_count = df[countk].sum()

                div = the_count - ddof
                mu = the_sum / the_count
                var = the_sos / div - (mu ** 2) * the_count / div

                outdf[k] = np.sqrt(var)

            return outdf

        return self._aggregation(
            lambda g: g.agg(['sum_of_squares', 'sum', 'count']),
            lambda g: g.apply(combine),
            split_every=None)


def _chunk_every(seq, every):
    group = []
    for x in seq:
        group.append(x)
        if every is not None and len(group) == every:
            yield group
            group = []
    yield group
