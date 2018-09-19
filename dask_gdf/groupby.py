import numpy as np

from dask.delayed import delayed
import pygdf

from .core import from_delayed


class Groupby(object):
    """The object returned by ``df.groupby()``.
    """
    def __init__(self, df, by, method):
        self._df = df
        self._by = tuple([by]) if isinstance(by, str) else tuple(by)
        self._grouped_cache = None
        self._method = method

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
        df = self._df.sort_values(firstkey, ignore_index=True)
        groups = df.to_delayed()

        # Second, do groupby internally for each partition.
        @delayed
        def _groupby(df, by, method):
            grouped = df.groupby(by=by, method=method)
            return grouped

        # Get the groupby objects
        grouped = [_groupby(g, self._by, self._method) for g in groups]
        return grouped

    def agg(self, mapping):
        second_mapping = {}
        for key, val in mapping.items():
            second_mapping[val+"_"+key] = mapping[key]
        return self._aggregation(lambda df: df.agg(mapping),
                                 lambda df: df.agg(second_mapping))

    def _aggregation(self, chunk, combine, split_every=4):
        by = self._by
        method = self._method

        @delayed
        def do_local_groupby(df, method):
            return chunk(df.groupby(by=by, method=method))

        @delayed
        def do_combine(dfs, method):
            return combine(pygdf.concat(dfs).groupby(by=by, method=method))

        parts = self._df.to_delayed()
        parts = [do_local_groupby(p, method) for p in parts]
        if split_every is not None:
            while len(parts) > 1:
                tasks, remains = parts[:split_every], parts[split_every:]
                out = do_combine(tasks, method)
                parts = remains + [out]
        else:
            parts = do_combine(parts, method)
        meta = combine(chunk(self._df._meta.groupby(by=by)).groupby(by=by))
        return from_delayed(parts, meta=meta).reset_index()

        # SHUFFLE VERSION
        # @delayed
        # def do_agg_prepare(gb):
        #     df = gb.as_df()[0]
        #     return df.set_index(df[by[0]])

        # fisrtgroupby = from_delayed(list(map(do_agg_prepare, self._grouped)),
        #                             meta=self._df._meta)
        # aligned, _ = fisrtgroupby._align_divisions()

        # @delayed
        # def do_local_groupby(df):
        #     return df.groupby(by)

        # tmp = map(do_local_groupby, aligned.to_delayed())
        # agg = map(delayed(chunk), tmp)
        # return from_delayed(list(agg), meta=self._df._meta).reset_index()

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
            outdf = df[:1].loc[:, list(self._by)].reset_index()
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

    def _compute_std_or_var(self, ddof=1, do_std=False):
        valcols = set(self._df.columns) - set(self._by)

        def combine(df):
            outdf = df[:1].loc[:, list(self._by)].reset_index()
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

                outdf[k] = np.sqrt(var) if do_std else var

            return outdf

        return self._aggregation(
            lambda g: g.agg(['sum_of_squares', 'sum', 'count']),
            lambda g: g.apply(combine),
            split_every=None)

    def std(self, ddof=1):
        return self._compute_std_or_var(ddof=ddof, do_std=True)

    def var(self, ddof=1):
        return self._compute_std_or_var(ddof=ddof, do_std=False)


def _chunk_every(seq, every):
    group = []
    for x in seq:
        group.append(x)
        if every is not None and len(group) == every:
            yield group
            group = []
    yield group
