import operator

import numpy as np

from dask.delayed import delayed
from dask import compute, persist

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
        df = self._df.sort_value(firstkey)
        groups = df.to_delayed()
        # Second, do groupby internally for each partition.
        @delayed
        def _groupby(df, by):
            grouped = df.groupby(by=by)
            ovdata = _extract_data_to_check_group_overlap(grouped, by)
            return grouped, ovdata

        grouped = [_groupby(g, self._by) for g in groups]
        # Persist the groupby operation to avoid duplicating the work
        grouped = persist(*grouped)
        # Get the groupby objects
        outgroups = list(map(delayed(operator.itemgetter(0)), grouped))
        _check_group_non_overlap_assumption(grouped)
        return outgroups

    def _aggregation(self, reducer):
        parts = [delayed(reducer)(g) for g in self._grouped]
        return from_delayed(parts).reset_index()

    def apply_grouped(self, *args, **kwargs):
        """Transform each group using a GPU function.

        Calls ``pygdf.Groupby.apply_grouped`` concurrently
        """
        @delayed
        def apply_to_group(grp):
            return grp.apply_grouped(*args, **kwargs)

        grouped = [apply_to_group(g) for g in self._grouped]
        return from_delayed(grouped)

    # Aggregation APIs

    def count(self):
        return self._aggregation(lambda g: g.count())

    def mean(self):
        return self._aggregation(lambda g: g.mean())

    def max(self):
        return self._aggregation(lambda g: g.max())

    def min(self):
        return self._aggregation(lambda g: g.min())

    def std(self):
        return self._aggregation(lambda g: g.std())


def _extract_data_to_check_group_overlap(grouped, by):
    """
    See _check_group_non_overlap_assumption()
    """
    interim, _ = grouped.as_df()
    limits = interim.loc[:, by].take(np.asarray([0, len(interim) - 1]))
    pdlimits = limits.to_pandas()
    first = pdlimits.iloc[0].to_dict()
    last = pdlimits.iloc[1].to_dict()
    return grouped, (first, last)


def _check_group_non_overlap_assumption(grouped):
    # Check that the groups do not overlap.
    # NOTE: the implementation of the aggregation functions is
    #       assuming non-overlapping groups.
    limits = compute(map(delayed(operator.itemgetter(1)), grouped))
    for (_, last), (cur, _) in zip(limits, limits[1:]):
        if last == cur:
            raise NotImplementedError("unexpected overlay of group")
