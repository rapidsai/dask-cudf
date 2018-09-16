import dask_gdf 
from toolz import partial 
import pygdf as gd 
class Accessor(object):
    """
    Base class for Accessor objects dt, (and (str, cat) once there are supported
    in future releases).
    Notes
    -----
    Subclasses should define the following attributes:
    * _accessor
    * _accessor_name
    """
    _not_implemented = set()

    def __init__(self, series):
        from .core import Series
        if not isinstance(series, Series):
            raise ValueError('Accessor cannot be initialized')
        self._series = series
        self._validate(series)

    def _validate(self, series):
        pass


    @staticmethod
    def _delegate_property(obj, accessor, attr):
        out = getattr(getattr(obj, accessor, obj), attr)
        return out 

    @staticmethod
    def _delegate_method(obj, accessor, attr, args, kwargs):
        out = getattr(getattr(obj, accessor, obj), attr)(*args, **kwargs)
        return out 

    def _property_map(self, attr):
        meta = self._delegate_property(self._series._meta,
                                       self._accessor_name, attr)
        token = '%s-%s' % (self._accessor_name, attr)
        return self._series.map_partitions(self._delegate_property,
                                           self._accessor_name, attr,
                                           token=token, meta=meta)

    def _function_map(self, attr, *args, **kwargs):
        meta = self._delegate_method(self._series._meta_nonempty,
                                     self._accessor_name, attr, args, kwargs)
        token = '%s-%s' % (self._accessor_name, attr)
        return self._series.map_partitions(self._delegate_method,
                                           self._accessor_name, attr, args,
                                           kwargs, meta=meta, token=token)

    @property
    def _delegates(self):
        return set(dir(self._accessor)).difference(self._not_implemented)

    def __dir__(self):
        o = self._delegates
        o.update(self.__dict__)
        o.update(dir(type(self)))
        return list(o)

    def __getattr__(self, key):
        if key in self._delegates:
            if isinstance(getattr(self._accessor, key), property):
                return self._property_map(key)
            else:
                return partial(self._function_map, key)
        else:
            raise AttributeError(key)


class DatetimeAccessor(Accessor):
    """ Accessor object for datetimelike properties of the Series values.
    """
    _accessor = gd.Series.dt
    _accessor_name = 'dt'

    def _validate(self, series):
        if not isinstance(series._meta._column, gd.datetime.DatetimeColumn):
            raise AttributeError("Can only use .dt accessor with datetimelike "
                                 "values")
    



