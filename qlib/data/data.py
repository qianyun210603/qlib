# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import re
import abc
import bisect
import copy
import multiprocessing
import queue
from collections import deque
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import struct

# For supporting multiprocessing in outer code, joblib is used
from joblib import delayed

from ..config import C
from ..log import get_module_logger
from ..utils import (
    Wrapper,
    code_to_fname,
    get_module_by_module_path,
    get_period_list,
    hash_args,
    init_instance_by_config,
    normalize_cache_fields,
    parse_field,
    get_period_offset,
    register_wrapper,
    time_to_slc_point,
)
from ..utils.paral import ParallelExt
from .cache import DiskDatasetCache, H, MemCacheLengthUnit
from .inst_processor import InstProcessor
from .ops import ExpressionOps, Operators  # pylint: disable=W0611  # noqa: F401


class ProviderBackendMixin:
    """
    This helper class tries to make the provider based on storage backend more convenient
    It is not necessary to inherent this class if that provider don't rely on the backend storage
    """

    def __init__(self, backend={}):
        self.backend = backend

    def get_default_backend(self):
        backend = {}
        provider_name: str = re.findall("[A-Z][^A-Z]*", self.__class__.__name__)[-2]
        # set default storage class
        backend.setdefault("class", f"File{provider_name}Storage")
        # set default storage module
        backend.setdefault("module_path", "qlib.data.storage.file_storage")
        return backend

    def backend_obj(self, **kwargs):
        backend = self.backend if self.backend else self.get_default_backend()
        backend = copy.deepcopy(backend)
        backend.setdefault("kwargs", {}).update(**kwargs)
        return init_instance_by_config(backend)


class CalendarProvider(abc.ABC):
    """Calendar provider base class

    Provide calendar data.
    """

    def calendar(self, start_time=None, end_time=None, freq="day", future=False):
        """Get calendar of certain market in given time range.

        Parameters
        ----------
        start_time : str
            start of the time range.
        end_time : str
            end of the time range.
        freq : str
            time frequency, available: year/quarter/month/week/day.
        future : bool
            whether including future trading day.

        Returns
        ----------
        list
            calendar list
        """
        _calendar, _calendar_index = self._get_calendar(freq, future)
        if start_time == "None":
            start_time = None
        if end_time == "None":
            end_time = None
        # strip
        if start_time:
            start_time = pd.Timestamp(start_time)
            if start_time > _calendar[-1]:
                return np.array([])
        else:
            start_time = _calendar[0]
        if end_time:
            end_time = pd.Timestamp(end_time)
            if end_time < _calendar[0]:
                return np.array([])
        else:
            end_time = _calendar[-1]
        _, _, si, ei = self.locate_index(start_time, end_time, freq, future)
        return _calendar[si : ei + 1]

    def locate_index(
        self, start_time: Union[pd.Timestamp, str], end_time: Union[pd.Timestamp, str], freq: str, future: bool = False
    ):
        """Locate the start time index and end time index in a calendar under certain frequency.

        Parameters
        ----------
        start_time : pd.Timestamp
            start of the time range.
        end_time : pd.Timestamp
            end of the time range.
        freq : str
            time frequency, available: year/quarter/month/week/day.
        future : bool
            whether including future trading day.

        Returns
        -------
        pd.Timestamp
            the real start time.
        pd.Timestamp
            the real end time.
        int
            the index of start time.
        int
            the index of end time.
        """
        start_time = pd.Timestamp(start_time)
        end_time = pd.Timestamp(end_time)
        calendar, calendar_index = self._get_calendar(freq=freq, future=future)
        if start_time not in calendar_index:
            try:
                start_time = calendar[bisect.bisect_left(calendar, start_time)]
            except IndexError as index_e:
                raise IndexError(
                    "`start_time` uses a future date, if you want to get future trading days, you can use: `future=True`"
                ) from index_e
        start_index = calendar_index[start_time]
        if end_time not in calendar_index:
            end_time = calendar[bisect.bisect_right(calendar, end_time) - 1]
        end_index = calendar_index[end_time]
        return start_time, end_time, start_index, end_index

    def _get_calendar(self, freq, future):
        """Load calendar using memcache.

        Parameters
        ----------
        freq : str
            frequency of read calendar file.
        future : bool
            whether including future trading day.

        Returns
        -------
        list
            list of timestamps.
        dict
            dict composed by timestamp as key and index as value for fast search.
        """
        flag = f"{freq}_future_{future}"
        if flag not in H["c"]:
            _calendar = np.array(self.load_calendar(freq, future))
            _calendar_index = {x: i for i, x in enumerate(_calendar)}  # for fast search
            H["c"][flag] = _calendar, _calendar_index
        return H["c"][flag]

    @staticmethod
    def _uri(start_time, end_time, freq, future=False):
        """Get the uri of calendar generation task."""
        return hash_args(start_time, end_time, freq, future)

    def load_calendar(self, freq, future):
        """Load original calendar timestamp from file.

        Parameters
        ----------
        freq : str
            frequency of read calendar file.
        future: bool

        Returns
        ----------
        list
            list of timestamps
        """
        raise NotImplementedError("Subclass of CalendarProvider must implement `load_calendar` method")


class InstrumentProvider(abc.ABC):
    """Instrument provider base class

    Provide instrument data.
    """

    @staticmethod
    def instruments(market: Union[List, str] = "all", filter_pipe: Union[List, None] = None):
        """Get the general config dictionary for a base market adding several dynamic filters.

        Parameters
        ----------
        market : Union[List, str]
            str:
                market/industry/index shortname, e.g. all/sse/szse/sse50/csi300/csi500.
            list:
                ["ID1", "ID2"]. A list of stocks
        filter_pipe : list
            the list of dynamic filters.

        Returns
        ----------
        dict: if isinstance(market, str)
            dict of stockpool config.

            {`market` => base market name, `filter_pipe` => list of filters}

            example :

            .. code-block::

                {'market': 'csi500',
                'filter_pipe': [{'filter_type': 'ExpressionDFilter',
                'rule_expression': '$open<40',
                'filter_start_time': None,
                'filter_end_time': None,
                'keep': False},
                {'filter_type': 'NameDFilter',
                'name_rule_re': 'SH[0-9]{4}55',
                'filter_start_time': None,
                'filter_end_time': None}]}

        list: if isinstance(market, list)
            just return the original list directly.
            NOTE: this will make the instruments compatible with more cases. The user code will be simpler.
        """
        if isinstance(market, list):
            return market
        from .filter import SeriesDFilter  # pylint: disable=C0415

        if filter_pipe is None:
            filter_pipe = []
        config = {"market": market, "filter_pipe": []}
        # the order of the filters will affect the result, so we need to keep
        # the order
        for filter_t in filter_pipe:
            if isinstance(filter_t, dict):
                _config = filter_t
            elif isinstance(filter_t, SeriesDFilter):
                _config = filter_t.to_config()
            else:
                raise TypeError(
                    f"Unsupported filter types: {type(filter_t)}! Filter only supports dict or isinstance(filter, SeriesDFilter)"
                )
            config["filter_pipe"].append(_config)
        return config

    @abc.abstractmethod
    def list_instruments(self, instruments, start_time=None, end_time=None, freq="day", as_list=False):
        """List the instruments based on a certain stockpool config.

        Parameters
        ----------
        instruments : dict
            stockpool config.
        start_time : str
            start of the time range.
        end_time : str
            end of the time range.
        as_list : bool
            return instruments as list or dict.

        Returns
        -------
        dict or list
            instruments list or dictionary with time spans
        """
        raise NotImplementedError("Subclass of InstrumentProvider must implement `list_instruments` method")

    @staticmethod
    def _uri(instruments, start_time=None, end_time=None, freq="day", as_list=False):
        return hash_args(instruments, start_time, end_time, freq, as_list)

    # instruments type
    LIST = "LIST"
    DICT = "DICT"
    CONF = "CONF"

    @classmethod
    def get_inst_type(cls, inst):
        if "market" in inst:
            return cls.CONF
        if isinstance(inst, dict):
            return cls.DICT
        if isinstance(inst, (list, tuple, pd.Index, np.ndarray)):
            return cls.LIST
        raise ValueError(f"Unknown instrument type {inst}")


class FeatureProvider(abc.ABC):
    """Feature provider class

    Provide feature data.
    """

    @abc.abstractmethod
    def feature(self, instrument, field, start_time, end_time, freq):
        """Get feature data.

        Parameters
        ----------
        instrument : str
            a certain instrument.
        field : str
            a certain field of feature.
        start_time : str
            start of the time range.
        end_time : str
            end of the time range.
        freq : str
            time frequency, available: year/quarter/month/week/day.

        Returns
        -------
        pd.Series
            data of a certain feature
        """
        raise NotImplementedError("Subclass of FeatureProvider must implement `feature` method")


class PITProvider(abc.ABC):
    @abc.abstractmethod
    def period_feature(
        self,
        instrument,
        field,
        start_index: int,
        end_index: int,
        cur_time: pd.Timestamp,
        period: Optional[int] = None,
    ) -> pd.Series:
        """
        get the historical periods data series between `start_index` and `end_index`

        Parameters
        ----------
        start_index: int
            start_index is a relative index to the latest period to cur_time

        end_index: int
            end_index is a relative index to the latest period to cur_time
            in most cases, the start_index and end_index will be a non-positive values
            For example, start_index == -3 end_index == 0 and current period index is cur_idx,
            then the data between [start_index + cur_idx, end_index + cur_idx] will be retrieved.

        period: int
            This is used for query specific period.
            The period is represented with int in Qlib. (e.g. 202001 may represent the first quarter in 2020)
            NOTE: `period`  will override `start_index` and `end_index`

        Returns
        -------
        pd.Series
            The index will be integers to indicate the periods of the data
            An typical examples will be
            TODO

        Raises
        ------
        FileNotFoundError
            This exception will be raised if the queried data do not exist.
        """
        raise NotImplementedError(f"Please implement the `period_feature` method")


class ExpressionProvider(abc.ABC):
    """Expression provider class

    Provide Expression data.
    """

    def __init__(self):
        self.population = []
        self.expression_instance_cache = {}

    def get_expression_instance(self, field):
        try:
            if field in self.expression_instance_cache:
                expression = self.expression_instance_cache[field]
            else:
                expression = eval(parse_field(field))
                self.expression_instance_cache[field] = expression
        except NameError as e:
            get_module_logger("data").exception(
                "ERROR: field [%s] contains invalid operator/variable [%s]" % (str(field), str(e).split()[1])
            )
            raise
        except (SyntaxError, TypeError):
            get_module_logger("data").exception("ERROR: field [%s] contains invalid syntax" % str(field))
            raise
        return expression

    @abc.abstractmethod
    def expression(self, instrument, field, start_time=None, end_time=None, freq="day", **kwargs) -> pd.Series:
        """Get Expression data.

        The responsibility of `expression`
        - parse the `field` and `load` the according data.
        - When loading the data, it should handle the time dependency of the data. `get_expression_instance` is commonly used in this method

        Parameters
        ----------
        instrument : str
            a certain instrument.
        field : str
            a certain field of feature.
        start_time : str
            start of the time range.
        end_time : str
            end of the time range.
        freq : str
            time frequency, available: year/quarter/month/week/day.

        Returns
        -------
        pd.Series
            data of a certain expression

            The data has two types of format

            1) expression with datetime index

            2) expression with integer index

                - because the datetime is not as good as
        """
        raise NotImplementedError("Subclass of ExpressionProvider must implement `Expression` method")


class DatasetProvider(abc.ABC):
    """Dataset provider class

    Provide Dataset data.
    """

    @abc.abstractmethod
    def dataset(self, instruments, fields, start_time=None, end_time=None, freq="day", inst_processors=[], **kwargs):
        """Get dataset data.

        Parameters
        ----------
        instruments : list or dict
            list/dict of instruments or dict of stockpool config.
        fields : list
            list of feature instances.
        start_time : str
            start of the time range.
        end_time : str
            end of the time range.
        freq : str
            time frequency.
        inst_processors:  Iterable[Union[dict, InstProcessor]]
            the operations performed on each instrument

        Returns
        ----------
        pd.DataFrame
            a pandas dataframe with <instrument, datetime> index.
        """
        raise NotImplementedError("Subclass of DatasetProvider must implement `Dataset` method")

    @staticmethod
    def _uri(
        instruments,
        fields,
        start_time=None,
        end_time=None,
        freq="day",
        disk_cache=1,
        inst_processors=[],
        **_,
    ):
        """Get task uri, used when generating rabbitmq task in qlib_server

        Parameters
        ----------
        instruments : list or dict
            list/dict of instruments or dict of stockpool config.
        fields : list
            list of feature instances.
        start_time : str
            start of the time range.
        end_time : str
            end of the time range.
        freq : str
            time frequency.
        disk_cache : int
            whether to skip(0)/use(1)/replace(2) disk_cache.

        """
        # TODO: qlib-server support inst_processors
        return DiskDatasetCache._uri(instruments, fields, start_time, end_time, freq, disk_cache, inst_processors)

    @staticmethod
    def get_instruments_d(instruments, freq):
        """
        Parse different types of input instruments to output instruments_d
        Wrong format of input instruments will lead to exception.

        """
        if isinstance(instruments, dict):
            if "market" in instruments:
                # dict of stockpool config
                instruments_d = Inst.list_instruments(instruments=instruments, freq=freq, as_list=False)
            else:
                # dict of instruments and timestamp
                instruments_d = instruments
        elif isinstance(instruments, (list, tuple, pd.Index, np.ndarray)):
            # list or tuple of a group of instruments
            instruments_d = list(instruments)
        else:
            raise ValueError("Unsupported input type for param `instrument`")
        return instruments_d

    @staticmethod
    def get_column_names(fields):
        """
        Get column names from input fields

        """
        if len(fields) == 0:
            raise ValueError("fields cannot be empty")
        fields = fields.copy()
        column_names = [str(f) for f in fields]
        return column_names

    @staticmethod
    def parse_fields(fields):
        # parse and check the input fields
        return [ExpressionD.get_expression_instance(f) for f in fields]

    @staticmethod
    def _analysis_features(column_names):
        normalize_column_names = normalize_cache_fields(column_names)

        def _parse_col_name(field):
            feature_instance = ExpressionD.get_expression_instance(field)
            return field, feature_instance, feature_instance.get_extended_window_size(), 0

        feature_queue = deque(_parse_col_name(field) for field in reversed(normalize_column_names))
        all_sub_features = {}
        cs_level_summary = {}
        feature_extended_windows = {}
        while len(feature_queue) > 0:
            this_feature_name, this_feature, extended_window, this_cs_level = feature_queue.pop()
            cs_level_summary.setdefault(this_cs_level, {})[this_feature_name] = this_feature
            all_sub_features.setdefault(str(this_feature), set()).add(this_cs_level)
            feature_extended_windows.setdefault(str(this_feature), set()).add(extended_window)
            next_cs_level = this_cs_level + 1 if this_feature.require_cs_info else this_cs_level
            for next_feature in this_feature.get_direct_dependents():
                feature_queue.append((str(next_feature), next_feature, extended_window, next_cs_level))

        level_shared_features = {}
        for feature, levels in all_sub_features.items():
            if len(levels) > 1:
                level_shared_features.setdefault(max(levels), set()).add(feature)

        return normalize_column_names, cs_level_summary, level_shared_features, feature_extended_windows

    @staticmethod
    def dataset_processor(instruments_d, column_names, start_time, end_time, freq, inst_processors=[]):
        """
        Load and process the data, return the data set.
        - default using multi-kernel method.

        """
        (
            normalize_column_names,
            cs_level_summary,
            level_shared_features,
            feature_extended_windows,
        ) = DatasetProvider._analysis_features(column_names)
        # One process for one task, so that the memory will be freed quicker.
        workers = max(min(C.get_kernels(freq), len(instruments_d)), 1)

        if isinstance(instruments_d, dict):
            if not (start_time is None and end_time is None):
                for inst in list(instruments_d):
                    spans = [s for s in instruments_d[inst] if start_time <= s[1] and s[0] <= end_time]
                    if not spans:
                        del instruments_d[inst]
                    else:
                        instruments_d[inst] = spans
            it = instruments_d.items()
        else:
            it = list(zip(instruments_d, [None] * len(instruments_d)))

        cs_levels = sorted(cs_level_summary, reverse=True)
        shared_data_cache = {}
        ts_cache = {}
        cs_cache = {}
        shared_mgr = None
        if len(cs_levels) > 1 and C["joblib_backend"] == "multiprocessing":  # pylint: disable=R1702
            shared_mgr = multiprocessing.Manager()
            shared_data_cache = shared_mgr.dict()

            for dep_level in cs_levels[:-1]:
                expressions = cs_level_summary[dep_level]
                level_shared_feature = level_shared_features.get(dep_level, set())
                cache_task_l = [
                    delayed(DatasetProvider.load_cache)(
                        inst,
                        start_time=start_time,
                        end_time=end_time,
                        freq=freq,
                        expressions=expressions,
                        feature_extended_windows=feature_extended_windows,
                        g_config=C,
                        population=instruments_d,
                        cache_data={**ts_cache.get(inst, {}), **cs_cache},
                        shared_cache=shared_data_cache,
                    )
                    for inst, _ in it
                ]
                cs_cache.clear()
                shared_data_cache.clear()
                result = ParallelExt(n_jobs=workers, backend=C.joblib_backend, maxtasksperchild=C.maxtasksperchild)(
                    cache_task_l
                )
                for inst_cache in result:
                    for k, v in inst_cache.items():
                        if k[0] in expressions:
                            if C["joblib_backend"] == "multiprocessing":
                                shared_data_cache[k] = v
                            else:
                                cs_cache[k] = v
                        elif k[0] in level_shared_feature:
                            ts_cache.setdefault(k[1], {})[k[0]] = v

        inst_l, task_l = zip(
            *list(
                (
                    inst,
                    delayed(DatasetProvider.inst_calculator)(
                        inst,
                        start_time=start_time,
                        end_time=end_time,
                        freq=freq,
                        column_names=normalize_column_names,
                        expressions=cs_level_summary[0],
                        spans=spans,
                        g_config=C,
                        inst_processors=inst_processors,
                        population=instruments_d,
                        cache_data={**ts_cache.get(inst, {}), **cs_cache},
                        shared_cache=shared_data_cache,
                    ),
                )
                for inst, spans in it
            )
        )

        data = dict(
            zip(
                inst_l,
                ParallelExt(n_jobs=workers, backend=C.joblib_backend, maxtasksperchild=C.maxtasksperchild)(task_l),
            )
        )

        if len(cs_levels) > 1 and C["joblib_backend"] == "multiprocessing":
            del shared_data_cache
            shared_mgr.shutdown()

        new_data = dict()
        for inst in sorted(data.keys()):
            if len(data[inst]) > 0:
                # NOTE: Python version >= 3.6; in versions after python3.6, dict will always guarantee the insertion order
                new_data[inst] = data[inst]

        if len(new_data) > 0:
            data = pd.concat(new_data, names=["instrument"], sort=False)
            data = DiskDatasetCache.cache_to_origin_data(data, column_names)
        else:
            data = pd.DataFrame(
                index=pd.MultiIndex.from_arrays([[], []], names=("instrument", "datetime")),
                columns=column_names,
                dtype=np.float32,
            )
        return data

    @staticmethod
    def load_cache(
        inst,
        start_time,
        end_time,
        freq,
        expressions,
        feature_extended_windows={},
        g_config=None,
        population={},
        shared_features=set(),
        cache_data=None,
        shared_cache=None,
    ):
        C.register_from_C(g_config)
        if cache_data is not None:
            H["f"].update(cache_data)
        if shared_cache is not None:
            H.create_shared_cache(shared_cache)
        for field, expression in expressions.items():
            #  The client does not have expression provider, the data will be loaded from cache using static method.
            for ext_windows in feature_extended_windows.get(str(expression), {(0, 0)}):
                ExpressionD.expression(
                    inst, expression, start_time, end_time, freq, instrument_d=population, extend_windows=ext_windows
                )

        obj = {}
        for k, v in H["f"].internal_data.items():
            if k[1] in inst and (k[0] in expressions or k[0] in shared_features):
                obj[k] = v
        return obj

    @staticmethod
    def inst_calculator(
        inst,
        start_time,
        end_time,
        freq,
        column_names,
        expressions,
        spans=None,
        g_config=None,
        inst_processors=(),
        population={},
        cache_data=None,
        shared_cache=None,
    ):
        """
        Calculate the expressions for **one** instrument, return a df result.
        If the expression has been calculated before, load from cache.

        return value: A data frame with index 'datetime' and other data columns.

        """
        # FIXME: Windows OS or MacOS using spawn: https://docs.python.org/3.8/library/multiprocessing.html?highlight=spawn#contexts-and-start-methods
        # NOTE: This place is compatible with windows, windows multi-process is spawn
        C.register_from_C(g_config)
        if cache_data is not None:
            H["f"].update(cache_data)

        if shared_cache is not None:
            H.create_shared_cache(shared_cache)

        obj = dict()
        for field in column_names:
            #  The client does not have expression provider, the data will be loaded from cache using static method.
            obj[field] = ExpressionD.expression(
                inst, expressions[field], start_time, end_time, freq, instrument_d=population
            )

        data = pd.DataFrame(obj)
        if not data.empty and not np.issubdtype(data.index.dtype, np.dtype("M")):
            # If the underlaying provides the data not in datatime formmat, we'll convert it into datetime format
            _calendar = Cal.calendar(freq=freq)
            data.index = _calendar[data.index.values.astype(int)]
        data.index.names = ["datetime"]

        if not data.empty and spans is not None:
            mask = np.zeros(len(data), dtype=bool)
            for begin, end in spans:
                mask |= (data.index >= begin) & (data.index <= end)
            data = data[mask]

        for _processor in inst_processors:
            if _processor:
                _processor_obj = init_instance_by_config(_processor, accept_types=InstProcessor)
                data = _processor_obj(data, instrument=inst)
        return data


class LocalCalendarProvider(CalendarProvider, ProviderBackendMixin):
    """Local calendar data provider class

    Provide calendar data from local data source.
    """

    def __init__(self, remote=False, backend={}):
        super().__init__()
        ProviderBackendMixin.__init__(self, backend)
        self.remote = remote
        # self.backend = backend

    def load_calendar(self, freq, future):
        """Load original calendar timestamp from file.

        Parameters
        ----------
        freq : str
            frequency of read calendar file.
        future: bool
        Returns
        ----------
        list
            list of timestamps
        """
        try:
            backend_obj = self.backend_obj(freq=freq, future=future).data
        except ValueError:
            if future:
                get_module_logger("data").warning(
                    f"load calendar error: freq={freq}, future={future}; return current calendar!"
                )
                get_module_logger("data").warning(
                    "You can get future calendar by referring to the following document: https://github.com/microsoft/qlib/blob/main/scripts/data_collector/contrib/README.md"
                )
                backend_obj = self.backend_obj(freq=freq, future=False).data
            else:
                raise

        return [pd.Timestamp(x) for x in backend_obj]


class LocalInstrumentProvider(InstrumentProvider, ProviderBackendMixin):
    """Local instrument data provider class

    Provide instrument data from local data source.
    """

    def __init__(self, backend={}) -> None:
        super().__init__()
        ProviderBackendMixin.__init__(self, backend)
        # self.backend = backend

    def _load_instruments(self, market, freq):
        return self.backend_obj(market=market, freq=freq).data

    def list_instruments(self, instruments, start_time=None, end_time=None, freq="day", as_list=False):
        market = instruments["market"]
        if market in H["i"]:
            _instruments = H["i"][market]
        else:
            _instruments = self._load_instruments(market, freq=freq)
            H["i"][market] = _instruments
        # strip
        # use calendar boundary
        cal = Cal.calendar(freq=freq)
        start_time = pd.Timestamp(start_time or cal[0])
        end_time = pd.Timestamp(end_time or cal[-1])
        _instruments_filtered = {
            inst: list(
                filter(
                    lambda x: x[0] <= x[1],
                    [(max(start_time, pd.Timestamp(x[0])), min(end_time, pd.Timestamp(x[1]))) for x in spans],
                )
            )
            for inst, spans in _instruments.items()
        }
        _instruments_filtered = {key: value for key, value in _instruments_filtered.items() if value}
        # filter
        filter_pipe = instruments["filter_pipe"]
        for filter_config in filter_pipe:
            from . import filter as F  # pylint: disable=C0415

            filter_t = getattr(F, filter_config["filter_type"]).from_config(filter_config)
            _instruments_filtered = filter_t(_instruments_filtered, start_time, end_time, freq)
        # as list
        if as_list:
            return list(_instruments_filtered)
        return _instruments_filtered


class LocalFeatureProvider(FeatureProvider, ProviderBackendMixin):
    """Local feature data provider class

    Provide feature data from local data source.
    """

    def __init__(self, remote=False, backend={}):
        super().__init__()
        self.remote = remote
        ProviderBackendMixin.__init__(self, backend)
        # self.backend = backend

    def feature(self, instrument, field, start_index, end_index, freq):
        # validate
        field = str(field)[1:]
        instrument = code_to_fname(instrument)
        return self.backend_obj(instrument=instrument, field=field, freq=freq)[start_index : end_index + 1]


class LocalPITProvider(PITProvider):
    # TODO: Add PIT backend file storage
    # NOTE: This class is not multi-threading-safe!!!!

    DATA_RECORDS = [
        ("date", C.pit_record_type["date"]),
        ("period", C.pit_record_type["period"]),
        ("value", C.pit_record_type["value"]),
        ("_next", C.pit_record_type["index"]),
    ]
    PERIOD_DTYPE = C.pit_record_type["period"]
    INDEX_DTYPE = C.pit_record_type["index"]

    def __init__(self):
        super().__init__()
        self._data = MemCacheLengthUnit(size_limit=20)
        self._indices = MemCacheLengthUnit(size_limit=20)
        self._series_cache = MemCacheLengthUnit(size_limit=5)

    def _load_data(self, instrument, field):
        if (instrument, field) not in self._data:
            index_path = C.dpm.get_data_uri() / "financial" / instrument.lower() / f"{field}.index"
            data_path = C.dpm.get_data_uri() / "financial" / instrument.lower() / f"{field}.data"
            if not (index_path.exists() and data_path.exists()):
                raise FileNotFoundError("No file is found.")
            self._data[(instrument, field)] = np.fromfile(data_path, dtype=self.DATA_RECORDS)

        return self._data[(instrument, field)]

    def _load_index(self, instrument, field):
        if (instrument, field) not in self._indices:
            index_path = C.dpm.get_data_uri() / "financial" / instrument.lower() / f"{field}.index"
            if not index_path.exists():
                raise FileNotFoundError("No index file found.")
            with open(index_path, "rb") as fi:
                (first_year,) = struct.unpack(self.PERIOD_DTYPE, fi.read(struct.calcsize(self.PERIOD_DTYPE)))
                all_periods = np.fromfile(fi, dtype=self.INDEX_DTYPE)
            self._indices[(instrument, field)] = (first_year, all_periods)

        return self._indices[(instrument, field)]

    def read_period_data(
        self, instrument, field, period, cur_date_int: int, pit_mode: str, last_period_index: int = None
    ):
        NAN_VALUE = C.pit_record_nan["value"]
        NAN_INDEX = C.pit_record_nan["index"]
        DATA_RECORDS_BYTES = struct.calcsize("".join(x[1] for x in self.DATA_RECORDS))

        # find the first index of linked revisions
        if last_period_index is None:
            first_year, all_periods = self._load_index(instrument, field)
            offset = get_period_offset(first_year, period, pit_mode)
            _next = all_periods[offset]
        else:
            _next = last_period_index

        prev_value = NAN_VALUE
        prev_next = _next

        data = self._load_data(instrument, field)
        while _next != NAN_INDEX:
            date, period, value, new_next = data[_next // DATA_RECORDS_BYTES]
            if date > cur_date_int:
                break
            prev_next = _next
            _next = new_next
            prev_value = value

        return prev_value, prev_next

    def period_feature(self, instrument, field, start_index, end_index, cur_time, period=None):
        if not isinstance(cur_time, pd.Timestamp):
            raise ValueError(
                f"Expected pd.Timestamp for `cur_time`, got '{cur_time}'. Advices: you can't query PIT data directly(e.g. '$$roewa_q'), you must use `P` operator to convert data to each day (e.g. 'P($$roewa_q)')"
            )

        assert end_index <= 0  # PIT don't support querying future data

        VALUE_DTYPE = C.pit_record_type["value"]

        field = str(field).lower()[2:]
        instrument = code_to_fname(instrument)

        _, pit_mode = field.rsplit("_", 1)
        if pit_mode not in {"a", "q", "m", "i"}:
            raise ValueError(
                """period field must have with following suffix:
            _a: annually data
            _q: quarterly data
            _m: monthly data
            _i: indefinite frequency data
            """
            )

        data = self._load_data(instrument, field)

        # find all revision periods before `cur_time`
        cur_time_int = int(cur_time.year) * 10000 + int(cur_time.month) * 100 + int(cur_time.day)
        loc = np.searchsorted(data["date"], cur_time_int, side="right")
        if loc <= 0:
            return pd.Series(dtype=C.pit_record_type["value"])

        if (instrument, field, start_index, end_index, cur_time, period) not in self._series_cache:
            period_list = get_period_list(data["period"][:loc], pit_mode)
            if period is not None:
                # NOTE: `period` has higher priority than `start_index` & `end_index`
                if period not in period_list:
                    return pd.Series(dtype=C.pit_record_type["value"])
                else:
                    period_list = [period]
            else:
                period_list = period_list[max(0, len(period_list) + start_index - 1) : len(period_list) + end_index]
            value = np.full((len(period_list),), np.nan, dtype=VALUE_DTYPE)
            for i, p in enumerate(period_list):
                value[i], now_period_index = self.read_period_data(
                    instrument, field, p, cur_time_int, pit_mode  # , last_period_index  # For acceleration
                )

            # NOTE: the index is period_list; So it may result in unexpected values(e.g. nan)
            # when calculation between different features and only part of its financial indicator is published
            self._series_cache[(instrument, field, start_index, end_index, cur_time, period)] = pd.Series(
                value, index=period_list, dtype=VALUE_DTYPE
            )
        return self._series_cache[(instrument, field, start_index, end_index, cur_time, period)]


class LocalExpressionProvider(ExpressionProvider):
    """Local expression data provider class

    Provide expression data from local data source.
    """

    def __init__(self, time2idx=True):
        super().__init__()
        self.time2idx = time2idx

    def expression(
        self, instrument, expression, start_time=None, end_time=None, freq="day", instrument_d={}, extend_windows=(0, 0)
    ):
        if isinstance(expression, str):
            expression = self.get_expression_instance(expression)

        start_time = time_to_slc_point(start_time)
        end_time = time_to_slc_point(end_time)

        # Two kinds of queries are supported
        # - Index-based expression: this may save a lot of memory because the datetime index is not saved on the disk
        # - Data with datetime index expression: this will make it more convenient to integrating with some existing databases
        if self.time2idx:
            _, _, start_index, end_index = Cal.locate_index(start_time, end_time, freq=freq, future=False)
            lft_etd, rght_etd = expression.get_extended_window_size()
            lft_etd = max(extend_windows[0], lft_etd)
            rght_etd = max(extend_windows[1], rght_etd)
            query_start, query_end = max(0, start_index - lft_etd), end_index + rght_etd
            if isinstance(instrument_d, dict):
                instrument_d = {
                    inst: [Cal.locate_index(span[0], span[1], freq=freq, future=False)[2:] for span in spans]
                    for inst, spans in instrument_d.items()
                }
        else:
            start_index, end_index = query_start, query_end = start_time, end_time

        if isinstance(expression, ExpressionOps):
            expression.set_population(instrument_d)

        try:
            series = expression.load(instrument, query_start, query_end, freq)
        except Exception as e:
            get_module_logger("data").debug(
                f"Loading expression error: "
                f"instrument={instrument}, expression=({str(expression)}), start_time={start_time}, end_time={end_time},"
                f" freq={freq}. "
                f"error info: {str(e)}"
            )
            raise
        # Ensure that each column type is consistent
        # FIXME:
        # 1) The stock data is currently float. If there is other types of data, this part needs to be re-implemented.
        # 2) The precision should be configurable
        try:
            series = series.astype(np.float32)
        except ValueError:
            pass
        except TypeError:
            pass
        if not series.empty:
            series = series.loc[start_index:end_index]
        return series


class LocalDatasetProvider(DatasetProvider):
    """Local dataset data provider class

    Provide dataset data from local data source.
    """

    def __init__(self, align_time: bool = True):
        """
        Parameters
        ----------
        align_time : bool
            Will we align the time to calendar
            the frequency is flexible in some dataset and can't be aligned.
            For the data with fixed frequency with a shared calendar, the aligned data to the calendar will provide following benefits

            - Align queries to the same parameters, so the cache can be shared.
        """
        super().__init__()
        self.align_time = align_time

    def dataset(self, instruments, fields, start_time=None, end_time=None, freq="day", inst_processors=[], **_):
        instruments_d = self.get_instruments_d(instruments, freq)
        column_names = self.get_column_names(fields)
        if self.align_time:
            # NOTE: if the frequency is a fixed value.
            # align the data to fixed calendar point
            cal = Cal.calendar(start_time, end_time, freq)
            if len(cal) == 0:
                return pd.DataFrame(
                    index=pd.MultiIndex.from_arrays([[], []], names=("instrument", "datetime")), columns=column_names
                )
            start_time = cal[0]
            end_time = cal[-1]
        data = self.dataset_processor(
            instruments_d, column_names, start_time, end_time, freq, inst_processors=inst_processors
        )

        return data

    @staticmethod
    def multi_cache_walker(instruments, fields, start_time=None, end_time=None, freq="day"):
        """
        This method is used to prepare the expression cache for the client.
        Then the client will load the data from expression cache by itself.

        """
        instruments_d = DatasetProvider.get_instruments_d(instruments, freq)
        column_names = DatasetProvider.get_column_names(fields)
        cal = Cal.calendar(start_time, end_time, freq)
        if len(cal) == 0:
            return
        start_time = cal[0]
        end_time = cal[-1]
        workers = max(min(C.kernels, len(instruments_d)), 1)

        ParallelExt(n_jobs=workers, backend=C.joblib_backend, maxtasksperchild=C.maxtasksperchild)(
            delayed(LocalDatasetProvider.cache_walker)(inst, start_time, end_time, freq, column_names)
            for inst in instruments_d
        )

    @staticmethod
    def cache_walker(inst, start_time, end_time, freq, column_names):
        """
        If the expressions of one instrument haven't been calculated before,
        calculate it and write it into expression cache.

        """
        for field in column_names:
            ExpressionD.expression(inst, field, start_time, end_time, freq)


class ClientCalendarProvider(CalendarProvider):
    """Client calendar data provider class

    Provide calendar data by requesting data from server as a client.
    """

    def __init__(self):
        self.conn = None
        self.queue = queue.Queue()

    def set_conn(self, conn):
        self.conn = conn

    def calendar(self, start_time=None, end_time=None, freq="day", future=False):
        self.conn.send_request(
            request_type="calendar",
            request_content={"start_time": str(start_time), "end_time": str(end_time), "freq": freq, "future": future},
            msg_queue=self.queue,
            msg_proc_func=lambda response_content: [pd.Timestamp(c) for c in response_content],
        )
        result = self.queue.get(timeout=C["timeout"])
        return result


class ClientInstrumentProvider(InstrumentProvider):
    """Client instrument data provider class

    Provide instrument data by requesting data from server as a client.
    """

    def __init__(self):
        self.conn = None
        self.queue = queue.Queue()

    def set_conn(self, conn):
        self.conn = conn

    def list_instruments(self, instruments, start_time=None, end_time=None, freq="day", as_list=False):
        def inst_msg_proc_func(response_content):
            if isinstance(response_content, dict):
                instrument = {
                    i: [(pd.Timestamp(s), pd.Timestamp(e)) for s, e in t] for i, t in response_content.items()
                }
            else:
                instrument = response_content
            return instrument

        self.conn.send_request(
            request_type="instrument",
            request_content={
                "instruments": instruments,
                "start_time": str(start_time),
                "end_time": str(end_time),
                "freq": freq,
                "as_list": as_list,
            },
            msg_queue=self.queue,
            msg_proc_func=inst_msg_proc_func,
        )
        result = self.queue.get(timeout=C["timeout"])
        if isinstance(result, Exception):
            raise result
        get_module_logger("data").debug("get result")
        return result


class ClientDatasetProvider(DatasetProvider):
    """Client dataset data provider class

    Provide dataset data by requesting data from server as a client.
    """

    def __init__(self):
        self.conn = None

    def set_conn(self, conn):
        self.conn = conn
        self.queue = queue.Queue()

    def dataset(
        self,
        instruments,
        fields,
        start_time=None,
        end_time=None,
        freq="day",
        return_uri=False,
        inst_processors=[],
        **kwargs,
    ):
        disk_cache = kwargs.pop("disk_cache", 0)
        if Inst.get_inst_type(instruments) == Inst.DICT:
            get_module_logger("data").warning(
                "Getting features from a dict of instruments is not recommended because the features will not be "
                "cached! "
                "The dict of instruments will be cleaned every day."
            )

        if disk_cache == 0:
            """
            Call the server to generate the expression cache.
            Then load the data from the expression cache directly.
            - default using multi-kernel method.

            """
            self.conn.send_request(
                request_type="feature",
                request_content={
                    "instruments": instruments,
                    "fields": fields,
                    "start_time": start_time,
                    "end_time": end_time,
                    "freq": freq,
                    "disk_cache": 0,
                },
                msg_queue=self.queue,
            )
            feature_uri = self.queue.get(timeout=C["timeout"])
            if isinstance(feature_uri, Exception):
                raise feature_uri
            else:
                instruments_d = self.get_instruments_d(instruments, freq)
                column_names = self.get_column_names(fields)
                cal = Cal.calendar(start_time, end_time, freq)
                if len(cal) == 0:
                    return pd.DataFrame(
                        index=pd.MultiIndex.from_arrays([[], []], names=("instrument", "datetime")),
                        columns=column_names,
                    )
                start_time = cal[0]
                end_time = cal[-1]

                data = self.dataset_processor(instruments_d, column_names, start_time, end_time, freq, inst_processors)
                if return_uri:
                    return data, feature_uri
                else:
                    return data
        else:

            """
            Call the server to generate the data-set cache, get the uri of the cache file.
            Then load the data from the file on NFS directly.
            - using single-process implementation.

            """
            # TODO: support inst_processors, need to change the code of qlib-server at the same time
            # FIXME: The cache after resample, when read again and intercepted with end_time, results in incomplete data date
            if inst_processors:
                raise ValueError(
                    f"{self.__class__.__name__} does not support inst_processor. "
                    f"Please use `D.features(disk_cache=0)` or `qlib.init(dataset_cache=None)`"
                )
            self.conn.send_request(
                request_type="feature",
                request_content={
                    "instruments": instruments,
                    "fields": fields,
                    "start_time": start_time,
                    "end_time": end_time,
                    "freq": freq,
                    "disk_cache": 1,
                },
                msg_queue=self.queue,
            )
            # - Done in callback
            feature_uri = self.queue.get(timeout=C["timeout"])
            if isinstance(feature_uri, Exception):
                raise feature_uri
            get_module_logger("data").debug("get result")
            try:
                # pre-mound nfs, used for demo
                mnt_feature_uri = C.dpm.get_data_uri(freq).joinpath(C.dataset_cache_dir_name, feature_uri)
                df = DiskDatasetCache.read_data_from_cache(mnt_feature_uri, start_time, end_time, fields)
                get_module_logger("data").debug("finish slicing data")
                if return_uri:
                    return df, feature_uri
                return df
            except AttributeError as attribute_e:
                raise IOError("Unable to fetch instruments from remote server!") from attribute_e


class BaseProvider:
    """Local provider class
    It is a set of interface that allow users to access data.
    Because PITD is not exposed publicly to users, so it is not included in the interface.

    To keep compatible with old qlib provider.
    """

    @staticmethod
    def calendar(start_time=None, end_time=None, freq="day", future=False):
        return Cal.calendar(start_time, end_time, freq, future=future)

    @staticmethod
    def instruments(market="all", filter_pipe=None, start_time=None, end_time=None):
        if start_time is not None or end_time is not None:
            get_module_logger("Provider").warning(
                "The instruments corresponds to a stock pool. "
                "Parameters `start_time` and `end_time` does not take effect now."
            )
        return InstrumentProvider.instruments(market, filter_pipe)

    @staticmethod
    def list_instruments(instruments, start_time=None, end_time=None, freq="day", as_list=False):
        return Inst.list_instruments(instruments, start_time, end_time, freq, as_list)

    @staticmethod
    def features(
        instruments,
        fields,
        start_time=None,
        end_time=None,
        freq="day",
        disk_cache=None,
        inst_processors=[],
    ):
        """
        Parameters
        ----------
        inst_processors
        freq
        end_time
        start_time
        fields
        instruments
        disk_cache : int
            whether to skip(0)/use(1)/replace(2) disk_cache


        This function will try to use cache method which has a keyword `disk_cache`,
        and will use provider method if a type error is raised because the DatasetD instance
        is a provider class.
        """
        disk_cache = C.default_disk_cache if disk_cache is None else disk_cache
        fields = list(fields)  # In case of tuple.
        try:
            return DatasetD.dataset(
                instruments, fields, start_time, end_time, freq, inst_processors=inst_processors, disk_cache=disk_cache
            )
        except TypeError:
            return DatasetD.dataset(instruments, fields, start_time, end_time, freq, inst_processors=inst_processors)


class LocalProvider(BaseProvider):
    @staticmethod
    def _uri(type_, **kwargs):
        """_uri
        The server hope to get the uri of the request. The uri will be decided
        by the dataprovider. For ex, different cache layer has different uri.

        :param type: The type of resource for the uri
        :param **kwargs:
        """
        if type_ == "calendar":
            return Cal._uri(**kwargs)
        elif type_ == "instrument":
            return Inst._uri(**kwargs)
        elif type_ == "feature":
            return DatasetD._uri(**kwargs)

    @staticmethod
    def features_uri(instruments, fields, start_time, end_time, freq, disk_cache=1):
        """features_uri

        Return the uri of the generated cache of features/dataset

        :param disk_cache:
        :param instruments:
        :param fields:
        :param start_time:
        :param end_time:
        :param freq:
        """
        return DatasetD._dataset_uri(instruments, fields, start_time, end_time, freq, disk_cache)


class ClientProvider(BaseProvider):
    """Client Provider

    Requesting data from server as a client. Can propose requests:

        - Calendar : Directly respond a list of calendars
        - Instruments (without filter): Directly respond a list/dict of instruments
        - Instruments (with filters):  Respond a list/dict of instruments
        - Features : Respond a cache uri

    The general workflow is described as follows:
    When the user use client provider to propose a request, the client provider will connect the server and send the request. The client will start to wait for the response. The response will be made instantly indicating whether the cache is available. The waiting procedure will terminate only when the client get the response saying `feature_available` is true.
    `BUG` : Everytime we make request for certain data we need to connect to the server, wait for the response and disconnect from it. We can't make a sequence of requests within one connection. You can refer to https://python-socketio.readthedocs.io/en/latest/client.html for documentation of python-socketIO client.
    """

    def __init__(self):
        def is_instance_of_provider(instance: object, cls: type):
            if isinstance(instance, Wrapper):
                p = getattr(instance, "_provider", None)

                return False if p is None else isinstance(p, cls)

            return isinstance(instance, cls)

        from .client import Client  # pylint: disable=C0415

        self.client = Client(C.flask_server, C.flask_port)
        self.logger = get_module_logger(self.__class__.__name__)
        if is_instance_of_provider(Cal, ClientCalendarProvider):
            Cal.set_conn(self.client)
        if is_instance_of_provider(Inst, ClientInstrumentProvider):
            Inst.set_conn(self.client)
        if hasattr(DatasetD, "provider"):
            DatasetD.provider.set_conn(self.client)
        else:
            DatasetD.set_conn(self.client)


import sys

if sys.version_info >= (3, 9):
    from typing import Annotated

    CalendarProviderWrapper = Annotated[CalendarProvider, Wrapper]
    InstrumentProviderWrapper = Annotated[InstrumentProvider, Wrapper]
    FeatureProviderWrapper = Annotated[FeatureProvider, Wrapper]
    PITProviderWrapper = Annotated[PITProvider, Wrapper]
    ExpressionProviderWrapper = Annotated[ExpressionProvider, Wrapper]
    DatasetProviderWrapper = Annotated[DatasetProvider, Wrapper]
    BaseProviderWrapper = Annotated[BaseProvider, Wrapper]
else:
    CalendarProviderWrapper = CalendarProvider
    InstrumentProviderWrapper = InstrumentProvider
    FeatureProviderWrapper = FeatureProvider
    PITProviderWrapper = PITProvider
    ExpressionProviderWrapper = ExpressionProvider
    DatasetProviderWrapper = DatasetProvider
    BaseProviderWrapper = BaseProvider

Cal: CalendarProviderWrapper = Wrapper()
Inst: InstrumentProviderWrapper = Wrapper()
FeatureD: FeatureProviderWrapper = Wrapper()
PITD: PITProviderWrapper = Wrapper()
ExpressionD: ExpressionProviderWrapper = Wrapper()
DatasetD: DatasetProviderWrapper = Wrapper()
D: BaseProviderWrapper = Wrapper()


def register_all_wrappers(C):
    """register_all_wrappers"""
    logger = get_module_logger("data")
    module = get_module_by_module_path("qlib.data")

    _calendar_provider = init_instance_by_config(C.calendar_provider, module)
    if getattr(C, "calendar_cache", None) is not None:
        _calendar_provider = init_instance_by_config(C.calendar_cache, module, provide=_calendar_provider)
    register_wrapper(Cal, _calendar_provider, "qlib.data")
    logger.debug(f"registering Cal {C.calendar_provider}-{C.calendar_cache}")

    _instrument_provider = init_instance_by_config(C.instrument_provider, module)
    register_wrapper(Inst, _instrument_provider, "qlib.data")
    logger.debug(f"registering Inst {C.instrument_provider}")

    if getattr(C, "feature_provider", None) is not None:
        feature_provider = init_instance_by_config(C.feature_provider, module)
        register_wrapper(FeatureD, feature_provider, "qlib.data")
        logger.debug(f"registering FeatureD {C.feature_provider}")

    if getattr(C, "pit_provider", None) is not None:
        pit_provider = init_instance_by_config(C.pit_provider, module)
        register_wrapper(PITD, pit_provider, "qlib.data")
        logger.debug(f"registering PITD {C.pit_provider}")

    if getattr(C, "expression_provider", None) is not None:
        # This provider is unnecessary in client provider
        _eprovider = init_instance_by_config(C.expression_provider, module)
        if getattr(C, "expression_cache", None) is not None:
            _eprovider = init_instance_by_config(C.expression_cache, module, provider=_eprovider)
        register_wrapper(ExpressionD, _eprovider, "qlib.data")
        logger.debug(f"registering ExpressionD {C.expression_provider}-{C.expression_cache}")

    _dprovider = init_instance_by_config(C.dataset_provider, module)
    if getattr(C, "dataset_cache", None) is not None:
        _dprovider = init_instance_by_config(C.dataset_cache, module, provider=_dprovider)
    register_wrapper(DatasetD, _dprovider, "qlib.data")
    logger.debug(f"registering DatasetD {C.dataset_provider}-{C.dataset_cache}")

    register_wrapper(D, C.provider, "qlib.data")
    logger.debug(f"registering D {C.provider}")
