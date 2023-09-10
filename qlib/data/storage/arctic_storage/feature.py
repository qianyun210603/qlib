import bisect
from .base import ArcticStorageMixin, qlib_symbol_to_db
from typing import List, Union
from arctic.date import DateRange
import numpy as np
import pandas as pd
from functools import lru_cache
from qlib.data import D

from qlib.data.storage import FeatureStorage


class ArcticFeatureStorage(ArcticStorageMixin, FeatureStorage):
    def __init__(self, instrument: str, field: str, freq: str, **kwargs):
        super().__init__(instrument, field, freq, **kwargs)
        self.arctic_store = self._get_arctic_store()
        self.alias_lib = self.arctic_store.get_library("feature_alias")
        self.field_lib_mapping = self.alias_lib.read("field_lib_mapping")
        self.name_alias = self.alias_lib.read("name_alias")
        self.name_inst_alias = self.alias_lib.read("name_inst_alias")
        self.db_field, self.db_inst, self.lib = self._get_arctic_path()

    @lru_cache
    def _get_inst_mapping(self, inst_mapping_name):
        inst_mapping = self.alias_lib.read(inst_mapping_name)
        return inst_mapping

    def _get_arctic_path(self):
        freq_suffix = "d" if self.freq == "day" else "1m"
        base_db_symbol = qlib_symbol_to_db(self.instrument)
        if self.field in self.name_alias:
            db_field = self.name_alias[self.field]
            db_inst = base_db_symbol
            lib = self.arctic_store.get_library(self.field_lib_mapping[db_field])
        elif self.field in self.name_inst_alias:
            db_field, inst_mapping_name = self.name_inst_alias[self.field]
            inst_mapping = self._get_inst_mapping(inst_mapping_name)
            db_inst = inst_mapping[base_db_symbol]
            lib = self.arctic_store.get_library(self.field_lib_mapping[db_field])
        else:
            db_field = self.field
            db_inst = base_db_symbol
            lib = self.arctic_store.get_library(self.field_lib_mapping[db_field])
        if self.field_lib_mapping[db_field] == "bar_data":
            db_inst = f"{db_inst}_{freq_suffix}"
        return db_field, db_inst, lib

    def clear(self):
        raise NotImplementedError("ArcticFeatureStorage is read-only")

    @property
    def data(self) -> pd.Series:
        return self[:]

    def write(self, data_array: Union[List, np.ndarray], index: int = None) -> None:
        raise NotImplementedError("ArcticFeatureStorage is read-only")

    @lru_cache(maxsize=1)
    def _get_chunk_ranges(self):
        ll = sorted(
            (pd.Timestamp(s.decode()).tz_convert(None), pd.Timestamp(e.decode()).tz_convert(None))
            for s, e in self.lib.get_chunk_ranges(self.db_inst)
        )
        return ll

    @property
    @lru_cache(maxsize=1)
    def start_index(self) -> Union[pd.Timestamp, None]:
        if self.lib.has_symbol(self.db_inst):
            ll = self._get_chunk_ranges()
            if len(ll) == 0:
                raise ValueError(f"{self.instrument} has no data")
            for lll in ll:
                df_min = self.lib.read(self.db_inst, chunk_range=DateRange(*lll))
                if not df_min.index.empty:
                    return df_min.index.min()
        return pd.Timestamp("1970-01-01")

    @property
    @lru_cache(maxsize=1)
    def end_index(self) -> Union[pd.Timestamp, None]:
        if self.lib.has_symbol(self.db_inst):
            ll = self._get_chunk_ranges()
            if len(ll) == 0:
                raise ValueError(f"{self.instrument} has no data")
            for lll in reversed(ll):
                df_max = self.lib.read(self.db_inst, chunk_range=DateRange(*lll))
                if not df_max.index.empty:
                    return df_max.index.max()
        return pd.Timestamp("2099-01-01")

    def _read_factor(self, start_index: pd.Timestamp, end_index: pd.Timestamp) -> pd.Series:
        inferred_index = pd.DatetimeIndex(
            D.calendar(start_time=start_index, end_time=end_index, freq=self.freq, future=True)
        )
        if not self.lib.has_symbol(self.db_inst):
            return pd.Series(1.0, index=inferred_index)
        ll = sorted(s for s, e in self._get_chunk_ranges())
        if len(ll) == 0:
            return pd.Series(1.0, index=inferred_index)
        idx = bisect.bisect(ll, start_index)
        if idx == 0:
            df = self.lib.read(self.db_inst, chunk_range=DateRange(start_index, end_index), columns=[self.db_field])
            series = df[self.db_field]
            series = series.reindex(inferred_index, method="ffill").fillna(1.0)
            return series
        df = self.lib.read(self.db_inst, chunk_range=DateRange(ll[idx - 1], end_index), columns=[self.db_field])
        series = df[self.db_field]
        series = series.reindex(inferred_index, method="ffill")
        return series

    # TODO: make vwap directly available
    def _read_vwap(self, start_index: pd.Timestamp, end_index: pd.Timestamp) -> pd.Series:
        df = self.lib.read(
            self.db_inst, chunk_range=DateRange(start_index, end_index), columns=["close_price", "volume", "turnover"]
        )
        series = df.apply(lambda x: x["turnover"] / x["volume"] if x["volume"] > 0 else x["close_price"], axis=1)
        return series

    def __getitem__(self, i: Union[pd.Timestamp, slice]) -> pd.Series:
        start_index = max(self.start_index, i.start)
        end_index = min(self.end_index, i.stop)
        if self.field == "factor":
            return self._read_factor(start_index, end_index)
        if self.field == "vwap":
            return self._read_vwap(start_index, end_index)
        if not self.lib.has_symbol(self.db_inst):
            raise KeyError(f"{self.instrument} not found in {self.lib}")
        df = self.lib.read(self.db_inst, chunk_range=DateRange(start_index, end_index), columns=[self.db_field])
        series = df[self.db_field]
        return series

    def __len__(self) -> int:
        info = self.lib.get_info(self.db_inst)
        return info["len"]
