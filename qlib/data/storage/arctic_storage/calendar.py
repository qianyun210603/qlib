from .base import ArcticStorageMixin
from typing import Iterable, List, Union
import pandas as pd
import numpy as np
from qlib.data.storage import CalendarStorage, CalVT


class ArcticCalendarStorage(ArcticStorageMixin, CalendarStorage):
    def __init__(self, freq: str, future: bool, calendar_type: str = None):
        super().__init__(freq, future)
        self.calendar_type = calendar_type
        self.arctic_stroe = self._get_arctic_store()
        self.cal_lib = self.arctic_stroe.get_library("trading_calendar")
        self.cal_symbol = f"{self.calendar_type}_{self.freq}{'' if not self.future else '_future'}"
        self._calendar = [pd.Timestamp(x).tz_localize(None) for x in self.cal_lib.read(self.cal_symbol)]

    def _write_calendar(self, values: List[CalVT]) -> None:
        self.cal_lib.write(self.cal_symbol, values)

    @property
    def data(self) -> List[CalVT]:
        _calendar = self._calendar
        # TODO: resample calendar
        # if Freq(self._freq_file) != Freq(self.freq):
        #     _calendar = resam_calendar(
        #         np.array(list(map(pd.Timestamp, _calendar))), self._freq_file, self.freq, self.region
        #     )
        return _calendar

    def _get_storage_freq(self) -> List[str]:
        storage_freqs = [sym.split("_")[1] for sym in self.cal_lib.list_symbols() if sym.startswith(self.calendar_type)]
        return storage_freqs

    # def extend(self, values: Iterable[CalVT]) -> None:
    #     self._write_calendar(values, mode="ab")

    def clear(self) -> None:
        self._write_calendar(values=[])

    def index(self, value: CalVT) -> int:
        calendar = self._calendar
        return int(np.argwhere(calendar == value)[0])

    def insert(self, index: int, value: CalVT):
        self.cal_lib.insert(self.cal_symbol, index, value)
        self._calendar = self.cal_lib.read(self.cal_symbol)

    def remove(self, value: CalVT) -> None:
        self.cal_lib.remove(value)
        self._calendar = self.cal_lib.read(self.cal_symbol)

    def __setitem__(self, i: Union[int, slice], values: Union[CalVT, Iterable[CalVT]]) -> None:
        if isinstance(i, int):
            self.cal_lib.set_nth_element(self.cal_symbol, i, values)
            self._calendar = self.cal_lib.read(self.cal_symbol)
        elif isinstance(i, slice):
            self._calendar[i] = values
            self._write_calendar(self._calendar)

    def __delitem__(self, i: Union[int, slice]) -> None:
        calendar = self._calendar
        calendar = np.delete(calendar, i)
        self._write_calendar(values=calendar)

    def __getitem__(self, i: Union[int, slice]) -> Union[CalVT, List[CalVT]]:
        return self._calendar[i]

    def __len__(self) -> int:
        return len(self.data)
