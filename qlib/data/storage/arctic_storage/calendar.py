from .base import ArcticStorageMixin
from typing import Iterable, List, Union
import pandas as pd
import numpy as np
from qlib.data.storage import CalendarStorage, CalVT


class ArcticCalendarStorage(ArcticStorageMixin, CalendarStorage):
    def __init__(self, freq: str, future: bool, calendar_type: str = None):
        super().__init__(freq, future)
        self.calendar_type = calendar_type
        self.cal_symbol = f"{self.calendar_type}_{self.freq}{'' if not self.future else '_future'}"
        self._calendar = None
        self._callib = None
        self._pid = -1

    @property
    def data(self) -> Iterable[CalVT]:
        if self._calendar is None:
            self._calendar = self._read_calendar()
        return self._calendar

    def get_cal_lib(self):
        arctic_stroe = self._get_arctic_store()
        _callib = arctic_stroe.get_library("trading_calendar")
        return _callib

    def _read_calendar(self) -> List[CalVT]:
        cal_lib = self.get_cal_lib()
        return [pd.Timestamp(x).tz_localize(None) for x in cal_lib.read(self.cal_symbol)]

    def _write_calendar(self, values: List[CalVT]) -> None:
        cal_lib = self.get_cal_lib()
        cal_lib.write(self.cal_symbol, values)

    def _get_storage_freq(self) -> List[str]:
        cal_lib = self.get_cal_lib()
        storage_freqs = [sym.split("_")[1] for sym in cal_lib.list_symbols() if sym.startswith(self.calendar_type)]
        return storage_freqs

    def clear(self) -> None:
        self._write_calendar(values=[])

    def index(self, value: CalVT) -> int:
        calendar = self._read_calendar()
        return int(np.argwhere(calendar == value)[0])

    def insert(self, index: int, value: CalVT):
        cal_lib = self.get_cal_lib()
        cal_lib.insert(self.cal_symbol, index, value)
        self._calendar = cal_lib.read(self.cal_symbol)

    def extend(self, values: Iterable[CalVT]) -> None:
        cal_lib = self.get_cal_lib()
        cal_lib.extend(self.cal_symbol, values)

    def remove(self, value: CalVT) -> None:
        cal_lib = self.get_cal_lib()
        cal_lib.remove(value)
        self._calendar = cal_lib.read(self.cal_symbol)

    def __setitem__(self, i: Union[int, slice], values: Union[CalVT, Iterable[CalVT]]) -> None:
        cal_lib = self.get_cal_lib()
        if isinstance(i, int):
            cal_lib.set_nth_element(self.cal_symbol, i, values)
            self._calendar = cal_lib.read(self.cal_symbol)
        elif isinstance(i, slice):
            self._calendar[i] = values
            self._write_calendar(self._calendar)

    def __delitem__(self, i: Union[int, slice]) -> None:
        calendar = self._read_calendar()
        calendar = np.delete(calendar, i)
        self._write_calendar(values=calendar)

    def __getitem__(self, i: Union[int, slice]) -> Union[CalVT, List[CalVT]]:
        self._read_calendar()
        return self._calendar[i]

    def __len__(self) -> int:
        return len(self.data)
