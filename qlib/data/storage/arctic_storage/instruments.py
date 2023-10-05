from .base import ArcticStorageMixin, db_symbol_to_qlib
from typing import Dict

import pandas as pd

from qlib.data.storage import InstKT, InstrumentStorage, InstVT
from qlib.log import get_module_logger

INDEX_MAPPING = {
    "csi050": "000016_SSE@COMPONENT",
    "csi300": "000300_SSE@COMPONENT",
    "csi500": "000905_SSE@COMPONENT",
    "csi1000": "000852_SSE@COMPONENT",
    "convert": "000832_SSE@COMPONENT",
}


class ArcticInstrumentStorage(ArcticStorageMixin, InstrumentStorage):
    def __init__(self, market: str, freq: str, **kwargs):
        super().__init__(market, freq, **kwargs)
        self.market = market
        self.inst_symbol = INDEX_MAPPING[market]
        self._instruments = None

    def _process_all(self, arctic_store):
        freq_suffix = "d" if self.freq == "day" else "1m"
        ov_lib = arctic_store.get_library("data_overview")
        if self.market == "cnstock_all":
            stock_meta_lib = arctic_store.get_library("stock_meta")
            population = stock_meta_lib.list_symbols()
        elif self.market == "cnconvert_all":
            convert_meta_lib = arctic_store.get_library("convert_meta")
            population = convert_meta_lib.list_symbols()
        else:
            raise NotImplementedError(f"market {self.market} not implemented")
        instruments = {}
        for db_symbol in population:
            tmp_sym = f"{db_symbol}_{freq_suffix}"
            if tmp_sym not in ov_lib.list_symbols():
                get_module_logger("arctic_storage").info(f"Skip {db_symbol} as no data")
                continue
            ov = ov_lib.read(tmp_sym)
            instruments[db_symbol_to_qlib(db_symbol)] = [
                [
                    pd.Timestamp(ov["start"]).tz_localize(None),
                    pd.Timestamp(ov["end"]).tz_localize(None).replace(hour=23, minute=59, second=59),
                ]
            ]
        return instruments

    def _read_instrument(self) -> Dict[InstKT, InstVT]:
        arctic_store = self._get_arctic_store()
        inst_lib = arctic_store.get_library("index_components")
        if self.market in ["cnstock_all", "cnconvert_all"]:
            return self._process_all(arctic_store)
        tmp = inst_lib.read(self.inst_symbol)

        def _parse_intervals(orig_intervals):
            intervals = []
            for interval in orig_intervals:
                if interval[0] == interval[1]:
                    continue
                intervals.append(
                    [
                        pd.Timestamp(interval[0]).tz_localize(None),
                        pd.Timestamp(interval[1]).tz_localize(None).replace(hour=23, minute=59, second=59),
                    ]
                )
            return intervals

        insts = {db_symbol_to_qlib(db_sym): _parse_intervals(intervals) for db_sym, intervals in tmp.items()}
        return insts

    def clear(self) -> None:
        raise NotImplementedError("ArcticInstrumentStorage is read-only")

    @property
    def data(self) -> Dict[InstKT, InstVT]:
        if self._instruments is None:
            self._instruments = self._read_instrument()
        return self._instruments

    def __setitem__(self, k: InstKT, v: InstVT) -> None:
        raise NotImplementedError("ArcticInstrumentStorage is read-only")

    def __delitem__(self, k: InstKT) -> None:
        raise NotImplementedError("ArcticInstrumentStorage is read-only")

    def __getitem__(self, k: InstKT) -> InstVT:
        return self._instruments[k]

    def update(self, *args, **kwargs) -> None:
        raise NotImplementedError("ArcticInstrumentStorage is read-only")

    def __len__(self) -> int:
        return len(self.data)
