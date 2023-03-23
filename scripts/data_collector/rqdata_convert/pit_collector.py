# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import sys
import importlib
from datetime import datetime
from pathlib import Path
from typing import List, Iterable, Optional, Union, cast

import fire
import pandas as pd
from loguru import logger
from vnpy.trader.database import get_database
from vnpy_arctic.arctic_database import ArcticDatabase

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent.parent))

from data_collector.base import BaseCollector, BaseRun, BaseNormalize
from data_collector.utils import get_calendar_list
from dump_pit import DumpPitData

class RqPitCollector(BaseCollector):
    DATE_FIELDS_IN_META = [
        'value_date', 'conversion_start_date', 'conversion_end_date', 'de_listed_date', 'listed_date',
        'maturity_date', 'stop_trading_date', 'callback_start_date', 'callback_end_date', 'putback_start_date',
        'putback_end_date', "no_call_peroid_end"
    ]
    REMOVE_FIELDS_IN_META = ["name", "short_name", "symbol", "exchange", "stock_code", "type", "round_lot",
                             "stock_exchange"]

    INTERVAL_QUARTERLY = "quarterly"
    INTERVAL_ANNUAL = "annual"
    INTERVAL_MONTH = "monthly"
    INTERVAL_INDEFINITE = "indefinite"

    def __init__(
        self,
        save_dir: Union[str, Path],
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "quarterly",
        max_workers: int = 1,
        max_collector_count: int = 1,
        delay: int = 0,
        check_data_length: bool = False,
        limit_nums: Optional[int] = None,
        symbol_regex: Optional[str] = None,
    ):

        self.symbol_regex = symbol_regex
        db_mgr = get_database()
        self.arctic_store = cast(ArcticDatabase, db_mgr).connection
        self.meta_lib = self.arctic_store.get_library("convert_meta")
        self.convert_price_lib = self.arctic_store.get_library("convert_convert_price")
        self.convert_cf_lib = self.arctic_store.get_library("convert_cash_flow")
        self.coupon_lib = self.arctic_store.get_library("convert_coupon")

        self.syms_with_meta = {}

        super().__init__(
            save_dir=save_dir,
            start=self.convert_datetime(start, self._timezone),
            end=self.convert_datetime(end, self._timezone),
            interval=interval,
            max_workers=max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        )

    def get_instrument_list(self) -> List[str]:

        def symbol_validation(_, v):
            if pd.isna(v['listed_date']):
                return False
            if pd.isna(v['de_listed_date']):
                return v['listed_date'] < self.end_datetime.normalize()
            stop_trading_date = v.get('stop_trading_date', None)
            if stop_trading_date is None:
                stop_trading_date = v['de_listed_date']
            # print(self.start_datetime, stop_trading_date, v)
            return self.start_datetime <= stop_trading_date and v['listed_date'] < \
                   min(self.end_datetime, pd.Timestamp.now('Asia/Shanghai')).normalize()

        logger.info("get HS converts symbols......")
        self.syms_with_meta = {x: self.meta_lib.read_history(x) for x in self.meta_lib.list_symbols()}
        symbols = [k for k, v in self.syms_with_meta.items() if symbol_validation(k, v.iloc[-1, 0])]
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    @staticmethod
    def normalize_symbol(symbol):
        symbol_s = symbol.split("_")
        symbol = f"sh{symbol_s[0]}" if symbol_s[-1] == "SSE" else f"sz{symbol_s[0]}"
        return symbol

    @property
    def _timezone(self):
        return "Asia/Shanghai"

    @staticmethod
    def convert_datetime(dt: Union[pd.Timestamp, str], timezone):
        if isinstance(dt, str):
            return pd.Timestamp(dt, tz=timezone)
        if dt.tzinfo is None:
            return dt.tz_localize(timezone)
        return dt.tz_convert(timezone)

    def get_meta(self, symbol, start_date: str, end_date: str):
        meta_hist = self.syms_with_meta[symbol]
        meta_hist.index = meta_hist.index.normalize()
        meta_hist = meta_hist.loc[start_date:end_date].tz_localize(None)

        parsed_meta = {}
        for d, content_row in meta_hist.iterrows():
            meta_dict = {k: v for k, v in content_row.iloc[0].items() if k not in self.REMOVE_FIELDS_IN_META}
            if pd.isna(meta_dict['listed_date']):
                continue
            if pd.isna(meta_dict['stop_trading_date']):
                meta_dict['stop_trading_date'] = meta_dict['maturity_date']
            for f in self.DATE_FIELDS_IN_META:
                if f in meta_dict:
                    meta_dict[f] = (meta_dict[f] - pd.Timestamp(year=1970, month=1, day=1, tz=meta_dict[f].tzinfo)).days
                else:
                    meta_dict[f] = np.nan
            parsed_meta[d] = meta_dict

        if len(parsed_meta) < 1:
            logger.warning(f"{symbol} has no valid meta")
            return pd.DataFrame()

        min_d = min(parsed_meta.keys())
        first_meta_dict = parsed_meta.pop(min_d)
        parsed_meta[max(pd.Timestamp(first_meta_dict['listed_date'], unit='d'), pd.Timestamp("2010-01-01"))] = first_meta_dict

        df = pd.DataFrame.from_dict(parsed_meta, orient='index').ffill().unstack().to_frame('value').sort_index()\
            .reset_index(level=0).rename(columns={'level_0': 'field'}).drop_duplicates(keep='first')
        df.index.name = 'date'
        df["symbol"] = self.normalize_symbol(symbol)
        df["period"] = (df.index - pd.Timestamp(year=1970, month=1, day=1)).days
        df.reset_index(inplace=True)

        return df


    def get_coupon(self):
        pass


    def get_data(
        self,
        symbol: str,
        interval: str,
        start_datetime: pd.Timestamp,
        end_datetime: pd.Timestamp,
    ) -> pd.DataFrame:
        df = self.get_meta(symbol, start_datetime, end_datetime)

        return df


class RqPitNormalize(BaseNormalize):
    def __init__(self, interval: str = "quarterly", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interval = interval

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.interval == RqPitCollector.INTERVAL_INDEFINITE:
            return df
        df["period"] = pd.to_datetime(df["period"])
        df["period"] = df["period"].apply(
            lambda x: x.year if self.interval == RqPitCollector.INTERVAL_ANNUAL else x.year * 100 + (x.month - 1) // 3 + 1
        )
        return df

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return get_calendar_list()


class Run(BaseRun):

    @property
    def _cur_module(self):
        return importlib.import_module("pit_collector")

    @property
    def collector_class_name(self) -> str:
        return f"RqPitCollector"

    @property
    def normalize_class_name(self) -> str:
        return f"RqPitNormalize"

    @property
    def default_base_dir(self) -> [Path, str]:
        return BASE_DIR

    def update_data_to_bin(
        self,
        qlib_dir,
        interval = "indefinite",
        trading_date: Union[str, pd.Timestamp, datetime] = None,
        end_date: Union[str, pd.Timestamp, datetime] = None,
    ):
        """update Rqdata data to bin

        Parameters
        ----------
        qlib_dir: str
            the qlib data to be updated for Rqdata, usually from:
            https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data

        trading_date: [str, pd.Timestamp, datetime.datetime]
            trading days to be updated, by default ``datetime.datetime.now().strftime("%Y-%m-%d")``
        end_date: [str, pd.Timestamp, datetime.datetime]
            end datetime, default ``pd.Timestamp(trading_date + pd.Timedelta(days=1))``; open interval(excluding end)

        Notes
        -----
            If the data in qlib_data_dir is incomplete, np.nan will be populated to trading_date for the previous
            trading day

        Examples
        -------
            $ python collector.py update_data_to_bin --qlib_data_1d_dir <user data dir> --trading_date <start date>
            --end_date <end date>
            # get 1m data
        """

        # start/end date
        if end_date is None:
            end_date = (pd.Timestamp.now() + pd.Timedelta(days=1))
            logger.info(f"end_date not specified, use the tomorrow: {end_date}")

        if trading_date is None:
            trading_date = (end_date - pd.Timedelta(days=8))
            logger.info(f"trading_date is None, use the one week before: {trading_date}")

        # NOTE: a larger max_workers setting here would be faster
        self.max_workers = 1
        # download data from Rqdata
        # NOTE: when downloading data from RqdataFinance, max_workers is recommended to be 1
        self.download_data(
            max_collector_count=self.max_workers,
            start=trading_date,
            end=end_date,
        )

        # normalize data
        self.normalize_data(interval=interval)

        # dump bin
        qlib_dir = Path(qlib_dir).expanduser().resolve()

        _dump = DumpPitData(
            csv_path=self.normalize_dir,
            qlib_dir=qlib_dir,
            exclude_fields="symbol,date",
            max_workers=self.max_workers,
        )

        _dump.dump(interval=interval,)



if __name__ == "__main__":
    fire.Fire(Run)

