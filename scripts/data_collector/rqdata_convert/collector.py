# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
import sys
import copy
import time
import datetime
import importlib
from abc import ABC
import multiprocessing
from pathlib import Path
from typing import Iterable
from arctic import Arctic
from arctic.date import DateRange

import fire
import requests
import numpy as np
import pandas as pd
from loguru import logger


from qlib.tests.data import GetData
from qlib.utils import code_to_fname, fname_to_code, exists_qlib_data
from qlib.constant import REG_CN as REGION_CN

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from dump_bin import DumpDataUpdate
from data_collector.base import BaseCollector, BaseNormalize, BaseRun, Normalize
from data_collector.utils import (
    deco_retry,
    get_calendar_list,
    get_hs_stock_symbols,
    get_us_stock_symbols,
    get_in_stock_symbols,
    generate_minutes_calendar_from_daily,
)

INDEX_BENCH_URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.{index_code}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg={begin}&end={end}"


class RqdataCollector(BaseCollector):

    def __init__(
        self,
        save_dir: [str, Path],
        start=None,
        end=None,
        interval="1d",
        max_workers=4,
        max_collector_count=2,
        delay=2,
        check_data_length: int = None,
        limit_nums: int = None,
    ):
        """

        Parameters
        ----------
        save_dir: str
            stock save dir
        max_workers: int
            workers, default 4
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0
        interval: str
            freq, value from ['d', '1d', '1m', '1min'], default d
        start: str
            start datetime, default None
        end: str
            end datetime, default None
        check_data_length: int
            check data length, by default None
        limit_nums: int
            using for debug, by default None
        """
        self.arctic_store = Arctic("127.0.0.1")
        self.bar_data_infos = self.arctic_store.get_library('data_overview')
        self.bar_lib = self.arctic_store.get_library('bar_data')
        self.ex_factor_lib = self.arctic_store.get_library("ex_factor")  # 复权因子
        self.split_lib = self.arctic_store.get_library("split")  # 拆分信息
        interval = 'd' if interval.endswith('d') else '1m'

        super(RqdataCollector, self).__init__(
            save_dir=save_dir,
            start=start,
            end=end,
            interval=interval,
            max_workers=max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        )

        self.init_datetime()

    def init_datetime(self):
        if self.interval not in {'d', '1d', '1m', '1min'}:
            raise ValueError(f"interval error: {self.interval}")

        self.start_datetime = self.convert_datetime(self.start_datetime, self._timezone)
        self.end_datetime = self.convert_datetime(self.end_datetime, self._timezone)

    @staticmethod
    def convert_datetime(dt: [pd.Timestamp, str], timezone):
        if isinstance(dt, str):
            return pd.Timestamp(dt, tz=timezone)
        if dt.tzinfo is None:
            return dt.tz_localize(timezone)
        return dt.tz_convert(timezone)

    def get_data(
        self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        if interval not in {'d', '1d', '1m', '1min'}:
            raise ValueError(f"cannot support {interval}")
        # start_datetime = self.convert_datetime(start_datetime, self._timezone)
        # end_datetime = self.convert_datetime(end_datetime, self._timezone)
        interval = 'd' if interval.endswith('d') else '1m'
        db_symbol = symbol + '_' + interval
        _result = self.bar_lib.read(db_symbol, chunk_range=DateRange(start_datetime.tz_localize(None),
                                                                     end_datetime.tz_localize(None)))
        try:
            _result.set_index('date', inplace=True)
        except:
            print(start_datetime, end_datetime, db_symbol, symbol, _result)

        if self.ex_factor_lib.has_symbol(symbol):
            ex_factors = self.ex_factor_lib.read(symbol)
            _result = pd.merge_asof(_result, ex_factors[['ex_cum_factor', 'ex_factor']], left_index=True,
                                    right_index=True)
        else:
            _result[['ex_cum_factor', 'ex_factor']] = 1.0

        if self.split_lib.has_symbol(symbol):
            split_factor = self.split_lib.read(symbol)
            _result = pd.merge_asof(_result, split_factor[['cum_factor']].rename(columns={'cum_factor': 'split_cum_factor'}),
                                 left_index=True, right_index=True)
        else:
            _result['split_cum_factor'] = 1.0

        time.sleep(self.delay)

        # _result = _result.tz_localize(self._timezone)

        return pd.DataFrame() if _result is None else _result

    def collector_data(self):
        """collector data"""
        super(RqdataCollector, self).collector_data()
        self.download_index_data()

    def get_instrument_list(self):
        logger.info("get HS stock symbols......")
        symbols = [x.rsplit('_', 1)[0] for x in self.bar_data_infos.list_symbols() if ('SSE' in x or 'SZSE' in x) and not x.startswith('INDEX') and x.endswith(self.interval)]
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def normalize_symbol(self, symbol):
        symbol_s = symbol.split("_")
        symbol = f"sh{symbol_s[0]}" if symbol_s[-1] == "SSE" else f"sz{symbol_s[0]}"
        return symbol

    @property
    def _timezone(self):
        return "Asia/Shanghai"

    def download_index_data(self):
        # TODO: from MSN
        _format = "%Y%m%d"
        _begin = self.start_datetime.tz_localize(None)
        _end = self.end_datetime.tz_localize(None)
        interval = 'd' if self.interval.endswith('d') else '1m'
        for _index_name, _index_code in {"csi300": "000300", "csi100": "000903", "csi500": "000905"}.items():
            logger.info(f"get bench data: {_index_name}({_index_code})......")
            try:
                exch_str = 'SZSE' if _index_code.startswith('INDEX399') else 'SSE'
                db_symbol = f'INDEX{_index_code}_{exch_str}_{interval}'
                df = self.bar_lib.read(db_symbol, chunk_range=DateRange(_begin, _end)).set_index('date')
                df[['ex_cum_factor', 'ex_factor', 'split_cum_factor']] = 1.0
            except Exception as e:
                logger.warning(f"get {_index_name} error: {e}")
                continue
            df["symbol"] = f"sh{_index_code}"
            _path = self.save_dir.joinpath(f"sh{_index_code}.csv")
            if _path.exists():
                _old_df = pd.read_csv(_path)
                df = pd.concat([_old_df[~_old_df.index.isin(df.index)], df], sort=True)
            df.to_csv(_path)
            time.sleep(1)

    def save_instrument(self, symbol, df: pd.DataFrame):
        """save instrument data to file

        Parameters
        ----------
        symbol: str
            instrument code
        df : pd.DataFrame
            df.columns must contain "symbol" and "datetime"
        """
        if df is None or df.empty:
            logger.warning(f"{symbol} is empty")
            return

        symbol = self.normalize_symbol(symbol)
        symbol = code_to_fname(symbol)
        instrument_path = self.save_dir.joinpath(f"{symbol}.csv")
        df["symbol"] = symbol
        if instrument_path.exists():
            _old_df = pd.read_csv(instrument_path)
            df = pd.concat([_old_df[~_old_df.index.isin(df.index)], df], sort=True)
        df.to_csv(instrument_path)


class RqdataNormalize(BaseNormalize):
    SOURCE_COLS = ["open_price", "close_price", "high_price", "low_price", "volume", 'turnover']
    COLUMNS = ["open", "close", "high", "low", "volume", 'money']
    DAILY_FORMAT = "%Y-%m-%d"

    @staticmethod
    def calc_change(df: pd.DataFrame, last_close: float) -> pd.Series:
        df = df.copy()
        _tmp_series = df["close"].fillna(method="ffill")
        _tmp_shift_series = _tmp_series.shift(1)
        if last_close is not None:
            _tmp_shift_series.iloc[0] = float(last_close)
        change_series = _tmp_series / _tmp_shift_series - 1
        return change_series

    @staticmethod
    def normalize_rqdata(
        df: pd.DataFrame,
        calendar_list: list = None,
        last_close: float = None,
    ):
        if df.empty:
            return df
        symbol = df.loc[df["symbol"].first_valid_index(), "symbol"]
        columns = copy.deepcopy(RqdataNormalize.COLUMNS)
        df['date'] = pd.to_datetime(df.date)
        df.set_index("date", inplace=True)
        df = df.rename(columns=dict((s, t) for s, t in zip(RqdataNormalize.SOURCE_COLS, RqdataNormalize.COLUMNS))).copy()

        duplicated_record = df.index.duplicated(keep="first")
        if duplicated_record.any():
            logger.warning(f"Duplicated record discovered for {symbol}")
            df = df[~duplicated_record]
        if calendar_list is not None:
            index_from_cal = pd.DatetimeIndex(
                [x for x in calendar_list if
                 df.index.min().replace(hour=0, minute=0, second=0) <= x
                 < df.index.max().replace(hour=23, minute=59, second=59)]
                , name='date')
            df = df.reindex(index_from_cal)

        # assign adjclose as close price adjusted by split only
        df["adjclose"] = df.close
        # adjust ohlc by split and dividends
        df[["open", "close", "high", "low"]] = df[["open", "close", "high", "low"]].multiply(df.ex_cum_factor, axis=0)
        df['factor'] = df.ex_cum_factor

        df.sort_index(inplace=True)


        df.loc[(df["volume"] <= 1e-10) | np.isnan(df["volume"]), list(set(df.columns) - {"symbol"})] = np.nan

        # NOTE: The data obtained by Rqdata finance sometimes has exceptions
        # WARNING: If it is normal for a `symbol(exchange)` to differ by a factor of *89* to *111* for consecutive trading days,
        # WARNING: the logic in the following line needs to be modified
        _count = 0
        change_series = RqdataNormalize.calc_change(df, last_close)
        while True:
            # NOTE: may appear unusual for many days in a row
            _mask = (change_series >= 89) & (change_series <= 111)
            if not _mask.any():
                break
            _tmp_cols = ["high", "close", "low", "open", "adjclose"]
            df.loc[_mask, _tmp_cols] = df.loc[_mask, _tmp_cols] / 100
            _count += 1
            if _count >= 10:
                _symbol = df.loc[df["symbol"].first_valid_index()]["symbol"]
                logger.warning(
                    f"{_symbol} `change` is abnormal for {_count} consecutive days, please check the specific data file carefully"
                )

        df["change"] = RqdataNormalize.calc_change(df, last_close)

        columns += ["change"]
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), columns] = np.nan

        df["symbol"] = symbol
        return df.drop(columns=['ex_cum_factor', 'ex_factor', 'split_cum_factor']).reset_index()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # normalize
        df = self.normalize_rqdata(df, self._calendar_list)
        return df

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return [x for x in get_calendar_list("ALL")]


class Run(BaseRun):
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="1d"):
        """

        Parameters
        ----------
        source_dir: str
            The directory where the raw data collected from the Internet is saved, default "Path(__file__).parent/source"
        normalize_dir: str
            Directory for normalize data, default "Path(__file__).parent/normalize"
        max_workers: int
            Concurrent number, default is 1; when collecting data, it is recommended that max_workers be set to 1
        interval: str
            freq, value from [1min, 1d], default 1d
        region: str
            region, value from ["CN", "US"], default "CN"
        """
        super().__init__(source_dir, normalize_dir, max_workers, interval)

    @property
    def collector_class_name(self):
        return f"RqdataCollector"

    @property
    def normalize_class_name(self):
        return f"RqdataNormalize"

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

    def download_data(
        self,
        max_collector_count=2,
        start=None,
        end=None,
        check_data_length=None,
        limit_nums=None,
    ):
        """download data from Internet

        Parameters
        ----------
        max_collector_count: int
            default 2
        start: str
            start datetime, default "2000-01-01"; closed interval(including start)
        end: str
            end datetime, default ``pd.Timestamp(datetime.datetime.now() + pd.Timedelta(days=1))``; open interval(excluding end)
        check_data_length: int
            check data length, if not None and greater than 0, each symbol will be considered complete if its data length is greater than or equal to this value, otherwise it will be fetched again, the maximum number of fetches being (max_collector_count). By default None.
        limit_nums: int
            using for debug, by default None

        Notes
        -----
            check_data_length, example:
                daily, one year: 252 // 4
                us 1min, a week: 6.5 * 60 * 5
                cn 1min, a week: 4 * 60 * 5

        Examples
        ---------
            # get daily data
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1d
            # get 1m data
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1m
        """
        super(Run, self).download_data(max_collector_count, 0.5, start, end, check_data_length, limit_nums)

    def normalize_data(
        self,
        qlib_data_1d_dir: str = None,
    ):
        """normalize data

        Parameters
        ----------
        qlib_data_1d_dir: str
            if interval==1min, qlib_data_1d_dir cannot be None, normalize 1min needs to use 1d data;

                qlib_data_1d can be obtained like this:
                    $ python scripts/get_data.py qlib_data --target_dir <qlib_data_1d_dir> --interval 1d
                    $ python scripts/data_collector/Rqdata/collector.py update_data_to_bin --qlib_data_1d_dir <qlib_data_1d_dir> --trading_date 2021-06-01
                or:
                    download 1d data, reference: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/Rqdata#1d-from-Rqdata

        Examples
        ---------
            $ python collector.py normalize_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize --region cn --interval 1d
            $ python collector.py normalize_data --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data --source_dir ~/.qlib/stock_data/source_cn_1min --normalize_dir ~/.qlib/stock_data/normalize_cn_1min --region CN --interval 1min
        """
        if self.interval.lower() == "1min":
            if qlib_data_1d_dir is None or not Path(qlib_data_1d_dir).expanduser().exists():
                raise ValueError(
                    "If normalize 1min, the qlib_data_1d_dir parameter must be set: --qlib_data_1d_dir <user qlib 1d data >, Reference: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/Rqdata#automatic-update-of-daily-frequency-datafrom-Rqdata-finance"
                )
        _class = getattr(self._cur_module, self.normalize_class_name)
        yc = Normalize(
            source_dir=self.source_dir,
            target_dir=self.normalize_dir,
            normalize_class=_class,
            max_workers=self.max_workers,
        )
        yc.normalize()

    def update_data_to_bin(
        self,
        qlib_data_1d_dir: str,
        trading_date: str = None,
        end_date: str = None,
    ):
        """update Rqdata data to bin

        Parameters
        ----------
        qlib_data_1d_dir: str
            the qlib data to be updated for Rqdata, usually from: https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data

        trading_date: str
            trading days to be updated, by default ``datetime.datetime.now().strftime("%Y-%m-%d")``
        end_date: str
            end datetime, default ``pd.Timestamp(trading_date + pd.Timedelta(days=1))``; open interval(excluding end)
        check_data_length: int
            check data length, if not None and greater than 0, each symbol will be considered complete if its data length is greater than or equal to this value, otherwise it will be fetched again, the maximum number of fetches being (max_collector_count). By default None.
        delay: float
            time.sleep(delay), default 1
        Notes
        -----
            If the data in qlib_data_dir is incomplete, np.nan will be populated to trading_date for the previous trading day

        Examples
        -------
            $ python collector.py update_data_to_bin --qlib_data_1d_dir <user data dir> --trading_date <start date> --end_date <end date>
            # get 1m data
        """

        if self.interval.lower() != "1d":
            logger.warning(f"currently supports 1d data updates: --interval 1d")

        # start/end date
        if trading_date is None:
            trading_date = pd.Timestamp.now().strftime("%Y-%m-%d")
            logger.warning(f"trading_date is None, use the current date: {trading_date}")

        if end_date is None:
            # noinspection PyTypeChecker
            end_date = (pd.Timestamp(trading_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        # NOTE: a larger max_workers setting here would be faster
        self.max_workers = (
            max(multiprocessing.cpu_count() - 2, 1)
            if self.max_workers is None or self.max_workers < 1
            else self.max_workers
        )
        # download data from Rqdata
        # NOTE: when downloading data from RqdataFinance, max_workers is recommended to be 1
        self.download_data(max_collector_count=self.max_workers, start=trading_date, end=end_date)

        # normalize data
        self.normalize_data(qlib_data_1d_dir)

        # dump bin
        _dump = DumpDataUpdate(
            csv_path=self.normalize_dir,
            qlib_dir=qlib_data_1d_dir,
            exclude_fields="symbol,date",
            max_workers=self.max_workers,
        )
        _dump.dump()

        # parse index
        index_list = ["CSI100", "CSI300", "CSI500"]
        get_instruments = getattr(
            importlib.import_module(f"data_collector.cn_index.collector"), "get_instruments"
        )
        for _index in index_list:
            get_instruments(str(qlib_data_1d_dir), _index)


if __name__ == "__main__":
    # fire.Fire(Run)
    runner = Run(
        source_dir=r"D:\Documents\TradeResearch\qlib_test\rqdata\source",
        normalize_dir=r"D:\Documents\TradeResearch\qlib_test\rqdata\normalize",
        max_workers=6
    )
    # runner.download_data(max_collector_count=1, start=pd.Timestamp("2014-01-01"), end=pd.Timestamp("2021-12-31"))
    # runner.normalize_data()
    runner.update_data_to_bin(qlib_data_1d_dir=r"D:\Documents\TradeResearch\qlib_test\rqdata", trading_date='2005-01-01', end_date="2022-04-01")
