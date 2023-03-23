# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import copy
import multiprocessing
import re
import sys
# import fire
import traceback
from datetime import time
from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import Iterable, Type, cast

import exchange_calendars as mcal
import numpy as np
import pandas as pd
from arctic.auth import Credential
from arctic.date import DateRange
from arctic.hooks import register_get_auth_hook
from loguru import logger
from pandas_helper import resample
from pandas_helper.tseries.offsets import CustomBusinessIntradayOffset
from vnpy.trader.database import SETTINGS, get_database
from vnpy_arctic.arctic_database import ArcticDatabase

from qlib.utils import code_to_fname

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))
from data_collector.base import BaseCollector, BaseNormalize, BaseRun, Normalize
from data_collector.utils import get_calendar_list
from dump_bin import DumpDataAll, DumpDataUpdate

SYMBOL_PATTERN = re.compile(r"([A-Za-z]+)(\d+)_([A-Za-z]+)")


def _process(tt):
    tl, tr = pd.Timestamp(tt[0]) - pd.Timedelta(minutes=1), pd.Timestamp(tt[1])
    return (
        tl.time() if tl.hour < 17 else (tl - pd.Timedelta(days=1)).time(),
        tr.time() if tr.hour < 17 else (tr - pd.Timedelta(days=1)).time(),
    )


def arctic_auth_hook(*_):
    if bool(SETTINGS.get("database.password", "")) and bool(SETTINGS.get("database.user", "")):
        return Credential(
            database="admin",
            user=SETTINGS["database.user"],
            password=SETTINGS["database.password"],
        )
    return None


register_get_auth_hook(arctic_auth_hook)


class RqdataCollector(BaseCollector):
    def __init__(
        self,
        save_dir: [str, Path],
        start=None,
        end=None,
        interval="1d",
        commod_only=0,
        max_workers=4,
        max_collector_count=2,
        delay=0.5,
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

        db_mgr = get_database()
        self.arctic_store = cast(ArcticDatabase, db_mgr).connection
        self.meta_lib = self.arctic_store.get_library("future_meta")
        self.bar_lib = self.arctic_store.get_library("bar_data")
        self.dom_lib = self.arctic_store.get_library("future_dominant")
        self.limit_lib = self.arctic_store.get_library("future_limit_up_down")
        self.fut_next_mon_lib = self.arctic_store.get_library("future_next_mon")

        self.symbol_dict = None

        self.commod_only = commod_only

        interval = "d" if interval.endswith("d") else interval

        super(RqdataCollector, self).__init__(
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
        if self.interval not in {"d", "1d", "1m", "1min", "5min", "15min", "30min", "1h", "1hour"}:
            raise ValueError(f"interval error: {self.interval}")

    @staticmethod
    def convert_datetime(dt: [pd.Timestamp, str], timezone):
        if isinstance(dt, str):
            return pd.Timestamp(dt, tz=timezone)
        if dt.tzinfo is None:
            return dt.tz_localize(timezone)
        return dt.tz_convert(timezone)

    def _cal_rolling_yields(self, und_code, exch, dtrange):
        def get_days_to_expiry(s):
            mymeta = self.meta_lib.read(s.contract)
            return (mymeta["maturity_date"].replace(tzinfo=None) - s.name).days

        doms = self.dom_lib.read(und_code, chunk_range=dtrange)
        doms["days_to_expiry"] = doms.apply(get_days_to_expiry, axis=1)
        dom_price_df = (
            self.bar_lib.read(und_code + "88_" + exch + "_d", chunk_range=dtrange, columns=["close_price", "date"])
            .set_index("date")
            .sort_index()
        )
        dom_contract = pd.merge(dom_price_df, doms, left_index=True, right_index=True, how="left")
        next_mon_df = (
            self.fut_next_mon_lib.read(und_code, chunk_range=dtrange, columns=["date", "close_price", "contract"])
            .set_index("date")
            .sort_index()
        )
        next_mon_df["days_to_expiry"] = next_mon_df.apply(get_days_to_expiry, axis=1)
        next_next_mon_df = (
            self.fut_next_mon_lib.read(
                und_code + "_next", chunk_range=dtrange, columns=["date", "close_price", "contract"]
            )
            .set_index("date")
            .sort_index()
        )
        next_next_mon_df["days_to_expiry"] = next_next_mon_df.apply(get_days_to_expiry, axis=1)
        dom_contract.loc[dom_contract.days_to_expiry == next_mon_df.days_to_expiry, :] = next_next_mon_df.loc[
            dom_contract.days_to_expiry == next_mon_df.days_to_expiry, :
        ]
        return (
            np.log(next_mon_df.close_price / dom_contract.close_price)
            * 365
            / (dom_contract.days_to_expiry - next_mon_df.days_to_expiry)
        ).rename("rolling_yield")

    def get_data(
        self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        if interval not in {"d", "1d", "1min", "15min"}:
            raise ValueError(f"cannot support {interval}")
        read_interval = "d" if interval.endswith("d") else "1m"
        obj = re.match(SYMBOL_PATTERN, symbol)
        if obj is None:
            logger.error(f"wrong sym format {symbol}")
            return pd.DataFrame()
        und_sym, time_code, exch_str = obj.groups()
        if und_sym == "FU":  # https://www.shfe.com.cn/news/notice/911330669.html
            start_datetime = max(start_datetime, pd.Timestamp("2018-07-16", tz="Asia/Shanghai"))
        if und_sym == "WR":  # https://www.shfe.com.cn/news/notice/911331232.html
            start_datetime = max(start_datetime, pd.Timestamp("2018-10-16", tz="Asia/Shanghai"))
        db_symbol = symbol + "_" + read_interval
        dt_range = DateRange(start_datetime.tz_localize(None), end_datetime.tz_localize(None))
        _result = self.bar_lib.read(db_symbol, chunk_range=dt_range)
        try:
            _result = _result.set_index("date").sort_index()
        except:
            logger.error(
                f"{start_datetime.strftime('%Y-%m-%d')}, {end_datetime.strftime('%Y-%m-%d')}, {db_symbol}, {symbol}"
            )
            return pd.DataFrame()

        if exch_str == "CZCE" and read_interval == "1m" and time_code not in {"888", "889", "890", "891"}:
            _result["turnover"] = (
                (_result["high_price"] + _result["low_price"] + _result["close_price"])
                / 3.0
                * _result["volume"]
                * self.symbol_dict[symbol]["contract_multiplier"]
            )

        _limits = self.limit_lib.read(
            symbol, chunk_range=dt_range, columns=["limit_up", "limit_down", "settlement"]
        ).sort_index()

        if time_code in {"888", "889", "890", "891"}:
            unadj_db_symbol = db_symbol.replace(time_code, "88")
            if exch_str == "CZCE" and read_interval == "1m":
                unadjclose = (
                    self.bar_lib.read(
                        unadj_db_symbol,
                        chunk_range=dt_range,
                        columns=[
                            "date",
                            "close_price",
                            "high_price",
                            "low_price",
                            "turnover",
                            "volume",
                            "open_interest",
                        ],
                    )
                    .set_index("date")
                    .sort_index()
                )
                unadjclose["turnover"] = (
                    (unadjclose["high_price"] + unadjclose["low_price"] + unadjclose["close_price"])
                    / 3.0
                    * unadjclose["volume"]
                    * self.symbol_dict[symbol]["contract_multiplier"]
                )
            else:
                unadjclose = (
                    self.bar_lib.read(
                        unadj_db_symbol,
                        chunk_range=dt_range,
                        columns=["date", "close_price", "turnover", "volume", "open_interest"],
                    )
                    .set_index("date")
                    .sort_index()
                )
            _result["unadjclose"] = unadjclose["close_price"]
            _result["volume"] = unadjclose["volume"]
            if exch_str == "CZCE":
                _result["turnover"] = unadjclose["turnover"]
            else:
                _result["turnover"] = unadjclose["turnover"]
            _result["open_interest"] = unadjclose["open_interest"]

            if time_code in ("888", "889"):
                _result["factor"] = _result["close_price"] - _result["unadjclose"]
            elif time_code in ("890", "891"):
                _result["factor"] = _result["close_price"] / _result["unadjclose"]
            else:
                raise ValueError("unknow time code")
        else:
            _result["factor"] = 1.0
            _result["unadjclose"] = _result["close_price"]

        if interval not in {"d", "1d"}:
            _limits.index = _limits.index - pd.Timedelta("6H")

        try:
            _result = pd.merge_asof(_result, _limits, left_index=True, right_index=True)
        except:
            logger.error(
                f"merge limits failed for {db_symbol} from {start_datetime} to {end_datetime}: {traceback.format_exc()}"
            )
            raise

        if self.commod_only % 2 == 1:
            day_only = True
        else:
            day_only = False
        if interval in {"15min"}:
            try:
                trading_peroids = [tuple(x.split("-")) for x in self.symbol_dict[symbol]["trading_hours"].split(",")]
                starts, ends = map(list, zip(*[_process(tt) for tt in trading_peroids]))

                # for financial futures before 2018
                if _result.index.time.min() == time(9, 15) and starts[0] != time(9, 15):
                    starts[0] = time(9, 15)
                if _result.index.time.max() == time(15, 14) and ends[-1] != time(15, 15):
                    ends[-1] = time(15, 15)

                cbi = CustomBusinessIntradayOffset(
                    n=1,
                    step=pd.Timedelta("15min"),
                    start=starts,
                    end=ends,
                    holidays=mcal.get_calendar("SSE").adhoc_holidays.tolist(),
                    normalize=False,
                    weekmask="Mon Tue Wed Thu Fri",
                )
                _result = (
                    resample(_result, rule=cbi)
                    .agg(
                        {
                            "open_price": "first",
                            "high_price": "max",
                            "low_price": "min",
                            "close_price": "last",
                            "unadjclose": "last",
                            "volume": "sum",
                            "turnover": "sum",
                            "open_interest": "last",
                            "factor": "mean",
                        }
                    )
                    .dropna(how="all", subset=["open_price", "high_price", "low_price", "close_price"])
                )
                if day_only:
                    _result = _result.between_time("08:00", "17:00")
            except Exception:
                logger.error(
                    f"resampling failed for {db_symbol} from {start_datetime} to {end_datetime}: {traceback.format_exc()}"
                )
                raise

        if obj is not None and obj[2] in ("888", "889"):
            _result["vwap"] = (
                _result["turnover"] / _result["volume"] / self.symbol_dict[symbol]["contract_multiplier"]
                + _result["factor"]
            )
        else:
            _result["vwap"] = (
                _result["turnover"]
                / _result["volume"]
                / self.symbol_dict[symbol]["contract_multiplier"]
                * _result["factor"]
            )

        _result["vwap"].fillna(_result["close_price"], inplace=True)
        if interval in {"d", "1d"}:
            _result["rolling_yield"] = self._cal_rolling_yields(und_code=und_sym, exch=exch_str, dtrange=dt_range)

        return pd.DataFrame() if _result is None else _result

    def get_instrument_list(self):
        def symbol_validation(_, v):
            if self.commod_only > 1 and v["underlying_symbol"] in {"IF", "IC", "IH", "IM", "T", "TS", "TF"}:
                return False
            if pd.isna(v["listed_date"]):
                return False
            if pd.isna(v["de_listed_date"]):
                return v["listed_date"] <= self.end_datetime
            if v["underlying_symbol"] == "FU" and v["listed_date"] < pd.Timestamp(
                "2018-07-15 23:00:00", tz="Asia/Shanghai"
            ):
                return False
            if v["underlying_symbol"] == "WR" and v["listed_date"] < pd.Timestamp(
                "2018-10-15 23:00:00", tz="Asia/Shanghai"
            ):
                return False
            return self.start_datetime <= v["de_listed_date"] and v["listed_date"].replace(hour=16) <= min(
                self.end_datetime, pd.Timestamp.now("Asia/Shanghai")
            )

        logger.info("get china future contracts......")
        self.symbol_dict = {x: self.meta_lib.read(x) for x in self.meta_lib.list_symbols()}
        symbols = [
            k for k, v in self.symbol_dict.items() if k != "_exchange_underlying_mapping" and symbol_validation(k, v)
        ]
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    @property
    def _timezone(self):
        return "Asia/Shanghai"

    @staticmethod
    def normalize_symbol(symbol: str):
        """normalize symbol"""
        return symbol

    # def download_index_data(self):
    #     _format = "%Y%m%d"
    #     _begin = self.start_datetime.tz_localize(None)
    #     _end = self.end_datetime.tz_localize(None)
    #     interval = 'd' if self.interval.endswith('d') else '1m'
    #     for _index_name, _index_code in INDEXES.items():
    #         logger.info(f"get bench data: {_index_name}({_index_code})......")
    #         try:
    #             exch_str = 'SZSE' if _index_code.startswith('INDEX399') else 'SSE'
    #             db_symbol = f'INDEX{_index_code}_{exch_str}_{interval}'
    #             df = self.bar_lib.read(db_symbol, chunk_range=DateRange(_begin, _end)).set_index('date')
    #             df[['ex_cum_factor', 'ex_factor', 'split_cum_factor']] = 1.0
    #             df[['call_announced']] = 0
    #         except Exception as e:
    #             logger.warning(f"get {_index_name} error: {e}")
    #             continue
    #         df["symbol"] = f"sh{_index_code}"
    #         _path = self.save_dir.joinpath(f"sh{_index_code}.csv")
    #         if _path.exists():
    #             _old_df = pd.read_csv(_path, index_col=['date'], parse_dates=['date'])
    #             df = pd.concat([_old_df[~_old_df.index.isin(df.index)], df], sort=True)
    #         df.to_csv(_path)
    #         time.sleep(1)

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

        symbol = code_to_fname(symbol)
        instrument_path = self.save_dir.joinpath(f"{symbol}.csv")
        df["symbol"] = symbol
        if instrument_path.exists():
            _old_df = pd.read_csv(instrument_path, index_col=["date"], parse_dates=["date"])
            df = pd.concat([_old_df[~_old_df.index.isin(df.index)], df], sort=True)
        df.to_csv(instrument_path)

    def _simple_collector(self, symbol: str):
        """

        Parameters
        ----------
        symbol: str

        """
        self.sleep()
        df = self.get_data(symbol, self.interval, self.start_datetime, self.end_datetime)
        # info = self.get_info(symbol)
        _result = self.NORMAL_FLAG
        if self.check_data_length > 0:
            _result = self.cache_small_data(symbol, df)
        if _result == self.NORMAL_FLAG:
            self.save_instrument(symbol, df)
        return _result


class RqdataNormalize(BaseNormalize):
    SOURCE_COLS = [
        "open_price",
        "close_price",
        "high_price",
        "low_price",
        "volume",
        "turnover",
        "open_interest",
    ]
    COLUMNS = [
        "open",
        "close",
        "high",
        "low",
        "volume",
        "money",
        "open_interest",
    ]
    DAILY_FORMAT = "%Y-%m-%d"

    def __init__(self, date_field_name: str = "date", symbol_field_name: str = "symbol", **kwargs):
        self.commod_only = kwargs.get("commod_only", 0)
        super(RqdataNormalize, self).__init__(date_field_name, symbol_field_name, **kwargs)

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
        df["date"] = pd.to_datetime(df.date)
        df.set_index("date", inplace=True)
        df = df.rename(
            columns=dict((s, t) for s, t in zip(RqdataNormalize.SOURCE_COLS, RqdataNormalize.COLUMNS))
        ).copy()

        duplicated_record = df.index.duplicated(keep="first")
        if duplicated_record.any():
            logger.warning(f"Duplicated record discovered for {symbol}")
            df = df[~duplicated_record]
        if calendar_list is not None:
            tmp_idx_cal = pd.DatetimeIndex(calendar_list, name="date").sort_values()
            index_from_cal = tmp_idx_cal[
                tmp_idx_cal.searchsorted(df.index.min().replace(hour=0, minute=0, second=0)) : tmp_idx_cal.searchsorted(
                    df.index.max().replace(hour=23, minute=59, second=59)
                )
            ]
            df = df.reindex(index_from_cal)

        df.sort_index(inplace=True)

        df["change"] = RqdataNormalize.calc_change(df, last_close)

        columns += ["change"]
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), columns] = np.nan
        df["symbol"] = symbol
        return df.reset_index()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # normalize
        df = self.normalize_rqdata(df, self._calendar_list)
        return df

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return [x for x in get_calendar_list("ALL")]


class RqdataNormalizeIntraday(RqdataNormalize):
    def __init__(
        self,
        qlib_data_1d_dir: [str, Path],
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        interval: str = "1min",
        **kwargs,
    ):
        self.qlib_data_1d_dir = qlib_data_1d_dir
        db_mgr = get_database()
        self.arctic_store = cast(ArcticDatabase, db_mgr).connection
        self.meta_lib = self.arctic_store.get_library("future_meta")
        self.interval = interval
        super(RqdataNormalizeIntraday, self).__init__(date_field_name, symbol_field_name, **kwargs)

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return self.generate_intraday_from_daily(self.calendar_list_1d)

    def _get_1d_calendar_list(self) -> Iterable[pd.Timestamp]:
        import qlib
        from qlib.data import D

        qlib.init(provider_uri=self.qlib_data_1d_dir)
        return list(D.calendar(freq="day"))

    @property
    def calendar_list_1d(self):
        calendar_list_1d = getattr(self, "_calendar_list_1d", None)
        if calendar_list_1d is None:
            calendar_list_1d = self._get_1d_calendar_list()
            setattr(self, "_calendar_list_1d", calendar_list_1d)
        return calendar_list_1d

    def _generate_intraday_intervals(self):
        def _merge_union_interval(open_periods_sets):
            def process(tt):
                tl, tr = pd.Timestamp(tt[0]) - pd.Timedelta(minutes=1), pd.Timestamp(tt[1]) - pd.Timedelta(minutes=1)
                return tl if tl.hour < 17 else tl - pd.Timedelta(days=1), tr if tr.hour < 17 else tr - pd.Timedelta(
                    days=1
                )

            tss = sorted(sum([[process(tt) for tt in one_set] for one_set in open_periods_sets], []), key=itemgetter(0))
            merged = []
            # Traverse all input Intervals starting from
            # second interval
            curr_lb, curr_rb = tss[0]
            for interval in tss[1:]:
                # If this is not first Interval and overlaps
                # with the previous one, Merge previous and
                # current Intervals
                if curr_rb >= interval[0]:
                    curr_rb = max(curr_rb, interval[1])
                else:
                    merged.append((curr_lb, curr_rb))
                    curr_lb, curr_rb = interval
            merged.append((curr_lb, curr_rb))

            return [
                (lb.time(), rb.time())
                for lb, rb in merged
                if self.commod_only < 2 or time(8, 0) < lb.time() < time(17, 0) and time(8, 0) < rb.time() < time(17, 0)
            ]

        metas = {x: self.meta_lib.read(x) for x in self.meta_lib.list_symbols() if "888" in x}
        trading_hours = {
            k: v["trading_hours"]
            for k, v in metas.items()
            if not self.commod_only > 1 or v["underlying_symbol"] not in {"IF", "IC", "IH", "IM", "T", "TS", "TF"}
        }
        open_periods_sets = [[tuple(x.split("-")) for x in th.split(",")] for th in trading_hours.values()]
        return _merge_union_interval(open_periods_sets)

    def generate_intraday_from_daily(self, calendars: Iterable) -> pd.Index:
        def _generate_offset_intervals(time_interval):
            left_time_delta = pd.Timedelta(
                hours=time_interval[0].hour, minutes=time_interval[0].minute, seconds=time_interval[0].second
            )
            right_time_delta = pd.Timedelta(
                hours=time_interval[1].hour, minutes=time_interval[1].minute, seconds=time_interval[1].second
            )
            if right_time_delta < left_time_delta:
                right_time_delta += pd.Timedelta(hours=24)
            return left_time_delta, right_time_delta

        intraday_ranges = self._generate_intraday_intervals()

        offset_intervals = [_generate_offset_intervals(interval) for interval in intraday_ranges]

        res = []
        for _day in calendars:
            for _range in offset_intervals:
                res.append(
                    pd.date_range(
                        pd.Timestamp(_day).normalize() + _range[0],
                        pd.Timestamp(_day).normalize() + _range[1],
                        freq=self.interval,
                    )
                )

        return pd.Index(sorted(set(np.hstack(res))))


class Run(BaseRun):
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="1d", commod_only=0):
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
        commod_only: int
            0=all species, all time; 1=all species, day only; 2=commodity only, all time, 3=commodity only, day only
        """
        self.suffix = "day" if interval in ("d", "1d") else interval
        self.commod_only = commod_only
        super().__init__(source_dir, normalize_dir, max_workers, interval)

    @property
    def collector_class_name(self):
        return f"RqdataCollector"

    @property
    def normalize_class_name(self):
        if self.suffix == "day":
            return f"RqdataNormalize"
        return f"RqdataNormalizeIntraday"

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

    def download_data(
        self,
        max_collector_count=2,
        start=None,
        end=None,
        check_data_length=0,
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
            end datetime, default ``pd.Timestamp(datetime.datetime.now() + pd.Timedelta(days=1))``;
            open interval(excluding end)
        check_data_length: int
            check data length, if not None and greater than 0, each symbol will be considered complete if its data
            length is greater than or equal to this value, otherwise it will be fetched again, the maximum number
             of fetches being (max_collector_count). By default, None.
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
        _class = getattr(self._cur_module, self.collector_class_name)  # type: Type[BaseCollector]
        _class(
            self.source_dir,
            max_workers=self.max_workers,
            max_collector_count=max_collector_count,
            start=start,
            end=end,
            interval=self.interval,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        ).collector_data()
        # super(Run, self).download_data(max_collector_count, 0.5, start, end, check_data_length, limit_nums)

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
            qlib_data_1d_dir=qlib_data_1d_dir,
            interval=self.interval,
            commod_only=self.commod_only,
        )
        yc.normalize()

    def update_data_to_bin(
        self,
        qlib_data_1d_dir: str,
        trading_date: str = None,
        end_date: str = None,
        check_data_length: int = None,
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
        Notes
        -----
            If the data in qlib_data_dir is incomplete, np.nan will be populated to trading_date for the previous trading day

        Examples
        -------
            $ python collector.py update_data_to_bin --qlib_data_1d_dir <user data dir> --trading_date <start date> --end_date <end date>
            # get 1m data
        """

        # if self.interval.lower() != "1d":
        #     logger.warning(f"currently supports 1d data updates: --interval 1d")

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
        # # download data from Rqdata
        self.download_data(
            max_collector_count=self.max_workers, start=trading_date, end=end_date, check_data_length=check_data_length
        )

        # # normalize data
        self.normalize_data(qlib_data_1d_dir)

        # dump bin
        qlib_dir = Path(qlib_data_1d_dir).expanduser().resolve()

        DumpClass = DumpDataUpdate if qlib_dir.joinpath(f"calendars/{self.suffix}.txt").exists() else DumpDataAll
        _dump = DumpClass(
            csv_path=self.normalize_dir,
            qlib_dir=qlib_dir,
            exclude_fields="symbol,date",
            max_workers=self.max_workers,
            freq=self.suffix,
        )
        _dump.dump()

        if self.suffix == "day":
            logger.info("generate continuous population")
            instruments_dir = _dump._instruments_dir
            # all_instrument_path = instrument_dir.joinpath(_dump.INSTRUMENTS_FILE_NAME)
            all_instrument_date_range = pd.read_csv(
                instruments_dir.joinpath(_dump.INSTRUMENTS_FILE_NAME), sep="\t", header=None
            )

            def check_symbol(symbol, suffix, commod_only):
                base, _ = symbol.rsplit("_", 1)
                und_code = base[: -len(suffix)]
                if commod_only > 1 and und_code in {"IF", "IC", "IH", "IM", "T", "TS", "TF"}:
                    return False
                return base.endswith(suffix) and und_code.isalpha()

            for suffix, name in [
                ("88", "continuous"),
                (
                    "888",
                    "cont_prev_close_spread",
                ),
                ("889", "cont_open_spread"),
                (
                    "890",
                    "cont_prev_close_ratio",
                ),
                ("891", "cont_open_ratio"),
            ]:
                this_kind_conti = all_instrument_date_range[
                    all_instrument_date_range.iloc[:, 0].apply(
                        partial(check_symbol, suffix=suffix, commod_only=self.commod_only)
                    )
                ]
                this_kind_conti.to_csv(
                    instruments_dir.joinpath(f"{name}.txt"), header=False, sep=_dump.INSTRUMENTS_SEP, index=False
                )


if __name__ == "__main__":
    # fire.Fire(Run)
    interval = "d"
    runner = Run(
        source_dir=r"D:\Documents\TradeResearch\qlib_test\rqdata_fut\source" + interval,
        normalize_dir=r"D:\Documents\TradeResearch\qlib_test\rqdata_fut\normalize" + interval,
        max_workers=6,
        interval=interval,
        commod_only=3,
    )
    # runner.download_data(max_collector_count=6, start=pd.Timestamp('2017-01-01'), end=pd.Timestamp.now().strftime("%Y-%m-%d"))
    # runner.normalize_data(qlib_data_1d_dir=r"D:\Documents\TradeResearch\qlib_test\rqdata_fut_commod")
    runner.update_data_to_bin(
        qlib_data_1d_dir=r"D:\Documents\TradeResearch\qlib_test\rqdata_fut",
        trading_date="2012-01-01",
        end_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
    )
