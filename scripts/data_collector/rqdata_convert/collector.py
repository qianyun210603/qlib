# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import copy
import time
import multiprocessing
import traceback
from pathlib import Path
from typing import Iterable, Optional
from arctic import Arctic
from arctic.date import DateRange
from tqdm import tqdm

import fire
import numpy as np
import pandas as pd
from loguru import logger
logger.add("rqdata2qlib.log", rotation="12:00")
from qlib.utils import code_to_fname


CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

INDEXES = {"csi300": "000300", "csi100": "000903", "csi500": "000905", 'csiconvert': "000832"}

INDICATOR_COLS = ['remaining_size', 'turnover_rate', "call_status",
                  'convertible_market_cap_ratio', 'conversion_price_reset_status']

from dump_bin import DumpDataUpdate, DumpDataAll, DumpDataFix
from data_collector.base import BaseCollector, BaseNormalize, BaseRun, Normalize
from data_collector.utils import get_calendar_list
from qlib.data.inst_info import ConvertInstrumentInfo


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
        import pytz
        self.arctic_store = Arctic("127.0.0.1", tz_aware=True, tzinfo=pytz.timezone('Asia/Shanghai'))
        self.meta_lib = self.arctic_store.get_library("convert_meta")
        self.convert_price_lib = self.arctic_store.get_library("convert_convert_price")
        self.convert_cf_lib = self.arctic_store.get_library("convert_cash_flow")
        self.coupon_lib = self.arctic_store.get_library("convert_coupon")

        self.bar_lib = self.arctic_store.get_library('bar_data')
        self.ex_factor_lib = self.arctic_store.get_library("ex_factor")  # 复权因子
        self.split_lib = self.arctic_store.get_library("split")  # 拆分信息

        self.derived_lib = self.arctic_store.get_library("convert_derived")
        self.indicator_lib = self.arctic_store.get_library("convert_indicator")

        interval = 'd' if interval.endswith('d') else '1m'

        # self.init_datetime()

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
        if self.interval not in {'d', '1d', '1m', '1min'}:
            raise ValueError(f"interval error: {self.interval}")

    @staticmethod
    def convert_datetime(dt: [pd.Timestamp, str], timezone):
        if isinstance(dt, str):
            return pd.Timestamp(dt, tz=timezone)
        if dt.tzinfo is None:
            return dt.tz_localize(timezone)
        return dt.tz_convert(timezone)

    def get_info(self, symbol):
        meta = self.meta_lib.read(symbol)
        coupon = self.coupon_lib.read(symbol)
        cash_flow = self.convert_cf_lib.read(symbol)
        coupon.index += pd.Timedelta(days=1)
        if cash_flow.index.max() != coupon.index.max():
            call_date = cash_flow.index.max()
        else:
            call_date = pd.Timestamp(year=2200, month=1, day=1)

        return ConvertInstrumentInfo(cash_flow.cash_flow, coupon.coupon_rate, meta['maturity_date'].replace(tzinfo=None), call_date=call_date)

    def get_data(
        self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> Optional[pd.DataFrame]:
        if interval not in {'d', '1d', '1m', '1min'}:
            raise ValueError(f"cannot support {interval}")
        # start_datetime = self.convert_datetime(start_datetime, self._timezone)
        # end_datetime = self.convert_datetime(end_datetime, self._timezone)
        metadata = self.meta_lib.read(symbol)
        interval = 'd' if interval.endswith('d') else '1m'
        db_symbol = symbol + '_' + interval
        _resultb = self.bar_lib.read(db_symbol, chunk_range=DateRange(start_datetime.tz_localize(None),
                                                                      end_datetime.tz_localize(None)))
        try:
            _resultb.set_index('date', inplace=True)
            convert_prices = self.convert_price_lib.read(symbol).set_index('effective_date')
            _resultb = pd.merge_asof(_resultb, convert_prices[['conversion_price']], left_index=True, right_index=True)
            stock_symbol =  metadata['stock_code'] + '_' + metadata['stock_exchange']
            db_stock_symbol = stock_symbol + '_' + interval
            _results = self.bar_lib.read(
                db_stock_symbol, chunk_range=DateRange(start_datetime.tz_localize(None), end_datetime.tz_localize(None)))
            _results.set_index('date', inplace=True)
        except Exception:
            logger.error(f"{start_datetime}, {end_datetime}, {db_symbol}, {symbol}:\n {sys.exc_info()}")
            return None

        if self.ex_factor_lib.has_symbol(stock_symbol):
            ex_factors = self.ex_factor_lib.read(stock_symbol)
            _results = pd.merge_asof(_results, ex_factors[['ex_cum_factor', 'ex_factor']], left_index=True,
                                    right_index=True)
        else:
            _results[['ex_cum_factor', 'ex_factor']] = 1.0

        if self.split_lib.has_symbol(stock_symbol):
            split_factor = self.split_lib.read(stock_symbol)
            _results = pd.merge_asof(
                _results, split_factor[['cum_factor']].rename(columns={'cum_factor': 'split_cum_factor'}),
                left_index=True, right_index=True)
        else:
            _results['split_cum_factor'] = 1.0

        _result = pd.merge(
            _resultb, _results.rename(columns={c: c+'_stock' for c in _results.columns}), left_index=True, right_index=True,
            how = 'left'
        )

        derived = self.derived_lib.read(
            symbol, chunk_range=DateRange(start_datetime.tz_localize(None), end_datetime.tz_localize(None))
        )

        indicator = self.indicator_lib.read(
            symbol, chunk_range=DateRange(start_datetime.tz_localize(None), end_datetime.tz_localize(None)),
            columns=INDICATOR_COLS
        )
        try:
            indicator['call_announced'] = indicator.call_status.apply(lambda x: 1.0 if x == 3 else 0).ffill()
            indicator['call_satisfied'] = indicator.call_status.apply(lambda x: 1.0 if x >= 2 else 0).ffill()
        except:
            logger.error(f"no call announced for {db_symbol}")
            return pd.DataFrame()

        indicator_cols = copy.deepcopy(INDICATOR_COLS)
        indicator_cols.remove("call_status")

        _resulta = pd.concat(
            [derived, indicator[indicator_cols + ['call_announced', 'call_satisfied']]],
            axis=1)

        _result = pd.merge(_result, _resulta, left_index=True, right_index=True, how = 'left')

        time.sleep(self.delay)

        return pd.DataFrame() if _result is None else _result

    def collector_data(self):
        """collector data"""
        super(RqdataCollector, self).collector_data()
        self.download_index_data()

    def get_instrument_list(self):
        def symbol_validation(_, v):
            if pd.isna(v['listed_date']):
                return False
            if pd.isna(v['de_listed_date']):
                return v['listed_date']<=self.end_datetime
            return self.start_datetime<=v['de_listed_date'] and v['listed_date'].replace(hour=16) <= min(self.end_datetime, pd.Timestamp.now('Asia/Shanghai')) #- pd.Timedelta(days=1)

        logger.info("get HS converts symbols......")
        symbol_dict = {x: self.meta_lib.read(x) for x in self.meta_lib.list_symbols()}
        symbols = [k for k, v in symbol_dict.items() if symbol_validation(k, v)]
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

    def download_index_data(self):
        _format = "%Y%m%d"
        _begin = self.start_datetime.tz_localize(None)
        _end = self.end_datetime.tz_localize(None)
        interval = 'd' if self.interval.endswith('d') else '1m'
        for _index_name, _index_code in INDEXES.items():
            logger.info(f"get bench data: {_index_name}({_index_code})......")
            try:
                exch_str = 'SZSE' if _index_code.startswith('INDEX399') else 'SSE'
                db_symbol = f'INDEX{_index_code}_{exch_str}_{interval}'
                df = self.bar_lib.read(db_symbol, chunk_range=DateRange(_begin, _end)).set_index('date')
                df[['ex_cum_factor', 'ex_factor', 'split_cum_factor']] = 1.0
                df[['call_announced']] = 0
            except Exception as e:
                logger.warning(f"get {_index_name} error: {e}")
                continue
            df["symbol"] = f"sh{_index_code}"
            _path = self.save_dir.joinpath(f"sh{_index_code}.csv")
            if _path.exists():
                _old_df = pd.read_csv(_path, index_col=['date'], parse_dates=['date'])
                df = pd.concat([_old_df[~_old_df.index.isin(df.index)], df], sort=True)
            df.to_csv(_path)
            time.sleep(1)

    def save_instrument(self, symbol, df: pd.DataFrame, info: ConvertInstrumentInfo):
        """save instrument data to file

        Parameters
        ----------
        symbol: str
            instrument code
        df: pd.DataFrame
            df.columns must contain "symbol" and "datetime"
        info:
            the information of the convert bond
        """
        if df is None or df.empty:
            logger.warning(f"{symbol} is empty")
            return

        symbol = self.normalize_symbol(symbol)
        symbol = code_to_fname(symbol)
        instrument_path = self.save_dir.joinpath(f"{symbol}.csv")
        df["symbol"] = symbol
        if instrument_path.exists():
            _old_df = pd.read_csv(instrument_path, index_col=['date'], parse_dates=['date'])
            df = pd.concat([_old_df[~_old_df.index.isin(df.index)], df], sort=True)
        df.to_csv(instrument_path)

        info_path = self.save_dir.joinpath(f"{symbol}.pkl")
        info.to_pickle(info_path)

    def _simple_collector(self, symbol: str):
        """

        Parameters
        ----------
        symbol: str

        """
        self.sleep()
        df = self.get_data(symbol, self.interval, self.start_datetime, self.end_datetime)
        info = self.get_info(symbol)
        _result = self.NORMAL_FLAG
        if self.check_data_length > 0:
            _result = self.cache_small_data(symbol, df)
        if _result == self.NORMAL_FLAG:
            self.save_instrument(symbol, df, info)
        return _result


class RqdataNormalize(BaseNormalize):
    SOURCE_COLS = [
        "open_price", "close_price", "high_price", "low_price", "volume", 'turnover', 'conversion_price',
        'open_price_stock', 'high_price_stock', 'low_price_stock', 'close_price_stock', 'volume_stock',
        'turnover_stock'
    ]
    COLUMNS = [
        "open", "close", "high", "low", "volume", 'money', 'conversionprice', 'openstock',
        'highstock', 'lowstock', 'closestock', 'volumestock', 'turnoverstock'
    ]
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
            tmp_idx_cal = pd.DatetimeIndex(calendar_list, name='date').sort_values()
            index_from_cal = tmp_idx_cal[
                tmp_idx_cal.searchsorted(df.index.min().replace(hour=0, minute=0, second=0)):
                tmp_idx_cal.searchsorted(df.index.max().replace(hour=23, minute=59, second=59))
            ]
            df = df.reindex(index_from_cal)

        # assign adjclose as close price adjusted by split only
        if 'closestock' in df.columns:
            df["adjclosestock"] = df.closestock
            # adjust ohlc by split and dividends
            df[['openstock', 'highstock', 'lowstock', 'closestock']] = df[['openstock', 'highstock', 'lowstock', 'closestock']].multiply(df.ex_cum_factor_stock, axis=0)
            df['factorstock'] = df.ex_cum_factor_stock

        df.sort_index(inplace=True)

        df.loc[(df["volume"] <= 1e-10) | np.isnan(df["volume"]), list(set(df.columns) - {"symbol"})] = np.nan

        _count = 0

        df["change"] = RqdataNormalize.calc_change(df, last_close)

        columns += ["change"]
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), columns] = np.nan

        df["symbol"] = symbol
        return df.drop(
            columns=['ex_cum_factor_stock', 'ex_factor_stock', 'split_cum_factor_stock',
                     'ex_cum_factor', 'ex_factor', 'split_cum_factor'],
            errors='ignore'
        ).reset_index()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # normalize
        df = self.normalize_rqdata(df, self._calendar_list)
        return df

    @staticmethod
    def _get_calendar_list() -> Iterable[pd.Timestamp]:
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
        check_data_length: bool = False,
        is_fix: bool = False
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
        self.download_data(max_collector_count=self.max_workers, start=trading_date, end=end_date, check_data_length=check_data_length)

        # normalize data
        self.normalize_data(qlib_data_1d_dir)

        # dump bin
        qlib_dir = Path(qlib_data_1d_dir).expanduser().resolve()

        DumpClass = DumpDataFix if is_fix else DumpDataUpdate if qlib_dir.joinpath(r"calendars\day.txt").exists() else DumpDataAll
        _dump = DumpClass(
            csv_path=self.normalize_dir,
            qlib_dir=qlib_data_1d_dir,
            exclude_fields="symbol,date",
            max_workers=self.max_workers,
        )
        _dump.dump()

        logger.info("Start copy contract info files")
        contract_spec_dir = qlib_dir.joinpath("contract_specs")
        contract_spec_dir.mkdir(parents=True, exist_ok=True)
        all_info_files = list(self.source_dir.glob('*.pkl'))
        for info_file in tqdm(all_info_files):
            contract_spec_dir.joinpath(info_file.name).write_bytes(info_file.read_bytes())
        logger.info("Copy contract info files done")

        logger.info("Exclude indexes from `all` to formulate `convert` population")
        # noinspection PyProtectedMember
        instruments_dir = _dump._instruments_dir
        #all_instrument_path = instrument_dir.joinpath(_dump.INSTRUMENTS_FILE_NAME)
        all_instrument_w_index = pd.read_csv(
            instruments_dir.joinpath(_dump.INSTRUMENTS_FILE_NAME),
            sep='\t', header=None
        )
        indexes = ['SH' + x for x in INDEXES.values()]
        all_instrument_wo_index = all_instrument_w_index[~all_instrument_w_index.iloc[:, 0].isin(indexes)]
        all_instrument_wo_index.to_csv(instruments_dir.joinpath("converts.txt"), header=False, sep=_dump.INSTRUMENTS_SEP, index=False)


if __name__ == "__main__":
    #fire.Fire(Run)
    runner = Run(
        source_dir=r"D:\Documents\TradeResearch\qlib_test\rqdata_convert\source",
        normalize_dir=r"D:\Documents\TradeResearch\qlib_test\rqdata_convert\normalize",
        max_workers=6
    )
    today =  pd.Timestamp.now().normalize()
    # runner.update_data_to_bin(
    #     qlib_data_1d_dir=r"D:\Documents\TradeResearch\qlib_test\rqdata_convert",
    #     trading_date=pd.Timestamp("2010-01-01"), end_date=today.strftime("%Y-%m-%d")
    # )

    runner.update_data_to_bin(
        qlib_data_1d_dir=r"D:\Documents\TradeResearch\qlib_test\rqdata_convert",
        trading_date=(today - pd.Timedelta(days=7)).strftime("%Y-%m-%d"), end_date=today.strftime("%Y-%m-%d")
    )
