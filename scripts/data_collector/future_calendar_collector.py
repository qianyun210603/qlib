# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
import importlib
from pathlib import Path
from typing import Union, Iterable, List

import fire
import numpy as np
import pandas as pd
import exchange_calendars as xcal

# pip install baostock
import baostock as bs
from loguru import logger

from utils import generate_minutes_calendar_from_daily


class CollectorFutureCalendar:
    calendar_format = "%Y-%m-%d"
    intraday_format = "%Y-%m-%d %H:%M:%S"

    def __init__(self, qlib_dir: Union[str, Path], start_date: str = None, end_date: str = None, freq="day"):
        """

        Parameters
        ----------
        qlib_dir:
            qlib data directory
        start_date
            start date
        end_date
            end date
        """
        assert freq in {"day", "1min"}, "freq must be either `day` or `1min`."
        self.freq = freq
        self.qlib_dir = Path(qlib_dir).expanduser().absolute()
        self.calendar_path = self.qlib_dir.joinpath(f"calendars/{freq}.txt")
        self.future_path = self.qlib_dir.joinpath(f"calendars/{freq}_future.txt")
        self._calendar_list = self.calendar_list
        _latest_date = self._calendar_list[-1]
        self.start_date = _latest_date if start_date is None else pd.Timestamp(start_date)
        self.end_date = _latest_date + pd.Timedelta(days=365 * 2) if end_date is None else pd.Timestamp(end_date)

    @property
    def calendar_list(self) -> List[pd.Timestamp]:
        # load old calendar
        if not self.calendar_path.exists():
            raise ValueError(f"calendar does not exist: {self.calendar_path}")
        calendar_df = pd.read_csv(self.calendar_path, header=None)
        calendar_df.columns = ["date"]
        calendar_df["date"] = pd.to_datetime(calendar_df["date"])
        return calendar_df["date"].to_list()

    def _format_datetime(self, datetime_d: [str, pd.Timestamp], freq="day"):
        datetime_d = pd.Timestamp(datetime_d)
        format_str = self.calendar_format if freq == "day" else self.intraday_format
        return datetime_d.strftime(format_str)

    def write_calendar(self, calendar: List):
        calendars_list = list(
            map(lambda x: self._format_datetime(x, self.freq), sorted(set(self.calendar_list + calendar)))
        )
        np.savetxt(self.future_path, X=np.array(calendars_list), fmt="%s", encoding="utf-8")

    @abc.abstractmethod
    def collector(self) -> List[pd.Timestamp]:
        """

        Returns
        -------

        """
        raise NotImplementedError(f"Please implement the `collector` method")


class CollectorFutureCalendarCN(CollectorFutureCalendar):
    def collector(self) -> List[pd.Timestamp]:
        lg = bs.login()
        if lg.error_code != "0":
            raise ValueError(f"login respond error_msg: {lg.error_msg}")
        rs = bs.query_trade_dates(
            start_date=self._format_datetime(self.start_date), end_date=self._format_datetime(self.end_date)
        )
        if rs.error_code != "0":
            raise ValueError(f"query_trade_dates respond error_msg: {rs.error_msg}")
        data_list = []
        while (rs.error_code == "0") & rs.next():
            data_list.append(rs.get_row_data())
        calendar = pd.DataFrame(data_list, columns=rs.fields)
        calendar["is_trading_day"] = calendar["is_trading_day"].astype(int)
        trading_dates = pd.to_datetime(calendar[calendar["is_trading_day"] == 1]["calendar_date"]).to_list()
        if trading_dates[-1] - pd.Timestamp.now().normalize() < pd.Timedelta(60):
            cal = xcal.get_calendar("XSHG")
            trading_dates += cal.sessions_in_range(
                trading_dates[-1] + pd.Timedelta(days=1), pd.Timestamp.now().normalize() + pd.Timedelta(days=60)
            ).to_list()
        if self.freq == "1min":
            trading_dates = generate_minutes_calendar_from_daily(trading_dates, self.freq).to_list()
        return trading_dates


class CollectorFutureCalendarUS(CollectorFutureCalendar):
    def collector(self) -> List[pd.Timestamp]:
        # TODO: US future calendar
        raise ValueError("Us calendar is not supported")


def run(
    qlib_dir: Union[str, Path], region: str = "cn", start_date: str = None, end_date: str = None, freq: str = "day"
):
    """Collect future calendar(day)

    Parameters
    ----------
    qlib_dir:
        qlib data directory
    region:
        cn/CN or us/US
    start_date
        start date
    end_date
        end date
    freq
        freq of calendar

    Examples
    -------
        # get cn future calendar
        $ python future_calendar_collector.py --qlib_data_1d_dir <user data dir> --region cn
    """
    logger.info(f"collector future calendar: region={region}")
    _cur_module = importlib.import_module("future_calendar_collector")
    _class = getattr(_cur_module, f"CollectorFutureCalendar{region.upper()}")
    collector = _class(qlib_dir=qlib_dir, start_date=start_date, end_date=end_date, freq=freq)
    collector.write_calendar(collector.collector())


if __name__ == "__main__":
    fire.Fire(run)
