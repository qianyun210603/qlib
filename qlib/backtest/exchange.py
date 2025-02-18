# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Type, Union, cast

from ..utils.index_data import IndexData

if TYPE_CHECKING:
    from .account import Account

import random

import numpy as np
import pandas as pd

from qlib.backtest.position import BasePosition

from ..config import C
from ..constant import REG_CN, REG_TW
from ..data.data import D
from ..log import get_module_logger
from .decision import Order, OrderDir, OrderHelper
from .high_performance_ds import BaseQuote, NumpyQuote


class Exchange:
    # `quote_dfs` is a pd.DataFrame class that contains basic information for backtesting
    # After some processing, the data will later be maintained by `quote_cls` object for faster data retrieving.
    # Some conventions for `quote_dfs`
    # - $close is for calculating the total value at end of each day.
    #   - if $close is None, the stock on that day is regarded as suspended.
    # - $factor is for rounding to the trading unit;
    #   - if any $factor is missing when $close exists, trading unit rounding will be disabled
    quote_dfs: Dict[str, pd.DataFrame] = {}

    def __init__(
        self,
        freq: str = "day",
        start_time: Union[pd.Timestamp, str] = None,
        end_time: Union[pd.Timestamp, str] = None,
        codes: Union[list, str] = "all",
        deal_price: Union[str, Tuple[str, str], List[str]] = None,
        subscribe_fields: Union[Iterable, Dict[str, Iterable]] = (),
        subscribe_fields_alias: dict = None,
        volume_threshold: Union[tuple, dict] = None,
        open_cost: float = 0.0015,
        close_cost: float = 0.0025,
        min_cost: float = 5.0,
        impact_cost: float = 0.0,
        extra_quote: pd.DataFrame = None,
        quote_cls: Type[BaseQuote] = NumpyQuote,
        **kwargs: Any,
    ) -> None:
        """__init__
        :param freq:             frequency of data
        :param start_time:       closed start time for backtest
        :param end_time:         closed end time for backtest
        :param codes:            list stock_id list or a string of instruments(i.e. all, csi500, sse50)
        :param deal_price:      Union[str, Tuple[str, str], List[str]]
                                The `deal_price` supports following two types of input
                                - <deal_price> : str
                                - (<buy_price>, <sell_price>): Tuple[str] or List[str]
                                <deal_price>, <buy_price> or <sell_price> := <price>
                                <price> := str
                                - for example '$close', '$open', '$vwap' ("close" is OK. `Exchange` will help to prepend
                                  "$" to the expression)
        :param subscribe_fields: Iterable (convertible to set), subscribe fields. This expressions will be added to the
                                 query and `self.quote`.
                                 It is useful when users want more fields to be queried
        :param subscribe_fields_alias: dict, subscribe fields alias. The keys are expressions and values are alias names.
        :param limit_threshold: Union[Tuple[str, str], List[str], float, None]
                                1) `None`: no limitation
                                2) float, 0.1 for example, default None
                                3) Tuple[str, str]: (<the expression for buying stock limitation>,
                                                     <the expression for sell stock limitation>)
                                                    `False` value indicates the stock is tradable
                                                    `True` value indicates the stock is limited and not tradable
        :param volume_threshold: Union[
                                    Dict[
                                        "all": ("cum" or "current", limit_str),
                                        "buy": ("cum" or "current", limit_str),
                                        "sell":("cum" or "current", limit_str),
                                    ],
                                    ("cum" or "current", limit_str),
                                 ]
                                1) ("cum" or "current", limit_str) denotes a single volume limit.
                                    - limit_str is qlib data expression which is allowed to define your own Operator.
                                    Please refer to qlib/contrib/ops/high_freq.py, here are any custom operator for
                                    high frequency, such as DayCumsum. !!!NOTE: if you want you use the custom
                                    operator, you need to register it in qlib_init.
                                    - "cum" means that this is a cumulative value over time, such as cumulative market
                                    volume. So when it is used as a volume limit, it is necessary to subtract the dealt
                                    amount.
                                    - "current" means that this is a real-time value and will not accumulate over time,
                                    so it can be directly used as a capacity limit.
                                    e.g. ("cum", "0.2 * DayCumsum($volume, '9:45', '14:45')"), ("current", "$bidV1")
                                2) "all" means the volume limits are both buying and selling.
                                "buy" means the volume limits of buying. "sell" means the volume limits of selling.
                                Different volume limits will be aggregated with min(). If volume_threshold is only
                                ("cum" or "current", limit_str) instead of a dict, the volume limits are for
                                both by default. In other words, it is same as {"all": ("cum" or "current", limit_str)}.
                                3) e.g. "volume_threshold": {
                                            "all": ("cum", "0.2 * DayCumsum($volume, '9:45', '14:45')"),
                                            "buy": ("current", "$askV1"),
                                            "sell": ("current", "$bidV1"),
                                        }
        :param open_cost:        cost rate for open, default 0.0015
        :param close_cost:       cost rate for close, default 0.0025
        :param trade_unit:       trade unit, 100 for China A market.
                                 None for disable trade unit.
                                 **NOTE**: `trade_unit` is included in the `kwargs`. It is necessary because we must
                                 distinguish `not set` and `disable trade_unit`
        :param min_cost:         min cost, default 5
        :param impact_cost:     market impact cost rate (a.k.a. slippage). A recommended value is 0.1.
        :param extra_quote:     pandas, dataframe consists of
                                    columns: like ['$vwap', '$close', '$volume', '$factor', 'limit_sell', 'limit_buy'].
                                            The limit indicates that the etf is tradable on a specific day.
                                            Necessary fields:
                                                $close is for calculating the total value at end of each day.
                                            Optional fields:
                                                $volume is only necessary when we limit the trade amount or calculate
                                                PA(vwap) indicator
                                                $vwap is only necessary when we use the $vwap price as the deal price
                                                $factor is for rounding to the trading unit
                                                limit_sell will be set to False by default (False indicates we can sell
                                                this target on this day).
                                                limit_buy will be set to False by default (False indicates we can buy
                                                this target on this day).
                                    index: MultipleIndex(instrument, pd.Datetime)
        """
        self.freq = freq
        self.start_time = pd.Timestamp(start_time)
        self.end_time = pd.Timestamp(end_time)

        self.trade_unit = kwargs.pop("trade_unit", C.trade_unit)

        # limit_threshold here shall be in kwargs to distinguish 'not provided' and `None` as `None` is a valid option here.
        # not setting in kwargs will lead to `None` explicitly provided here, which should shield the one in global config,
        # being overwrittin by the global config, which is counterintuitive.
        # Also allow input as list here since yaml cannot direct provide tuple, in such case explicitly convert to tuple.
        self.limit_threshold = kwargs.pop("limit_threshold", C.limit_threshold)
        if isinstance(self.limit_threshold, list):
            self.limit_threshold = tuple(self.limit_threshold)

        if deal_price is None:
            deal_price = C.deal_price

        # we have some verbose information here. So logging is enabled
        self.logger = get_module_logger("online operator")

        # TODO: the quote, trade_dates, codes are not necessary.
        # It is just for performance consideration.
        self.limit_type = self._get_limit_type()
        if self.limit_threshold is None:
            if C.region in [REG_CN, REG_TW]:
                self.logger.warning(f"limit_threshold not set. The stocks hit the limit may be bought/sold")
        elif self.limit_type == self.LT_FLT and abs(cast(float, self.limit_threshold)) > 0.1:
            if C.region in [REG_CN, REG_TW]:
                self.logger.warning(f"limit_threshold may not be set to a reasonable value")

        if isinstance(deal_price, str):
            if deal_price[0] != "$":
                deal_price = "$" + deal_price
            self.buy_price = self.sell_price = deal_price
        elif isinstance(deal_price, (tuple, list)):
            self.buy_price, self.sell_price = cast(Tuple[str, str], deal_price)
        else:
            raise NotImplementedError(f"This type of input is not supported")

        if isinstance(codes, str):
            codes = D.instruments(codes)
        self.codes = codes
        # Necessary fields
        # $close is for calculating the total value at end of each day.
        # - if $close is None, the stock on that day is regarded as suspended.
        # $factor is for rounding to the trading unit
        # $change is for calculating the limit of the stock

        # 　get volume limit from kwargs
        self.buy_vol_limit, self.sell_vol_limit, vol_lt_fields = self._get_vol_limit(volume_threshold)

        necessary_fields = {self.buy_price, self.sell_price, "$close", "$change", "$factor", "$volume"}
        if self.limit_type == self.LT_TP_EXP:
            assert isinstance(self.limit_threshold, tuple)
            for exp in self.limit_threshold:
                necessary_fields.add(exp)
        if not isinstance(subscribe_fields, Dict):
            subscribe_fields = {self.freq: subscribe_fields}

        self.all_fields = {
            freq: list(necessary_fields | set(vol_lt_fields) | set(s_fields))
            for freq, s_fields in subscribe_fields.items()
        }

        self.open_cost = open_cost
        self.close_cost = close_cost
        self.min_cost = min_cost
        self.impact_cost = impact_cost

        self.volume_threshold = volume_threshold
        self.extra_quote = extra_quote
        self.get_quote_from_qlib(subscribe_fields_alias)

        # init quote by quote_dfs
        self.quote_cls = quote_cls
        self.quotes: Dict[str, BaseQuote] = {
            freq: self.quote_cls(self.quote_dfs[freq], freq) for freq in self.all_fields.keys()
        }
        self.instrument_info = {}
        dpm_uri = C.dpm.provider_uri.get("day", C.dpm.provider_uri.get(C.DEFAULT_FREQ, None))
        if dpm_uri is not None and C.dpm.get_uri_type(dpm_uri) == "local":
            from pathlib import Path  # pylint: disable=import-outside-toplevel

            instrument_info_path = Path(dpm_uri).joinpath("contract_specs")
            if instrument_info_path.exists():
                import pickle  # pylint: disable=import-outside-toplevel

                self.instrument_info.update(
                    {
                        p.stem.upper(): pickle.load(open(p, "rb"))  # pylint: disable=consider-using-with
                        for p in instrument_info_path.glob("*.pkl")
                    }
                )

    @property
    def quote_df(self) -> pd.DataFrame:
        return self.quote_dfs[self.freq]

    @property
    def quote(self) -> BaseQuote:
        return self.quotes[self.freq]

    def get_quote_from_qlib(self, subscribe_fields_alias: Dict[str, str] = None) -> None:
        """
        init quote_dfs by qlib dataloader

        Parameters
        ----------
        subscribe_fields_alias: rename the fields in quote_dfs

        Returns
        -------

        """
        # get stock data from qlib
        if len(self.codes) == 0:
            self.codes = D.instruments()
        for freq in self.all_fields.keys():
            if freq in self.quote_dfs:
                min_date = self.quote_dfs[freq].index.get_level_values("datetime").min().normalize()
                max_date = self.quote_dfs[freq].index.get_level_values("datetime").max().normalize() + pd.Timedelta(
                    days=1
                )
                missing_fields = set(self.all_fields[freq]) - set(self.quote_dfs[freq].columns)

                if min_date <= self.start_time and max_date >= self.end_time and not bool(missing_fields):
                    self.logger.info(f"Using cached {freq} data,")
                    continue
                self.logger.info(f"{min_date}-{self.start_time}; {max_date}-{self.end_time}; {str(missing_fields)}")
                self.logger.info(f"clear insufficient {freq} data")
                del self.quote_dfs[freq]
            self.quote_dfs[freq] = D.features(
                self.codes,
                self.all_fields[freq],
                self.start_time,
                self.end_time,
                freq=freq,
                disk_cache=True,
            )
            if subscribe_fields_alias is not None:
                self.quote_dfs[freq].rename(columns=subscribe_fields_alias, inplace=True)
            self.quote_dfs[freq].columns = self.all_fields[freq]
            if not (C.get("ohlc_adjusted", True) or self.quote_dfs[freq]["$factor"].isna().all()):
                for f in {"$open", "$close", "$high", "$low"}.intersection(self.quote_dfs[freq].columns):
                    self.quote_dfs[freq][f] = self.quote_dfs[freq][f] * self.quote_dfs[freq]["$factor"]
                if "$volume" in self.quote_dfs[freq].columns:
                    self.quote_dfs[freq]["$volume"] = self.quote_dfs[freq]["$volume"] / self.quote_dfs[freq]["$factor"]

            # check buy_price data and sell_price data
            for attr in ("buy_price", "sell_price"):
                pstr = getattr(self, attr)  # price string
                if self.quote_dfs[freq][pstr].isna().any():
                    self.logger.warning("{} field data contains nan.".format(pstr))

        # update trade_w_adj_price
        if (self.quote_dfs[self.freq]["$factor"].isna() & ~self.quote_dfs[self.freq]["$close"].isna()).any():
            # The 'factor.day.bin' file not exists, and `factor` field contains `nan`
            # Use adjusted price
            self.trade_w_adj_price = True
            self.logger.warning("factor.day.bin file not exists or factor contains `nan`. Order using adjusted_price.")
            # if self.trade_unit is not None:
            #     self.logger.warning(f"trade unit {self.trade_unit} is not supported in adjusted_price mode.")
        else:
            # The `factor.day.bin` file exists and all data `close` and `factor` are not `nan`
            # Use normal price
            self.trade_w_adj_price = False
        # update limit
        self._update_limit()

        # concat extra_quote
        if self.extra_quote is not None:
            # process extra_quote
            if "$close" not in self.extra_quote:
                raise ValueError("$close is necessray in extra_quote")
            for attr in "buy_price", "sell_price":
                pstr = getattr(self, attr)  # price string
                if pstr not in self.extra_quote.columns:
                    self.extra_quote[pstr] = self.extra_quote["$close"]
                    self.logger.warning(f"No {pstr} set for extra_quote. Use $close as {pstr}.")
            if "$factor" not in self.extra_quote.columns:
                self.extra_quote["$factor"] = 1.0
                self.logger.warning("No $factor set for extra_quote. Use 1.0 as $factor.")
            if "limit_sell" not in self.extra_quote.columns:
                self.extra_quote["limit_sell"] = False
                self.logger.warning("No limit_sell set for extra_quote. All stock will be able to be sold.")
            if "limit_buy" not in self.extra_quote.columns:
                self.extra_quote["limit_buy"] = False
                self.logger.warning("No limit_buy set for extra_quote. All stock will be able to be bought.")
            assert set(self.extra_quote.columns) == set(self.quote_dfs[self.freq].columns) - {"$change"}
            self.quote_dfs[self.freq] = pd.concat([self.quote_dfs[self.freq], self.extra_quote], sort=False, axis=0)

    LT_TP_EXP = "(exp)"  # Tuple[str, str]:  the limitation is calculated by a Qlib expression.
    LT_FLT = "float"  # float:  the trading limitation is based on `abs($change) < limit_threshold`
    LT_NONE = "none"  # none:  there is no trading limitation

    def get_instrument_info(self, symbol: str) -> Any:
        return self.instrument_info.get(symbol, None)

    def _get_limit_type(self) -> str:
        """get limit type"""
        assert hasattr(self, "limit_threshold"), "self.limit_threshold must be set first in __init__."
        if isinstance(self.limit_threshold, tuple):
            return self.LT_TP_EXP
        elif isinstance(self.limit_threshold, float):
            return self.LT_FLT
        elif self.limit_threshold is None:
            return self.LT_NONE
        else:
            raise NotImplementedError(f"This type of `limit_threshold` is not supported")

    def _update_limit(self) -> None:
        assert hasattr(self, "limit_type") and hasattr(
            self, "limit_threshold"
        ), "self.limit_type and self.limit_threshold must be set first."
        # $close may contain NaN, the nan indicates that the stock is not tradable at that timestamp
        suspended = self.quote_dfs[self.freq]["$volume"].isna() | np.isclose(self.quote_dfs[self.freq]["$volume"], 0)
        # check limit_threshold
        limit_type = self.limit_type
        if limit_type == self.LT_NONE:
            self.quote_dfs[self.freq]["limit_buy"] = suspended
            self.quote_dfs[self.freq]["limit_sell"] = suspended
        elif limit_type == self.LT_TP_EXP:
            # set limit
            expressions = cast(tuple, self.limit_threshold)
            if len(expressions) == 2:
                lb_row, ls_row = cast(tuple, self.limit_threshold)
                # astype bool is necessary, because quote_dfs is an expression and could be float
                self.quote_dfs[self.freq]["limit_buy"] = self.quote_dfs[self.freq][lb_row].astype("bool") | suspended
                self.quote_dfs[self.freq]["limit_sell"] = self.quote_dfs[self.freq][ls_row].astype("bool") | suspended
            elif len(expressions) == 4:
                lb_col, ls_col, wlb_col, wls_col = cast(tuple, self.limit_threshold)
                self.quote_dfs[self.freq]["limit_buy"] = self.quote_dfs[self.freq][lb_col].astype("bool") | suspended
                self.quote_dfs[self.freq]["limit_sell"] = self.quote_dfs[self.freq][ls_col].astype("bool") | suspended
                self.quote_dfs[self.freq]["whole_limit_buy"] = (
                    self.quote_dfs[self.freq][wlb_col].astype("bool") | suspended
                )
                self.quote_dfs[self.freq]["whole_limit_sell"] = (
                    self.quote_dfs[self.freq][wls_col].astype("bool") | suspended
                )
        elif limit_type == self.LT_FLT:
            limit_ratio: float = cast(float, self.limit_threshold)
            self.quote_dfs[self.freq]["limit_buy"] = self.quote_dfs[self.freq]["$change"].ge(limit_ratio) | suspended
            self.quote_dfs[self.freq]["limit_sell"] = (
                self.quote_dfs[self.freq]["$change"].le(-limit_ratio) | suspended  # pylint: disable=E1130
            )

    @staticmethod
    def _get_vol_limit(volume_threshold: Union[tuple, dict, None]) -> Tuple[Optional[list], Optional[list], set]:
        """
        preprocess the volume limit.
        get the fields need to get from qlib.
        get the volume limit list of buying and selling which is composed of all limits.
        Parameters
        ----------
        volume_threshold :
            please refer to the doc of exchange.
        Returns
        -------
        fields: set
            the fields need to get from qlib.
        buy_vol_limit: List[Tuple[str]]
            all volume limits of buying.
        sell_vol_limit: List[Tuple[str]]
            all volume limits of selling.
        Raises
        ------
        ValueError
            the format of volume_threshold is not supported.
        """
        if volume_threshold is None:
            return None, None, set()

        fields = set()
        buy_vol_limit = []
        sell_vol_limit = []
        if isinstance(volume_threshold, tuple):
            volume_threshold = {"all": volume_threshold}

        assert isinstance(volume_threshold, dict)
        for key, vol_limit in volume_threshold.items():
            assert isinstance(vol_limit, tuple)
            fields.add(vol_limit[1])

            if key in ("buy", "all"):
                buy_vol_limit.append(vol_limit)
            if key in ("sell", "all"):
                sell_vol_limit.append(vol_limit)

        return buy_vol_limit, sell_vol_limit, fields

    def check_stock_limit(
        self,
        stock_id: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        direction: int | None = None,
        freq: str | None = None,
    ) -> bool:
        """
        Parameters
        ----------
        stock_id : str
        start_time: pd.Timestamp
        end_time: pd.Timestamp
        direction : int, optional
            trade direction, by default None
            - if direction is None, check if tradable for buying and selling.
            - if direction == Order.BUY, check the if tradable for buying
            - if direction == Order.SELL, check the sell limit for selling.
        freq : str | None
        Returns
        -------
        True: the trading of the stock is limited (maybe hit the highest/lowest price), hence the stock is not tradable
        False: the trading of the stock is not limited, hence the stock may be tradable
        """
        # NOTE:
        # **all** is used when checking limitation.
        # For example, the stock trading is limited in a day if every minute is limited in a day if every minute is limited.
        quote = self.quote if freq is None else self.quotes[freq]
        if direction is None:
            # The trading limitation is related to the trading direction
            # if the direction is not provided, then any limitation from buy or sell will result in trading limitation
            buy_limit = quote.get_data(stock_id, start_time, end_time, field="limit_buy", method="all")
            sell_limit = quote.get_data(stock_id, start_time, end_time, field="limit_sell", method="all")
            return bool(buy_limit or sell_limit)
        elif direction == Order.BUY:
            return cast(bool, quote.get_data(stock_id, start_time, end_time, field="limit_buy", method="all"))
        elif direction == Order.SELL:
            return cast(bool, quote.get_data(stock_id, start_time, end_time, field="limit_sell", method="all"))
        else:
            raise ValueError(f"direction {direction} is not supported!")

    def check_stock_suspended(
        self, stock_id: str, start_time: pd.Timestamp, end_time: pd.Timestamp, freq: str | None = None
    ) -> bool:
        """if stock is suspended(hence not tradable), True will be returned"""
        quote = self.quote if freq is None else self.quotes[freq]
        # is suspended
        if stock_id in quote.get_all_stock():
            # suspended stocks are represented by None $close stock
            # The $close may contain NaN,
            volume = quote.get_data(stock_id, start_time, end_time, "$volume")
            if volume is None:
                # if no volume record exists
                return True
            elif isinstance(volume, IndexData):
                # **any** non-NaN $close represents trading opportunity may exist
                #  if all returned is nan, then the stock is suspended
                idx_data = cast(IndexData, volume)
                return cast(bool, idx_data.isna().all() or np.isclose(idx_data.data, 0).all())
            else:
                # it is single value, make sure is not None
                return cast(bool, np.isnan(volume) or np.isclose(volume, 0))
        else:
            # if the stock is not in the stock list, then it is not tradable and regarded as suspended
            return True

    def is_stock_tradable(
        self,
        stock_id: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        direction: int | None = None,
        freq: str | None = None,
    ) -> bool:
        # check if stock can be traded
        return not (
            self.check_stock_suspended(stock_id, start_time, end_time, freq)
            or self.check_stock_limit(stock_id, start_time, end_time, direction, freq)
        )

    def check_order(self, order: Order) -> bool:
        # check limit and suspended
        if isinstance(order.price, float):
            return True
        return self.is_stock_tradable(order.stock_id, order.start_time, order.end_time, order.direction)

    def deal_order(
        self,
        order: Order,
        trade_account: Account | None = None,
        position: BasePosition | None = None,
        dealt_order_amount: Dict[str, float] = defaultdict(float),
    ) -> Tuple[float, float, float]:
        """
        Deal order when the actual transaction
        the results section in `Order` will be changed.
        :param order:  Deal the order.
        :param trade_account: Trade account to be updated after dealing the order.
        :param position: position to be updated after dealing the order.
        :param dealt_order_amount: the dealt order amount dict with the format of {stock_id: float}
        :return: trade_val, trade_cost, trade_price
        """
        # check order first.
        if not self.check_order(order):
            order.deal_amount = 0.0
            # using np.nan instead of None to make it more convenient to show the value in format string
            self.logger.debug(f"Order failed due to trading limitation: {order}")
            return 0.0, 0.0, np.nan

        if trade_account is not None and position is not None:
            raise ValueError("trade_account and position can only choose one")

        # NOTE: order will be changed in this function
        trade_price, trade_val, trade_cost = self._calc_trade_info_by_order(
            order,
            trade_account.current_position if trade_account else position,
            dealt_order_amount,
        )
        if trade_val > 1e-5:
            # If the order can only be deal 0 value. Nothing to be updated
            # Otherwise, it will result in
            # 1) some stock with 0 value in the position
            # 2) `trade_unit` of trade_cost will be lost in user account
            if trade_account:
                trade_account.update_order(order=order, trade_val=trade_val, cost=trade_cost, trade_price=trade_price)
            elif position:
                position.update_order(order=order, trade_val=trade_val, cost=trade_cost, trade_price=trade_price)

        return trade_val, trade_cost, trade_price

    def get_quote_info(
        self,
        stock_id: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        field: str,
        method: str = "ts_data_last",
        freq: str | None = None,
    ) -> Union[None, int, float, bool, IndexData]:
        quote = self.quote if freq is None else self.quotes[freq]
        return quote.get_data(stock_id, start_time, end_time, field=field, method=method)

    def get_close(
        self,
        stock_id: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        method: str = "ts_data_last",
        freq: str | None = None,
    ) -> Union[None, int, float, bool, IndexData]:
        quote = self.quote if freq is None else self.quotes[freq]
        return quote.get_data(stock_id, start_time, end_time, field="$close", method=method)

    def get_volume(
        self,
        stock_id: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        method: Optional[str] = "sum",
        freq: str | None = None,
    ) -> Union[None, int, float, bool, IndexData]:
        """get the total deal volume of stock with `stock_id` between the time interval [start_time, end_time)"""
        quote = self.quote if freq is None else self.quotes[freq]
        return quote.get_data(stock_id, start_time, end_time, field="$volume", method=method)

    def get_deal_price(
        self,
        stock_id: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        direction: OrderDir,
        method: Optional[str] = "ts_data_last",
    ) -> Union[None, int, float, bool, IndexData]:
        if direction == OrderDir.SELL:
            pstr = self.sell_price
        elif direction == OrderDir.BUY:
            pstr = self.buy_price
        else:
            raise NotImplementedError(f"This type of input is not supported")

        deal_price = self.quote.get_data(stock_id, start_time, end_time, field=pstr, method=method)
        if method is not None and (deal_price is None or np.isnan(deal_price) or deal_price <= 1e-08):
            self.logger.warning(f"(stock_id:{stock_id}, trade_time:{(start_time, end_time)}, {pstr}): {deal_price}!!!")
            self.logger.warning(f"setting deal_price to close price")
            deal_price = self.get_close(stock_id, start_time, end_time, method)
        return deal_price

    def get_factor(
        self,
        stock_id: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        freq: str | None = None,
    ) -> Optional[float]:
        """
        Returns
        -------
        Optional[float]:
            `None`: if the stock is suspended `None` may be returned
            `float`: return factor if the factor exists
        """
        assert start_time is not None and end_time is not None, "the time range must be given"
        quote = self.quote if freq is None else self.quotes[freq]
        if stock_id not in quote.get_all_stock():
            return None
        return quote.get_data(stock_id, start_time, end_time, field="$factor", method="ts_data_last")

    def generate_amount_position_from_weight_position(
        self,
        weight_position: dict,
        cash: float,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        direction: OrderDir = OrderDir.BUY,
    ) -> dict:
        """
        Generates the target position according to the weight and the cash.
        NOTE: All the cash will be assigned to the tradable stock.
        Parameter:
        weight_position : dict {stock_id : weight}; allocate cash by weight_position
            among then, weight must be in this range: 0 < weight < 1
        cash : cash
        start_time : the start time point of the step
        end_time : the end time point of the step
        direction : the direction of the deal price for estimating the amount
                    # NOTE: this function is used for calculating target position. So the default direction is buy
        """

        # calculate the total weight of tradable value
        tradable_weight = 0.0
        for stock_id, wp in weight_position.items():
            if self.is_stock_tradable(stock_id=stock_id, start_time=start_time, end_time=end_time):
                # weight_position must be greater than 0 and less than 1
                if wp < 0 or wp > 1:
                    raise ValueError(
                        "weight_position is {}, " "weight_position is not in the range of (0, 1).".format(wp),
                    )
                tradable_weight += wp

        if tradable_weight - 1.0 >= 1e-5:
            raise ValueError("tradable_weight is {}, can not greater than 1.".format(tradable_weight))

        amount_dict = {}
        for stock_id in weight_position:
            if weight_position[stock_id] > 0.0 and self.is_stock_tradable(
                stock_id=stock_id,
                start_time=start_time,
                end_time=end_time,
            ):
                amount_dict[stock_id] = (
                    cash
                    * weight_position[stock_id]
                    / tradable_weight
                    // self.get_deal_price(
                        stock_id=stock_id,
                        start_time=start_time,
                        end_time=end_time,
                        direction=direction,
                    )
                )
        return amount_dict

    def get_real_deal_amount(self, current_amount: float, target_amount: float, factor: float | None = None) -> float:
        """
        Calculate the real adjust deal amount when considering the trading unit
        :param current_amount:
        :param target_amount:
        :param factor:
        :return real_deal_amount;  Positive deal_amount indicates buying more stock.
        """
        if current_amount == target_amount:
            return 0
        elif current_amount < target_amount:
            deal_amount = target_amount - current_amount
            deal_amount = self.round_amount_by_trade_unit(deal_amount, factor)
            return deal_amount
        else:
            if target_amount == 0:
                return -current_amount
            else:
                deal_amount = current_amount - target_amount
                deal_amount = self.round_amount_by_trade_unit(deal_amount, factor)
                return -deal_amount

    def generate_order_for_target_amount_position(
        self,
        target_position: dict,
        current_position: dict,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ) -> List[Order]:
        """
        Note: some future information is used in this function
        Parameter:
        target_position : dict { stock_id : amount }
        current_position : dict { stock_id : amount}
        trade_unit : trade_unit
        down sample : for amount 321 and trade_unit 100, deal_amount is 300
        deal order on trade_date
        """
        # split buy and sell for further use
        buy_order_list = []
        sell_order_list = []
        # three parts: kept stock_id, dropped stock_id, new stock_id
        # handle kept stock_id

        # because the order of the set is not fixed, the trading order of the stock is different, so that the backtest
        # results of the same parameter are different;
        # so here we sort stock_id, and then randomly shuffle the order of stock_id
        # because the same random seed is used, the final stock_id order is fixed
        sorted_ids = sorted(set(list(current_position.keys()) + list(target_position.keys())))
        random.seed(0)
        random.shuffle(sorted_ids)
        for stock_id in sorted_ids:
            # Do not generate order for the non-tradable stocks
            if not self.is_stock_tradable(stock_id=stock_id, start_time=start_time, end_time=end_time):
                continue

            target_amount = target_position.get(stock_id, 0)
            current_amount = current_position.get(stock_id, 0)
            factor = self.get_factor(stock_id, start_time=start_time, end_time=end_time)

            deal_amount = self.get_real_deal_amount(current_amount, target_amount, factor)
            if deal_amount == 0:
                continue
            if deal_amount > 0:
                # buy stock
                buy_order_list.append(
                    Order(
                        stock_id=stock_id,
                        amount=deal_amount,
                        direction=Order.BUY,
                        start_time=start_time,
                        end_time=end_time,
                        factor=factor,
                    ),
                )
            else:
                # sell stock
                sell_order_list.append(
                    Order(
                        stock_id=stock_id,
                        amount=abs(deal_amount),
                        direction=Order.SELL,
                        start_time=start_time,
                        end_time=end_time,
                        factor=factor,
                    ),
                )
        # return order_list : buy + sell
        return sell_order_list + buy_order_list

    def calculate_amount_position_value(
        self,
        amount_dict: dict,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        only_tradable: bool = False,
        direction: OrderDir = OrderDir.SELL,
    ) -> float:
        """Parameter
        position : Position()
        amount_dict : {stock_id : amount}
        direction : the direction of the deal price for estimating the amount
                    # NOTE:
                    This function is used for calculating current position value.
                    So the default direction is sell.
        """
        value = 0
        for stock_id in amount_dict:
            if not only_tradable or (
                not self.check_stock_suspended(stock_id=stock_id, start_time=start_time, end_time=end_time)
                and not self.check_stock_limit(stock_id=stock_id, start_time=start_time, end_time=end_time)
            ):
                value += (
                    self.get_deal_price(
                        stock_id=stock_id,
                        start_time=start_time,
                        end_time=end_time,
                        direction=direction,
                    )
                    * amount_dict[stock_id]
                )
        return value

    def _get_factor_or_raise_error(
        self,
        factor: float | None = None,
        stock_id: str | None = None,
        start_time: pd.Timestamp = None,
        end_time: pd.Timestamp = None,
    ) -> float:
        """Please refer to the docs of get_amount_of_trade_unit"""
        if factor is None:
            if stock_id is not None and start_time is not None and end_time is not None:
                factor = self.get_factor(stock_id=stock_id, start_time=start_time, end_time=end_time)
            else:
                raise ValueError(f"`factor` and (`stock_id`, `start_time`, `end_time`) can't both be None")
        assert factor is not None
        return factor

    def get_amount_of_trade_unit(
        self,
        factor: float | None = None,
        stock_id: str | None = None,
        start_time: pd.Timestamp = None,
        end_time: pd.Timestamp = None,
    ) -> Optional[float]:
        """
        get the trade unit of amount based on **factor**
        the factor can be given directly or calculated in given time range and stock id.
        `factor` has higher priority than `stock_id`, `start_time` and `end_time`
        Parameters
        ----------
        factor : float
            the adjusted factor
        stock_id : str
            the id of the stock
        start_time :
            the start time of trading range
        end_time :
            the end time of trading range
        """
        if self.trade_unit is not None:
            factor = (
                self._get_factor_or_raise_error(
                    factor=factor, stock_id=stock_id, start_time=start_time, end_time=end_time
                )
                if not self.trade_w_adj_price
                else 1
            )

            return self.trade_unit / factor
        else:
            return None

    def round_amount_by_trade_unit(
        self,
        deal_amount: float,
        factor: float | None = None,
        stock_id: str | None = None,
        start_time: pd.Timestamp = None,
        end_time: pd.Timestamp = None,
    ) -> float:
        """Parameter
        Please refer to the docs of get_amount_of_trade_unit
        deal_amount : float, adjusted amount
        factor : float, adjusted factor
        return : float, real amount
        """
        if self.trade_unit is not None:
            # the minimal amount is 1. Add 0.1 for solving precision problem.
            factor = (
                self._get_factor_or_raise_error(
                    factor=factor, stock_id=stock_id, start_time=start_time, end_time=end_time
                )
                if not self.trade_w_adj_price
                else 1
            )

            return (deal_amount * factor + 0.1) // self.trade_unit * self.trade_unit / factor
        return deal_amount

    def _clip_amount_by_volume(self, order: Order, dealt_order_amount: dict) -> Optional[float]:
        """parse the capacity limit string and return the actual amount of orders that can be executed.
        NOTE:
            this function will change the order.deal_amount **inplace**
            - This will make the order info more accurate
        Parameters
        ----------
        order : Order
            the order to be executed.
        dealt_order_amount : dict
            :param dealt_order_amount: the dealt order amount dict with the format of {stock_id: float}
        """
        vol_limit = self.buy_vol_limit if order.direction == Order.BUY else self.sell_vol_limit

        if vol_limit is None:
            return order.deal_amount

        vol_limit_num: List[float] = []
        for limit in vol_limit:
            assert isinstance(limit, tuple)
            if limit[0] == "current":
                limit_value = self.quote.get_data(
                    order.stock_id,
                    order.start_time,
                    order.end_time,
                    field=limit[1],
                    method="sum",
                )
                vol_limit_num.append(cast(float, limit_value))
            elif limit[0] == "cum":
                limit_value = self.quote.get_data(
                    order.stock_id,
                    order.start_time,
                    order.end_time,
                    field=limit[1],
                    method="ts_data_last",
                )
                vol_limit_num.append(limit_value - dealt_order_amount[order.stock_id])
            else:
                raise ValueError(f"{limit[0]} is not supported")
        vol_limit_min = min(vol_limit_num)
        orig_deal_amount = order.deal_amount
        order.deal_amount = max(min(vol_limit_min, orig_deal_amount), 0)
        if vol_limit_min < orig_deal_amount:
            self.logger.debug(f"Order clipped due to volume limitation: {order}, {list(zip(vol_limit_num, vol_limit))}")

        return None

    def _get_buy_amount_by_cash_limit(self, trade_price: float, cash: float, cost_ratio: float) -> float:
        """return the real order amount after cash limit for buying.
        Parameters
        ----------
        trade_price : float
        cash : float
        cost_ratio : float

        Return
        ----------
        float
            the real order amount after cash limit for buying.
        """
        max_trade_amount = 0.0
        if cash >= self.min_cost:
            # critical_price means the stock transaction price when the service fee is equal to min_cost.
            critical_price = self.min_cost / cost_ratio + self.min_cost
            if cash >= critical_price:
                # the service fee is equal to cost_ratio * trade_amount
                max_trade_amount = cash / (1 + cost_ratio) / trade_price
            else:
                # the service fee is equal to min_cost
                max_trade_amount = (cash - self.min_cost) / trade_price
        return max_trade_amount

    def _calc_trade_info_by_order(
        self,
        order: Order,
        position: Optional[BasePosition],
        dealt_order_amount: dict,
    ) -> Tuple[float, float, float]:
        """
        Calculation of trade info
        **NOTE**: Order will be changed in this function
        :param order:
        :param position: Position
        :param dealt_order_amount: the dealt order amount dict with the format of {stock_id: float}
        :return: trade_price, trade_val, trade_cost
        """
        if isinstance(order.price, str):
            trade_price = self.quote.get_data(
                order.stock_id, order.start_time, order.end_time, field=order.price, method="ts_data_last"
            )
        elif not pd.isna(order.price):
            trade_price = order.price
        else:
            trade_price = cast(
                float,
                self.get_deal_price(order.stock_id, order.start_time, order.end_time, direction=order.direction),
            )
        total_trade_val = cast(float, self.get_volume(order.stock_id, order.start_time, order.end_time)) * trade_price
        order.factor = self.get_factor(order.stock_id, order.start_time, order.end_time)
        order.deal_amount = order.amount  # set to full amount and clip it step by step
        # Clipping amount first
        # - It simulates that the order is rejected directly by the exchange due to large order
        # Another choice is placing it after rounding the order
        # - It simulates that the large order is submitted, but partial is dealt regardless of rounding by trading unit.
        self._clip_amount_by_volume(order, dealt_order_amount)

        # TODO: the adjusted cost ratio can be overestimated as deal_amount will be clipped in the next steps
        trade_val = order.deal_amount * trade_price
        if not total_trade_val or np.isnan(total_trade_val):
            # TODO: assert trade_val == 0, f"trade_val != 0, total_trade_val: {total_trade_val}; order info: {order}"
            adj_cost_ratio = self.impact_cost
        else:
            adj_cost_ratio = self.impact_cost * (trade_val / total_trade_val) ** 2

        if order.direction == Order.SELL:
            cost_ratio = self.close_cost + adj_cost_ratio
            # sell
            # if we don't know current position, we choose to sell all
            # Otherwise, we clip the amount based on current position
            if position is not None:
                current_amount = (
                    position.get_stock_amount(order.stock_id) if position.check_stock(order.stock_id) else 0
                )
                if not np.isclose(order.deal_amount, current_amount):
                    # when not selling last stock. rounding is necessary
                    order.deal_amount = self.round_amount_by_trade_unit(
                        min(current_amount, order.deal_amount),
                        order.factor,
                    )

                # in case of negative value of cash
                if position.get_cash() + order.deal_amount * trade_price < max(
                    order.deal_amount * trade_price * cost_ratio,
                    self.min_cost,
                ):
                    order.deal_amount = 0
                    self.logger.debug(f"Order clipped due to cash limitation: {order}")

        elif order.direction == Order.BUY:
            cost_ratio = self.open_cost + adj_cost_ratio
            # buy
            if position is not None:
                cash = position.get_cash()
                trade_val = order.deal_amount * trade_price
                if cash < max(trade_val * cost_ratio, self.min_cost):
                    # cash cannot cover cost
                    order.deal_amount = 0
                    self.logger.debug(f"Order clipped due to cost higher than cash: {order}")
                elif cash < trade_val + max(trade_val * cost_ratio, self.min_cost):
                    # The money is not enough
                    max_buy_amount = self._get_buy_amount_by_cash_limit(trade_price, cash, cost_ratio)
                    order.deal_amount = self.round_amount_by_trade_unit(
                        min(max_buy_amount, order.deal_amount),
                        order.factor,
                    )
                    self.logger.debug(f"Order clipped due to cash limitation: {order}")
                else:
                    # The money is enough
                    order.deal_amount = self.round_amount_by_trade_unit(order.deal_amount, order.factor)
            else:
                # Unknown amount of money. Just round the amount
                order.deal_amount = self.round_amount_by_trade_unit(order.deal_amount, order.factor)

        else:
            raise NotImplementedError("order direction {} error".format(order.direction))

        trade_val = order.deal_amount * trade_price
        trade_cost = max(trade_val * cost_ratio, self.min_cost)
        if trade_val <= 1e-5:
            # if dealing is not successful, the trade_cost should be zero.
            trade_cost = 0
        return trade_price, trade_val, trade_cost

    def get_order_helper(self) -> OrderHelper:
        if not hasattr(self, "_order_helper"):
            # cache to avoid recreate the same instance
            self._order_helper = OrderHelper(self)
        return self._order_helper
