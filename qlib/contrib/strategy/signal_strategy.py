# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import abc
import copy
import os
import warnings
from abc import ABC
from typing import Dict, List, Text, Tuple, Union, cast

import numpy as np
import pandas as pd

from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.backtest.position import Position
from qlib.backtest.signal import Signal, create_signal_from
from qlib.contrib.strategy.optimizer import EnhancedIndexingOptimizer
from qlib.contrib.strategy.order_generator import OrderGenerator, OrderGenWOInteract
from qlib.data import D
from qlib.data.dataset import Dataset
from qlib.log import get_module_logger
from qlib.model.base import BaseModel
from qlib.strategy.base import BaseStrategy
from qlib.utils import get_pre_trading_date, load_dataset


class BaseSignalStrategy(BaseStrategy, ABC):
    def __init__(
        self,
        *,
        signal: Union[Signal, Tuple[BaseModel, Dataset], List, Dict, Text, pd.Series, pd.DataFrame] = None,
        model=None,
        dataset=None,
        risk_degree: float = 0.95,
        trade_exchange=None,
        level_infra=None,
        common_infra=None,
        **kwargs,
    ):
        """
        Parameters
        -----------
        signal :
            the information to describe a signal. Please refer to the docs of `qlib.backtest.signal.create_signal_from`
            the decision of the strategy will base on the given signal
        risk_degree : float
            position percentage of total value.
        trade_exchange : Exchange
            exchange that provides market info, used to deal order and generate report
            - If `trade_exchange` is None, self.trade_exchange will be set with common_infra
            - It allowes different trade_exchanges is used in different executions.
            - For example:
                - In daily execution, both daily exchange and minutely are usable, but the daily exchange is recommended because it runs faster.
                - In minutely execution, the daily exchange is not usable, only the minutely exchange is recommended.

        """
        super().__init__(
            outer_trade_decision=kwargs.get("outer_trade_decision", None),
            level_infra=level_infra,
            common_infra=common_infra,
            trade_exchange=trade_exchange,
        )

        self.risk_degree = risk_degree

        # This is trying to be compatible with previous version of qlib task config
        if model is not None and dataset is not None:
            warnings.warn("`model` `dataset` is deprecated; use `signal`.", DeprecationWarning)
            signal = model, dataset

        self.signal: Signal = create_signal_from(signal)

    def get_risk_degree(self, trade_step=None):
        """get_risk_degree
        Return the proportion of your total value you will use in investment.
        Dynamically risk_degree will result in Market timing.
        """
        # It will use 95% amount of your total value by default
        return self.risk_degree

    @abc.abstractmethod
    def generate_trade_candidates(self, trade_time, *args, redundancy=0, **kwargs) -> pd.DataFrame:
        raise NotImplementedError("Implement in subclasses")


class BaseTopkStrategy(BaseSignalStrategy):
    """
    hold top k instruments with equal amount of market value based on score
    (to reduce turnover we may switch less than k instruments in one period, so it is not exactly equal amount of top k).
    """

    def __init__(self, topk, hold_thresh, market=None, forbid_all_trade_at_limit=True, **kwargs):
        self.hold_thresh = hold_thresh
        self.topk = topk
        if market is None:
            self.instruments = None
        else:
            if isinstance(market, str):
                market = D.instruments(market)
            if isinstance(market, dict) and "market" in market:
                self.instruments = D.list_instruments(market, freq="day")
                self.forbid_buy_days = kwargs.get("forbid_buy_days", -2)
                self.force_sell_days = kwargs.get("force_sell_days", -2)
            else:
                self.instruments = None
        self.forbid_all_trade_at_limit = forbid_all_trade_at_limit
        self.delist_schedule = {"SZ000418": pd.Timestamp("2019-05-07")}
        self.use_prev_close_for_amount = kwargs.get("use_prev_close_for_amount", False)
        super().__init__(**kwargs)

    def filter_instruments_by_market(self, pred_score, current_stock_list, trade_start_time, trade_end_time):
        def _check_delist(code):
            return code in self.delist_schedule and trade_end_time >= self.delist_schedule[code]

        if self.instruments is None:
            force_sell = set(code for code in current_stock_list if _check_delist(code))
            remain_holding = set(current_stock_list) - set(force_sell)

            forbid_buy = np.array([_check_delist(code) for code in pred_score.index])

            pred_score = pred_score[
                ~(pred_score.index.isin(force_sell) | forbid_buy) | pred_score.index.isin(remain_holding)
            ]

            return pred_score, list(remain_holding), list(force_sell)

        # FIXME better way to distinguish true end of range
        max_date = max(max(r[-1] for r in ranges) for ranges in self.instruments.values())

        def _days_to_remove(code):
            if code not in self.instruments:
                return -100000
            prev_ed = pd.Timestamp(year=1970, month=1, day=1)
            for st, ed in sorted(self.instruments[code]):
                if ed == max_date and trade_end_time >= ed:
                    return 1
                if trade_start_time <= ed <= trade_end_time:
                    return 0
                if st <= trade_end_time < ed:
                    return (ed.normalize() - trade_end_time.normalize()).days
                if trade_end_time < st:
                    return (prev_ed - trade_start_time.normalize()).days
                prev_ed = ed.normalize()
            return (prev_ed - trade_start_time.normalize()).days

        force_sell = set(
            code for code in current_stock_list if _days_to_remove(code) <= self.force_sell_days or _check_delist(code)
        )
        remain_holding = set(current_stock_list) - set(force_sell)

        forbid_buy = np.array(
            [
                (code not in remain_holding) and (_days_to_remove(code) <= self.forbid_buy_days or _check_delist(code))
                for code in pred_score.index
            ]
        )

        pred_score = pred_score[
            ~(pred_score.index.isin(force_sell) | forbid_buy) | pred_score.index.isin(remain_holding)
        ]

        return pred_score, list(remain_holding), list(force_sell)

    def _generate_buy_sell_list(
        self,
        pred_score,
        current_stock_list,
        trade_start_time,
        trade_end_time,
        pred_start_time,
        pred_end_time,
        redundancy=0,
    ) -> Tuple[List, List]:
        raise NotImplementedError("Please implement `_generate_buy_sell_list` method")

    def _generate_decisions_from_bs_list(
        self, current_pos, buy, sell, trade_start_time, trade_end_time, pred_start_time, pred_end_time
    ):
        sell_order_list = []
        buy_order_list = []
        cash = current_pos.get_cash()
        current_stock_list = current_pos.get_stock_list()
        for code in sell:
            if code not in current_stock_list:
                get_module_logger("TopkStrategy").warn(f"Strategy tries to sell non-holding name {code}")
                continue
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL,
            ):
                continue
            # check hold limit
            time_per_step = self.trade_calendar.get_freq()
            if cast(Position, current_pos).get_stock_count(code, bar=time_per_step) < self.hold_thresh:
                continue
            # sell order
            sell_amount = current_pos.get_stock_amount(code=code)
            # factor = self.trade_exchange.get_factor(
            #     stock_id=code, start_time=trade_start_time, end_time=trade_end_time
            # )
            # sell_amount_rounded = self.trade_exchange.round_amount_by_trade_unit(sell_amount, factor)
            sell_order = Order(
                stock_id=code,
                amount=sell_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.SELL,  # 0 for sell, 1 for buy
            )
            # is order executable
            if self.trade_exchange.check_order(sell_order):
                sell_order_list.append(sell_order)
                if self.use_prev_close_for_amount:
                    cash += self.trade_position.get_stock_price(code) * self.trade_position.get_stock_amount(code)
                else:
                    trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                        sell_order, position=current_pos
                    )
                    # update cash
                    cash += trade_val - trade_cost
        # buy new stock
        # note the current has been changed
        vacancy = self.topk - len(current_stock_list) + len(sell)
        value = cash * self.risk_degree / vacancy if vacancy > 0 else 0.0

        # open_cost should be considered in the real trading environment, while the backtest in evaluate.py does not
        # consider it as the aim of demo is to accomplish same strategy as evaluate.py, so comment out this line
        # value = value / (1+self.trade_exchange.open_cost) # set open_cost limit
        for code in buy:
            # check is stock suspended
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.BUY,
            ):
                continue
            # buy order
            if self.use_prev_close_for_amount:
                buy_price = self.trade_exchange.get_close(
                    stock_id=code, start_time=pred_start_time, end_time=pred_end_time
                )
                factor = self.trade_exchange.get_factor(
                    stock_id=code, start_time=pred_start_time, end_time=pred_end_time
                )
            else:
                buy_price = self.trade_exchange.get_deal_price(
                    stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY
                )
                factor = self.trade_exchange.get_factor(
                    stock_id=code, start_time=trade_start_time, end_time=trade_end_time
                )

            buy_amount = value / buy_price

            buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
            if buy_amount != 0:
                buy_order = Order(
                    stock_id=code,
                    amount=buy_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=Order.BUY,  # 1 for buy
                )
                buy_order_list.append(buy_order)
        return TradeDecisionWO(sell_order_list + buy_order_list, self)

    def generate_trade_decision(self, execute_result=None):
        # get the number of trading step finished, trade_step can be [0, 1, 2, ..., trade_len - 1]
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        # NOTE: the current version of topk dropout strategy can't handle pd.DataFrame(multiple signal)
        # So it only leverage the first col of signal
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None:
            return TradeDecisionWO([], self)

        # generate order list for this adjust date
        current_temp = copy.deepcopy(self.trade_position)
        current_stock_list = current_temp.get_stock_list()

        buy, sell = self._generate_buy_sell_list(
            pred_score, current_stock_list, trade_start_time, trade_end_time, pred_start_time, pred_end_time
        )
        return self._generate_decisions_from_bs_list(
            current_temp, buy, sell, trade_start_time, trade_end_time, pred_start_time, pred_end_time
        )

    def generate_trade_candidates(self, trade_time, *args, redundancy=0, **kwargs) -> pd.DataFrame:
        trade_step, _ = self.trade_calendar.get_range_idx(trade_time, trade_time)
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None:
            return pd.DataFrame()
        # generate order list for this adjust date
        current_temp = copy.deepcopy(self.trade_position)
        current_stock_list = current_temp.get_stock_list()
        buy, sell = self._generate_buy_sell_list(
            pred_score,
            current_stock_list,
            trade_start_time,
            trade_end_time,
            pred_start_time,
            pred_end_time,
            redundancy=redundancy,
        )

        pred_df = pred_score.sort_values(ascending=False, kind="stable").to_frame("score")
        pred_df["rank"] = list(range(len(pred_df)))
        time_per_step = self.trade_calendar.get_freq()
        pred_df[f"count_{time_per_step}"] = [
            cast(Position, self.trade_position).get_stock_count(c, bar=time_per_step)
            if c in set(current_stock_list).union(sell).union(buy)
            else 0
            for c in pred_df.index
        ]
        sell_real = [c for c in sell if pred_df.loc[c, f"count_{time_per_step}"] >= self.hold_thresh]
        buy_real = buy[: len(buy) + len(sell_real) - len(sell)]
        pred_df["action"] = np.nan

        pred_df.loc[current_stock_list, "action"] = 0
        pred_df.loc[sell_real, "action"] = -1
        pred_df.loc[buy_real, "action"] = 1
        pred_df.dropna(subset=["action"], inplace=True)
        pred_df["prev_close"] = [
            self.trade_exchange.get_close(x, trade_start_time, trade_end_time) for x in pred_df.index
        ]
        pred_df["prev_factor"] = [
            self.trade_exchange.get_close(x, trade_start_time, trade_end_time) for x in pred_df.index
        ]
        cash = self.trade_position.get_cash()
        expected_sell_proceeds = sum(
            self.trade_position.get_stock_price(code) * self.trade_position.get_stock_amount(code) for code in sell_real
        )

        num_to_buy = len(buy_real) - redundancy
        estimate_value_target = (
            0 if num_to_buy == 0 else (cash + expected_sell_proceeds) * self.risk_degree / num_to_buy
        )

        def get_price_and_amount(stock_id, estimate_value_target, start_time, end_time, action):
            prev_close = self.trade_exchange.get_close(stock_id, start_time, end_time)
            factor = self.trade_exchange.get_factor(stock_id, start_time=start_time, end_time=end_time, freq="day")
            if pd.isna(factor):
                factor = 1.0
            if action == 1:
                adj_rounded_amount = self.trade_exchange.round_amount_by_trade_unit(
                    deal_amount=estimate_value_target / prev_close, factor=factor
                )
                return pd.Series([prev_close / factor, adj_rounded_amount * factor], index=["prev_close", "amount"])
            return pd.Series(
                [prev_close / factor, self.trade_position.get_stock_amount(stock_id)], index=["prev_close", "amount"]
            )

        df = pd.concat(
            [
                pred_df.drop(columns=["prev_close", "prev_factor"]),
                pred_df.apply(
                    lambda r: get_price_and_amount(
                        r.name, estimate_value_target, pred_start_time, pred_end_time, r["action"]
                    ),
                    axis=1,
                ),
            ],
            axis=1,
        )

        return df


class TopkDropoutStrategy(BaseTopkStrategy):
    # 1. Supporting leverage the get_range_limit result from the decision
    # 2. Supporting alter_outer_trade_decision
    # 3. Supporting checking the availability of trade decision
    def __init__(
        self,
        *,
        topk,
        n_drop,
        method_sell="bottom",
        method_buy="top",
        hold_thresh=1,
        only_tradable=False,
        market=None,
        **kwargs,
    ):
        """
        Parameters
        -----------
        topk : int
            the number of stocks in the portfolio.
        n_drop : int
            number of stocks to be replaced in each trading date.
        method_sell : str
            dropout method_sell, random/bottom.
        method_buy : str
            dropout method_buy, random/top.
        hold_thresh : int
            minimum holding days
            before sell stock , will check current.get_stock_count(order.stock_id) >= self.hold_thresh.
        only_tradable : bool
            will the strategy only consider the tradable stock when buying and selling.

            if only_tradable:

                strategy will make decision with the tradable state of the stock info and avoid buy and sell them.

            else:

                strategy will make buy sell decision without checking the tradable state of the stock.
        """
        super().__init__(topk=topk, hold_thresh=hold_thresh, market=market, **kwargs)
        self.n_drop = n_drop
        self.method_sell = method_sell
        self.method_buy = method_buy
        self.hold_thresh = hold_thresh
        self.only_tradable = only_tradable

    def _generate_buy_sell_list(
        self,
        pred_score,
        current_stock_list,
        trade_start_time,
        trade_end_time,
        pred_start_time,
        pred_end_time,
        redundancy=0,
    ):
        if self.only_tradable:
            # If The strategy only consider tradable stock when make decision
            # It needs following actions to filter stocks
            estimate_value = self.trade_position.calculate_value() / self.topk

            def get_first_n(li, n, reverse=False):
                cur_n = 0
                res = []
                for si in reversed(li) if reverse else li:
                    if trade_start_time >= self.trade_calendar.get_all_time()[
                        -1
                    ] or self.trade_exchange.is_stock_tradable(
                        stock_id=si, start_time=trade_start_time, end_time=trade_end_time
                    ):
                        test_buy_price = self.trade_exchange.get_close(
                            stock_id=si, start_time=pred_start_time, end_time=pred_end_time
                        )
                        if test_buy_price is not None:
                            factor = self.trade_exchange.get_factor(
                                stock_id=si, start_time=pred_start_time, end_time=pred_end_time
                            )
                            buy_amount = self.trade_exchange.round_amount_by_trade_unit(
                                estimate_value / test_buy_price, factor
                            )
                        else:
                            buy_amount = 100
                        if buy_amount > 0:
                            res.append(si)
                            cur_n += 1
                        if cur_n >= n:
                            break
                return res[::-1] if reverse else res

            def get_last_n(li, n):
                return get_first_n(li, n, reverse=True)

            def filter_stock(li):
                return [
                    si
                    for si in li
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=si, start_time=trade_start_time, end_time=trade_end_time
                    )
                ]

        else:
            # Otherwise, the stock will make decision without the stock tradable info
            def get_first_n(li, n):
                return list(li)[:n]

            def get_last_n(li, n):
                return list(li)[-n:]

            def filter_stock(li):
                return li

        pred_score, current_stock_list, removed_from_population = self.filter_instruments_by_market(
            pred_score, current_stock_list, trade_start_time, trade_end_time
        )

        last = pred_score[pred_score.index.isin(current_stock_list)].sort_values(ascending=False, kind="stable").index

        # The new stocks today want to buy **at most**
        if self.method_buy == "top":
            today = get_first_n(
                pred_score[~pred_score.index.isin(last)].sort_values(ascending=False, kind="stable").index,
                self.n_drop + self.topk + redundancy - len(last),
            )
        elif self.method_buy == "random":
            topk_candi = get_first_n(pred_score.sort_values(ascending=False, kind="stable").index, self.topk)
            candi = list(filter(lambda x: x not in last, topk_candi))
            n = self.n_drop + self.topk + redundancy - len(last)
            try:
                today = np.random.choice(candi, n, replace=False)
            except ValueError:
                today = candi
        else:
            raise NotImplementedError(f"This type of input is not supported")
        # combine(new stocks + last stocks),  we will drop stocks from this list
        # In case of dropping higher score stock and buying lower score stock.
        comb = pred_score.reindex(last.append(pd.Index(today))).sort_values(ascending=False, kind="stable").index

        # Get the stock list we really want to sell (After filtering the case that we sell high and buy low)
        if self.method_sell == "bottom":
            sell = last[last.isin(get_last_n(comb, self.n_drop))]
        elif self.method_sell == "random":
            candi = filter_stock(last)
            try:
                sell = pd.Index(np.random.choice(candi, self.n_drop, replace=False) if len(last) else [])
            except ValueError:  # No enough candidates
                sell = candi
        else:
            raise NotImplementedError(f"This type of input is not supported")
        # Get the stock list we really want to buy, this should be calculated before add back to
        # codes which are removed from population
        buy = today[: len(sell) + self.topk + redundancy - len(last)]
        # force sell names which are removed from population
        sell = sell.to_list() + removed_from_population

        return buy, sell


class TopkKeepnDropoutStrategy(BaseTopkStrategy):
    # TODO:
    # 1. Supporting leverage the get_range_limit result from the decision
    # 2. Supporting alter_outer_trade_decision
    # 3. Supporting checking the availability of trade decision
    def __init__(
        self,
        *,
        topk,
        keepn,
        forcedropnum=0,
        only_positive_score=False,
        only_tradable=False,
        market=None,
        hold_thresh=0,
        **kwargs,
    ):
        """
        Parameters
        -----------
        topk : int
            the number of stocks in the portfolio.
        n_drop : int
            number of stocks to be replaced in each trading date.
        hold_thresh : int
            minimum holding days
            before sell stock , will check current.get_stock_count(order.stock_id) >= self.hold_thresh.
        only_tradable : bool
            will the strategy only consider the tradable stock when buying and selling.
            if only_tradable:
                strategy will make buy sell decision without checking the tradable state of the stock.
            else:
                strategy will make decision with the tradable state of the stock info and avoid buy and sell them.
        """
        super().__init__(topk=topk, hold_thresh=hold_thresh, market=market, **kwargs)
        self.keepn = keepn
        assert keepn >= topk, "number to keep must larger than top k"
        self.only_positive_score = only_positive_score
        self.forcedropnum = forcedropnum
        self.only_tradable = only_tradable

    def _generate_buy_sell_list(
        self,
        pred_score,
        current_stock_list,
        trade_start_time,
        trade_end_time,
        pred_start_time,
        pred_end_time,
        redundancy=0,
    ):
        pred_score, current_stock_list, removed_from_population = self.filter_instruments_by_market(
            pred_score, current_stock_list, trade_start_time, trade_end_time
        )

        pred_df = pred_score.sort_values(ascending=False, kind="stable").to_frame()
        pred_df["current_hold"] = pred_df.index.isin(current_stock_list)
        pred_df["cum_current_hold"] = pred_df["current_hold"].cumsum()
        pred_df["tradestatusflag"] = (
            pred_df.apply(
                lambda x: 0
                if self.trade_exchange.is_stock_tradable(x.name, trade_start_time, trade_end_time)
                else 1
                if x.current_hold
                else -1,
                axis=1,
            )
            if self.only_tradable
            else 0
        )
        pred_df["rank"] = list(range(len(pred_df)))
        pred_df["keep"] = pred_df.apply(
            lambda x: x["current_hold"]
            and (
                x["rank"] < self.keepn
                and x["cum_current_hold"] <= self.topk - self.forcedropnum - len(removed_from_population)
                or not self.trade_exchange.is_stock_tradable(x.name, trade_start_time, trade_end_time)
            ),
            axis=1,
        )
        num_keep = pred_df.keep.sum()

        sell = pred_df[(~pred_df.keep) & pred_df.current_hold].index.union(removed_from_population).to_list()
        buy = pred_df[~pred_df.current_hold & pred_df.tradable].iloc[: self.topk - num_keep].index.tolist()

        return buy, sell


class WeightStrategyBase(BaseSignalStrategy):
    # TODO:
    # 1. Supporting leverage the get_range_limit result from the decision
    # 2. Supporting alter_outer_trade_decision
    # 3. Supporting checking the availability of trade decision
    def __init__(
        self,
        *,
        order_generator_cls_or_obj=OrderGenWOInteract,
        **kwargs,
    ):
        """
        signal :
            the information to describe a signal. Please refer to the docs of `qlib.backtest.signal.create_signal_from`
            the decision of the strategy will base on the given signal
        trade_exchange : Exchange
            exchange that provides market info, used to deal order and generate report

            - If `trade_exchange` is None, self.trade_exchange will be set with common_infra
            - It allowes different trade_exchanges is used in different executions.
            - For example:

                - In daily execution, both daily exchange and minutely are usable, but the daily exchange is recommended because it runs faster.
                - In minutely execution, the daily exchange is not usable, only the minutely exchange is recommended.
        """
        super().__init__(**kwargs)

        if isinstance(order_generator_cls_or_obj, type):
            self.order_generator: OrderGenerator = order_generator_cls_or_obj()
        else:
            self.order_generator: OrderGenerator = order_generator_cls_or_obj

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        """
        Generate target position from score for this date and the current position.The cash is not considered in the position

        Parameters
        -----------
        score : pd.Series
            pred score for this trade date, index is stock_id, contain 'score' column.
        current : Position()
            current position.
        trade_start_time: pd.Timestamp
        trade_end_time: pd.Timestamp
        """
        raise NotImplementedError()

    def generate_trade_decision(self, execute_result=None):
        # generate_trade_decision
        # generate_target_weight_position() and generate_order_list_from_target_weight_position() to generate order_list

        # get the number of trading step finished, trade_step can be [0, 1, 2, ..., trade_len - 1]
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if pred_score is None:
            return TradeDecisionWO([], self)
        current_temp = copy.deepcopy(self.trade_position)
        assert isinstance(current_temp, Position)  # Avoid InfPosition

        target_weight_position = self.generate_target_weight_position(
            score=pred_score, current=current_temp, trade_start_time=trade_start_time, trade_end_time=trade_end_time
        )
        order_list = self.order_generator.generate_order_list_from_target_weight_position(
            current=current_temp,
            trade_exchange=self.trade_exchange,
            risk_degree=self.get_risk_degree(trade_step),
            target_weight_position=target_weight_position,
            pred_start_time=pred_start_time,
            pred_end_time=pred_end_time,
            trade_start_time=trade_start_time,
            trade_end_time=trade_end_time,
        )
        return TradeDecisionWO(order_list, self)


class EnhancedIndexingStrategy(WeightStrategyBase):

    """Enhanced Indexing Strategy

    Enhanced indexing combines the arts of active management and passive management,
    with the aim of outperforming a benchmark index (e.g., S&P 500) in terms of
    portfolio return while controlling the risk exposure (a.k.a. tracking error).

    Users need to prepare their risk model data like below:

    .. code-block:: text

        ├── /path/to/riskmodel
        ├──── 20210101
        ├────── factor_exp.{csv|pkl|h5}
        ├────── factor_cov.{csv|pkl|h5}
        ├────── specific_risk.{csv|pkl|h5}
        ├────── blacklist.{csv|pkl|h5}  # optional

    The risk model data can be obtained from risk data provider. You can also use
    `qlib.model.riskmodel.structured.StructuredCovEstimator` to prepare these data.

    Args:
        riskmodel_path (str): risk model path
        name_mapping (dict): alternative file names
    """

    FACTOR_EXP_NAME = "factor_exp.pkl"
    FACTOR_COV_NAME = "factor_cov.pkl"
    SPECIFIC_RISK_NAME = "specific_risk.pkl"
    BLACKLIST_NAME = "blacklist.pkl"

    def __init__(
        self,
        *,
        riskmodel_root,
        market="csi500",
        turn_limit=None,
        name_mapping={},
        optimizer_kwargs={},
        verbose=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.logger = get_module_logger("EnhancedIndexingStrategy")

        self.riskmodel_root = riskmodel_root
        self.market = market
        self.turn_limit = turn_limit

        self.factor_exp_path = name_mapping.get("factor_exp", self.FACTOR_EXP_NAME)
        self.factor_cov_path = name_mapping.get("factor_cov", self.FACTOR_COV_NAME)
        self.specific_risk_path = name_mapping.get("specific_risk", self.SPECIFIC_RISK_NAME)
        self.blacklist_path = name_mapping.get("blacklist", self.BLACKLIST_NAME)

        self.optimizer = EnhancedIndexingOptimizer(**optimizer_kwargs)

        self.verbose = verbose

        self._riskdata_cache = {}

    def get_risk_data(self, date):
        if date in self._riskdata_cache:
            return self._riskdata_cache[date]

        root = self.riskmodel_root + "/" + date.strftime("%Y%m%d")
        if not os.path.exists(root):
            return None

        factor_exp = load_dataset(root + "/" + self.factor_exp_path, index_col=[0])
        factor_cov = load_dataset(root + "/" + self.factor_cov_path, index_col=[0])
        specific_risk = load_dataset(root + "/" + self.specific_risk_path, index_col=[0])

        if not factor_exp.index.equals(specific_risk.index):
            # NOTE: for stocks missing specific_risk, we always assume it has the highest volatility
            specific_risk = specific_risk.reindex(factor_exp.index, fill_value=specific_risk.max())

        universe = factor_exp.index.tolist()

        blacklist = []
        if os.path.exists(root + "/" + self.blacklist_path):
            blacklist = load_dataset(root + "/" + self.blacklist_path).index.tolist()

        self._riskdata_cache[date] = factor_exp.values, factor_cov.values, specific_risk.values, universe, blacklist

        return self._riskdata_cache[date]

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        trade_date = trade_start_time
        pre_date = get_pre_trading_date(trade_date, future=True)  # previous trade date

        # load risk data
        outs = self.get_risk_data(pre_date)
        if outs is None:
            self.logger.warning(f"no risk data for {pre_date:%Y-%m-%d}, skip optimization")
            return None
        factor_exp, factor_cov, specific_risk, universe, blacklist = outs

        # transform score
        # NOTE: for stocks missing score, we always assume they have the lowest score
        score = score.reindex(universe).fillna(score.min()).values

        # get current weight
        # NOTE: if a stock is not in universe, its current weight will be zero
        cur_weight = current.get_stock_weight_dict(only_stock=False)
        cur_weight = np.array([cur_weight.get(stock, 0) for stock in universe])
        assert all(cur_weight >= 0), "current weight has negative values"
        cur_weight = cur_weight / self.get_risk_degree(trade_date)  # sum of weight should be risk_degree
        if cur_weight.sum() > 1 and self.verbose:
            self.logger.warning(f"previous total holdings excess risk degree (current: {cur_weight.sum()})")

        # load bench weight
        bench_weight = D.features(
            D.instruments("all"), [f"${self.market}_weight"], start_time=pre_date, end_time=pre_date
        ).squeeze()
        bench_weight.index = bench_weight.index.droplevel(level="datetime")
        bench_weight = bench_weight.reindex(universe).fillna(0).values

        # whether stock tradable
        # NOTE: currently we use last day volume to check whether tradable
        tradable = D.features(D.instruments("all"), ["$volume"], start_time=pre_date, end_time=pre_date).squeeze()
        tradable.index = tradable.index.droplevel(level="datetime")
        tradable = tradable.reindex(universe).gt(0).values
        mask_force_hold = ~tradable

        # mask force sell
        mask_force_sell = np.array([stock in blacklist for stock in universe], dtype=bool)

        # optimize
        weight = self.optimizer(
            r=score,
            F=factor_exp,
            cov_b=factor_cov,
            var_u=specific_risk**2,
            w0=cur_weight,
            wb=bench_weight,
            mfh=mask_force_hold,
            mfs=mask_force_sell,
        )

        target_weight_position = {stock: weight for stock, weight in zip(universe, weight) if weight > 0}

        if self.verbose:
            self.logger.info("trade date: {:%Y-%m-%d}".format(trade_date))
            self.logger.info("number of holding stocks: {}".format(len(target_weight_position)))
            self.logger.info("total holding weight: {:.6f}".format(weight.sum()))

        return target_weight_position
