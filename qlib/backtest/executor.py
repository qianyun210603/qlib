import copy
import warnings
import pandas as pd
from typing import Union

from .order import Order, BaseTradeDecision
from .exchange import Exchange
from .utils import TradeCalendarManager, CommonInfrastructure, LevelInfrastructure

from ..utils import init_instance_by_config
from ..utils.time import Freq
from ..strategy.base import BaseStrategy


class BaseExecutor:
    """Base executor for trading"""

    def __init__(
        self,
        time_per_step: str,
        start_time: Union[str, pd.Timestamp] = None,
        end_time: Union[str, pd.Timestamp] = None,
        indicator_config: dict = {},
        generate_report: bool = False,
        verbose: bool = False,
        track_data: bool = False,
        common_infra: CommonInfrastructure = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        time_per_step : str
            trade time per trading step, used for genreate the trade calendar
        show_indicator: bool, optional
            whether to show indicators, :
            - 'pa', the price advantage
            - 'pos', the positive rate
            - 'ffr', the fulfill rate
        indicator_config: dict, optional
            config for calculating trade indicator, including the following fields:
            - 'show_indicator': whether to show indicators, optional, default by False. The indicators includes
                - 'pa', the price advantage
                - 'pos', the positive rate
                - 'ffr', the fulfill rate
            - 'pa_config': config for calculating price advantage(pa), optional
                - 'base_price': the based price than which the trading price is advanced, Optional, default by 'twap'
                    - If 'base_price' is 'twap', the based price is the time weighted average price
                    - If 'base_price' is 'vwap', the based price is the volume weighted average price
                - 'weight_method': weighted method when calculating total trading pa by different orders' pa in each step, optional, default by 'mean'
                    - If 'weight_method' is 'mean', calculating mean value of different orders' pa
                    - If 'weight_method' is 'amount_weighted', calculating amount weighted average value of different orders' pa
                    - If 'weight_method' is 'value_weighted', calculating value weighted average value of different orders' pa
            - 'ffr_config': config for calculating fulfill rate(ffr), optional
                - 'weight_method': weighted method when calculating total trading ffr by different orders' ffr in each step, optional, default by 'mean'
                    - If 'weight_method' is 'mean', calculating mean value of different orders' ffr
                    - If 'weight_method' is 'amount_weighted', calculating amount weighted average value of different orders' ffr
                    - If 'weight_method' is 'value_weighted', calculating value weighted average value of different orders' ffr
            Example:
                {
                    'show_indicator': True,
                    'pa_config': {
                        'base_value': 'twap',
                        'weight_method': 'value_weighted',
                    },
                    'ffr_config':{
                        'weight_method': 'value_weighted',
                    }
                }
        generate_report : bool, optional
            whether to generate report, by default False
        verbose : bool, optional
            whether to print trading info, by default False
        track_data : bool, optional
            whether to generate trade_decision, will be used when training rl agent
            - If `self.track_data` is true, when making data for training, the input `trade_decision` of `execute` will be generated by `collect_data`
            - Else,  `trade_decision` will not be generated
        common_infra : CommonInfrastructure, optional:
            common infrastructure for backtesting, may including:
            - trade_account : Account, optional
                trade account for trading
            - trade_exchange : Exchange, optional
                exchange that provides market info

        """
        self.time_per_step = time_per_step
        self.indicator_config = indicator_config
        self.generate_report = generate_report
        self.verbose = verbose
        self.track_data = track_data
        self.reset(start_time=start_time, end_time=end_time, track_data=track_data, common_infra=common_infra)

    def reset_common_infra(self, common_infra):
        """
        reset infrastructure for trading
            - reset trade_account
        """
        if not hasattr(self, "common_infra"):
            self.common_infra = common_infra
        else:
            self.common_infra.update(common_infra)

        if common_infra.has("trade_account"):
            self.trade_account = copy.copy(common_infra.get("trade_account"))
            self.trade_account.reset(freq=self.time_per_step, init_report=True)

    def reset(self, track_data: bool = None, common_infra: CommonInfrastructure = None, **kwargs):
        """
        - reset `start_time` and `end_time`, used in trade calendar
        - reset `track_data`, used when making data for multi-level training
        - reset `common_infra`, used to reset `trade_account`, `trade_exchange`, .etc
        """

        if track_data is not None:
            self.track_data = track_data

        if "start_time" in kwargs or "end_time" in kwargs:
            start_time = kwargs.get("start_time")
            end_time = kwargs.get("end_time")
            self.trade_calendar = TradeCalendarManager(
                freq=self.time_per_step, start_time=start_time, end_time=end_time
            )

        if common_infra is not None:
            self.reset_common_infra(common_infra)

    def get_level_infra(self):
        return LevelInfrastructure(trade_calendar=self.trade_calendar)

    def finished(self):
        return self.trade_calendar.finished()

    def execute(self, trade_decision):
        """execute the trade decision and return the executed result

        Parameters
        ----------
        trade_decision : BaseTradeDecision

        Returns
        ----------
        execute_result : List[object]
            the executed result for trade decision
        """
        raise NotImplementedError("execute is not implemented!")

    def collect_data(self, trade_decision):
        """Generator for collecting the trade decision data for rl training

        Parameters
        ----------
        trade_decision : BaseTradeDecision

        Returns
        ----------
        execute_result : List[object]
            the executed result for trade decision

        Yields
        -------
        object
            trade decision
        """
        if self.track_data:
            yield trade_decision
        return self.execute(trade_decision)

    def get_report(self):
        """get the history report and postions instance"""
        if self.generate_report:
            _report = self.trade_account.report.generate_report_dataframe()
            _positions = self.trade_account.get_positions()
            return _report, _positions
        else:
            raise ValueError("generate_report should be True if you want to generate report")

    def get_trade_indicator(self):
        """get the trade indicator instance, which has pa/pos/ffr info."""
        return self.trade_account.indicator

    def get_all_executors(self):
        """get all executors"""
        return [self]


class NestedExecutor(BaseExecutor):
    """
    Nested Executor with inner strategy and executor
    - At each time `execute` is called, it will call the inner strategy and executor to execute the `trade_decision` in a higher frequency env.
    """

    def __init__(
        self,
        time_per_step: str,
        inner_executor: Union[BaseExecutor, dict],
        inner_strategy: Union[BaseStrategy, dict],
        start_time: Union[str, pd.Timestamp] = None,
        end_time: Union[str, pd.Timestamp] = None,
        indicator_config: dict = {},
        generate_report: bool = False,
        verbose: bool = False,
        track_data: bool = False,
        trade_exchange: Exchange = None,
        common_infra: CommonInfrastructure = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        inner_executor : BaseExecutor
            trading env in each trading bar.
        inner_strategy : BaseStrategy
            trading strategy in each trading bar
        trade_exchange : Exchange
            exchange that provides market info, used to generate report
            - If generate_report is None, trade_exchange will be ignored
            - Else If `trade_exchange` is None, self.trade_exchange will be set with common_infra
        """
        self.inner_executor = init_instance_by_config(
            inner_executor, common_infra=common_infra, accept_types=BaseExecutor
        )
        self.inner_strategy = init_instance_by_config(
            inner_strategy, common_infra=common_infra, accept_types=BaseStrategy
        )

        super(NestedExecutor, self).__init__(
            time_per_step=time_per_step,
            start_time=start_time,
            end_time=end_time,
            indicator_config=indicator_config,
            generate_report=generate_report,
            verbose=verbose,
            track_data=track_data,
            common_infra=common_infra,
            **kwargs,
        )

        if trade_exchange is not None:
            self.trade_exchange = trade_exchange

    def reset_common_infra(self, common_infra):
        """
        reset infrastructure for trading
            - reset trade_exchange
            - reset inner_strategyand inner_executor common infra
        """
        super(NestedExecutor, self).reset_common_infra(common_infra)

        if common_infra.has("trade_exchange"):
            self.trade_exchange = common_infra.get("trade_exchange")

        self.inner_executor.reset_common_infra(common_infra)
        self.inner_strategy.reset_common_infra(common_infra)

    def _init_sub_trading(self, trade_decision):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        self.inner_executor.reset(start_time=trade_start_time, end_time=trade_end_time)
        sub_level_infra = self.inner_executor.get_level_infra()
        self.inner_strategy.reset(level_infra=sub_level_infra, outer_trade_decision=trade_decision)

    def execute(self, trade_decision):
        return_value = {}
        for _decision in self.collect_data(trade_decision, return_value):
            pass
        return return_value.get("execute_result")

    def collect_data(self, trade_decision: BaseTradeDecision, return_value=None):
        if self.track_data:
            yield trade_decision
        self._init_sub_trading(trade_decision)
        execute_result = []
        inner_order_indicators = []
        _inner_execute_result = None
        while not self.inner_executor.finished():
            # outter strategy have chance to update decision each iterator
            updated_trade_decision = trade_decision.update(self.inner_executor.trade_calendar)
            if updated_trade_decision is not None:
                trade_decision = updated_trade_decision
                # NEW UPDATE
                # create a hook for inner strategy to update outter decision
                self.inner_strategy.alter_decision(trade_decision)

            _inner_trade_decision = self.inner_strategy.generate_trade_decision(_inner_execute_result)

            # NOTE: Trade Calendar will step forward in the follow line
            _inner_execute_result = yield from self.inner_executor.collect_data(trade_decision=_inner_trade_decision)

            execute_result.extend(_inner_execute_result)
            inner_order_indicators.append(self.inner_executor.get_trade_indicator().get_order_indicator)

        if hasattr(self, "trade_account"):
            trade_step = self.trade_calendar.get_trade_step()
            trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
            self.trade_account.update_bar_end(
                trade_start_time,
                trade_end_time,
                self.trade_exchange,
                atomic=False,
                generate_report=self.generate_report,
                inner_order_indicators=inner_order_indicators,
                indicator_config=self.indicator_config,
            )

        self.trade_calendar.step()
        if return_value is not None:
            return_value.update({"execute_result": execute_result})
        return execute_result

    def get_all_executors(self):
        """get all executors, including self and inner_executor.get_all_executors()"""
        return [self, *self.inner_executor.get_all_executors()]


class SimulatorExecutor(BaseExecutor):
    """Executor that simulate the true market"""

    def __init__(
        self,
        time_per_step: str,
        start_time: Union[str, pd.Timestamp] = None,
        end_time: Union[str, pd.Timestamp] = None,
        indicator_config: dict = {},
        generate_report: bool = False,
        verbose: bool = False,
        track_data: bool = False,
        trade_exchange: Exchange = None,
        common_infra: CommonInfrastructure = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        trade_exchange : Exchange
            exchange that provides market info, used to deal order and generate report
            - If `trade_exchange` is None, self.trade_exchange will be set with common_infra
        """
        super(SimulatorExecutor, self).__init__(
            time_per_step=time_per_step,
            start_time=start_time,
            end_time=end_time,
            indicator_config=indicator_config,
            generate_report=generate_report,
            verbose=verbose,
            track_data=track_data,
            common_infra=common_infra,
            **kwargs,
        )
        if trade_exchange is not None:
            self.trade_exchange = trade_exchange

    def reset_common_infra(self, common_infra):
        """
        reset infrastructure for trading
            - reset trade_exchange
        """
        super(SimulatorExecutor, self).reset_common_infra(common_infra)
        if common_infra.has("trade_exchange"):
            self.trade_exchange = common_infra.get("trade_exchange")

    def execute(self, trade_decision: BaseTradeDecision):

        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        execute_result = []
        for order in trade_decision.get_decision():
            if self.trade_exchange.check_order(order) is True:
                # execute the order
                trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                    order, trade_account=self.trade_account
                )
                execute_result.append((order, trade_val, trade_cost, trade_price))
                if self.verbose:
                    if order.direction == Order.SELL:  # sell
                        print(
                            "[I {:%Y-%m-%d %H:%M:%S}]: sell {}, price {:.2f}, amount {}, deal_amount {}, factor {}, value {:.2f}.".format(
                                trade_start_time,
                                order.stock_id,
                                trade_price,
                                order.amount,
                                order.deal_amount,
                                order.factor,
                                trade_val,
                            )
                        )
                    else:
                        print(
                            "[I {:%Y-%m-%d %H:%M:%S}]: buy {}, price {:.2f}, amount {}, deal_amount {}, factor {}, value {:.2f}.".format(
                                trade_start_time,
                                order.stock_id,
                                trade_price,
                                order.amount,
                                order.deal_amount,
                                order.factor,
                                trade_val,
                            )
                        )

            else:
                if self.verbose:
                    print("[W {:%Y-%m-%d %H:%M:%S}]: {} wrong.".format(trade_start_time, order.stock_id))
                # do nothing
                pass

        self.trade_account.update_bar_end(
            trade_start_time,
            trade_end_time,
            self.trade_exchange,
            atomic=True,
            generate_report=self.generate_report,
            trade_info=execute_result,
            indicator_config=self.indicator_config,
        )
        self.trade_calendar.step()
        return execute_result
