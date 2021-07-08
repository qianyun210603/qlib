# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations
import copy
from typing import List, Tuple, Union, TYPE_CHECKING

from .account import Account

if TYPE_CHECKING:
    from ..strategy.base import BaseStrategy
from .exchange import Exchange
from .executor import BaseExecutor
from .backtest import backtest_loop
from .backtest import collect_data_loop
from .utils import CommonInfrastructure
from .order import Order
from ..utils import init_instance_by_config
from ..log import get_module_logger
from ..config import C


logger = get_module_logger("backtest caller")


def get_exchange(
    exchange=None,
    freq="day",
    start_time=None,
    end_time=None,
    codes="all",
    subscribe_fields=[],
    open_cost=0.0015,
    close_cost=0.0025,
    min_cost=5.0,
    limit_threshold=None,
    deal_price: Union[str, Tuple[str], List[str]] = None,
    **kwargs,
):
    """get_exchange

    Parameters
    ----------

    # exchange related arguments
    exchange: Exchange().
    subscribe_fields: list
        subscribe fields.
    open_cost : float
        open transaction cost.
    close_cost : float
        close transaction cost.
    min_cost : float
        min transaction cost.
    trade_unit : int
        Included in kwargs.  Please refer to the docs of `__init__` of `Exchange`
    deal_price: Union[str, Tuple[str], List[str]]
                The `deal_price` supports following two types of input
                - <deal_price> : str
                - (<buy_price>, <sell_price>): Tuple[str] or List[str]

                <deal_price>, <buy_price> or <sell_price> := <price>
                <price> := str
                - for example '$close', '$open', '$vwap' ("close" is OK. `Exchange` will help to prepend
                  "$" to the expression)
    limit_threshold : float
        limit move 0.1 (10%) for example, long and short with same limit.

    Returns
    -------
    :class: Exchange
    an initialized Exchange object
    """

    if limit_threshold is None:
        limit_threshold = C.limit_threshold
    if exchange is None:
        logger.info("Create new exchange")

        exchange = Exchange(
            freq=freq,
            start_time=start_time,
            end_time=end_time,
            codes=codes,
            deal_price=deal_price,
            subscribe_fields=subscribe_fields,
            limit_threshold=limit_threshold,
            open_cost=open_cost,
            close_cost=close_cost,
            min_cost=min_cost,
            **kwargs,
        )
        return exchange
    else:
        return init_instance_by_config(exchange, accept_types=Exchange)


def create_account_instance(
    start_time, end_time, benchmark: str, account: float, pos_type: str = "Position"
) -> Account:
    """
    # TODO: is very strange pass benchmark_config in the account(maybe for report)
    # There should be a post-step to process the report.

    Parameters
    ----------
    start_time :
        start time of the benchmark
    end_time :
        end time of the benchmark
    benchmark : str
        the benchmark for reporting
    account : Union[float, str]
        information for describing how to creating the account
        For `float`
            Using Account with a normal position
        For `str`:
            Using account with a specific Position
    """
    kwargs = {
        "init_cash": account,
        "benchmark_config": {
            "benchmark": benchmark,
            "start_time": start_time,
            "end_time": end_time,
        },
        "pos_type": pos_type,
    }
    return Account(**kwargs)


def get_strategy_executor(
    start_time,
    end_time,
    strategy: BaseStrategy,
    executor: BaseExecutor,
    benchmark: str = "SH000300",
    account: Union[float, str] = 1e9,
    exchange_kwargs: dict = {},
    pos_type: str = "Position",
):

    # NOTE:
    # - for avoiding recursive import
    # - typing annotations is not reliable
    from ..strategy.base import BaseStrategy

    trade_account = create_account_instance(
        start_time=start_time, end_time=end_time, benchmark=benchmark, account=account, pos_type=pos_type
    )

    exchange_kwargs = copy.copy(exchange_kwargs)
    if "start_time" not in exchange_kwargs:
        exchange_kwargs["start_time"] = start_time
    if "end_time" not in exchange_kwargs:
        exchange_kwargs["end_time"] = end_time
    trade_exchange = get_exchange(**exchange_kwargs)

    common_infra = CommonInfrastructure(trade_account=trade_account, trade_exchange=trade_exchange)
    trade_strategy = init_instance_by_config(strategy, accept_types=BaseStrategy, common_infra=common_infra)
    trade_executor = init_instance_by_config(executor, accept_types=BaseExecutor, common_infra=common_infra)

    return trade_strategy, trade_executor


def backtest(
    start_time,
    end_time,
    strategy,
    executor,
    benchmark="SH000300",
    account=1e9,
    exchange_kwargs={},
    pos_type: str = "Position",
):

    trade_strategy, trade_executor = get_strategy_executor(
        start_time,
        end_time,
        strategy,
        executor,
        benchmark,
        account,
        exchange_kwargs,
        pos_type=pos_type,
    )
    report_dict, indicator_dict = backtest_loop(start_time, end_time, trade_strategy, trade_executor)

    return report_dict, indicator_dict


def collect_data(
    start_time,
    end_time,
    strategy,
    executor,
    benchmark="SH000300",
    account=1e9,
    exchange_kwargs={},
    pos_type: str = "Position",
):

    trade_strategy, trade_executor = get_strategy_executor(
        start_time,
        end_time,
        strategy,
        executor,
        benchmark,
        account,
        exchange_kwargs,
        pos_type=pos_type,
    )
    yield from collect_data_loop(start_time, end_time, trade_strategy, trade_executor)
