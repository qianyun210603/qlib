# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import pandas as pd
from typing import List, Union


from ..model.base import BaseModel
from ..data.dataset import DatasetH
from ..data.dataset.utils import convert_index_format
from ..contrib.backtest.order import Order
from ..rl.interpreter import ActionInterpreter, StateInterpreter
from ..utils import init_instance_by_config


class BaseStrategy:
    """Base strategy for trading"""

    def __init__(
        self,
        rely_trade_decision: object = None,
        level_infra: dict = {},
        common_infra: dict = {},
    ):
        """
        Parameters
        ----------
        rely_trade_decision : object, optional
            the high-level trade decison on which the startegy rely, and it will be traded in [start_time , end_time] , by default None
            - If the strategy is used to split trade decison, it will be used
            - If the strategy is used for portfolio management, it can be ignored
        level_infra : dict, optional
            level shared infrastructure for backtesting, including trade_calendar
        common_infra : dict, optional
            common infrastructure for backtesting, including trade_account, trade_exchange, .etc
        """

        self.reset(level_infra=level_infra, common_infra=common_infra, rely_trade_decision=rely_trade_decision)

    def reset_level_infra(self, level_infra):
        if not hasattr(self, "level_infra"):
            self.level_infra = level_infra
        else:
            self.level_infra.update(level_infra)

        if "trade_calendar" in level_infra:
            self.trade_calendar = level_infra.get("trade_calendar")

    def reset_common_infra(self, common_infra):
        if not hasattr(self, "common_infra"):
            self.common_infra = common_infra
        else:
            self.common_infra.update(common_infra)

        if "trade_account" in common_infra:
            self.trade_position = common_infra.get("trade_account").current

    def reset(self, level_infra: dict = None, common_infra: dict = None, rely_trade_decision=None, **kwargs):
        """
        - reset `level_infra`, used to reset trade_calendar, .etc
        - reset `common_infra`, used to reset `trade_account`, `trade_exchange`, .etc
        - reset `rely_trade_decision`, used to make split decison
        """
        if level_infra is not None:
            self.reset_level_infra(level_infra)

        if common_infra is not None:
            self.reset_common_infra(common_infra)

        if rely_trade_decision is not None:
            self.rely_trade_decision = rely_trade_decision

    def generate_trade_decision(self, execute_state):
        """Generate trade decision in each trading bar"""
        raise NotImplementedError("generate_trade_decision is not implemented!")


class RuleStrategy(BaseStrategy):
    """Rule-based Trading strategy"""

    pass


class ModelStrategy(BaseStrategy):
    """Model-based trading strategy, use model to make predictions for trading"""

    def __init__(
        self,
        model: BaseModel,
        dataset: DatasetH,
        rely_trade_decision: object = None,
        level_infra: dict = {},
        common_infra: dict = {},
        **kwargs,
    ):
        """
        Parameters
        ----------
        model : BaseModel
            the model used in when making predictions
        dataset : DatasetH
            provide test data for model
        kwargs : dict
            arguments that will be passed into `reset` method
        """
        super(ModelStrategy, self).__init__(rely_trade_decision, level_infra, common_infra, **kwargs)
        self.model = model
        self.dataset = dataset
        self.pred_scores = convert_index_format(self.model.predict(dataset), level="datetime")

    def _update_model(self):
        """
        When using online data, pdate model in each bar as the following steps:
            - update dataset with online data, the dataset should support online update
            - make the latest prediction scores of the new bar
            - update the pred score into the latest prediction
        """
        raise NotImplementedError("_update_model is not implemented!")


class RLStrategy(BaseStrategy):
    """RL-based strategy"""

    def __init__(
        self,
        policy,
        rely_trade_decision: object = None,
        level_infra: dict = {},
        common_infra: dict = {},
        **kwargs,
    ):
        """
        Parameters
        ----------
        policy :
            RL policy for generate action
        """
        super(RLStrategy, self).__init__(rely_trade_decision, level_infra, common_infra, **kwargs)
        self.policy = policy


class RLIntStrategy(RLStrategy):
    """(RL)-based (Strategy) with (Int)erpreter"""

    def __init__(
        self,
        policy,
        state_interpreter: StateInterpreter,
        action_interpreter: ActionInterpreter,
        rely_trade_decision: object = None,
        level_infra: dict = {},
        common_infra: dict = {},
        **kwargs,
    ):
        """
        Parameters
        ----------
        state_interpreter : StateInterpreter
            interpretor that interprets the qlib execute result into rl env state.
        action_interpreter : ActionInterpreter
            interpretor that interprets the rl agent action into qlib order list
        start_time : Union[str, pd.Timestamp], optional
            start time of trading, by default None
        end_time : Union[str, pd.Timestamp], optional
            end time of trading, by default None
        """
        super(RLIntStrategy, self).__init__(policy, rely_trade_decision, level_infra, common_infra, **kwargs)

        self.policy = policy
        self.state_interpreter = init_instance_by_config(state_interpreter)
        self.action_interpreter = init_instance_by_config(action_interpreter)

    def generate_trade_decision(self, execute_state):
        super(RLStrategy, self).step()
        _interpret_state = self.state_interpretor.interpret(execute_result=execute_state)
        _policy_action = self.policy.step(_interpret_state)
        _order_list = self.action_interpreter.interpret(action=_policy_action)
        return _order_list
