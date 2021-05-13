# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
OnlineManager can manage a set of `Online Strategy <#Online Strategy>`_ and run them dynamically.

With the change of time, the decisive models will be also changed. In this module, we call those contributing models as `online` models.
In every routine(such as everyday or every minutes), the `online` models maybe changed and the prediction of them need to be updated.
So this module provide a series methods to control this process. 

This module also provide a method to simulate `Online Strategy <#Online Strategy>`_ in the history.
Which means you can verify your strategy or find a better one.
"""

import logging
from typing import Callable, Dict, List, Union

import pandas as pd
from qlib import get_module_logger
from qlib.data.data import D
from qlib.log import set_global_logger_level
from qlib.model.ens.ensemble import AverageEnsemble
from qlib.model.trainer import DelayTrainerR, Trainer
from qlib.utils import flatten_dict
from qlib.utils.serial import Serializable
from qlib.workflow.online.strategy import OnlineStrategy
from qlib.workflow.task.collect import MergeCollector


class OnlineManager(Serializable):
    """
    OnlineManager can manage online models with `Online Strategy <#Online Strategy>`_.
    It also provide a history recording which models are onlined at what time.
    """

    def __init__(
        self,
        strategies: Union[OnlineStrategy, List[OnlineStrategy]],
        trainer: Trainer = None,
        begin_time: Union[str, pd.Timestamp] = None,
        freq="day",
    ):
        """
        Init OnlineManager.
        One OnlineManager must have at least one OnlineStrategy.

        Args:
            strategies (Union[OnlineStrategy, List[OnlineStrategy]]): an instance of OnlineStrategy or a list of OnlineStrategy
            begin_time (Union[str,pd.Timestamp], optional): the OnlineManager will begin at this time. Defaults to None for using latest date.
            trainer (Trainer): the trainer to train task. None for using DelayTrainerR.
            freq (str, optional): data frequency. Defaults to "day".
        """
        self.logger = get_module_logger(self.__class__.__name__)
        if not isinstance(strategies, list):
            strategies = [strategies]
        self.strategies = strategies
        self.freq = freq
        if begin_time is None:
            begin_time = D.calendar(freq=self.freq).max()
        self.begin_time = pd.Timestamp(begin_time)
        self.cur_time = self.begin_time
        # OnlineManager will recorder the history of online models, which is a dict like {begin_time, {strategy, [online_models]}}. begin_time means when online_models are onlined.
        self.history = {}
        if trainer is None:
            trainer = DelayTrainerR()
        self.trainer = trainer
        self.signals = None

    def first_train(self, strategies: List[OnlineStrategy] = None, model_kwargs: dict = {}):
        """
        Get tasks from every strategy's first_tasks method and train them.
        If using DelayTrainer, it can finish training all together after every strategy's first_tasks.

        Args:
            strategies (List[OnlineStrategy]): the strategies list (need this param when adding strategies). None for use default strategies.
            model_kwargs (dict): the params for `prepare_online_models`
        """
        models_list = []
        if strategies is None:
            strategies = self.strategies
        for strategy in strategies:
            self.logger.info(f"Strategy `{strategy.name_id}` begins first training...")
            tasks = strategy.first_tasks()
            models = self.trainer.train(tasks, experiment_name=strategy.name_id)
            models_list.append(models)

        for strategy, models in zip(strategies, models_list):
            self.prepare_online_models(strategy, models, model_kwargs=model_kwargs)

    def routine(
        self,
        cur_time: Union[str, pd.Timestamp] = None,
        delay: bool = False,
        task_kwargs: dict = {},
        model_kwargs: dict = {},
        signal_kwargs: dict = {},
    ):
        """
        Run typical update process for every strategy and record the online history.

        The typical update process after a routine, such as day by day or month by month.
        The process is: Prepare signals -> Prepare tasks -> Prepare online models.

        If using DelayTrainer, it can finish training all together after every strategy's prepare_tasks.

        Args:
            cur_time (Union[str,pd.Timestamp], optional): run routine method in this time. Defaults to None.
            delay (bool): if delay prepare signals and models
            task_kwargs (dict): the params for `prepare_tasks`
            model_kwargs (dict): the params for `prepare_online_models`
            signal_kwargs (dict): the params for `prepare_signals`
        """
        if cur_time is None:
            cur_time = D.calendar(freq=self.freq).max()
        self.cur_time = pd.Timestamp(cur_time)  # None for latest date
        models_list = []
        for strategy in self.strategies:
            self.logger.info(f"Strategy `{strategy.name_id}` begins routine...")
            if not delay:
                strategy.tool.update_online_pred()

            tasks = strategy.prepare_tasks(self.cur_time, **task_kwargs)
            models = self.trainer.train(tasks)
            self.logger.info(f"Finished training {len(models)} models.")
            models_list.append(models)

        for strategy, models in zip(self.strategies, models_list):
            self.prepare_online_models(strategy, models, delay=delay, model_kwargs=model_kwargs)

        if not delay:
            self.prepare_signals(**signal_kwargs)

    def prepare_online_models(
        self, strategy: OnlineStrategy, models: list, delay: bool = False, model_kwargs: dict = {}
    ):
        """
        Prepare online model for strategy, including end_train, reset_online_tag and add history.

        Args:
            strategy (OnlineStrategy): the instance of strategy.
            models (list): a list of models.
            delay (bool, optional): if delay prepare models. Defaults to False.
            model_kwargs (dict, optional): the params for `prepare_online_models`.
        """
        if not delay:
            models = self.trainer.end_train(models, experiment_name=strategy.name_id)
        online_models = strategy.prepare_online_models(models, **model_kwargs)
        self.history.setdefault(self.cur_time, {})[strategy] = online_models

    def get_collector(self) -> MergeCollector:
        """
        Get the instance of `Collector <../advanced/task_management.html#Task Collecting>`_ to collect results from every strategy.
        This collector can be a basis as the signals preparation.

        Returns:
            MergeCollector: the collector to merge other collectors.
        """
        collector_dict = {}
        for strategy in self.strategies:
            collector_dict[strategy.name_id] = strategy.get_collector()
        return MergeCollector(collector_dict, process_list=[])

    def add_strategy(self, strategies: Union[OnlineStrategy, List[OnlineStrategy]]):
        """
        Add some new strategies to online manager.

        Args:
            strategy (Union[OnlineStrategy, List[OnlineStrategy]]): a list of OnlineStrategy
        """
        if not isinstance(strategies, list):
            strategies = [strategies]
        self.first_train(strategies)
        self.strategies.extend(strategies)

    def prepare_signals(self, prepare_func: Callable = AverageEnsemble(), over_write=False):
        """
        After perparing the data of last routine (a box in box-plot) which means the end of the routine, we can prepare trading signals for next routine.

        NOTE: Given a set prediction, all signals before these prediction end time will be prepared well.

        Even if the latest signal already exists, the latest calculation result will be overwritten.

        .. note::

            Given a prediction of a certain time, all signals before this time will be prepared well.

        Args:
            prepare_func (Callable, optional): Get signals from a dict after collecting. Defaults to AverageEnsemble(), the results after mergecollector must be {xxx:pred}.
            over_write (bool, optional): If True, the new signals will overwrite. If False, the new signals will append to the end of signals. Defaults to False.

        Returns:
            pd.DataFrame: the signals.
        """
        signals = prepare_func(self.get_collector()())
        old_signals = self.signals
        if old_signals is not None and not over_write:
            old_max = old_signals.index.get_level_values("datetime").max()
            new_signals = signals.loc[old_max:]
            signals = pd.concat([old_signals, new_signals], axis=0)
        else:
            new_signals = signals
        self.logger.info(f"Finished preparing new {len(new_signals)} signals.")
        self.signals = signals
        return new_signals

    def get_signals(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Get prepared online signals.

        Returns:
            Union[pd.Series, pd.DataFrame]: pd.Series for only one signals every datetime.
            pd.DataFrame for multiple signals, for example, buy and sell operation use different trading signal.
        """
        return self.signals

    SIM_LOG_LEVEL = logging.INFO + 1
    SIM_LOG_NAME = "SIMULATE_INFO"

    def simulate(self, end_time, frequency="day", task_kwargs={}, model_kwargs={}, signal_kwargs={}):
        """
        Starting from current time, this method will simulate every routine in OnlineManager until end time.

        Considering the parallel training, the models and signals can be perpared after all routine simulating.

        The delay training way can be ``DelayTrainer`` and the delay preparing signals way can be ``delay_prepare``.

        Args:
            end_time: the time the simulation will end
            frequency: the calendar frequency
            task_kwargs (dict): the params for `prepare_tasks`
            model_kwargs (dict): the params for `prepare_online_models`
            signal_kwargs (dict): the params for `prepare_signals`

        Returns:
            HyperCollector: the OnlineManager's collector
        """
        cal = D.calendar(start_time=self.cur_time, end_time=end_time, freq=frequency)
        self.first_train()

        simulate_level = self.SIM_LOG_LEVEL
        set_global_logger_level(simulate_level)
        logging.addLevelName(simulate_level, self.SIM_LOG_NAME)

        for cur_time in cal:
            self.logger.log(level=simulate_level, msg=f"Simulating at {str(cur_time)}......")
            self.routine(
                cur_time,
                delay=self.trainer.is_delay(),
                task_kwargs=task_kwargs,
                model_kwargs=model_kwargs,
                signal_kwargs=signal_kwargs,
            )
        # delay prepare the models and signals
        if self.trainer.is_delay():
            self.delay_prepare(model_kwargs=model_kwargs, signal_kwargs=signal_kwargs)

        # FIXME: get logging level firstly and restore it here
        set_global_logger_level(logging.DEBUG)
        self.logger.info(f"Finished preparing signals")
        return self.get_collector()

    def delay_prepare(self, model_kwargs={}, signal_kwargs={}):
        """
        Prepare all models and signals if there are something waiting for prepare.

        Args:
            model_kwargs: the params for `prepare_online_models`
            signal_kwargs: the params for `prepare_signals`
        """
        for cur_time, strategy_models in self.history.items():
            self.cur_time = cur_time
            for strategy, models in strategy_models.items():
                self.prepare_online_models(strategy, models, delay=False, model_kwargs=model_kwargs)
            # NOTE: Assumption: the predictions of online models need less than next cur_time, or this method will work in a wrong way.
            self.prepare_signals(**signal_kwargs)
