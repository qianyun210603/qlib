# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import logging
from typing import Optional, Text, Dict, Any
import re
from logging import config as logging_config
from time import time
from contextlib import contextmanager

from .config import C


class MetaLogger(type):
    def __init__(self, name, bases, dic):
        super().__init__(name, bases, dic)

    def __new__(cls, name, bases, dict):
        wrapper_dict = type(logging.getLogger("module_name")).__dict__.copy()
        wrapper_dict.update(dict)
        wrapper_dict["__doc__"] = logging.getLogger("module_name").__doc__
        return type.__new__(cls, name, bases, wrapper_dict)

    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls)
        cls.__init__(cls, *args, **kwargs)
        return obj


class QlibLogger(metaclass=MetaLogger):
    """
    Customized logger for Qlib.
    """

    def __init__(self, module_name):
        self.module_name = module_name
        self.level = 0

    @property
    def logger(self):
        logger = logging.getLogger(self.module_name)
        logger.setLevel(self.level)
        return logger

    def setLevel(self, level):
        self.level = level

    def __getattr__(self, name):
        return self.logger.__getattribute__(name)


def get_module_logger(module_name, level: Optional[int] = None):
    """
    Get a logger for a specific module.

    :param module_name: str
        Logic module name.
    :param level: int
    :return: Logger
        Logger object.
    """
    if level is None:
        level = C.logging_level

    module_name = "qlib.{}".format(module_name)
    # Get logger.
    module_logger = QlibLogger(module_name)
    module_logger.setLevel(level)
    return module_logger


class TimeInspector:

    timer_logger = get_module_logger("timer", level=logging.WARNING)

    time_marks = []

    @classmethod
    def set_time_mark(cls):
        """
        Set a time mark with current time, and this time mark will push into a stack.
        :return: float
            A timestamp for current time.
        """
        _time = time()
        cls.time_marks.append(_time)
        return _time

    @classmethod
    def pop_time_mark(cls):
        """
        Pop last time mark from stack.
        """
        return cls.time_marks.pop()

    @classmethod
    def get_cost_time(cls):
        """
        Get last time mark from stack, calculate time diff with current time.
        :return: float
            Time diff calculated by last time mark with current time.
        """
        cost_time = time() - cls.time_marks.pop()
        return cost_time

    @classmethod
    def log_cost_time(cls, info="Done"):
        """
        Get last time mark from stack, calculate time diff with current time, and log time diff and info.
        :param info: str
            Info that will be log into stdout.
        """
        cost_time = time() - cls.time_marks.pop()
        cls.timer_logger.info("Time cost: {0:.3f}s | {1}".format(cost_time, info))

    @classmethod
    @contextmanager
    def logt(cls, name="", show_start=False):
        """logt.
        Log the time of the inside code

        Parameters
        ----------
        name :
            name
        show_start :
            show_start
        """
        if show_start:
            cls.timer_logger.info(f"{name} Begin")
        cls.set_time_mark()
        try:
            yield None
        finally:
            pass
        cls.log_cost_time(info=f"{name} Done")


def set_log_with_config(log_config: Dict[Text, Any]):
    """set log with config

    :param log_config:
    :return:
    """
    logging_config.dictConfig(log_config)


class LogFilter(logging.Filter):
    def __init__(self, param=None):
        self.param = param

    @staticmethod
    def match_msg(filter_str, msg):
        match = False
        try:
            if re.match(filter_str, msg):
                match = True
        except Exception:
            pass
        return match

    def filter(self, record):
        allow = True
        if isinstance(self.param, str):
            allow = not self.match_msg(self.param, record.msg)
        elif isinstance(self.param, list):
            allow = not any([self.match_msg(p, record.msg) for p in self.param])
        return allow
