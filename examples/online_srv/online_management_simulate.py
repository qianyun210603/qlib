# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This examples is about how can simulate the OnlineManager based on rolling tasks. 
"""

import fire
import qlib
from qlib.model.trainer import DelayTrainerRM
from qlib.workflow.online.manager import OnlineManager
from qlib.workflow.online.strategy import RollingAverageStrategy
from qlib.workflow.task.gen import RollingGen
from qlib.workflow.task.manage import TaskManager


data_handler_config = {
    "start_time": "2018-01-01",
    "end_time": "2018-10-31",
    "fit_start_time": "2018-01-01",
    "fit_end_time": "2018-03-31",
    "instruments": "csi100",
}

dataset_config = {
    "class": "DatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": data_handler_config,
        },
        "segments": {
            "train": ("2018-01-01", "2018-03-31"),
            "valid": ("2018-04-01", "2018-05-31"),
            "test": ("2018-06-01", "2018-09-10"),
        },
    },
}

record_config = [
    {
        "class": "SignalRecord",
        "module_path": "qlib.workflow.record_temp",
    },
    {
        "class": "SigAnaRecord",
        "module_path": "qlib.workflow.record_temp",
    },
]

# use lgb model
task_lgb_config = {
    "model": {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
    },
    "dataset": dataset_config,
    "record": record_config,
}

# use xgboost model
task_xgboost_config = {
    "model": {
        "class": "XGBModel",
        "module_path": "qlib.contrib.model.xgboost",
    },
    "dataset": dataset_config,
    "record": record_config,
}


class OnlineSimulationExample:
    def __init__(
        self,
        provider_uri="~/.qlib/qlib_data/cn_data",
        region="cn",
        exp_name="rolling_exp",
        task_url="mongodb://10.0.0.4:27017/",
        task_db_name="rolling_db",
        task_pool="rolling_task",
        rolling_step=80,
        start_time="2018-09-10",
        end_time="2018-10-31",
        tasks=[task_xgboost_config, task_lgb_config],
    ):
        """
        Init OnlineManagerExample.

        Args:
            provider_uri (str, optional): the provider uri. Defaults to "~/.qlib/qlib_data/cn_data".
            region (str, optional): the stock region. Defaults to "cn".
            exp_name (str, optional): the experiment name. Defaults to "rolling_exp".
            task_url (str, optional): your MongoDB url. Defaults to "mongodb://10.0.0.4:27017/".
            task_db_name (str, optional): database name. Defaults to "rolling_db".
            task_pool (str, optional): the task pool name (a task pool is a collection in MongoDB). Defaults to "rolling_task".
            rolling_step (int, optional): the step for rolling. Defaults to 80.
            start_time (str, optional): the start time of simulating. Defaults to "2018-09-10".
            end_time (str, optional): the end time of simulating. Defaults to "2018-10-31".
            tasks (dict or list[dict]): a set of the task config waiting for rolling and training
        """
        self.exp_name = exp_name
        self.task_pool = task_pool
        self.start_time = start_time
        self.end_time = end_time
        mongo_conf = {
            "task_url": task_url,
            "task_db_name": task_db_name,
        }
        qlib.init(provider_uri=provider_uri, region=region, mongo=mongo_conf)
        self.rolling_gen = RollingGen(
            step=rolling_step, rtype=RollingGen.ROLL_SD, modify_end_time=False
        )  # The rolling tasks generator, modify_end_time is false because we just need simulate to 2018-10-31.
        self.trainer = DelayTrainerRM(self.exp_name, self.task_pool)
        self.task_manager = TaskManager(self.task_pool)  # A good way to manage all your tasks
        self.rolling_online_manager = OnlineManager(
            RollingAverageStrategy(
                exp_name, task_template=tasks, rolling_gen=self.rolling_gen, trainer=self.trainer, need_log=False
            ),
            begin_time=self.start_time,
            need_log=False,
        )
        self.tasks = tasks

    # Run this to run all workflow automatically
    def main(self):
        print("========== reset ==========")
        self.rolling_online_manager.reset()
        print("========== simulate ==========")
        self.rolling_online_manager.simulate(end_time=self.end_time)
        print("========== collect results ==========")
        print(self.rolling_online_manager.get_collector()())
        print("========== signals ==========")
        print(self.rolling_online_manager.get_signals())
        print("========== online history ==========")
        print(self.rolling_online_manager.get_online_history(self.exp_name))


if __name__ == "__main__":
    ## to run all workflow automatically with your own parameters, use the command below
    # python online_management_simulate.py main --experiment_name="your_exp_name" --rolling_step=60
    fire.Fire(OnlineSimulationExample)
