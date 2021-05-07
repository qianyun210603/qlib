# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
OnlineTool is a module to set and unset a series of `online` models.
The `online` models are some decisive models in some time point, which can be changed with the change of time.
This allows us to use efficient submodels as the market style changing.
"""

from typing import List, Union

from qlib.log import get_module_logger
from qlib.workflow.online.update import PredUpdater
from qlib.workflow.recorder import Recorder
from qlib.workflow.task.utils import list_recorders


class OnlineTool:
    """
    OnlineTool will manage `online` models in an experiment which includes the models recorder.
    """

    ONLINE_KEY = "online_status"  # the online status key in recorder
    ONLINE_TAG = "online"  # the 'online' model
    OFFLINE_TAG = "offline"  # the 'offline' model, not for online serving

    def __init__(self, need_log=True):
        """
        Init OnlineTool.

        Args:
            need_log (bool, optional): print log or not. Defaults to True.
        """
        self.logger = get_module_logger(self.__class__.__name__)
        self.need_log = need_log

    def set_online_tag(self, tag, recorder: Union[list, object]):
        """
        Set `tag` to the model to sign whether online.

        Args:
            tag (str): the tags in `ONLINE_TAG`, `OFFLINE_TAG`
            recorder (Union[list,object]): the model's recorder
        """
        raise NotImplementedError(f"Please implement the `set_online_tag` method.")

    def get_online_tag(self, recorder: object) -> str:
        """
        Given a model recorder and return its online tag.

        Args:
            recorder (Object): the model's recorder

        Returns:
            str: the online tag
        """
        raise NotImplementedError(f"Please implement the `get_online_tag` method.")

    def reset_online_tag(self, recorder: Union[list, object]):
        """
        Offline all models and set the recorders to 'online'.

        Args:
            recorder (Union[list,object]):
                the recorder you want to reset to 'online'.

        """
        raise NotImplementedError(f"Please implement the `reset_online_tag` method.")

    def online_models(self) -> list:
        """
        Get current `online` models

        Returns:
            list: a list of `online` models.
        """
        raise NotImplementedError(f"Please implement the `online_models` method.")

    def update_online_pred(self, to_date=None):
        """
        Update the predictions of `online` models to a date.

        Args:
            to_date (pd.Timestamp): the pred before this date will be updated. None for update to latest.

        """
        raise NotImplementedError(f"Please implement the `update_online_pred` method.")


class OnlineToolR(OnlineTool):
    """
    The implementation of OnlineTool based on (R)ecorder.
    """

    def __init__(self, experiment_name:str, need_log=True):
        """
        Init OnlineToolR.

        Args:
            experiment_name (str): the experiment name.
            need_log (bool, optional): print log or not. Defaults to True.
        """
        super().__init__(need_log=need_log)
        self.exp_name = experiment_name

    def set_online_tag(self, tag, recorder: Union[Recorder, List]):
        """
        Set `tag` to the model's recorder to sign whether online.

        Args:
            tag (str): the tags in `ONLINE_TAG`, `NEXT_ONLINE_TAG`, `OFFLINE_TAG`
            recorder (Union[Recorder, List]): a list of Recorder or an instance of Recorder
        """
        if isinstance(recorder, Recorder):
            recorder = [recorder]
        for rec in recorder:
            rec.set_tags(**{self.ONLINE_KEY: tag})
        if self.need_log:
            self.logger.info(f"Set {len(recorder)} models to '{tag}'.")

    def get_online_tag(self, recorder: Recorder) -> str:
        """
        Given a model recorder and return its online tag.

        Args:
            recorder (Recorder): an instance of recorder

        Returns:
            str: the online tag
        """
        tags = recorder.list_tags()
        return tags.get(self.ONLINE_KEY, self.OFFLINE_TAG)

    def reset_online_tag(self, recorder: Union[Recorder, List]):
        """
        Offline all models and set the recorders to 'online'.

        Args:
            recorder (Union[Recorder, List]):
                the recorder you want to reset to 'online'.

        """
        if isinstance(recorder, Recorder):
            recorder = [recorder]
        recs = list_recorders(self.exp_name)
        self.set_online_tag(self.OFFLINE_TAG, list(recs.values()))
        self.set_online_tag(self.ONLINE_TAG, recorder)

    def online_models(self) -> list:
        """
        Get current `online` models

        Returns:
            list: a list of `online` models.
        """
        return list(list_recorders(self.exp_name, lambda rec: self.get_online_tag(rec) == self.ONLINE_TAG).values())

    def update_online_pred(self, to_date=None):
        """
        Update the predictions of online models to a date.

        Args:
            to_date (pd.Timestamp): the pred before this date will be updated. None for update to latest time in Calendar.
        """
        online_models = self.online_models()
        for rec in online_models:
            PredUpdater(rec, to_date=to_date, need_log=self.need_log).update()

        if self.need_log:
            self.logger.info(f"Finished updating {len(online_models)} online model predictions of {self.exp_name}.")
