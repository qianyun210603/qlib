import sys
import qlib
import pandas as pd
from pathlib import Path
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SignalSeriesRecord
from qlib.utils import flatten_dict
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import check_transform_proc
from qlib.walkforward.walkforward_handler import WFDataHandler

market = "all"
benchmark = "SH000832"

if __name__ == '__main__':
    provider_uri = r"D:\Documents\TradeResearch\qlib_test\rqdata_convert"  # target_dir
    if exists_qlib_data(provider_uri):
        qlib.init(provider_uri=provider_uri, region=REG_CN)
    else:
        print("data not exist")

    wf_segments = [
        # {
        #     "train": ("2017-01-01", "2019-12-31"),
        #     # "valid": ("2018-01-01", "2018-12-31"),
        #     "test": ("2020-01-01", "2020-06-30"),
        # },
        # {
        #     "train": ("2017-07-01", "2020-06-30"),
        #     # "valid": ("2018-01-01", "2018-12-31"),
        #     "test": ("2020-07-01", "2020-12-31"),
        # },
        # {
        #     "train": ("2018-01-01", "2020-12-31"),
        #     # "valid": ("2018-01-01", "2018-12-31"),
        #     "test": ("2021-01-01", "2021-06-30"),
        # },
        # {
        #     "train": ("2018-07-01", "2021-06-30"),
        #     # "valid": ("2018-01-01", "2018-12-31"),
        #     "test": ("2021-07-01", "2021-12-31"),
        # },
        {
            "train": ("2019-01-01", "2021-12-31"),
            # "valid": ("2018-01-01", "2018-12-31"),
            "test": ("2022-01-01", "2022-06-30"),
        },
    ]

    features = ['$open', '$close', '$high', '$low',  'Ref($open, -2)', 'Ref($open, -1)']
    feature_labels = ['ref0', 'close', 'high', 'low',"ref2", 'ref1']

    assert len(feature_labels) == len(features), "'features' and its labels must have same length"

    data_loader = {
        "class": "QlibDataLoader",
        "module_path": "qlib.data.dataset.loader",
        "kwargs": {
            "config": {
                "feature": (features, feature_labels),
                "label": (["Ref($open, -2)/Ref($open, -1) - 1"], ["LABEL0"]),
            },
        },
    }

    data_loader = init_instance_by_config(data_loader)

    print(data_loader.load(['SH113643'], "2022-01-01", "2022-06-30"))
