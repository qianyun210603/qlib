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

    features = ['($adjclosestock * 100 / $conversionprice) / $close - 1', '$pure_bond_ytm']
    feature_labels = ["convertion_premium", 'pure_bond_ytm']

    assert len(feature_labels) == len(features), "'features' and its labels must have same length"

    data_loader = {
        "class": "QlibDataLoader",
        "kwargs": {
            "config": {
                "feature": (features, feature_labels),
                "label": (["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]),
            },
        },
    }

    pre_handler = DataHandlerLP(
        instruments='all',
        start_time=wf_segments[0]['train'][0],
        end_time=wf_segments[-1]['test'][-1],
        data_loader=data_loader,
        infer_processors=[],
        learn_processors=[],
    )

    model_config = {
        "class": "LinearModel",
        "module_path": "qlib.contrib.model.linear",
        "kwargs": {
            "estimator": "ols",
        },
    }

    trained_models = []
    datasets = []
    pred_scores = []

    for segments in wf_segments:
        infer_processors = check_transform_proc([
            {"class": 'RobustZScoreNorm', "kwargs": {"fields_group": "feature", 'clip_outlier': True}},
            {"class": 'Fillna', "kwargs": {"fields_group": "feature"}}
        ], segments["train"][0], segments["train"][1])
        learn_processors = check_transform_proc([
            {"class": "DropnaLabel"},
            {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
        ], segments["train"][0], segments["train"][1])
        h = WFDataHandler(
            start_time=segments["train"][0],
            end_time=segments["test"][1],
            fit_start_time=segments["train"][0],
            fit_end_time=segments["train"][1],
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            data_loader_kwargs={"handler_config": pre_handler, }
        )
        dataset = DatasetH(
            handler=h, segments=segments
        )
        datasets.append(dataset)

        dataset.prepare("test", col_set="label")

    #     model = init_instance_by_config(model_config)
    #     model.fit(dataset)
    #     trained_models.append(model)
    #
    #     pred_score = model.predict(dataset)
    #     pred_scores.append(pred_score)
    #
    # port_analysis_config = {
    #     "executor": {
    #         "class": "SimulatorExecutor",
    #         "module_path": "qlib.backtest.executor",
    #         "kwargs": {
    #             "time_per_step": "day",
    #             "generate_portfolio_metrics": True,
    #         },
    #     },
    #     "strategy": {
    #         "class": "TopkDropout4ConvertStrategy",  # 4Convert
    #         "module_path": "qlib.contrib.strategy.signal_strategy",
    #         "kwargs": {
    #             "signal": pd.concat(pred_scores).sort_index(),
    #             "topk": 20,
    #             "n_drop": 5,
    #             "only_tradable": False
    #             # "forcedropnum": 5,
    #         },
    #     },
    #     "backtest": {
    #         "start_time": wf_segments[0]["test"][0],
    #         "end_time": wf_segments[-1]["test"][-1],
    #         "account": 1000000,
    #         "benchmark": benchmark,
    #         "exchange_kwargs": {
    #             "freq": "day",
    #             "limit_threshold": 0.095,
    #             "deal_price": "close",
    #             "subscribe_fields": ["$call_announced"],
    #             "open_cost": 0.00015,
    #             "close_cost": 0.00015,
    #             "min_cost": 0.1,
    #             "trade_unit": 10,
    #             "instrument_info_path": r"D:\Documents\TradeResearch\qlib_test\rqdata_convert\contract_specs",
    #         },
    #     },
    # }
    #
    # # backtest and analysis
    # with R.start(experiment_name="backtest_analysis"):
    #
    #     recorder = R.get_recorder()
    #     ba_rid = recorder.id
    #     sr = SignalSeriesRecord(models=trained_models, datasets=datasets, recorder=recorder)
    #     sr.generate()
    #     #     recorder.set_tags(strategy=port_analysis_config["strategy"])
    #     # backtest & analysis
    #     par = PortAnaRecord(recorder, port_analysis_config, "day")
    #     par.generate()
