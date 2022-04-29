import qlib
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data
from qlib.workflow import R
from functools import lru_cache
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import check_transform_proc

segments = {
    "train": ("2017-01-01", "2020-01-01"),
    "test": ("2020-01-01", "2020-07-01"),
}

@lru_cache
def _generate_dataset():

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
    infer_processors = check_transform_proc([
        {"class": 'RobustZScoreNorm', "kwargs": {"fields_group": "feature", 'clip_outlier': True}},
        {"class": 'Fillna', "kwargs": {"fields_group": "feature"}}
    ], segments["train"][0], segments["train"][1])
    learn_processors = check_transform_proc([
        {"class": "DropnaLabel"},
        {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
    ], segments["train"][0], segments["train"][1])
    process_type = DataHandlerLP.PTYPE_A

    h = DataHandlerLP(
        instruments='all',
        start_time=segments["train"][0],
        end_time=segments["test"][1],
        data_loader=data_loader,
        infer_processors=infer_processors,
        learn_processors=learn_processors,
        process_type=process_type,
    )
    dataset = DatasetH(
        handler=h, segments=segments
    )
    return dataset


if __name__ == '__main__':
    provider_uri = r"D:\Documents\TradeResearch\qlib_test\rqdata_convert"
    if exists_qlib_data(provider_uri):
        qlib.init(provider_uri=provider_uri, region=REG_CN)
    else:
        print("data not exist")

    dataset = _generate_dataset()

    with R.start(experiment_name="backtest_analysis", uri='file:D:\\Documents\\TradeResearch\\qlib\\examples\\mlruns'):
        model_recorder = R.get_recorder(experiment_name="train_model", recorder_id='a0154477e767421e8fcc54c469639ab1')
        model = model_recorder.load_object("trained_model")
        port_analysis_config = {
            "executor": {
                "class": "SimulatorExecutor",
                "module_path": "qlib.backtest.executor",
                "kwargs": {
                    "time_per_step": "day",
                    "generate_portfolio_metrics": True,
                },
            },
            "strategy": {
                "class": "TopkDropout4ConvertStrategy", #4Convert
                "module_path": "qlib.contrib.strategy.signal_strategy",
                "kwargs": {
                    "model": model,
                    "dataset": dataset,
                    "topk": 20,
                    "n_drop": 5,
                    #"only_tradable": True
                },
            },
            "backtest": {
                "start_time": segments["test"][0],
                "end_time": segments["test"][1],
                "account": 100000000,
                "benchmark": "SH000832",
                "exchange_kwargs": {
                    "freq": "day",
                    "limit_threshold": 0.095,
                    "deal_price": "close",
                    "subscribe_fields": ["$call_announced"],
                    "open_cost": 0.00015,
                    "close_cost": 0.00015,
                    "min_cost": 0.1,
                    "trade_unit": 10,
                    "instrument_info_path": r"D:\Documents\TradeResearch\qlib_test\rqdata_convert\contract_specs"
                },
            },
        }

        # prediction
        recorder = R.get_recorder()
        ba_rid = recorder.id
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # backtest & analysis
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()