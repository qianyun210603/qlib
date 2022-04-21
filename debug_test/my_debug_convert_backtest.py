import qlib
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from functools import lru_cache
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import check_transform_proc


@lru_cache
def _generate_dataset():
    fit_start_time = "2010-01-01"
    fit_end_time = "2015-12-31"
    learn_processors = check_transform_proc([
        {"class": "DropnaLabel"},
        {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
    ], fit_start_time, fit_end_time)
    data_loader = {
        "class": "QlibDataLoader",
        "kwargs": {
            "config": {
                "feature": (['$open', '$high', '$low', '$close', '$hv20', '$hv60', '$iv', '$closestock',
                             '$conversion_price_reset_status', '$remaining_size', '$pure_bond_ytm', '$turnover_rate',
                             '$call_announced', '$call_satisfied'],
                            ['open', 'high', 'low', 'close', 'hv20', 'hv60', 'iv', 'closestock',
                             'CPRS', 'remaining_size', 'pure_bond_ytm', 'turnover_rate',
                             'call_announced', 'call_satisfied']),
                "label": (["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]),
            },
        },
    }

    h = DataHandlerLP(
        instruments='all',
        start_time="2010-01-01",
        end_time="2021-12-31",
        data_loader=data_loader,
        infer_processors=[],
        learn_processors=learn_processors,
        process_type=DataHandlerLP.PTYPE_A,
    )
    dataset = DatasetH(
        handler=h, segments={
            "train": ("2010-01-01", "2015-12-31"),
            "valid": ("2016-01-01", "2017-12-31"),
            "test": ("2018-01-01", "2021-12-31"),
        }
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
        model_recorder = R.get_recorder(experiment_name="train_model", recorder_id='169d854168214dd6a333180d7c49de9c')
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
                "class": "TopkDropoutStrategy",
                "module_path": "qlib.contrib.strategy.signal_strategy",
                "kwargs": {
                    "model": model,
                    "dataset": dataset,
                    "topk": 20,
                    "n_drop": 5,
                },
            },
            "backtest": {
                "start_time": "2018-01-01",
                "end_time": "2021-12-31",
                "account": 100000000,
                "benchmark": "SH000300",
                "exchange_kwargs": {
                    "freq": "day",
                    "limit_threshold": 0.095,
                    "deal_price": "close",
                    "open_cost": 0.00015,
                    "close_cost": 0.00015,
                    "min_cost": 0.1,
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