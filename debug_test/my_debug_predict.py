import qlib
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data
from qlib.workflow import R
from qlib.backtest import get_strategy_executor

if __name__ == '__main__':
    market = "converts"
    benchmark = "SH000832"
    provider_uri = r"D:\Documents\TradeResearch\qlib_test\rqdata_convert"  # target_dir
    if exists_qlib_data(provider_uri):
        qlib.init(provider_uri=provider_uri, region=REG_CN)
    else:
        print("data not exist")
    with R.start(experiment_name="backtest_analysis", uri='file:D:\\Documents\\TradeResearch\\qlib\\examples\\mlruns'):
        recorder = R.get_recorder(recorder_id='803e4bd543db402c92e250769bc2322c',)
        pred_df = recorder.load_object("pred.pkl")

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
            "class": "TopkDropout4ConvertStrategy",  # 4Convert
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": pred_df.sort_index(),
                "topk": 20,
                "n_drop": 5,
                "only_tradable": False
                # "forcedropnum": 5,
            },
        },
        "backtest": {
            "start_time": "2022-04-29",
            "end_time": "2022-05-05",
            "account": 100000,
            "benchmark": benchmark,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "open",
                "subscribe_fields": ["$call_announced"],
                "open_cost": 0.00015,
                "close_cost": 0.00015,
                "min_cost": 0.1,
                "trade_unit": 10,
                "instrument_info_path": r"D:\Documents\TradeResearch\qlib_test\rqdata_convert\contract_specs",
            },
        },
    }
    trade_strategy, trade_executor = get_strategy_executor(
        port_analysis_config["backtest"]['start_time'],
        port_analysis_config["backtest"]['end_time'],
        port_analysis_config["strategy"], port_analysis_config["executor"], benchmark=benchmark,
        account=100000, exchange_kwargs=port_analysis_config["backtest"]['exchange_kwargs']
    )
    trade_executor.reset(start_time=port_analysis_config["backtest"]['start_time'], end_time=port_analysis_config["backtest"]['end_time'])
    trade_strategy.reset_level_infra(trade_executor.get_level_infra())
    trade_strategy.generate_trade_decision()
