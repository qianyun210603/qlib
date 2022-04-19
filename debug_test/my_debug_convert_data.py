import qlib
from qlib.constant import REG_CN
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import check_transform_proc

_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]

if __name__ == "__main__":
    provider_uri = r"D:\Documents\TradeResearch\qlib_test\rqdata_convert"
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    fit_start_time = "2010-01-01"
    fit_end_time = "2015-12-31"
    learn_processors = check_transform_proc(_DEFAULT_LEARN_PROCESSORS, fit_start_time, fit_end_time)
    data_loader = {
        "class": "QlibDataLoader",
        "kwargs": {
            "config": {
                "feature": (['$open', '$high', '$low', '$close', '$volume', '$hv20', '$hv60', '$iv', '$closestock',
                          '$conversion_price_reset_status', '$remaining_size', '$pure_bond_ytm', '$turnover_rate',
                          '$call_announced', '$call_satisfied'],
                           ['open', 'high', 'low', 'close', 'volume', 'hv20', 'hv60', 'iv', 'closestock',
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

    # get all the columns of the data
    print(h.get_cols())

    # fetch all the labels
    print(h.fetch(col_set="label"))

    # fetch all the features
    print(h.fetch(col_set="feature"))