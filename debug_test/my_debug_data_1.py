import qlib
from qlib.constant import REG_CN
from qlib.contrib.data.handler import Alpha158

data_handler_config = {
    "start_time": "2008-01-01",
    "end_time": "2020-08-01",
    "fit_start_time": "2008-01-01",
    "fit_end_time": "2014-12-31",
    "instruments": "csi300",
}

if __name__ == "__main__":
    provider_uri = r"D:\Documents\TradeResearch\qlib_test\data"
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    h = Alpha158(**data_handler_config)

    # get all the columns of the data
    print(h.get_cols())

    # fetch all the labels
    print(h.fetch(col_set="label"))

    # fetch all the features
    print(h.fetch(col_set="feature"))