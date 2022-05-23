import qlib
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data
from qlib.data import D


if __name__ == '__main__':
    provider_uri = r"D:\Documents\TradeResearch\qlib_test\data"  # target_dir
    if exists_qlib_data(provider_uri):
        qlib.init(provider_uri=provider_uri, region=REG_CN)
    else:
        print("data not exist")

    fields = ["Ref($close, 1)", "Ref($close, -1)", "$close - $open", "Log($volume)"]
    ohlc_data = D.features(["SH000300", "SH000903", "SH000905"], fields, start_time='2021-01-01', end_time='2021-12-31', freq='day')
    print(ohlc_data)
