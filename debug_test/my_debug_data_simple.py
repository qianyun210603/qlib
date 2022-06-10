import qlib
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data
from qlib.data import D


if __name__ == '__main__':
    provider_uri = r"D:\Documents\TradeResearch\qlib_test\rqdata"  # target_dir
    if exists_qlib_data(provider_uri):
        qlib.init(provider_uri=provider_uri, region=REG_CN)
    else:
        print("data not exist")

    fields = ['$close', '$high', '$low', "TrailingStop($close, $high, $low, -20, 1, (0.05, 0.02, 0.08))"]
    ohlc_data = D.features(["SH000300"], fields, start_time='2021-01-01', end_time='2021-12-31', freq='day')
    print(ohlc_data)
    ohlc_data.to_csv(r"D:\test_trailing_stoploss_m1.csv")
