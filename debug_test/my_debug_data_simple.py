import qlib
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data
from qlib.data import D
from loguru import logger
from qlib.config import C


if __name__ == '__main__':
    provider_uri = r"D:\Documents\TradeResearch\qlib_test\rqdata"  # target_dir
    logging_config = C["logging_config"]
    # logging_config["handlers"]["console"]["class"] = 'logging.FileHandler'
    # logging_config["handlers"]["console"]['filename'] = r'E:\textlog.log'

    if exists_qlib_data(provider_uri):
        qlib.init(provider_uri=provider_uri, region=REG_CN, mem_cache_size_limit=500000, logging_config=logging_config)
    else:
        print("data not exist")
    instruments = D.instruments('csi300')

    fields = ['Sum($close, 3)', '$high+$low', "ChangeInstrument('SH000300', $close)", "CSRank($close/$high)"]
    # fields = ['$close', '$high+$low', "ChangeInstrument('SH000300', $close)", "$close/$high"]
    ohlc_data = D.features(instruments, fields, start_time='2021-01-01', end_time='2021-12-31', freq='day')
    print(ohlc_data)
    ohlc_data.to_csv(r"D:\test_data.csv")
