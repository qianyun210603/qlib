import sys
import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict

if __name__ == '__main__':
    provider_uri = r"D:\Documents\TradeResearch\qlib_test\rqdata"
    print(exists_qlib_data(provider_uri))