from datetime import datetime

from ..utils.serial import Serializable


class BaseInstrumentInfo(Serializable):
    def __init__(self, **kwargs):
        pass


class ConvertInstrumentInfo(BaseInstrumentInfo):
    def __init__(
        self,
        cash_flow_schedule,
        coupon_schedule,
        maturity_date,
        call_date=datetime(2200, 1, 1),
        principle=100,
        stop_trading_date=datetime(2200, 1, 1),
    ):
        self.cash_flow_schedule = cash_flow_schedule
        self.maturity_date = maturity_date
        self.call_date = call_date
        self.coupon_schedule = coupon_schedule
        self.principle = principle
        self.stop_trading_date = stop_trading_date
