# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from inspect import getfullargspec

from ...data.dataset import processor as processor_module
from ...data.dataset.handler import DataHandlerLP
from ...data.dataset.processor import Processor
from ...utils import get_callable_kwargs


def check_transform_proc(proc_l, fit_start_time, fit_end_time):
    new_l = []
    for p in proc_l:
        if not isinstance(p, Processor):
            klass, pkwargs = get_callable_kwargs(p, processor_module)
            args = getfullargspec(klass).args
            if "fit_start_time" in args and "fit_end_time" in args:
                assert (
                    fit_start_time is not None and fit_end_time is not None
                ), "Make sure `fit_start_time` and `fit_end_time` are not None."
                pkwargs.update(
                    {
                        "fit_start_time": fit_start_time,
                        "fit_end_time": fit_end_time,
                    }
                )
            proc_config = {"class": klass.__name__, "kwargs": pkwargs}
            if isinstance(p, dict) and "module_path" in p:
                proc_config["module_path"] = p["module_path"]
            new_l.append(proc_config)
        else:
            new_l.append(p)
    return new_l


_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "DropTouchLimits"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]
_DEFAULT_INFER_PROCESSORS = [
    {"class": "ProcessInf", "kwargs": {}},
    {"class": "ZScoreNorm", "kwargs": {}},
    {"class": "Fillna", "kwargs": {}},
]


class Alpha360(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=_DEFAULT_INFER_PROCESSORS,
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            learn_processors=learn_processors,
            infer_processors=infer_processors,
            **kwargs,
        )

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]

    @staticmethod
    def get_feature_config():
        # NOTE:
        # Alpha360 tries to provide a dataset with original price data
        # the original price data includes the prices and volume in the last 60 days.
        # To make it easier to learn models from this dataset, all the prices and volume
        # are normalized by the latest price and volume data ( dividing by $close, $volume)
        # So the latest normalized $close will be 1 (with name CLOSE0), the latest normalized $volume will be 1 (with name VOLUME0)
        # If further normalization are executed (e.g. centralization),  CLOSE0 and VOLUME0 will be 0.
        fields = []
        names = []

        for i in range(59, 0, -1):
            fields += ["Ref($close, %d)/$close" % i]
            names += ["CLOSE%d" % i]
        fields += ["$close/$close"]
        names += ["CLOSE0"]
        for i in range(59, 0, -1):
            fields += ["Ref($open, %d)/$close" % i]
            names += ["OPEN%d" % i]
        fields += ["$open/$close"]
        names += ["OPEN0"]
        for i in range(59, 0, -1):
            fields += ["Ref($high, %d)/$close" % i]
            names += ["HIGH%d" % i]
        fields += ["$high/$close"]
        names += ["HIGH0"]
        for i in range(59, 0, -1):
            fields += ["Ref($low, %d)/$close" % i]
            names += ["LOW%d" % i]
        fields += ["$low/$close"]
        names += ["LOW0"]
        for i in range(59, 0, -1):
            fields += ["Ref($vwap, %d)/$close" % i]
            names += ["VWAP%d" % i]
        fields += ["$vwap/$close"]
        names += ["VWAP0"]
        for i in range(59, 0, -1):
            fields += ["Ref($volume, %d)/($volume+1e-12)" % i]
            names += ["VOLUME%d" % i]
        fields += ["$volume/($volume+1e-12)"]
        names += ["VOLUME0"]

        return fields, names


class Alpha360vwap(Alpha360):
    def get_label_config(self):
        return ["Ref($vwap, -2)/Ref($vwap, -1) - 1"], ["LABEL0"]


class Alpha158(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        if "limits" in kwargs:
            data_loader["kwargs"]["config"]["limits"] = kwargs.pop("limits")
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
        return self.parse_config_to_fields(conf)

    @staticmethod
    def get_label_config():
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]

    @staticmethod
    def parse_config_to_fields(config):
        """create factors from config

        config = {
            'kbar': {}, # whether to use some hard-code kbar features
            'price': { # whether to use raw price features
                'windows': [0, 1, 2, 3, 4], # use price at n days ago
                'feature': ['OPEN', 'HIGH', 'LOW'] # which price field to use
            },
            'volume': { # whether to use raw volume features
                'windows': [0, 1, 2, 3, 4], # use volume at n days ago
            },
            'rolling': { # whether to use rolling operator based features
                'windows': [5, 10, 20, 30, 60], # rolling windows size
                'include': ['ROC', 'MA', 'STD'], # rolling operator to use
                #if include is None we will use default operators
                'exclude': ['RANK'], # rolling operator not to use
            }
        }
        """
        fields = []
        names = []
        if "kbar" in config:
            fields += [
                "($close-$open)/$open",
                "($high-$low)/$open",
                "($close-$open)/($high-$low+1e-12)",
                "($high-Greater($open, $close))/$open",
                "($high-Greater($open, $close))/($high-$low+1e-12)",
                "(Less($open, $close)-$low)/$open",
                "(Less($open, $close)-$low)/($high-$low+1e-12)",
                "(2*$close-$high-$low)/$open",
                "(2*$close-$high-$low)/($high-$low+1e-12)",
            ]
            names += [
                "KMID",
                "KLEN",
                "KMID2",
                "KUP",
                "KUP2",
                "KLOW",
                "KLOW2",
                "KSFT",
                "KSFT2",
            ]
        if "price" in config:
            windows = config["price"].get("windows", range(5))
            feature = config["price"].get("feature", ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"])
            for field in feature:
                field = field.lower()
                fields += ["Ref($%s, %d)/$close" % (field, d) if d != 0 else "$%s/$close" % field for d in windows]
                names += [field.upper() + str(d) for d in windows]
        if "volume" in config:
            windows = config["volume"].get("windows", range(5))
            fields += ["Ref($volume, %d)/($volume+1e-12)" % d if d != 0 else "$volume/($volume+1e-12)" for d in windows]
            names += ["VOLUME" + str(d) for d in windows]
        if "rolling" in config:
            windows = config["rolling"].get("windows", [5, 10, 20, 30, 60])
            include = config["rolling"].get("include", None)
            exclude = config["rolling"].get("exclude", [])
            # `exclude` in dataset config unnecessary filed
            # `include` in dataset config necessary field

            def use(x):
                return x not in exclude and (include is None or x in include)

            # Some factor ref: https://guorn.com/static/upload/file/3/134065454575605.pdf
            if use("ROC"):
                # https://www.investopedia.com/terms/r/rateofchange.asp
                # Rate of change, the price change in the past d days, divided by latest close price to remove unit
                fields += ["Ref($close, %d)/$close" % d for d in windows]
                names += ["ROC%d" % d for d in windows]
            if use("MA"):
                # https://www.investopedia.com/ask/answers/071414/whats-difference-between-moving-average-and-weighted-moving-average.asp
                # Simple Moving Average, the simple moving average in the past d days, divided by latest close price to remove unit
                fields += ["Mean($close, %d)/$close" % d for d in windows]
                names += ["MA%d" % d for d in windows]
            if use("STD"):
                # The standard diviation of close price for the past d days, divided by latest close price to remove unit
                fields += ["Std($close, %d)/$close" % d for d in windows]
                names += ["STD%d" % d for d in windows]
            if use("BETA"):
                # The rate of close price change in the past d days, divided by latest close price to remove unit
                # For example, price increase 10 dollar per day in the past d days, then Slope will be 10.
                fields += ["Slope($close, %d)/$close" % d for d in windows]
                names += ["BETA%d" % d for d in windows]
            if use("RSQR"):
                # The R-sqaure value of linear regression for the past d days, represent the trend linear
                fields += ["Rsquare($close, %d)" % d for d in windows]
                names += ["RSQR%d" % d for d in windows]
            if use("RESI"):
                # The redisdual for linear regression for the past d days, represent the trend linearity for past d days.
                fields += ["Resi($close, %d)/$close" % d for d in windows]
                names += ["RESI%d" % d for d in windows]
            if use("MAX"):
                # The max price for past d days, divided by latest close price to remove unit
                fields += ["Max($high, %d)/$close" % d for d in windows]
                names += ["MAX%d" % d for d in windows]
            if use("LOW"):
                # The low price for past d days, divided by latest close price to remove unit
                fields += ["Min($low, %d)/$close" % d for d in windows]
                names += ["MIN%d" % d for d in windows]
            if use("QTLU"):
                # The 80% quantile of past d day's close price, divided by latest close price to remove unit
                # Used with MIN and MAX
                fields += ["Quantile($close, %d, 0.8)/$close" % d for d in windows]
                names += ["QTLU%d" % d for d in windows]
            if use("QTLD"):
                # The 20% quantile of past d day's close price, divided by latest close price to remove unit
                fields += ["Quantile($close, %d, 0.2)/$close" % d for d in windows]
                names += ["QTLD%d" % d for d in windows]
            if use("RANK"):
                # Get the percentile of current close price in past d day's close price.
                # Represent the current price level comparing to past N days, add additional information to moving average.
                fields += ["Rank($close, %d)" % d for d in windows]
                names += ["RANK%d" % d for d in windows]
            if use("RSV"):
                # Represent the price position between upper and lower resistent price for past d days.
                fields += ["($close-Min($low, %d))/(Max($high, %d)-Min($low, %d)+1e-12)" % (d, d, d) for d in windows]
                names += ["RSV%d" % d for d in windows]
            if use("IMAX"):
                # The number of days between current date and previous highest price date.
                # Part of Aroon Indicator https://www.investopedia.com/terms/a/aroon.asp
                # The indicator measures the time between highs and the time between lows over a time period.
                # The idea is that strong uptrends will regularly see new highs, and strong downtrends will regularly see new lows.
                fields += ["IdxMax($high, %d)/%d" % (d, d) for d in windows]
                names += ["IMAX%d" % d for d in windows]
            if use("IMIN"):
                # The number of days between current date and previous lowest price date.
                # Part of Aroon Indicator https://www.investopedia.com/terms/a/aroon.asp
                # The indicator measures the time between highs and the time between lows over a time period.
                # The idea is that strong uptrends will regularly see new highs, and strong downtrends will regularly see new lows.
                fields += ["IdxMin($low, %d)/%d" % (d, d) for d in windows]
                names += ["IMIN%d" % d for d in windows]
            if use("IMXD"):
                # The time period between previous lowest-price date occur after highest price date.
                # Large value suggest downward momemtum.
                fields += ["(IdxMax($high, %d)-IdxMin($low, %d))/%d" % (d, d, d) for d in windows]
                names += ["IMXD%d" % d for d in windows]
            if use("CORR"):
                # The correlation between absolute close price and log scaled trading volume
                fields += ["Corr($close, Log($volume+1), %d)" % d for d in windows]
                names += ["CORR%d" % d for d in windows]
            if use("CORD"):
                # The correlation between price change ratio and volume change ratio
                fields += ["Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), %d)" % d for d in windows]
                names += ["CORD%d" % d for d in windows]
            if use("CNTP"):
                # The percentage of days in past d days that price go up.
                fields += ["Mean($close>Ref($close, 1), %d)" % d for d in windows]
                names += ["CNTP%d" % d for d in windows]
            if use("CNTN"):
                # The percentage of days in past d days that price go down.
                fields += ["Mean($close<Ref($close, 1), %d)" % d for d in windows]
                names += ["CNTN%d" % d for d in windows]
            if use("CNTD"):
                # The diff between past up day and past down day
                fields += ["Mean($close>Ref($close, 1), %d)-Mean($close<Ref($close, 1), %d)" % (d, d) for d in windows]
                names += ["CNTD%d" % d for d in windows]
            if use("SUMP"):
                # The total gain / the absolute total price changed
                # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
                fields += [
                    "Sum(Greater($close-Ref($close, 1), 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d)
                    for d in windows
                ]
                names += ["SUMP%d" % d for d in windows]
            if use("SUMN"):
                # The total lose / the absolute total price changed
                # Can be derived from SUMP by SUMN = 1 - SUMP
                # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
                fields += [
                    "Sum(Greater(Ref($close, 1)-$close, 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d)
                    for d in windows
                ]
                names += ["SUMN%d" % d for d in windows]
            if use("SUMD"):
                # The diff ratio between total gain and total lose
                # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
                fields += [
                    "(Sum(Greater($close-Ref($close, 1), 0), %d)-Sum(Greater(Ref($close, 1)-$close, 0), %d))"
                    "/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d, d)
                    for d in windows
                ]
                names += ["SUMD%d" % d for d in windows]
            if use("VMA"):
                # Simple Volume Moving average: https://www.barchart.com/education/technical-indicators/volume_moving_average
                fields += ["Mean($volume, %d)/($volume+1e-12)" % d for d in windows]
                names += ["VMA%d" % d for d in windows]
            if use("VSTD"):
                # The standard deviation for volume in past d days.
                fields += ["Std($volume, %d)/($volume+1e-12)" % d for d in windows]
                names += ["VSTD%d" % d for d in windows]
            if use("WVMA"):
                # The volume weighted price change volatility
                fields += [
                    "Std(Abs($close/Ref($close, 1)-1)*$volume, %d)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, %d)+1e-12)"
                    % (d, d)
                    for d in windows
                ]
                names += ["WVMA%d" % d for d in windows]
            if use("VSUMP"):
                # The total volume increase / the absolute total volume changed
                fields += [
                    "Sum(Greater($volume-Ref($volume, 1), 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)"
                    % (d, d)
                    for d in windows
                ]
                names += ["VSUMP%d" % d for d in windows]
            if use("VSUMN"):
                # The total volume increase / the absolute total volume changed
                # Can be derived from VSUMP by VSUMN = 1 - VSUMP
                fields += [
                    "Sum(Greater(Ref($volume, 1)-$volume, 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)"
                    % (d, d)
                    for d in windows
                ]
                names += ["VSUMN%d" % d for d in windows]
            if use("VSUMD"):
                # The diff ratio between total volume increase and total volume decrease
                # RSI indicator for volume
                fields += [
                    "(Sum(Greater($volume-Ref($volume, 1), 0), %d)-Sum(Greater(Ref($volume, 1)-$volume, 0), %d))"
                    "/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)" % (d, d, d)
                    for d in windows
                ]
                names += ["VSUMD%d" % d for d in windows]

        return fields, names


class Alpha158vwap(Alpha158):
    def get_label_config(self):
        return ["Ref($vwap, -2)/Ref($vwap, -1) - 1"], ["LABEL0"]


class Alpha101(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)
        feature_formula, feature_name = self.get_feature_config()

        data_loader = kwargs.get(
            "data_loader",
            {
                "class": "QlibDataLoader",
                "kwargs": {
                    "config": {
                        "feature": (feature_formula, feature_name),
                        "label": kwargs.get("label", self.get_label_config()),
                    },
                    "filter_pipe": filter_pipe,
                    "freq": freq,
                    "inst_processor": kwargs.get("inst_processor", None),
                },
            },
        )
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
        )

    def get_feature_config(self):
        return self.parse_config_to_fields()

    @staticmethod
    def get_label_config():
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]

    @staticmethod
    def parse_config_to_fields():
        """create factors from config"""
        f_return = "($close/Ref($close, 1)-1)"
        f_adv5 = "Mean($money, 5)"
        f_adv10 = "Mean($money, 10)"
        f_adv15 = "Mean($money, 15)"
        f_adv20 = "Mean($money, 20)"
        f_adv30 = "Mean($money, 30)"
        f_adv40 = "Mean($money, 40)"
        f_adv50 = "Mean($money, 50)"
        f_adv60 = "Mean($money, 60)"
        f_adv120 = "Mean($money, 120)"
        f_adv180 = "Mean($money, 180)"

        alpha_components = {
            "alpha001": f"CSRank(IdxMax(Power(If({f_return}<0, Std({f_return}, 20), $close), 2), 5))-0.5",
            "alpha002": "-1*Corr(CSRank(Delta(Log($volume), 2)), CSRank(($close-$open)/$open), 6)",
            "alpha003": "-1*Corr(CSRank($open), CSRank($volume), 10)",
            "alpha004": "-1*Rank(CSRank($low), 9)",
            "alpha005": f"CSRank($open-Sum($vwap, 10)/10)*(-1*CSRank($close-$vwap))",
            "alpha006": "-1*Corr($open, $volume, 10)",
            "alpha007": f"If({f_adv20}<$volume, 0-Rank(Abs(Delta($close, 7)), 60) * Sign(Delta($close, 7)), -1)",
            "alpha008": f"-1*CSRank(Sum($open, 5) * Sum({f_return}, 5) - Ref(Sum($open, 5) * Sum({f_return}, 5), 10))",
            "alpha009": f"If(0 < Min(Delta($close, 1), 5), Delta($close, 1), If(Max(Delta($close, 1), 5) < 0, Delta($close, 1), -1*Delta($close, 1)))",
            "alpha010": f"CSRank(If(0 < Min(Delta($close, 1), 4), Delta($close, 1), If(Max(Delta($close, 1), 4) < 0, Delta($close, 1), -1*Delta($close, 1))))",
            "alpha011": f"(CSRank(Max($vwap-$close, 3)) + CSRank(Min($vwap-$close, 3))) * CSRank(Delta($volume, 3))",
            "alpha012": f"Sign(Delta($volume, 1)) * (-1 * Delta($close, 1))",
            "alpha013": "-1*CSRank(Cov(CSRank($close), CSRank($volume), 5))",
            "alpha014": f"-1*CSRank(Delta({f_return}, 3))*Corr($open, $volume, 10)",
            "alpha015": "-1*Sum(CSRank(Corr(CSRank($high), CSRank($volume), 3)), 3)",
            "alpha016": "-1*CSRank(Cov(CSRank($high), CSRank($volume), 5))",
            "alpha017": f"-1*CSRank(Rank($close, 10))*CSRank(Delta(Delta($close, 1), 1))*CSRank(Rank($volume/{f_adv20}, 5))",
            "alpha018": f"-1*CSRank(Std(Abs($close-$open), 5) + ($close-$open) + Corr($close, $open, 10))",
            "alpha019": f"-1*Sign((($close - Ref($close, 7)) + Delta($close, 7)))*(1+CSRank(1+Sum({f_return}, 250)))",
            "alpha020": "-1*CSRank($open-Ref($high, 1))*CSRank($open-Ref($close, 1))*CSRank($open-Ref($low, 1))",
            "alpha021": f"If(Mean($close, 8) + Std($close, 8) < Mean($close, 2), -1, If(Mean($close, 2) < Mean($close, 8) - Std($close, 8), 1, If($volume <= {f_adv20}, 1, -1)))",
            "alpha022": "-1*Delta(Corr($high, $volume, 5), 5)*CSRank(Std($close, 20))",
            "alpha023": "If(Mean($high, 20)<$high, -1*Delta($high, 2), 0)",
            "alpha024": "If((Delta(Mean($close, 100) , 100) / Ref($close, 100)) <= 0.05, (Min($close,100)-$close),  -1*Delta($close, 3))",
            "alpha025": f"CSRank(-1*{f_return}*{f_adv20}*$vwap*($high-$close))",
            "alpha026": "0-Max(Corr(Rank($volume, 5), Rank($high, 5), 5), 3)",
            "alpha027": "If(0.5<CSRank(Mean(Corr(CSRank($volume), CSRank($vwap), 6), 2)), -1, 1)",
            "alpha028": f"CSScale((Corr({f_adv20}, $low, 5) + (($high + $low) / 2)) - $close)",
            "alpha029": f"Min(Prod(CSRank(Sum(Min(CSRank(-Delta($close-1,5)), 2), 1)),1), 5) + Rank(Ref(-1*{f_return}, 6), 5)",
            "alpha030": "(1.0-CSRank(Sign($close - Ref($close, 1))+Sign(Ref($close, 1) - Ref($close, 2))+Sign(Ref($close, 2) - Ref($close, 3)))) * Sum($volume, 5) / Sum($volume, 20)",
            "alpha031": f"CSRank(WMA(-CSRank(Delta($close, 10)), 10)) + CSRank(-Delta($close, 3)) + Sign(Corr({f_adv20}, $low, 12))",
            "alpha032": "CSScale(Mean($close, 7)-$close) + 20*CSScale(Corr($vwap, Ref($close, 5), 230))",
            "alpha033": "CSRank($close / ($close - $open))",
            "alpha034": f"CSRank(2 - CSRank(Std({f_return}, 2) / Std({f_return}, 5)) - CSRank(Delta($close, 1)))",
            "alpha035": f"Rank($volume, 32) * (1 - Rank(($close + $high) - $low, 16)) * (1 - Rank({f_return}, 32))",
            "alpha036": f"2.21*CSRank(Corr($close - $open, Ref($volume, 1), 15))+0.7*CSRank($open-$close)+0.73*CSRank(Rank(Ref(-1*{f_return}, 6), 5))+"
            f"CSRank(Abs(Corr($vwap, {f_adv20}, 6)))+0.6*CSRank((Mean($close, 200)-$open)*($close-$open))",
            "alpha037": "CSRank(Corr(Ref($open-$close, 1), $close, 200))+CSRank($open-$close)",
            "alpha038": "-CSRank(Rank($close, 10))*CSRank($close/$open)",
            "alpha039": f"-CSRank(Delta($close, 7) * (1-CSRank(WMA($volume / {f_adv20}, 9)))) * (1+CSRank(Sum({f_return},250)))",
            "alpha040": "-CSRank(Std($high, 10))*Corr($high, $volume, 10)",
            "alpha041": "Power($high*$low, 0.5)-$vwap",
            "alpha042": "CSRank($vwap-$close)/CSRank($vwap+$close)",
            "alpha043": f"Rank($volume / {f_adv20}, 20) * Rank(-1*Delta($close, 7), 8)",
            "alpha044": "-1*Corr($high, CSRank($volume), 5)",
            "alpha045": "-CSRank(Mean(Ref($close, 5), 20))*Corr($close, $volume, 2)*CSRank(Corr(Sum($close, 5), Sum($close, 20), 2))",
            "alpha046": "If(0.25 < (Ref($close, 20) - Ref($close, 10)) / 10 - (Ref($close, 10)-$close) / 10, -1, If((Ref($close, 20) "
            "- Ref($close, 10)) / 10 - (Ref($close, 10) - $close) / 10 < 0, 1, -1 * ($close - Ref($close, 1))))",
            "alpha047": f"CSRank(1/$close)*CSRank($high-$close)*$volume/{f_adv20}*$high/Mean($high, 5)-CSRank($vwap - Ref($vwap, 5))",
            # # 'alpha048': use  indneutralize
            "alpha049": "If((Ref($close, 20) - Ref($close, 10)) / 10 - (Ref($close, 10) - $close) / 10 < -0.1, 1, -1*($close - Ref($close, 1)))",
            "alpha050": f"-1*Max(CSRank(Corr(CSRank($volume), CSRank($vwap), 5)), 5)",
            "alpha051": "If((Ref($close, 20) - Ref($close, 10)) / 10 - (Ref($close, 10) - $close) / 10 < -0.05, 1, -1*($close - Ref($close, 1)))",
            "alpha052": f"(Ref(Min($low, 5), 5)-Min($low, 5))*CSRank((Sum({f_return}, 240) - Sum({f_return}, 20)) / 220)*Rank($volume, 5)",
            "alpha053": "-1*Delta((($close - $low) - ($high - $close)) / ($close*1.000001 - $low), 9)",
            "alpha054": "-1*($low - $close) * Power($open, 5) / (($low*1.000001 - $high) * Power($close, 5))",
            "alpha055": "-1*Corr(CSRank(($close - Min($low, 12)) / (Max($high, 12) - Min($low, 12))), CSRank($volume), 6)",
            "alpha056": f"-1*CSRank(Sum({f_return}, 10)/Sum(Sum({f_return}, 2), 3))*CSRank({f_return}*$open_interest)",
            "alpha057": f"-1*($close-$vwap)/WMA(CSRank(IdxMax($close, 30)), 2)",
            # # 'alpha058': use  indneutralize
            # # 'alpha059': use  indneutralize
            "alpha060": "CSScale(CSRank(IdxMax($close, 10))) - 2*CSScale(CSRank(((($close - $low) - ($high - $close)) / ($high - $low)) * $volume))",
            "alpha061": f"If(CSRank($vwap - Min($vwap, 16)) < CSRank(Corr($vwap, {f_adv180}, 17)), 1, 0)",
            "alpha062": f"If(CSRank(Corr($vwap, Sum({f_adv20}, 22), 9)) < CSRank(If(2*CSRank($open) < CSRank(($high+$low)/2) + CSRank($high), 1, 0)), -1, 0)",
            # # 'alpha063': use  indneutralize
            "alpha064": f"If(CSRank(Corr(Sum((($open * 0.178404) + ($low * (1 - 0.178404))), 12), Sum({f_adv120}, 12), 16))"
            f"<CSRank(Delta((((($high + $low) / 2) * 0.178404) + ($vwap * (1 - 0.178404))), 3)), -1, 0)",
            "alpha065": f"If(CSRank(Corr(($open * 0.00817205) + ($vwap * (1 - 0.00817205)), Sum({f_adv60}, 8), 6))<CSRank($open - Min($open, 13)), -1, 0)",
            "alpha066": f"-(CSRank(WMA(Delta($vwap, 3), 7))+Rank(WMA(($low * 0.96633 + $low * (1 - 0.96633) - $vwap) / ($open - ($high + $low) / 2), 11), 6))",
            # # 'alpha067': use  indneutralize
            "alpha068": f"If(Rank(Corr(CSRank($high), CSRank({f_adv15}), 9), 14)<CSRank(Delta($close * 0.518371 + $low * (1 - 0.518371), 1)), -1, 0)",
            # # 'alpha069': use  indneutralize
            # # 'alpha070': use  indneutralize
            "alpha071": f"Greater(Rank(WMA(Corr(Rank($close, 3), Rank({f_adv180},12), 18), 4), 15), Rank(WMA(Power(CSRank($low + $open - 2*$vwap), 2), 16), 4))",
            "alpha072": f"CSRank(WMA(Corr(($high + $low) / 2, {f_adv40}, 8), 10))/CSRank(WMA(Corr(Rank($vwap, 3), Rank($volume, 18), 6), 2))",
            "alpha073": f"-Greater(CSRank(WMA(Delta($vwap, 4), 3)), Rank(WMA(-1*(Delta($open * 0.147155 + $low * (1 - 0.147155), 2) / ($open *0.147155 + $low * (1 - 0.147155))), 3), 16))",
            "alpha074": f"If(CSRank(Corr($close, Sum({f_adv30}, 37), 15))<CSRank(Corr(CSRank($high * 0.0261661 + $vwap * (1 - 0.0261661)), CSRank($volume), 11)), -1, 0)",
            "alpha075": f"If(CSRank(Corr($vwap, $volume, 4))<CSRank(Corr(CSRank($low), CSRank({f_adv50}), 12)), 1, 0)",
            # # 'alpha076': use  indneutralize
            "alpha077": f"Less(CSRank(WMA(($high + $low) / 2 - $vwap, 20)), CSRank(WMA(Corr(($high + $low) / 2, {f_adv40}, 3), 5)))",
            "alpha078": f"Power(CSRank(Corr(Sum($low * 0.352233 + $vwap * (1 - 0.352233), 19), Sum({f_adv40}, 19), 6)), CSRank(Corr(CSRank($vwap), CSRank($volume), 6)))",
            # # 'alpha079': use  indneutralize
            # # 'alpha080': use  indneutralize
            "alpha081": f"If(CSRank(Sum(Log(CSRank(Power(CSRank(Corr($vwap, Sum({f_adv10}, 49), 8)), 4))), 15))<CSRank(Corr(CSRank($vwap), CSRank($volume), 5)), -1, 0)",
            # # 'alpha082': use  indneutralize
            "alpha083": "CSRank(Ref(($high - $low) / Mean($close, 5), 2))*CSRank($volume)*Mean($close, 5) * ($vwap - $close) / ($high*1.0000001-$low)",
            # "alpha084": f"Power(Rank($vwap - Max($vwap, 15), 20), Delta($close, 4))", # meaning less
            "alpha085": f"Power(CSRank(Corr(($high * 0.876703) + ($close * (1 - 0.876703)), {f_adv30}, 9)), CSRank(Corr(Rank(($high + $low) / 2, 3), Rank($volume, 10),7)))",
            "alpha086": f"If(Rank(Corr($close, Sum({f_adv20}, 14), 6), 20)<CSRank(($open+ $close) - ($vwap + $open)), -1, 0)",
            # 'alpha087': use  indneutralize
            "alpha088": f"Less(CSRank(WMA((CSRank($open)+CSRank($low))-(CSRank($high)+CSRank($close)), 8)), Rank(WMA(Corr(Rank($close,8), Rank({f_adv60},20), 8), 6), 2))",
            # # 'alpha089': use  indneutralize
            # # 'alpha090': use  indneutralize
            # # 'alpha091': use  indneutralize
            "alpha092": f"Less(Rank(WMA(If((((($high + $low) / 2) + $close) < ($low + $open)), 1, 0), 14), 18), Rank(WMA(Corr(CSRank($low), CSRank({f_adv30}), 8), 7), 7))",
            # # 'alpha093': use  indneutralize
            "alpha094": f"-Power(CSRank($vwap - Min($vwap, 11)), Rank(Corr(Rank($vwap,19), Rank({f_adv60}, 4), 18), 2))",
            "alpha095": f"If(CSRank($open - Min($open, 12))<Rank(CSRank(Corr(Sum(($high + $low)/ 2, 19), Sum({f_adv40}, 19), 12)), 12), 1, 0)",
            "alpha096": f"-Greater(Rank(WMA(Corr(CSRank($vwap), CSRank($volume), 4), 4), 8), Rank(WMA(IdxMax(Corr(Rank($close, 7), Rank({f_adv60}, 4), 3), 12), 14), 13))",
            # # 'alpha097': use  indneutralize
            "alpha098": f"CSRank(WMA(Corr($vwap, Sum({f_adv5}, 26), 4), 7))-CSRank(WMA(Rank(IdxMin(Corr(CSRank($open), CSRank({f_adv15}), 21), 9), 7), 8))",
            "alpha099": f"If(CSRank(Corr(Sum(($high + $low) / 2, 19), Sum({f_adv60}, 19), 8))<CSRank(Corr($low, $volume, 6)), -1, 0)",
            # # 'alpha100': use  indneutralize
            "alpha101": "(($close - $open) / (($high - $low) + 0.001))",
        }

        return list(alpha_components.values()), list(alpha_components.keys())


class Alpha191(DataHandlerLP):
    def __init__(
        self,
        benchmark="SH000300",
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        **kwargs,
    ):
        self.benchmark = benchmark
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)
        feature_formula, feature_name = self.get_feature_config()

        data_loader = kwargs.get(
            "data_loader",
            {
                "class": "QlibDataLoader",
                "kwargs": {
                    "config": {
                        "feature": (feature_formula, feature_name),
                        "label": kwargs.get("label", self.get_label_config()),
                    },
                    "filter_pipe": filter_pipe,
                    "freq": freq,
                    "inst_processor": kwargs.get("inst_processor", None),
                },
            },
        )
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
        )

    def get_feature_config(self):
        return self.parse_config_to_fields()

    @staticmethod
    def get_label_config():
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]

    def parse_config_to_fields(self):
        """create factors from config"""
        f_return = "($close/Ref($close, 1)-1)"
        f_hd = "$high-Ref($high, 1)"
        f_ld = "Ref($low, 1)-$low"
        f_dtm = "If($open<=Ref($open, 1), 0, Greater($high-$open, $open-Ref($open, 1)))"
        f_dbm = "If($open>=Ref($open, 1), 0, Greater($open-$low, $open-Ref($open, 1)))"
        f_tr = "Greater(Greater($high-$low, Abs($high-Ref($close, 1))), Abs($low-Ref($close, 1)))"

        alpha_components = {
            "alpha191_001": "-1*Corr(CSRank(Delta(Log($volume), 2)), CSRank(($close-$open)/$open), 6)",
            "alpha191_002": "-Delta((2*$close-$high-$low)/($high+$low+1e-10), 1)",
            "alpha191_003": f"Sum(If($close==Ref($close,1), 0, $close-If($close>Ref($close,1), Less($low,Ref($close,1)), Greater($high,Ref($close,1)))),6)",
            "alpha191_004": f"If(Sum($close, 8) / 8 + Std($close, 8) < Sum($close, 2) / 2, -1, If(Sum($close, 2) / 2 < Sum($close, 8) / 8 - Std($close, 8), 1, If(Or((1 < ($volume / Mean($volume,20))), (($volume / Mean($volume,20)) == 1)), 1, (-1 * 1))))",
            "alpha191_005": f"(-1 * Max(Corr(Rank($volume, 5), Rank($high, 5), 5), 3))",
            "alpha191_006": f"(CSRank(Sign(Delta(((($open * 0.85) + ($high * 0.15))), 4)))* -1)",
            "alpha191_007": f"((CSRank(Greater(($vwap-$close), 3)) + CSRank(Less(($vwap - $close), 3))) * "
            f"CSRank(Delta($volume, 3)))",
            "alpha191_008": f"CSRank(Delta((((($high + $low) / 2) * 0.2) + ($vwap * 0.8)), 4) * -1)",
            "alpha191_009": f"EMA((($high+$low)/2-(Ref($high,1)+Ref($low,1))/2)*($high-$low)/$volume, 2/7)",  # TODO check sma/ema
            "alpha191_010": f"CSRank(Greater(Power(If({f_return} < 0, Std({f_return}, 20), $close), 2),5))",
            "alpha191_011": f"Sum((($close-$low)-($high-$close))/($high-$low)*$volume,6)",
            "alpha191_012": f"CSRank(($open - (Sum($vwap, 10) / 10))) * (-CSRank(Abs(($close - $vwap))))",
            "alpha191_013": f"Power($high * $low, 0.5) - $vwap",
            "alpha191_014": f"$close-Ref($close,5)",
            "alpha191_015": f"$open/Ref($close,1)-1",
            "alpha191_016": f"(-1 * Max(CSRank(Corr(CSRank($volume), CSRank($vwap), 5)), 5))",
            "alpha191_017": f"Power(CSRank(($vwap - Greater($vwap, 15))), Delta($close, 5))",
            "alpha191_018": f"$close/Ref($close,5)",
            "alpha191_019": f"If($close<Ref($close,5), ($close-Ref($close,5))/Ref($close,5), If($close==Ref($close,5), 0, ($close-Ref($close,5))/$close))",
            "alpha191_020": f"($close-Ref($close,6))/Ref($close,6)*100",
            "alpha191_021": f"Slope(Mean($close,6), 6)",
            "alpha191_022": f"EMA((($close-Mean($close,6))/Mean($close,6)-Ref(($close-Mean($close,6))/Mean($close,6),3)),1/12)",
            "alpha191_023": f"EMA(If($close>Ref($close,1), Std($close,20), 0), 1/20)/(EMA(If($close>Ref($close,1), Std($close,20), 0),1/20)+EMA(If($close<=Ref($close,1),Std($close,20),0),1/20))*100",
            "alpha191_024": f"EMA($close-Ref($close,5), 1/5)",
            "alpha191_025": f"((-1 * CSRank((Delta($close, 7) * (1 - CSRank(WMA(($volume / Mean($volume,20)), 9)))))) * (1 +CSRank(Sum({f_return}, 250))))",
            "alpha191_026": f"((Sum($close, 7) / 7) - $close) + Corr($vwap, Ref($close, 5), 230)",
            "alpha191_027": f"WMA(($close-Ref($close,3))/Ref($close,3)*100+($close-Ref($close,6))/Ref($close,6)*100,12)",
            "alpha191_028": f"3*EMA(($close-Min($low,9))/(Max($high,9)-Min($low,9))*100,1/3)-2*EMA(EMA(($close-Min($low,9))/(Greater($high,9)-Max($low,9))*100,1/3),1/3)",
            "alpha191_029": f"($close-Ref($close,6))/Ref($close,6)*$volume",
            # "alpha191_030": f"WMA((REGRESI($close/Ref($close)-1,MKT,SMB,HML, 60))^2,20)", ## cannot calculate multi variate reg now
            "alpha191_031": f"($close-Mean($close,12))/Mean($close,12)*100",
            "alpha191_032": f"(-1 * Sum(CSRank(Corr(CSRank($high), CSRank($volume), 3)), 3))",
            "alpha191_033": f"((((-1 * Min($low, 5)) + Ref(Min($low, 5), 5)) * CSRank(((Sum({f_return}, 240) - Sum({f_return}, 20)) / 220))) *Rank($volume, 5))",
            "alpha191_034": f"Mean($close,12)/$close",
            "alpha191_035": f"(Less(CSRank(WMA(Delta($open, 1), 15)), CSRank(WMA(Corr(($volume), (($open * 0.65) +($open *0.35)), 17),7))) * -1)",
            "alpha191_036": f"CSRank(Sum(Corr(CSRank($volume), CSRank($vwap), 6), 2))",
            "alpha191_037": f"(-1 * CSRank(((Sum($open, 5) * Sum({f_return}, 5)) - Ref((Sum($open, 5) * Sum({f_return}, 5)), 10))))",
            "alpha191_038": f"If((Sum($high,20)/20)<$high,-1*Delta($high,2),0)",
            "alpha191_039": f"((CSRank(WMA(Delta(($close), 2),8)) - CSRank(WMA(Corr((($vwap * 0.3) + ($open * 0.7)),Sum(Mean($volume,180), 37), 14), 12))) * -1)",
            "alpha191_040": f"Sum(If($close>Ref($close,1),$volume,0),26)/Sum(If($close<=Ref($close,1),$volume,0),26)*100",
            "alpha191_041": f"(CSRank(Greater(Delta(($vwap), 3), 5))* -1)",
            "alpha191_042": f"((-1 * CSRank(Std($high, 10))) * Corr($high, $volume, 10))",
            "alpha191_043": f"Sum(If($close>Ref($close,1),$volume,If($close<Ref($close,1),-$volume,0)),6)",
            "alpha191_044": f"(Rank(WMA(Corr((($low )), Mean($volume,10), 7), 6),4) + Rank(WMA(Delta(($vwap),3), 10), 15))",
            "alpha191_045": f"(CSRank(Delta(((($close * 0.6) + ($open *0.4))), 1)) * CSRank(Corr($vwap, Mean($volume,150), 15)))",
            "alpha191_046": f"(Mean($close,3)+Mean($close,6)+Mean($close,12)+Mean($close,24))/(4*$close)",
            "alpha191_047": f"EMA((Max($high,6)-$close)/(Max($high,6)-Min($low,6))*100,1/9)",
            "alpha191_048": f"(-1*((CSRank(((Sign(($close - Ref($close, 1))) + Sign((Ref($close, 1)  - Ref($close, 2)))) +Sign((Ref($close, 2) - Ref($close, 3)))))) * Sum($volume, 5)) / Sum($volume, 20))",
            "alpha191_049": f"Sum(If(($high+$low)>=(Ref($high,1)+Ref($low,1)), 0, Greater(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))),12)/(Sum(If(($high+$low)>=(Ref($high,1)+Ref($low,1)), 0, Greater(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))),12)+Sum(If(($high+$low)<=(Ref($high,1)+Ref($low,1)), 0, Greater(Abs($high-Ref($high,1)), Abs($low-Ref($low,1)))),12))",
            "alpha191_050": f"Sum(If(($high+$low)<=(Ref($high,1)+Ref($low,1)), 0, Greater(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))),12)/(Sum(If(($high+$low)<=(Ref($high,1)+Ref($low,1)), 0, Greater(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))),12)+Sum(If(($high+$low)>=(Ref($high,1)+Ref($low,1)), 0, Greater(Abs($high-Ref($high,1)), Abs($low-Ref($low,1)))),12))-Sum(If(($high+$low)>=(Ref($high,1)+Ref($low,1)), 0, Greater(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))),12)/(Sum(If($high+$low>=Ref($high,1)+Ref($low,1), 0, Greater(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))),12)+Sum(If(($high+$low)<=(Ref($high,1)+Ref($low,1)), 0, Greater(Abs($high-Ref($high,1)), Abs($low-Ref($low,1)))),12))",
            "alpha191_051": f"Sum(If(($high+$low)<=(Ref($high,1)+Ref($low,1)), 0, Greater(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))),12)/(Sum(If(($high+$low)<=(Ref($high,1)+Ref($low,1)), 0, Greater(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))),12)+Sum(If(($high+$low)>=(Ref($high,1)+Ref($low,1)), 0, Greater(Abs($high-Ref($high,1)), Abs($low-Ref($low,1)))),12))",
            "alpha191_052": f"Sum(Greater(0,$high-Ref(($high+$low+$close)/3,1)),26)/Sum(Greater(0,Ref(($high+$low+$close)/3,1)-$low),26)*100",
            "alpha191_053": f"Sum($close>Ref($close,1),12)/12*100",
            "alpha191_054": f"(-1 * CSRank((Std(Abs($close - $open), 252) + ($close - $open)) + Corr($close, $open,10)))",
            "alpha191_055": f"Sum(16*($close-Ref($close,1)+($close-$open)/2+Ref($close,1)-Ref($open,1))/If(And(Abs($high-Ref($close,1))>Abs($low-Ref($close,1)), Abs($high-Ref($close,1))>Abs($high-Ref($low,1))), Abs($high-Ref($close,1))+Abs($low-Ref($close,1))/2+Abs(Ref($close,1)-Ref($open,1))/4, If(And(Abs($low-Ref($close,1))>Abs($high-Ref($low,1)), Abs($low-Ref($close,1))>Abs($high-Ref($close,1))), Abs($low-Ref($close,1))+Abs($high-Ref($close,1))/2+Abs(Ref($close,1)-Ref($open,1))/4, Abs($high-Ref($low,1))+Abs(Ref($close,1)-Ref($open,1))/4))*Greater(Abs($high-Ref($close,1)),Abs($low-Ref($close,1))),20)",
            "alpha191_056": f"(CSRank(($open-Min($open,12)))<CSRank(Power(CSRank(Corr(Sum((($high+$low)/2),19),Sum(Mean($volume,40), 19), 13)), 5)))",
            "alpha191_057": f"EMA(($close-Min($low,9))/(Max($high,9)-Min($low,9))*100,1/3)",
            "alpha191_058": f"Sum($close>Ref($close,1),20)/20*100",  # count if
            "alpha191_059": f"Sum(If($close==Ref($close,1), 0, $close-If($close>Ref($close,1), Less($low,Ref($close,1)), Greater($high,Ref($close,1)))),20)",
            "alpha191_060": f"Sum((($close-$low)-($high-$close))/($high-$low)*$volume,20)",
            "alpha191_061": f"-Greater(CSRank(WMA(Delta($vwap,1),12)),CSRank(WMA(CSRank(Corr(($low),Mean($volume,80), 8)), 17)))",
            "alpha191_062": f"(-1 * Corr($high, CSRank($volume), 5))",
            "alpha191_063": f"EMA(Greater($close-Ref($close,1),0),1/6)/EMA(Abs($close-Ref($close,1)),1/6)*100",
            "alpha191_064": f"(Greater(CSRank(WMA(Corr(CSRank($vwap),CSRank($volume),4),4)),CSRank(WMA(Greater(Corr(CSRank($close), CSRank(Mean($volume,60)), 4), 13), 14))) * -1)",
            "alpha191_065": f"Mean($close,6)/$close",
            "alpha191_066": f"($close-Mean($close,6))/Mean($close,6)*100",
            "alpha191_067": f"EMA(Greater($close-Ref($close,1),0),1/24)/EMA(Abs($close-Ref($close,1)),1/24)*100",
            "alpha191_068": f"EMA((($high+$low)/2-(Ref($high,1)+Ref($low,1))/2)*($high-$low)/$volume,2/15)",
            "alpha191_069": f"If(Sum({f_dtm},20)>Sum({f_dbm},20), (Sum({f_dtm},20)-Sum({f_dbm},20))/Sum({f_dtm},20), If(Sum({f_dtm},20)==Sum({f_dbm},20), 0, (Sum({f_dtm},20)-Sum({f_dbm},20))/Sum({f_dbm},20)))",
            "alpha191_070": f"Std($money,6)",
            "alpha191_071": f"($close-Mean($close,24))/Mean($close,24)*100",
            "alpha191_072": f"EMA((Max($high,6)-$close)/(Max($high,6)-Min($low,6))*100,1/15)",
            "alpha191_073": f"-(Rank(WMA(WMA(Corr(($close),$volume,10),16),4),5)-CSRank(WMA(Corr($vwap, Mean($volume,30), 4),3)))",
            "alpha191_074": f"(CSRank(Corr(Sum((($low*0.35)+($vwap*0.65)),20),Sum(Mean($volume,40),20),7))+CSRank(Corr(CSRank($vwap), CSRank($volume), 6)))",
            "alpha191_075": f"Sum(And($close>$open, ChangeInstrument('{self.benchmark}', $close)<ChangeInstrument('{self.benchmark}', $open)), 50)/Sum(ChangeInstrument('{self.benchmark}', $close)<ChangeInstrument('{self.benchmark}', $open) ,50)",  # count if
            "alpha191_076": f"Std(Abs(($close/Ref($close,1)-1))/$volume,20)/Mean(Abs(($close/Ref($close,1)-1))/$volume,20)",
            "alpha191_077": f"Less(CSRank(WMA((((($high+$low)/2)+$high)-($vwap+$high)),20)),CSRank(WMA(Corr((($high + $low) / 2), Mean($volume,40), 3), 6)))",
            "alpha191_078": f"(($high+$low+$close)/3-Mean(($high+$low+$close)/3,12))/(0.015*Mean(Abs($close-Mean(($high+$low+$close)/3,12)),12))",
            "alpha191_079": f"EMA(Greater($close-Ref($close,1),0),1/12)/EMA(Abs($close-Ref($close,1)),1/12)*100",
            "alpha191_080": f"($volume-Ref($volume,5))/Ref($volume,5)*100",
            "alpha191_081": f"EMA($volume,2/21)",
            "alpha191_082": f"EMA((Max($high,6)-$close)/(Max($high,6)-Min($low,6))*100,1/20)",
            "alpha191_083": f"(-1 * CSRank(Cov(CSRank($high), CSRank($volume), 5)))",
            "alpha191_084": f"Sum(If($close>Ref($close,1), $volume, If($close<Ref($close,1), -$volume, 0)),20)",
            "alpha191_085": f"(Rank(($volume / Mean($volume,20)), 20) * Rank((-1 * Delta($close, 7)), 8))",
            "alpha191_086": f"If((0.25 < (((Ref($close, 20) - Ref($close, 10)) / 10) - ((Ref($close, 10) - $close) / 10))), (-1 * 1), If(((((Ref($close, 20) - Ref($close, 10)) / 10) - ((Ref($close, 10) - $close) / 10)) < 0), 1, - ($close - Ref($close, 1))))",
            "alpha191_087": f"((CSRank(WMA(Delta($vwap, 4), 7)) + Rank(WMA((((($low * 0.9) + ($low * 0.1)) - $vwap) /($open - (($high + $low) / 2))), 11), 7)) * -1)",
            "alpha191_088": f"($close-Ref($close,20))/Ref($close,20)*100",
            "alpha191_089": f"2*(EMA($close,2/13)-EMA($close,2/27)-EMA(EMA($close,2/13)-EMA($close,2/27),2/10))",
            "alpha191_090": f"(CSRank(Corr(CSRank($vwap), CSRank($volume), 5)) * -1)",
            "alpha191_091": f"((CSRank(($close - Greater($close, 5)))*CSRank(Corr((Mean($volume,40)), $low, 5))) * -1)",
            "alpha191_092": f"(Greater(CSRank(WMA(Delta((($close*0.35)+($vwap*0.65)),2),3)),Rank(WMA(Abs(Corr((Mean($volume,180)), $close, 13)), 5), 15)) * -1)",
            "alpha191_093": f"Sum(If($open>=Ref($open,1), 0, Greater(($open-$low),($open-Ref($open,1)))),20)",
            "alpha191_094": f"Sum(If($close>Ref($close,1), $volume, If($close<Ref($close,1), -$volume, 0)),30)",
            "alpha191_095": f"Std($money,20)",
            "alpha191_096": f"EMA(EMA(($close-Min($low,9))/(Max($high,9)-Min($low,9))*100,1/3),1/3)",
            "alpha191_097": f"Std($volume,10)",
            "alpha191_098": f"If(Or(((Delta((Sum($close, 100) / 100), 100) / Ref($close, 100)) < 0.05), ((Delta((Sum($close, 100) / 100), 100) /Ref($close, 100)) == 0.05)), -1 * ($close - Min($close, 100)), -1 * Delta($close, 3))",
            "alpha191_099": f"(-1 * CSRank(Cov(CSRank($close), CSRank($volume), 5)))",
            "alpha191_100": f"Std($volume,20)",
            "alpha191_101": f"((CSRank(Corr($close, Sum(Mean($volume,30), 37), 15)) < CSRank(Corr(CSRank((($high * 0.1) + ($vwap * 0.9))),CSRank($volume), 11))) * -1)",
            "alpha191_102": f"EMA(Greater($volume-Ref($volume,1),0),1/6)/EMA(Abs($volume-Ref($volume,1)),1/6)*100",
            "alpha191_103": f"(IdxMin($low,20)/20)*100",
            "alpha191_104": f"(-1 * (Delta(Corr($high, $volume, 5), 5) * CSRank(Std($close, 20))))",
            "alpha191_105": f"(-1 * Corr(CSRank($open), CSRank($volume), 10))",
            "alpha191_106": f"$close-Ref($close,20)",
            "alpha191_107": f"(((-1 * CSRank(($open - Ref($high, 1)))) * CSRank(($open - Ref($close, 1)))) * CSRank(($open - Ref($low, 1))))",
            "alpha191_108": f"(Power(CSRank(($high - Less($high, 2))), CSRank(Corr(($vwap), (Mean($volume,120)), 6))) * -1)",
            "alpha191_109": f"EMA($high-$low,2/10)/EMA(EMA($high-$low,2/10),2/10)",
            "alpha191_110": f"Sum(Greater(0,$high-Ref($close,1)),20)/Sum(Greater(0,Ref($close,1)-$low),20)*100",
            "alpha191_111": f"EMA($volume*(($close-$low)-($high-$close))/($high-$low),2/11)-EMA($volume*(($close-$low)-($high-$close))/($high-$low),2/4)",
            "alpha191_112": f"(Sum(If($close-Ref($close,1)>0, $close-Ref($close,1), 0),12)-Sum(If($close-Ref($close,1)<0, Abs($close-Ref($close,1)), 0),12))/(Sum(If($close-Ref($close,1)>0, $close-Ref($close,1), 0),12)+Sum(If($close-Ref($close,1)<0, Abs($close-Ref($close,1)), 0),12))*100",
            "alpha191_113": f"(-1 * ((CSRank((Sum(Ref($close, 5), 20) / 20)) * Corr($close, $volume, 2)) * CSRank(Corr(Sum($close, 5),Sum($close, 20), 2))))",
            "alpha191_114": f"((CSRank(Ref((($high - $low) / (Sum($close, 5) / 5)), 2)) * CSRank(CSRank($volume))) / ((($high - $low) /(Sum($close, 5) / 5)) / ($vwap - $close)))",
            "alpha191_115": f"Power(CSRank(Corr((($high * 0.9) + ($close * 0.1)), Mean($volume,30), 10)), CSRank(Corr(Rank((($high + $low) /2), 4), Rank($volume, 10), 7)))",
            "alpha191_116": f"Slope($close, 20)",
            "alpha191_117": f"((Rank($volume, 32) * (1 - Rank((($close + $high) - $low), 16))) * (1 - Rank({f_return}, 32)))",
            "alpha191_118": f"Sum($high-$open,20)/Sum($open-$low,20)*100",
            "alpha191_119": f"(CSRank(WMA(Corr($vwap,Sum(Mean($volume,5),26),5),7))-CSRank(WMA(Rank(Less(Corr(CSRank($open), CSRank(Mean($volume,15)), 21), 9), 7), 8)))",
            "alpha191_120": f"(CSRank(($vwap - $close)) / CSRank(($vwap + $close)))",
            "alpha191_121": f"(Power(CSRank(($vwap - Less($vwap, 12))), Rank(Corr(Rank($vwap, 20), Rank(Mean($volume,60), 2), 18), 3)) *-1)",
            "alpha191_122": f"(EMA(EMA(EMA(Log($close),2/13),2/13),2/13)-Ref(EMA(EMA(EMA(Log($close),2/13),2/13),2/13),1))/Ref(EMA(EMA(EMA(Log($close),2/13),2/13),2/13),1)",
            "alpha191_123": f"((CSRank(Corr(Sum((($high + $low) / 2), 20), Sum(Mean($volume,60), 20), 9)) < CSRank(Corr($low, $volume,6))) * -1)",
            "alpha191_124": f"($close - $vwap) / WMA(CSRank(Max($close, 30)),2)",
            "alpha191_125": f"(CSRank(WMA(Corr(($vwap), Mean($volume,80),17), 20)) / CSRank(WMA(Delta((($close * 0.5)+ ($vwap * 0.5)), 3), 16)))",
            "alpha191_126": f"($close+$high+$low)/3",
            "alpha191_127": f"Power(Mean(Power(100*($close-Greater($close,12))/(Greater($close,12)),2), 12), 1/2)",
            "alpha191_128": f"100-(100/(1+Sum(If(($high+$low+$close)/3>Ref(($high+$low+$close)/3,1), ($high+$low+$close)/3*$volume, 0),14)/Sum(If(($high+$low+$close)/3<Ref(($high+$low+$close)/3,1), ($high+$low+$close)/3*$volume, 0), 14)))",
            "alpha191_129": f"Sum(If($close-Ref($close,1)<0, Abs($close-Ref($close,1)), 0),12)",
            "alpha191_130": f"(CSRank(WMA(Corr((($high+$low)/2),Mean($volume,40),9),10))/CSRank(WMA(Corr(CSRank($vwap), CSRank($volume), 7),3)))",
            "alpha191_131": f"Power(CSRank(Ref($vwap, 1)), Rank(Corr($close,Mean($volume,50), 18), 18))",
            "alpha191_132": f"Mean($money,20)",
            "alpha191_133": f"(IdxMax($high,20)/20)*100-(IdxMin($low,20)/20)*100",
            "alpha191_134": f"($close-Ref($close,12))/Ref($close,12)*$volume",
            "alpha191_135": f"EMA(Ref($close/Ref($close,20),1),1/20)",
            "alpha191_136": f"((-1 * CSRank(Delta({f_return}, 3))) * Corr($open, $volume, 10))",
            "alpha191_137": f"16*($close-Ref($close,1)+($close-$open)/2+Ref($close,1)-Ref($open,1))/(If(And(Abs($high-Ref($close,1))>Abs($low-Ref($close,1)), Abs($high-Ref($close,1))>Abs($high-Ref($low,1))), Abs($high-Ref($close,1))+Abs($low-Ref($close,1))/2+Abs(Ref($close,1)-Ref($open,1))/4, If(And(Abs($low-Ref($close,1))>Abs($high-Ref($low,1)), Abs($low-Ref($close,1))>Abs($high-Ref($close,1))), Abs($low-Ref($close,1))+Abs($high-Ref($close,1))/2+Abs(Ref($close,1)-Ref($open,1))/4, Abs($high-Ref($low,1))+Abs(Ref($close,1)-Ref($open,1))/4)))*Greater(Abs($high-Ref($close,1)),Abs($low-Ref($close,1)))",
            "alpha191_138": f"((CSRank(WMA(Delta(((($low*0.7)+($vwap*0.3))),3),20))-Rank(WMA(Rank(Corr(Rank($low, 8), Rank(Mean($volume,60), 17), 5), 19), 16), 7)) * -1)",
            "alpha191_139": f"(-1 * Corr($open, $volume, 10))",
            "alpha191_140": f"Less(CSRank(WMA(((CSRank($open)+CSRank($low))-(CSRank($high)+CSRank($close))), 8)),Rank(WMA(Corr(Rank($close, 8), Rank(Mean($volume,60), 20), 8), 7), 3))",
            "alpha191_141": f"(CSRank(Corr(CSRank($high), CSRank(Mean($volume,15)), 9))* -1)",
            "alpha191_142": f"(((-1 * CSRank(Rank($close, 10))) * CSRank(Delta(Delta($close, 1), 1))) * CSRank(Rank(($volume/Mean($volume,20)), 5)))",
            # "alpha191_143": f"If($close>Ref($close,1), ($close-Ref($close,1))/Ref($close,1)*SELF, SELF)",  初值无法确定
            "alpha191_144": f"Sum(If($close<Ref($close,1), Abs($close/Ref($close,1)-1)/$money, 0),20)/Sum($close<Ref($close,1),20)",  # count if
            "alpha191_145": f"(Mean($volume,9)-Mean($volume,26))/Mean($volume,12)*100",
            "alpha191_146": f"Mean(($close-Ref($close,1))/Ref($close,1)-EMA(($close-Ref($close,1))/Ref($close,1),2/61),20)*(($close-Ref($close,1))/Ref($close,1)-EMA(($close-Ref($close,1))/Ref($close,1),2/61))/EMA(((($close-Ref($close,1))/Ref($close,1)-EMA(($close-Ref($close,1))/Ref($close,1),2/61))), 2 / 61)",
            "alpha191_147": f"Slope(Mean($close,12), 12)",
            "alpha191_148": f"((CSRank(Corr(($open), Sum(Mean($volume,60), 9), 6)) < CSRank(($open - Min($open, 14)))) * -1)",
            # "alpha191_149": f"REGBETA(FILTER($close/Ref($close,1)-1,BANCHMARKINDEX$close<Ref(BANCHMARKINDEX$close,1)),FILTER(BANCHMARKINDEX$close/Ref(BANCHMARKINDEX$close,1)-1,BANCHMARKINDEX$close<DELAY(BANCHMARKINDEX$close,1)),252)", REGBETA
            "alpha191_150": f"($close+$high+$low)/3*$volume",
            "alpha191_151": f"EMA($close-Ref($close,20),1/20)",
            "alpha191_152": f"EMA(Mean(Ref(EMA(Ref($close/Ref($close,9),1),1/9),1),12)-Mean(Ref(EMA(Ref($close/Ref($close,9),1),1/9),1),26),1/9)",
            "alpha191_153": f"(Mean($close,3)+Mean($close,6)+Mean($close,12)+Mean($close,24))/4",
            "alpha191_154": f"((($vwap - Less($vwap, 16))) < (Corr($vwap, Mean($volume,180), 18)))",
            "alpha191_155": f"EMA($volume,2/13)-EMA($volume,2/27)-EMA(EMA($volume,2/13)-EMA($volume,2/27),2/10)",
            "alpha191_156": f"(Greater(CSRank(WMA(Delta($vwap, 5), 3)), CSRank(WMA(((Delta((($open * 0.15) + ($low *0.85)),2) / (($open * 0.15) + ($low * 0.85))) * -1), 3))) * -1)",
            "alpha191_157": f"(Less(Prod(CSRank(CSRank(Log(Sum(Min(CSRank(CSRank((-1 * CSRank(Delta(($close - 1), 5))))), 2), 1)))), 1), 5) +Rank(Ref((-1 * {f_return}), 6), 5))",
            "alpha191_158": f"(($high-EMA($close,2/15))-($low-EMA($close,2/15)))/$close",
            "alpha191_159": f"(($close-Sum(Less($low,Ref($close,1)),6))/Sum(Greater($high, Ref($close,1))-Less($low,Ref($close,1)),6)*12*24+($close-Sum(Less($low,Ref($close,1)),12))/Sum(Greater($high,Ref($close,1))-Less($low,Ref($close,1)),12)*6*24+($close-Sum(Less($low,Ref($close,1)),24))/Sum(Greater($high,Ref($close,1))-Less($low,Ref($close,1)),24)*6*24)*100/(6*12+6*24+12*24)",
            "alpha191_160": f"EMA(If($close<=Ref($close,1), Std($close,20), 0),1/20)",
            "alpha191_161": f"Mean(Greater(Greater(($high-$low),Abs(Ref($close,1)-$high)),Abs(Ref($close,1)-$low)),12)",
            "alpha191_162": f"(EMA(Greater($close-Ref($close,1),0),1/12)/EMA(Abs($close-Ref($close,1)),1/12)*100-Less(EMA(Greater($close-Ref($close,1),0),1/12)/EMA(Abs($close-Ref($close,1)),1/12)*100,12))/(Greater(EMA(Greater($close-Ref($close,1),0),1/12)/EMA(Abs($close-Ref($close,1)),1/12)*100,12)-Less(EMA(Greater($close-Ref($close,1),0),1/12)/EMA(Abs($close-Ref($close,1)),1/12)*100,12))",
            # "alpha191_163": f"CSRank(((((-1 * {f_return}) * Mean($volume,20)) * $vwap) * ($high - $close)))",
            "alpha191_164": f"EMA((If(($close>Ref($close,1)),1/($close-Ref($close,1)),1)-Less(If(($close>Ref($close,1)),1/($close-Ref($close,1)),1),12))/($high-$low)*100,2/13)",
            # "alpha191_165": f"Greater(SumAC($close-Mean($close,48)))-Less(SumAC($close-Mean($close,48)))/Std($close,48) unknown function sumac
            "alpha191_166": f"-87.17797887*Sum($close/Ref($close,1)-Mean($close/Ref($close,1),20),20)/(18*Power(Sum(Power(($close/Ref($close,1)-Mean($close/Ref($close,1), 20)),2),20),1.5))",
            "alpha191_167": f"Sum(If($close-Ref($close,1)>0, $close-Ref($close,1), 0),12)",
            "alpha191_168": f"(-1*$volume/Mean($volume,20))",
            "alpha191_169": f"EMA(Mean(Ref(EMA($close-Ref($close,1),1/9),1),12)-Mean(Ref(EMA($close-Ref($close,1),1/9),1),26),1/10)",
            "alpha191_170": f"((((CSRank((1 / $close)) * $volume) / Mean($volume,20)) * (($high * CSRank(($high - $close))) / (Sum($high, 5) /5))) - CSRank(($vwap - Ref($vwap, 5))))",
            "alpha191_171": f"((-1 * (($low - $close) * Power($open, 5))) / (($close - $high) * Power($close, 5)))",
            "alpha191_172": f"Mean(Abs(Sum(If(And({f_ld}>0, {f_ld}>{f_hd}), {f_ld}, 0),14)*100/Sum({f_tr},14)-Sum(If(And({f_hd}>0, {f_hd}>{f_ld}), {f_hd}, 0),14)*100/Sum({f_tr},14))/(Sum(If(And({f_ld}>0, {f_ld}>{f_hd}), {f_ld}, 0),14)*100/Sum({f_tr},14)+Sum(If(And({f_hd}>0, {f_hd}>{f_ld}),{f_hd},0),14)*100/Sum({f_tr},14))*100,6)",
            "alpha191_173": f"3*EMA($close,2/13)-2*EMA(EMA($close, 2/13), 2/13)+EMA(EMA(EMA(Log($close),2/13),2/13),2/13)",
            "alpha191_174": f"EMA(If($close>Ref($close,1), Std($close,20), 0),1/20)",
            "alpha191_175": f"Mean(Greater(Greater(($high-$low),Abs(Ref($close,1)-$high)),Abs(Ref($close,1)-$low)),6)",
            "alpha191_176": f"Corr(CSRank((($close - Min($low, 12)) / (Max($high, 12) - Min($low,12)))), CSRank($volume), 6)",
            "alpha191_177": f"(IdxMax($high,20)/20)*100",
            "alpha191_178": f"($close-Ref($close,1))/Ref($close,1)*$volume",
            "alpha191_179": f"(CSRank(Corr($vwap, $volume, 4)) *CSRank(Corr(CSRank($low), CSRank(Mean($volume,50)), 12)))",
            "alpha191_180": f"If(Mean($volume,20) < $volume, -1 * Rank(Abs(Delta($close, 7)), 60) * Sign(Delta($close, 7)), (-1 *$volume))",
            "alpha191_181": f"Sum((($close/Ref($close,1)-1)-Mean(($close/Ref($close,1)-1),20))-Power(ChangeInstrument('{self.benchmark}', $close)-Mean(ChangeInstrument('{self.benchmark}', $close),20), 2),20)/Sum(Power(ChangeInstrument('{self.benchmark}', $close)-Mean(ChangeInstrument('{self.benchmark}', $close),20), 3),20)",
            "alpha191_182": f"Sum(Or(And($close>$open, ChangeInstrument('{self.benchmark}', $close)>ChangeInstrument('{self.benchmark}', $open)), And($close<$open, ChangeInstrument('{self.benchmark}', $close)<ChangeInstrument('{self.benchmark}', $open))),20)/20",
            # "alpha191_183": f"Greater(SumAC($close-Mean($close,24)))-Less(SumAC($close-Mean($close,24)))/Std($close,24)", unknow sumac
            "alpha191_184": f"(CSRank(Corr(Ref(($open - $close), 1), $close, 200)) + CSRank(($open - $close)))",
            "alpha191_185": f"CSRank(-1 * Power(1 - ($open / $close), 2))",
            "alpha191_186": f"(Mean(Abs(Sum(If(And({f_ld}>0, {f_ld}>{f_hd}), {f_ld}, 0),14)*100/Sum({f_tr},14)-Sum(If(And({f_hd}>0,{f_hd}>{f_ld}), {f_hd}, 0),14)*100/Sum({f_tr},14))/(Sum(If(And({f_ld}>0, {f_ld}>{f_hd}), {f_ld}, 0),14)*100/Sum({f_tr},14)+Sum(If(And({f_hd}>0, {f_hd}>{f_ld}), {f_hd}, 0),14)*100/Sum({f_tr},14))*100, 6)+Ref(Mean(Abs(Sum(If(And({f_ld}>0,{f_ld}>{f_hd}), {f_ld}, 0), 14)*100/Sum({f_tr},14)-Sum(If(And({f_hd}>0, {f_hd}>{f_ld}), {f_hd}, 0),14)*100/Sum({f_tr},14))/(Sum(If(And({f_ld}>0, {f_ld}>{f_hd}), {f_ld}, 0),14)*100/Sum({f_tr},14)+Sum(If(And({f_hd}>0, {f_hd}>{f_ld}), {f_hd}, 0),14)*100/Sum({f_tr},14))*100,6),6))/2",
            "alpha191_187": f"Sum(If($open<=Ref($open,1), 0, Greater(($high-$open),($open-Ref($open,1)))),20)",
            "alpha191_188": f"(($high-$low-EMA($high-$low, 2/11))/EMA($high-$low, 2/11))*100",
            "alpha191_189": f"Mean(Abs($close-Mean($close,6)),6)",
            "alpha191_190": f"Log((Sum($close/Ref($close, 1)>Power($close/Ref($close,19), 1/20),20)-1)*Sum(If($close/Ref($close, 1)<Power($close/Ref($close,19), 1/20),  Power($close/Ref($close, 1)-Power($close/Ref($close,19),1/20), 2), 0), 20 ) / (Sum($close/Ref($close, 1)<Power($close/Ref($close,19), 1/20),20) * (Sum(If($close/Ref($close, 1)>Power($close/Ref($close,19), 1/20), Power($close/Ref($close,1)-Power($close/Ref($close,19), 1/20), 2), 0), 20)))+1e-16)",
            "alpha191_191": f"((Corr(Mean($volume,20), $low, 5) + (($high + $low) / 2)) - $close)",
        }

        return list(alpha_components.values()), list(alpha_components.keys())
