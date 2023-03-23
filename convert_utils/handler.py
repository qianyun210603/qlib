from qlib.contrib.data.handler import _DEFAULT_LEARN_PROCESSORS, check_transform_proc
from qlib.data.dataset.handler import DataHandlerLP


class AlphaConvert1(DataHandlerLP):
    def __init__(
        self,
        instruments="converts",
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
    def get_label_config(return_field="open"):
        return [f"Ref(${return_field}, -2)/Ref(${return_field}, -1) - 1"], ["LABEL0"]

    def parse_config_to_fields(self):
        """create factors from config"""

        alpha_components = {
            "convertion_premium": "($rawclosestock * 100 / $conversionprice)/$close - 1",
            "pure_bond_ytm": "$pure_bond_ytm",
            "remaining_size": "$remaining_size",
            "turnover_rate": "$turnover_rate",
            "ROC5": "Ref($close, 5)/$close",
            "MA10": "Mean($close, 10)/$close",
            "RESI30": "Resi($close, 30)/$close",
            "CNTP20": "Mean($close>Ref($close, 1), 20)",
            "WVMA5": "Std(Abs($close/Ref($close, 1)-1)*$volume, 5)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 5)+1e-12)",
            "alpha191_003": "-Sum(If($close==Ref($close,1), 0, $close-If($close>Ref($close,1), Less($low,Ref($close,1)), Greater($high,Ref($close,1)))),6)",
            "alpha191_006": "(CSRank(Sign(Delta(((($open * 0.85) + ($high * 0.15))), 4)))* -1)",
            "alpha191_008": "CSRank(Delta((((($high + $low) / 2) * 0.2) + ($vwap * 0.8)), 4) * -1)",
            "alpha191_009": "EMA((($high+$low)/2-(Ref($high,1)+Ref($low,1))/2)*($high-$low)/$volume, 2/7)",
            "alpha191_011": "Sum((($close-$low)-($high-$close))/($high-$low)*$volume,6)",
            "alpha191_012": "CSRank(($open - (Sum($vwap, 10) / 10))) * (-CSRank(Abs(($close - $vwap))))",
            "alpha191_017": "Power(CSRank(($vwap - Greater($vwap, 15))), Delta($close, 5))",
            "alpha191_019": "If($close<Ref($close,5), ($close-Ref($close,5))/Ref($close,5), If($close==Ref($close,5), 0, ($close-Ref($close,5))/$close))",
            "alpha191_022": "EMA((($close-Mean($close,6))/Mean($close,6)-Ref(($close-Mean($close,6))/Mean($close,6),3)),1/12)",
            "alpha191_029": "($close-Ref($close,6))/Ref($close,6)*$volume",
            "alpha191_037": "(-1 * CSRank(((Sum($open, 5) * Sum(($close/Ref($close, 1)-1), 5)) - Ref((Sum($open, 5) * Sum(($close/Ref($close, 1)-1), 5)), 10))))",
            "alpha191_044": "(Rank(WMA(Corr((($low )), Mean($volume,10), 7), 6),4) + Rank(WMA(Delta(($vwap),3), 10), 15))",
            "alpha191_046": "(Mean($close,3)+Mean($close,6)+Mean($close,12)+Mean($close,24))/(4*$close)",
            "alpha191_064": "(Greater(CSRank(WMA(Corr(CSRank($vwap),CSRank($volume),4),4)),CSRank(WMA(Greater(Corr(CSRank($close), CSRank(Mean($volume,60)), 4), 13), 14))) * -1)",
            "alpha191_073": "-(Rank(WMA(WMA(Corr(($close),$volume,10),16),4),5)-CSRank(WMA(Corr($vwap, Mean($volume,30), 4),3)))",
            "alpha191_087": "((CSRank(WMA(Delta($vwap, 4), 7)) + Rank(WMA((((($low * 0.9) + ($low * 0.1)) - $vwap) /($open - (($high + $low) / 2))), 11), 7)) * -1)",
            "alpha191_089": "2*(EMA($close,2/13)-EMA($close,2/27)-EMA(EMA($close,2/13)-EMA($close,2/27),2/10))",
            "alpha191_091": "((CSRank(($close - Greater($close, 5)))*CSRank(Corr((Mean($volume,40)), $low, 5))) * -1)",
            "alpha191_155": "EMA($volume,2/13)-EMA($volume,2/27)-EMA(EMA($volume,2/13)-EMA($volume,2/27),2/10)",
            "diff_premium_02": "($rawclosestock * 100 / $conversionprice)/$close - Ref(($rawclosestock * 100 / $conversionprice)/$close, 2)",
            "diff_premium_05": "($rawclosestock * 100 / $conversionprice)/$close - Ref(($rawclosestock * 100 / $conversionprice)/$close, 5)",
        }

        return list(alpha_components.values()), list(alpha_components.keys())
