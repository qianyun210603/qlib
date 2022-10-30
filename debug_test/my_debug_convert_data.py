import qlib
from qlib.constant import REG_CN
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import check_transform_proc

_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]

if __name__ == "__main__":
    provider_uri = r"D:\Documents\TradeResearch\qlib_test\rqdata_convert"
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    segments = {
        "train": ("2017-01-01", "2019-01-01"),
        "valid": ("2019-01-01", "2020-12-31"),
        "test": ("2020-01-01", "2020-07-01"),
    }
    features = ['($adjclosestock * 100 / $conversionprice) / $close - 1', '$pure_bond_ytm']
    feature_labels = ["convertion_premium", 'pure_bond_ytm']

    assert len(feature_labels) == len(features), "'features' and its labels must have same length"

    data_loader = {
        "class": "QlibDataLoader",
        "kwargs": {
            "config": {
                "feature": (features, feature_labels),
                "label": (["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]),
            },
        },
    }

    infer_processors = check_transform_proc([
        {"class": 'RobustZScoreNorm', "kwargs": {"fields_group": "feature", 'clip_outlier': True}},
        {"class": 'Fillna', "kwargs": {"fields_group": "feature"}}
    ], segments["train"][0], segments["train"][1])
    learn_processors = check_transform_proc([
        {"class": "DropnaLabel"},
        {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
    ], segments["train"][0], segments["train"][1])

    process_type = DataHandlerLP.PTYPE_A

    h = DataHandlerLP(
        instruments='all',
        start_time=segments["train"][0],
        end_time=segments["test"][1],
        data_loader=data_loader,
        infer_processors=infer_processors,
        learn_processors=learn_processors,
        process_type=process_type,
    )
    dataset = DatasetH(
        handler=h, segments=segments
    )

    dataset.prepare('train', col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    #dataset.prepare('valid', col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    # dataset.prepare('test', col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
    h.fetch()

