# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Text, Union, cast

import numpy as np
import pandas as pd
from sklearn.svm import SVR

from qlib.data.dataset.weight import Reweighter

from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...model.base import Model


class SVMRegression(Model):
    """SVM Regression Model"""

    def __init__(
        self,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        tol=1e-3,
        C=1.0,
        epsilon=0.1,
        shrinking=True,
        cache_size=200,
        verbose=False,
        max_iter=-1,
    ):
        self.predictor = SVR(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=C,
            epsilon=epsilon,
            shrinking=shrinking,
            cache_size=cache_size,
            verbose=verbose,
            max_iter=max_iter,
        )
        self.factor_names_ = None

    def fit(self, dataset: DatasetH, reweighter: Reweighter = None):
        df_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if df_train.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")
        X, y = df_train["feature"].values, np.squeeze(df_train["label"].values)
        w = None if reweighter is None else cast(pd.Series, reweighter.reweight(df_train)).value
        self.factor_names_ = df_train["feature"].columns
        self.predictor.fit(X, y, w)
        return self

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.predictor.fit_status_ != 0:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        x_test = x_test[self.factor_names_]
        return pd.Series(self.predictor.predict(x_test), index=x_test.index)
