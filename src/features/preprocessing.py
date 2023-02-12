import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
import category_encoders as ce


class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._feat_scaler = StandardScaler()
        self._num_cols = None

    def fit(self, X, y=None):
        self._num_cols = list(set(X.select_dtypes(include='float64').columns) - {'month_sin', 'month_cos'})
        self._feat_scaler = self._feat_scaler.fit(X[self._num_cols])
        return self

    def transform(self, X, y=None):
        X[self._num_cols] = self._feat_scaler.transform(X[self._num_cols])
        return X

    def fit_transform(self, X, y=None):
        self._num_cols = list(set(X.select_dtypes(include='float64').columns) - {'month_sin', 'month_cos'})
        self._feat_scaler = self._feat_scaler.fit(X[self._num_cols])
        X[self._num_cols] = self._feat_scaler.transform(X[self._num_cols])
        return X


class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=1.0, min_samples_leaf=1):
        self._encoder = ce.TargetEncoder(min_samples_leaf=min_samples_leaf, smoothing=smoothing)
        self._cat_cols = None

    def fit(self, X, y=None):
        self._cat_cols = X.select_dtypes(include='int64').columns
        X[self._cat_cols] = X[self._cat_cols].astype(object)
        self._encoder = self._encoder.fit(X[self._cat_cols], y)
        return self

    def transform(self, X, y=None):
        X[self._cat_cols] = self._encoder.transform(X[self._cat_cols])
        return X


class TargetTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._scaler = StandardScaler()

    def fit(self, X, y=None):
        self._scaler = self._scaler.fit(np.log(X))
        return self

    def transform(self, X, y=None):
        X = np.log(X)
        X = self._scaler.transform(X)
        return X

    def inverse_transform(self, X, y=None):
        X = self._scaler.inverse_transform(X)
        X = np.exp(X)
        return X


class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, prob=0.03):
        self._prob = prob
        self._threshold = None

    def fit(self, X, y=None):
        self._threshold = np.mean(y) / self._prob
        return self

    def fit_transform(self, X, y=None):
        self._threshold = np.mean(y) / self._prob
        X = X[y < self._threshold]
        y = y[y < self._threshold]
        return X, y

    def transform(self, X, y=None):
        X = X[y < self._threshold]
        y = y[y < self._threshold]
        return X, y


def get_preprocessing_pipeline(steps_only: bool = False, enc_smoothing=1.0, enc_min_samples_leaf=1) -> Pipeline:
    steps = [('scaler', FeatureScaler()),
             ('encoder', TargetEncoder(enc_smoothing, enc_min_samples_leaf))]
    if steps_only:
        return steps
    else:
        return Pipeline(steps=steps)


def preprocess_data(train_x: pd.DataFrame, train_y: pd.DataFrame,
                    val_x: pd.DataFrame, val_y: pd.DataFrame,
                    outlier_prob=0.03, enc_smoothing=1.0, enc_min_samples_leaf=1) -> (pd.DataFrame, pd.DataFrame,
                                                                                      pd.DataFrame, pd.DataFrame):

    outlier_remover = OutlierRemover(prob=outlier_prob)
    train_x, train_y = outlier_remover.fit_transform(train_x, train_y)
    val_x, val_y = outlier_remover.transform(val_x, val_y)

    prep_pipeline = get_preprocessing_pipeline()
    train_x = prep_pipeline.fit_transform(train_x, train_y)
    val_x = prep_pipeline.transform(val_x)

    return train_x, train_y, val_x, val_y


