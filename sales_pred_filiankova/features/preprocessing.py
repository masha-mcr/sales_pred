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
        self._cat_cols = ['shop_id', 'city_id', 'item_id', 'item_category_id', 'item_global_category_id']

    def fit(self, X, y=None):
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
    def __init__(self, target='item_cnt_month', prob=0.03):
        self._prob = prob
        self._threshold = None
        self._target = target

    def fit(self, X, y=None):
        self._threshold = np.mean(X[self._target]) / self._prob
        return self

    def fit_transform(self, X, y=None):
        self._threshold = np.mean(X[self._target]) / self._prob
        X = X[X[self._target] < self._threshold]
        return X

    def transform(self, X, y=None):
        X = X[X[self._target] < self._threshold]
        return X


class Preprocessor:
    def __init__(self, target_col, outlier_prob=0.03, **encoder_params):
        self.target_transformer = TargetTransformer()
        self._outlier_remover = OutlierRemover(prob=outlier_prob, target=target_col)
        self._target_col = target_col
        self._feature_prep_pipeline = Pipeline(steps=[('scaler', FeatureScaler()),
                                                      ('encoder', TargetEncoder(**encoder_params))])

    def preprocess_data(self, train_data=None, val_data=None, test_data=None):

        output = []
        if train_data is not None:
            train = self._outlier_remover.fit_transform(train_data)
            train_y = train[self._target_col]
            train_x = train.drop([self._target_col])
            train_y = self.target_transformer.fit_transform(train_y.values.reshape(-1, 1))
            train_x = self._feature_prep_pipeline.fit_transform(train_x, train_y)
            output.append(train_x)
            output.append(train_y)
        if val_data is not None:
            val = self._outlier_remover.transform(val_data)
            val_y = val[self._target_col]
            val_x = val.drop([self._target_col])
            val_y = self.target_transformer.transform(val_y.values.reshape(-1, 1))
            val_x = self._feature_prep_pipeline.transform(val_x)
            output.append(val_x)
            output.append(val_y)
        if test_data is not None:
            test_x = self._feature_prep_pipeline.transform(test_data)
            output.append(test_x)

        return output


