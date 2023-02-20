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
        self._feat_scaler = self._feat_scaler.fit(X.loc[:, self._num_cols])
        return self

    def transform(self, X, y=None):
        X.loc[:, self._num_cols] = self._feat_scaler.transform(X.loc[:, self._num_cols])
        return X


class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=1.0, min_samples_leaf=1):
        self._encoder = ce.TargetEncoder(min_samples_leaf=min_samples_leaf, smoothing=smoothing)
        self._cat_cols = ['shop_id', 'city_id', 'item_id', 'item_category_id', 'item_global_category_id']

    def fit(self, X, y=None):
        X.loc[:, self._cat_cols] = X.loc[:, self._cat_cols].astype(object)
        self._encoder = self._encoder.fit(X.loc[:, self._cat_cols], y)
        return self

    def transform(self, X, y=None):
        X.loc[:, self._cat_cols] = self._encoder.transform(X.loc[:, self._cat_cols])
        return X


class TargetTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, no_log=False):
        self._scaler = StandardScaler()
        self._no_log = no_log

    def fit(self, X, y=None):
        if self._no_log:
            self._scaler = self._scaler.fit(X)
        else:
            self._scaler = self._scaler.fit(np.log(X))
        return self

    def transform(self, X, y=None):
        if not self._no_log:
            X = np.log(X)
        X = self._scaler.transform(X)
        return X

    def inverse_transform(self, X, y=None):
        X = self._scaler.inverse_transform(X)
        if not self._no_log:
            X = np.exp(X)
        return X


class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, target='item_cnt_month', prob=0.03):
        self._prob = prob
        self._threshold = None
        self._target = target

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


class Preprocessor:
    def __init__(self, target_col, no_log_target=False, outlier_prob=0.03, **encoder_params):
        self.target_transformer = TargetTransformer(no_log=no_log_target)
        self._outlier_remover = OutlierRemover(prob=outlier_prob, target=target_col)
        self._target_col = target_col
        self._feature_prep_pipeline = Pipeline(steps=[('scaler', FeatureScaler()),
                                                      ('encoder', TargetEncoder(**encoder_params))])

    def preprocess_data(self, train_data=None, val_data=None, test_data=None):

        output = []
        if train_data is not None:
            train_x, train_y = train_data
            train_x, train_y = self._outlier_remover.fit_transform(train_x, train_y)
            train_y_transformed = self.target_transformer.fit_transform(train_y.values.reshape(-1, 1))
            train_x = self._feature_prep_pipeline.fit_transform(train_x, train_y)
            output.append(train_x)
            output.append(pd.DataFrame(train_y_transformed))
            output.append(pd.DataFrame(train_y))
        if val_data is not None:
            val_x, val_y = val_data
            val_x, val_y = self._outlier_remover.transform(val_x, val_y)
            val_y_transformed = self.target_transformer.transform(val_y.values.reshape(-1, 1))
            assert (val_x.columns == train_x.columns).all()
            assert type(val_x) == type(train_x)
            val_x = self._feature_prep_pipeline.transform(val_x)
            output.append(val_x)
            output.append(pd.DataFrame(val_y_transformed))
            output.append(pd.DataFrame(val_y))
        if test_data is not None:
            test_x = self._feature_prep_pipeline.transform(test_data)
            output.append(test_x)

        return output
