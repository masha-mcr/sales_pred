from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from datetime import date
from tqdm import tqdm


class MonthFromDate(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['month'] = X['date_block_num'] % 12 + 1
        X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12.0)
        X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12.0)
        X.drop(['month'], axis=1, inplace=True)
        print('Month encoding stage complete')
        return X


class ShopItemAge(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['shop_age_month'] = (X['date'] - X['first_shop_appearance']).dt.days / 30
        X['item_age_month'] = (X['date'] - X['first_item_appearance']).dt.days / 30
        X.drop(['first_shop_appearance', 'first_item_appearance', 'date'], axis=1, inplace=True)

        assert all(X['shop_age_month'] >= 0)
        assert all(X['item_age_month'] >= 0)
        assert not X.isna().any().any()

        # print('Item and shop ages calculation complete')
        return X


class MonthlySales(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.groupby(['date_block_num', 'item_id', 'shop_id']).agg({'item_cnt_day': 'sum',
                                                                     'date': 'max',
                                                                     }).reset_index().rename(
            {'item_cnt_day': 'item_cnt_month'}, axis=1)
        # print('Monthly sales calculated')
        return X


class MonthlySalesTest(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['date_block_num'] = 34
        X['date'] = date(2015, 34 % 12 + 1, 28)
        X['date'] = pd.to_datetime(X['date'], format="%Y-%m-%d")
        # print('Monthly sales calculated')
        return X


class MergeTables(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        if not all(df in kwargs.keys() for df in ['items', 'shops', 'item_categories']):
            raise ValueError('Not enough dataframes provided')
        self._items = kwargs['items']
        self._shops = kwargs['shops']
        self._item_cat = kwargs['item_categories']

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        old_shape = X.shape[0]
        X = X.merge(self._items, on='item_id')  # , how='left'
        assert X.shape[0] == old_shape
        X = X.merge(self._shops, on='shop_id')  # , how='left'
        assert X.shape[0] == old_shape
        X = X.merge(self._item_cat, on='item_category_id')  # , how='left'
        assert X.shape[0] == old_shape
        assert not X.isna().any().any()

        X.drop([col for col in X.columns if X[col].dtype == object], axis=1, inplace=True)
        # print('Merging stage complete')
        return X


# Lag feature transformers


class GetItemSalesLag(BaseEstimator, TransformerMixin):
    def __init__(self, lags, lag_data):
        self._lags = lags
        self._lag_data = lag_data

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        date_block = X['date_block_num'].values[0]
        old_shape = X.shape[0]
        X = X.merge(self._lag_data, on=['item_id', 'shop_id'], how='left')  #
        assert old_shape == X.shape[0]
        X = X.fillna(0)

        for lag in self._lags:
            if date_block - lag >= 0:
                X[f'item_sales_lag_{lag}'] = X.loc[:, str(int(date_block - lag))]
            else:
                X[f'item_sales_lag_{lag}'] = 0

        return X


class GetPriceLag(BaseEstimator, TransformerMixin):
    def __init__(self, lags, lag_data):
        self._lags = lags
        self._lag_data = lag_data

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        date_block = X['date_block_num'].values[0]
        old_shape = X.shape[0]
        X = X.merge(self._lag_data, on=['item_id', 'shop_id'], how='left')  #
        assert old_shape == X.shape[0]
        X = X.fillna(0)
        for lag in self._lags:
            if date_block - lag >= 0:
                X[f'price_lag_{lag}'] = X.loc[:, str(int(date_block - lag))]
            else:
                X[f'price_lag_{lag}'] = 0

        X.drop([str(i) for i in range(34)], axis=1, inplace=True)
        return X


class GetTotalSalesLag(BaseEstimator, TransformerMixin):
    def __init__(self, lags, lag_data):
        self._lags = lags
        self._lag_data = lag_data

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        date_block = X['date_block_num'].values[0]
        old_shape = X.shape[0]
        X = X.merge(self._lag_data, on=['shop_id'], how='left')  #
        assert X.shape[0] == old_shape
        X = X.fillna(0)
        for lag in self._lags:
            if date_block - lag >= 0:
                X[f'total_sales_lag_{lag}'] = X.loc[:, str(int(date_block - lag))]
            else:
                X[f'total_sales_lag_{lag}'] = 0

        X.drop([str(i) for i in range(34)], axis=1, inplace=True)
        return X


class GetShopVarietyLag(BaseEstimator, TransformerMixin):
    def __init__(self, lags, lag_data):
        self._lags = lags
        self._lag_data = lag_data

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        date_block = X['date_block_num'].values[0]
        old_shape = X.shape[0]
        X = X.merge(self._lag_data, on=['shop_id'], how='left')
        assert X.shape[0] == old_shape
        X = X.fillna(0)
        for lag in self._lags:
            if date_block - lag >= 0:
                X[f'shop_variety_lag_{lag}'] = X.loc[:, str(int(date_block - lag))]
            else:
                X[f'shop_variety_lag_{lag}'] = 0

        X.drop([str(i) for i in range(34)], axis=1, inplace=True)
        return X


class GetItemSpreadLag(BaseEstimator, TransformerMixin):
    def __init__(self, lags, lag_data):
        self._lags = lags
        self._lag_data = lag_data

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        date_block = X['date_block_num'].values[0]
        old_shape = X.shape[0]
        X = X.merge(self._lag_data, on=['item_id'], how='left')  #
        assert X.shape[0] == old_shape
        X = X.fillna(0)
        for lag in self._lags:
            if date_block - lag >= 0:
                X[f'item_spread_lag_{lag}'] = X.loc[:, str(int(date_block - lag))]
            else:
                X[f'item_spread_lag_{lag}'] = 0

        X.drop([str(i) for i in range(34)], axis=1, inplace=True)
        return X


class GetSumOfSales(BaseEstimator, TransformerMixin):
    def __init__(self, lags):
        self._lags = lags

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        date_block = X['date_block_num'].values[0]
        for lag in self._lags:
            cols_to_sum = [str(int(date_block - i)) for i in range(1, lag + 1) if date_block - i >= 0]
            X[f'sum_sales_lag_{lag}'] = X.loc[:, cols_to_sum].sum(axis=1)

        X.drop(['date_block_num'], axis=1, inplace=True)
        X.drop([str(i) for i in range(34)], axis=1, inplace=True)
        return X


class FeatureExtractor:
    def __init__(self, feature_datasets: dict, lag_datasets: dict):
        self._feature_datasets = feature_datasets
        self._lag_datasets = lag_datasets

        self._extraction_pipeline_train = self.init_extraction_pipeline(test_set=False)
        self._extraction_pipeline_test = self.init_extraction_pipeline(test_set=True)
        self._lag_extraction_pipeline = self.init_lag_features_pipeline()

    def init_extraction_pipeline(self, test_set: bool = False) -> Pipeline:
        extraction_steps = [
            ("monthly_sales", MonthlySales()) if not test_set else ("monthly_sales", MonthlySalesTest()),
            ("month_feat", MonthFromDate()),
            ("merge", MergeTables(**self._feature_datasets)),
            ("ages", ShopItemAge())]

        return Pipeline(steps=extraction_steps)

    def init_lag_features_pipeline(self) -> Pipeline:
        lag_extraction_steps = [
            ("shop_variety", GetShopVarietyLag(lags=[1, 2, 3], lag_data=self._lag_datasets['shop_variety'])),
            ("item_spread", GetItemSpreadLag(lags=[1, 2, 3], lag_data=self._lag_datasets['item_spread'])),
            ("past_month_price", GetPriceLag(lags=[1, 2, 3], lag_data=self._lag_datasets['item_price'])),
            ("past_month_total_sales", GetTotalSalesLag(lags=[1, 3, 6], lag_data=self._lag_datasets['total_sales'])),
            (
                "past_month_item_sales",
                GetItemSalesLag(lags=[1, 2, 3, 6, 12], lag_data=self._lag_datasets['item_sales'])),
            ("sum_of_sales", GetSumOfSales(lags=[6, 12]))
        ]

        return Pipeline(steps=lag_extraction_steps)

    def extract_features(self, train_data: pd.DataFrame=None, val_data: pd.DataFrame=None,
                               test_data: pd.DataFrame=None) -> (pd.DataFrame, pd.DataFrame):
        output = []
        if train_data is not None:
            train = self._extraction_pipeline_train.fit_transform(train_data)
            train_per_month = [
                self._lag_extraction_pipeline.fit_transform(train.loc[train['date_block_num'] == block].copy())
                for block in tqdm(train['date_block_num'].unique(), desc='Lag features extraction')]
            train = pd.concat(train_per_month, axis=0)
            output.append(train)
        if val_data is not None:
            val = self._extraction_pipeline_train.transform(val_data)
            val = self._lag_extraction_pipeline.transform(val)
            output.append(val)
        if test_data is not None:
            test = self._extraction_pipeline_test.fit_transform(test_data)
            test = self._lag_extraction_pipeline.transform(test)
            output.append(test)

        return output

