import pandas as pd
from datetime import date, timedelta


class ValidationSchema:
    def __init__(self, data):
        if 'date' not in data.columns:
            raise ValueError('Dataset doesnt have ''date'' feature')
        self._data = data
        self._train_start = data['date'].min()
        self._val_start = data['date'].max() - timedelta(days=30)  # last month
        self._val_end = data['date'].max()

    def split_data(self, train_start: date = None, val_start: date = None) -> (pd.DataFrame, pd.DataFrame):
        if train_start is not None:
            self._train_start = pd.Timestamp(train_start)
        if val_start is not None:
            self._val_start = pd.Timestamp(val_start)
            self._val_end = pd.Timestamp(val_start + timedelta(days=30))

        if train_start > val_start:
            raise ValueError('Validation fold can''t be located before training fold')
        if (val_start - train_start).days < 365:
            raise ValueError('Train start and validation start must be at least a year apart')

        return self._data[(self._data['date'] >= self._train_start) & (self._data['date'] < self._val_start)], \
            self._data[(self._data['date'] >= self._val_start) & (self._data['date'] < self._val_end)]

    def get_val(self, val_start: date) -> pd.DataFrame:
        val_start = pd.Timestamp(val_start)
        val_end = val_start + timedelta(days=30)

        return self._data[(self._data['date'] >= val_start) & (self._data['date'] < val_end)]

    @staticmethod
    def x_y_split(train: pd.DataFrame, val: pd.DataFrame, target: str) -> (pd.DataFrame, pd.DataFrame,
                                                                                pd.DataFrame, pd.DataFrame):
        return train.drop(columns=[target], axis=1), train[target], \
               val.drop(columns=[target], axis=1), val[target]
