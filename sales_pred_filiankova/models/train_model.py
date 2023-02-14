import pandas as pd
import numpy as np
import os
import sys

from sklearn.metrics import mean_squared_error


def fit_eval_model(model, t_x, t_y, v_x, v_y, target_transformer):
    model.fit(t_x, t_y.ravel())
    train_pred = target_transformer.inverse_transform(model.predict(t_x))
    train_rmse = mean_squared_error(train_y, train_pred, squared=False)
    val_pred = target_transformer.inverse_transform(model.predict(v_x))
    val_rmse = mean_squared_error(val_y, val_pred, squared=False)
    print(f'train RMSE - {np.round(train_rmse, 3)}, val RMSE - {np.round(val_rmse,3)}')
    return model, train_rmse, val_rmse


if __name__ == '__main__':
    train_x = pd.read_csv(os.path.join(sys.argv[1], 'train_x.csv'))
    val_x = pd.read_csv(os.path.join(sys.argv[1], 'val_x.csv'))
    train_y = pd.read_csv(os.path.join(sys.argv[1], 'train_y.csv'))
    val_y = pd.read_csv(os.path.join(sys.argv[1], 'val_y.csv'))




