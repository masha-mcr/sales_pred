from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from xgboost import XGBRegressor
from functools import partial
from sklearn.metrics import mean_squared_error

DEFAULT_XGBR_SPACE = {
    'n_estimators': hp.quniform('n_estimators', 10, 1000, 1),
    'max_depth': hp.quniform('max_depth', 3, 18, 1),
    'grow_policy': hp.choice('grow_policy', [0, 1]),
    'learning_rate': hp.quniform('learning_rate', 0.025, 0.5, 0.025),
    'booster': 'gbtree',
    'tree_method': hp.choice('tree_method', ['exact', 'approx', 'hist']),
    'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
    'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
    'colsample_bylevel': hp.quniform('colsample_bylevel', 0.5, 1, 0.05),
    'colsample_bynode': hp.quniform('colsample_bynode', 0.5, 1, 0.05),
}


def xgbr_score(params, **data):
    model = XGBRegressor(n_estimators=int(params['n_estimators']),
                         max_depth=int(params['max_depth']),
                         learning_rate=params['learning_rate'],
                         booster=params['booster'],
                         tree_method=params['tree_method'],
                         gamma=params['gamma'],
                         min_child_weight=int(params['min_child_weight']),
                         subsample=params['subsample'],
                         colsample_bytree=params['colsample_bytree'],
                         colsample_bylevel=params['colsample_bylevel'],
                         colsample_bynode=params['colsample_bynode'],
                         random_state=1001
                         )

    model.fit(data['t_x'], data['t_y'])
    pred = model.predict(data['v_x'])
    mse = mean_squared_error(data['v_y'], pred, squared=False)
    return {'loss': mse, 'status': STATUS_OK, 'model': model}


def bayesian_tuning(space, **data):

    trials = Trials()
    best = fmin(fn=partial(xgbr_score, **data),
                space=space,
                algo=tpe.suggest,
                max_evals=50,
                trials=trials)

    return best
