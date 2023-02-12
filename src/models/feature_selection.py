from sklearn.inspection import permutation_importance
import pandas as pd
import matplotlib.pyplot as plt
from boruta import BorutaPy


def impurity_decrease_importance(model, col_names):
    importances = pd.Series(model.feature_importances_, index=col_names)
    fig, ax = plt.subplots()
    fig.set_size_inches(7, 5)
    importances.plot.bar(ax=ax).get_figure()
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    return model.feature_importances_, fig


def permutations_importance(model, val_x, val_y):
    r = permutation_importance(model, val_x, val_y,
                               n_repeats=30,
                               random_state=1001)
    fig, ax = plt.subplots()
    fig.set_size_inches(7, 5)
    pd.Series(r.importances_mean, index=val_x.columns).plot.bar(yerr=r.importances_std, ax=ax)
    ax.set_title("Feature importances using Permutations")
    fig.tight_layout()
    return r.importances_mean, fig


def boruta_selection(model, X, y, max_iter=50, verbose=0):
    boruta_selector = BorutaPy(model, n_estimators='auto', verbose=verbose, random_state=1001, max_iter=max_iter)
    boruta_selector.fit(X.values, y.ravel())
    accept = X.columns[boruta_selector.support_].to_list()
    irresolution = X.columns[boruta_selector.support_weak_].to_list()
    print(f'Accepted features: {accept}')
    print(f'Accepted features (weak): {irresolution}')
    return accept, irresolution